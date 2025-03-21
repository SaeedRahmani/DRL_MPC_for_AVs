""" Pure MPC agent without collision avoidance """

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env
from shapely import LineString

from .base_agent import Agent
from .utils import MPC_Action, Vehicle

class PureMPC_Agent(Agent):
    weight_components = [
        "state",
        "speed", 
        "control", 
        # "distance", 
        # "collision", 
        "input_diff"
        # "final_state"
    ]

    def __init__(
        self, 
        env: Env,
        cfg: dict, 
    ) -> None:
        """
        Initializer.
        
        Args:
            env: gym.Env, a highway-env gymnasium environment to retrieve the configuration.
            cfg: configuration dict for pure mpc agent.
        """
        super().__init__(env, cfg)

        # Load weights for each cost component from config
        self.default_weights = {
            f"weight_{key}": self.config[f"weight_{key}"]
            for key in PureMPC_Agent.weight_components
        }

        # Initialize the matplotlib fig and ax if rendering is enabled
        if self.config.get("render", False):
            window_size = self.config.get("render_window_size", 5)
            self.fig, self.ax = plt.subplots(figsize=(window_size, window_size))
            # Set the position of the figure window (x, y)
            manager = plt.get_current_fig_manager()
            manager.window.wm_geometry("+50+50")  
        
        self.last_acc = 0

    def __str__(self) -> str:
        return "Pure MPC agent [Receding Horizon Control], Solved by `CasADi` "
      
    def predict(
            self, 
            obs, 
            return_numpy = True,
            weights_from_RL = None,
            ref_speed = None,
        ):
        self._parse_obs(obs)
        mpc_action = self._solve(weights_from_RL, ref_speed)
        return mpc_action.numpy() if return_numpy else mpc_action
      
    def _solve(self, weights_from_RL: dict[str, float]=None, ref_speed_from_RL=None) -> MPC_Action:
        # MPC parameters
        N = self.horizon
        
        # Define symbolic variables for states and controls
        n_states = 4    # [x, y, theta, v]
        n_controls = 2  # [acceleration, steer_angle]
        
        # Create symbolic variables for state and control trajectories
        x = ca.SX.sym('x', n_states, N + 1)  # State trajectory over N + 1 time steps
        u = ca.SX.sym('u', n_controls, N)    # Control inputs over N time steps

        """ Update weights """
        if weights_from_RL is None:
            # Use default weights from configuration file
            weights = self.default_weights
        else:
            # Use dynamic weights from RL agent
            weights = {
                f"weight_{key}": weights_from_RL[0, i] 
                for i, key in enumerate(PureMPC_Agent.weight_components)
            }

        # Get the index on the reference trajectory for ego vehicle
        self.ego_index = np.argmin(
            [np.linalg.norm(self.ego_vehicle.position - trajectory_point) 
             for trajectory_point in self.reference_trajectory]
        )
        
        # Update reference speed from RL if provided
        ref = np.copy(self.reference_states)
        if ref_speed_from_RL is not None:
            safe_speed = np.clip(ref_speed_from_RL[0,0], 0, 30.0)  # Clip between 0 and max speed
            ref[:, 2] = safe_speed
        
        closest_index = self.ego_index

        # Define the cost function
        total_cost = 0
        state_cost = 0
        control_cost = 0
        input_diff_cost = 0
        final_state_cost = 0
        
        for k in range(N):
            ref_traj_index = min(closest_index + k, ref.shape[0] - 1)

            dx = x[0, k] - ref[ref_traj_index,0]
            dy = x[1, k] - ref[ref_traj_index,1]

            ref_v = ref[ref_traj_index,2]
            ref_heading = ref[ref_traj_index,3]
            perp_deviation = dx * ca.sin(ref_heading) - dy * ca.cos(ref_heading)
            para_deviation = dx * ca.cos(ref_heading) + dy * ca.sin(ref_heading)
            
            # State cost
            state_cost += (
                4 * perp_deviation**2 + 
                2 * para_deviation**2 +
                weights["weight_speed"] * (x[3, k] - ref_v)**2 + 
                0.5 * (x[2, k] - ref_heading)**2
            )

            # Control cost
            control_cost += 0.01 * u[0, k]**2 + 0.01 * u[1, k]**2
            
            # Input difference cost
            if k > 0:
                input_diff_cost += 0.01 * ((u[0, k] - u[0, k-1])**2 + (u[1, k] - u[1, k-1])**2)

        # final state cost
        ref_traj_index = min(closest_index + N, ref.shape[0] - 1)
        desired_final_state = ref[ref_traj_index, :]        
        final_state_cost += 100 * (
            (x[0, -1] - desired_final_state[0])**2 + 
            (x[1, -1] + desired_final_state[1])**2 + 
            20 * (x[3, -1] - desired_final_state[2])**2 +    # ref speed
            (x[2, -1] - desired_final_state[3])**2      # heading angle
        )

        total_cost = (
            state_cost * weights["weight_state"] +
            control_cost * weights["weight_control"] +
            input_diff_cost * weights["weight_input_diff"] 
            # final_state_cost * weights["weight_final_state"]
        )

        # Define the vehicle dynamics using the Kinematic Bicycle Model
        def vehicle_model(x, u):
            beta = ca.atan(Vehicle.LENGTH_REAR / Vehicle.LENGTH * ca.tan(u[1]))  # Slip angle
            x_next = ca.vertcat(
                x[3] * ca.cos(x[2] + beta),                 # x_dot
                x[3] * ca.sin(x[2] + beta),                 # y_dot
                (x[3] / Vehicle.LENGTH) * ca.sin(beta),     # theta_dot
                u[0]                                        # v_dot (acceleration)
            )
            return x_next
        
        # Constraints
        g = []  # Constraints vector
        
        state = np.array([
            self.ego_vehicle.position[0], 
            self.ego_vehicle.position[1], 
            self.ego_vehicle.heading, 
            self.ego_vehicle.speed
        ])

        x0_states = np.tile(state, (N + 1, 1)).flatten()
        u0_controls = np.zeros(n_controls * N)
        x0 = np.concatenate((x0_states, u0_controls))

        # Initial condition constraint
        g.append(x[:, 0] - state)
    
        # State-update constraints for the entire horizon
        for k in range(N):
            x_next = x[:, k] + vehicle_model(x[:, k], u[:, k]) * self.dt
            g.append(x[:, k + 1] - x_next)
        
        # Flatten constraints
        g = ca.vertcat(*g)

        # Optimization variables
        opt_variables = ca.vertcat(ca.reshape(x, -1, 1), ca.reshape(u, -1, 1))

        # Define bounds
        lbg = [0] * g.size1()
        ubg = [0] * g.size1()

        # Bounds on state and control variables
        lbx = []
        ubx = []
        
        for _ in range(N + 1):
            lbx += [-500, -500, -ca.pi, 0]
            ubx += [500, 500, ca.pi, 30]

        for _ in range(N):
            lbx += [-5, -ca.pi / 3]
            ubx += [5, ca.pi / 3]
        
        # Create and solve the optimization problem
        nlp = {
            'x': opt_variables,
            'f': total_cost,
            'g': g,
        }
        
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0,
            'ipopt.max_iter': 1000,
            'ipopt.tol': 1e-6,
        }
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        
        if not solver.stats()['success']:
            print("WARNING: Optimization failed to find a solution")
            
        u_opt = sol['x'][-N * n_controls:].full().reshape(N, n_controls)
        self.last_acc = u_opt[0, 0]
        return MPC_Action(acceleration=u_opt[0, 0], steer=u_opt[0, 1])
    
    def plot(self):
        """ Visualize the MPC solving process """
        x_range = self.config.get("render_axis_range", 50)
        
        self.ax.clear()
        
        self.ax.set_xlim([-x_range, x_range])
        self.ax.set_ylim([-x_range, x_range])
        self.ax.invert_yaxis()
        self.ax.grid(True)

        self.ax.set_xlabel("X (meter)")
        self.ax.set_ylabel("Y (meter)")
        self.ax.set_title("Pure MPC Agent")

        # Plot reference trajectory
        self.ax.scatter(
            x=self.reference_trajectory[:, 0], 
            y=self.reference_trajectory[:, 1], 
            color='grey', 
            s=1,
        )
        
        # Plot ego vehicle
        ego_x, ego_y = self.ego_vehicle.position
        self.ax.scatter(
            x=ego_x, 
            y=ego_y, 
            color='blue', 
            s=5
        )

        # Plot other vehicles
        for agent_vehicle in self.agent_vehicles:
            agent_x, agent_y = agent_vehicle.position
            self.ax.scatter(
                x=agent_x, 
                y=agent_y, 
                color='red', 
                s=5
            )

        plt.pause(0.1)