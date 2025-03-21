import copy
import numpy as np
import highway_env
import casadi as ca

from highway_env.envs import IntersectionEnv
from highway_env.envs.common.action import (
    Action,
    PureMpcAction,
    DynamicWeightsAction
)
from highway_env.envs.common.abstract import Observation
from src_ray_version.utils.vehicle import Vehicle
# from utils.vehicle import Vehicle


class IntersectionMpcEnv(IntersectionEnv):
    """ An intersection environment with MPC solver inside. """
    def __init__(self, config: dict = None, render_mode: str | None = None):
        super().__init__(config=config, render_mode=render_mode)
        # MPC parameters
        self.horizon: int = 16
        self.dt: float = 0.1 
        self.weight_components = [
            "state",
            "speed",
            "control",
            # "distance",
            # "collision",
            "input_diff"
            # "final_state"
        ]
        self.default_weights = {
            "weight_speed": 1,
            "weight_control": 1,
            "weight_final_state": 1,
            "weight_input_diff": 1,
            "weight_distance": 10,
            "weight_collision": 1,
            "weight_state": 10,
        }

        self.reference_trajectory = self.reference_states[:, :2]
        self.last_acc = 0

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 10,
                    "features": ["presence", "x", "y", "vx", "vy", "heading", "sin_h", "cos_h"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20],
                        "heading": [-1 * np.pi, np.pi],
                        "sin_h": [-1, 1],
                        "cos_h": [-1, 1],
                    },
                    "absolute": True,
                    "normalize": False,
                    "order": "sorted",
                },
                "action": {
                    "type": "PureMpcAction",
                    "acceleration_range": [-5.0, 5.0],
                    "steering_range": [-np.pi / 4, np.pi / 4],
                },
                # vehicle spawning
                "vehicles_count": 10,
                "initial_vehicle_count": 5,
                "spawn_probability": 0.3,
                # time
                "duration": 200,            # [s]
                "policy_frequency": 10,
                "simulation_frequency": 30,
                # rendering
                "scaling": 3,
                "screen_width": 600,
                "screen_height": 600,
                # MPC
                "horizon": 16,                
            }
        )
        return config

    def __str__(self) -> str:
        return f"<IntersectionMpcEnv instance>"

    def __repr__(self) -> str:
        return self.__str__()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[Observation, dict]:
        """
        Reset the environment to it's initial configuration

        :param seed: The seed that is used to initialize the environment's PRNG
        :param options: Allows the environment configuration to specified through `options["config"]`
        :return: the observation of the reset state
        """
        super().reset(seed=seed, options=options)
        if options and "config" in options:
            self.configure(options["config"])
        self.update_metadata()
        # First, to set the controlled vehicle class depending on action space
        self.define_spaces()

        self.time_index = 0
        self.current_speed_idx = 0

        self.time = self.steps = 0
        self.done = False
        self._reset()
        # Second, to link the obs and actions to the vehicles once the scene is created
        self.define_spaces()
        obs = self.observation_type.observe()
        info = self._info(obs, action=self.action_space.sample())
        if self.render_mode == "human":
            self.render()
        self.current_observation = obs
        return obs, info

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError(
                "The road and vehicle must be initialized in the environment implementation"
            )

        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)
        self.time_index += 1

        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)
        if self.render_mode == "human":
            self.render()

        self.current_observation = obs

        return obs, reward, terminated, truncated, info

    def _simulate(self, action: Action | None = None) -> None:
        """Perform several steps of simulation with constant action."""
        frames = int(
            self.config["simulation_frequency"] // self.config["policy_frequency"]
        )

        # MPCRL
        mpc_action = self._predict_mpc_action(action)
        # print(mpc_action)

        for frame in range(frames):
            # Forward action to the vehicle
            if (
                action is not None
                and not self.config["manual_control"]
                and self.steps
                % int(
                    self.config["simulation_frequency"]
                    // self.config["policy_frequency"]
                )
                == 0
            ):

                self.action_type.act(mpc_action)

            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.steps += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            if (
                frame < frames - 1
            ):  # Last frame will be rendered through env.render() as usual
                self._automatic_rendering()

        self.enable_auto_render = False

    def _predict_mpc_action(self, action: Action) -> Action:
        """ Predict the action of ego vehicle using MPC. """
        self._prepare_obs()

        test_speeds = [5.0, 10.0, 15.0]  # Test three different speeds
        steps_per_speed = 33  # Change speed every 33 steps
        if self.time_index % steps_per_speed == 0 and self.current_speed_idx < len(test_speeds):
            self.ref_speed = test_speeds[self.current_speed_idx]
            self.current_speed_idx = (self.current_speed_idx + 1) % len(test_speeds)

        if action is None:
            mpc_action = self._solve_mpc(weights=None, ref_speed=np.array([[self.ref_speed]]))
        # elif isinstance(self.action_type, DynamicWeightsAction):
        #     weights = action
        #     mpc_action = self._solve_mpc(weights=weights, ref_speed=None)
        # elif isinstance(self.action_type, ReferenceSpeedAction):
        #     reference_speed = action
        #     mpc_action = self._solve_mpc(weights=None, ref_speed=reference_speed)
        else:
            raise TypeError("Wrong self.action_type")
        return mpc_action

    def _solve_mpc(
        self,
        weights: np.array,
        ref_speed: np.array
    ) -> np.ndarray:
        """ 
        Use Casadi to solve the MPC. 

        @arguments:
        - weights: the dynamics weights of each component recommanded by RL agent (v1).
        - ref_speed: the reference speed recommended by RL agent (v0).

        @return:
        - mpc_action: np.ndarray with shape of (2,), consisted of acceleration and steering.
        """
        # MPC parameters
        N = self.horizon

        # Define symbolic variables for states and controls
        n_states = 4    # [x, y, theta, v]
        n_controls = 2  # [acceleration, steer_angle]

        # Create symbolic variables for state and control trajectories
        # State trajectory over N + 1 time steps
        x = ca.SX.sym('x', n_states, N + 1)
        u = ca.SX.sym('u', n_controls, N)    # Control inputs over N time steps

        """ Update weights """
        if weights is None:
            # Use default weights from configuration file
            weights_dict = self.default_weights
        else:
            # Use dynamic weights from RL agent
            weights_dict = {
                f"{key}": weights[i]
                for i, key in enumerate(self.weight_components)
            }

        # Get the index on the reference trajectory for ego vehicle
        self.ego_index = np.argmin(
            [np.linalg.norm(self.ego_vehicle.position - trajectory_point) 
             for trajectory_point in self.reference_trajectory]
        )
        
        # Update reference speed from RL if provided
        ref = np.copy(self.reference_states)
        if ref_speed is not None:
            safe_speed = np.clip(ref_speed[0,0], 0, 30.0)  # Clip between 0 and max speed
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

            dx = x[0, k] - ref[ref_traj_index, 0]
            dy = x[1, k] - ref[ref_traj_index, 1]

            ref_v = ref[ref_traj_index,2]
            ref_heading = ref[ref_traj_index,3]
            perp_deviation = dx * ca.sin(ref_heading) - dy * ca.cos(ref_heading)
            para_deviation = dx * ca.cos(ref_heading) + dy * ca.sin(ref_heading)
 
            # State cost
            state_cost += (
                4 * perp_deviation**2 +
                2 * para_deviation**2 +
                weights_dict["weight_speed"] * (x[3, k] - ref_v)**2 +
                0.5 * (x[2, k] - ref_heading)**2
            )

            # Control cost
            control_cost += 0.01 * u[0, k]**2 + 0.01 * u[1, k]**2

            # Input difference cost
            if k > 0:
                input_diff_cost += 0.01 * \
                    ((u[0, k] - u[0, k-1])**2 + (u[1, k] - u[1, k-1])**2)

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
            state_cost * weights_dict["weight_state"] +
            control_cost * weights_dict["weight_control"] +
            input_diff_cost * weights_dict["weight_input_diff"]
            # final_state_cost * weights_dict["weight_final_state"]
        )

        # Define the vehicle dynamics using the Kinematic Bicycle Model
        def vehicle_model(x, u):
            beta = ca.atan(Vehicle.LENGTH_REAR /
                           Vehicle.LENGTH * ca.tan(u[1]))  # Slip angle
            x_next = ca.vertcat(
                x[3] * ca.cos(x[2] + beta),                 # x_dot
                x[3] * ca.sin(x[2] + beta),                 # y_dot
                (x[3] / Vehicle.LENGTH) * ca.sin(beta),     # theta_dot
                # v_dot (acceleration)
                u[0]
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

        acceleration, steering = u_opt[0, 0], u_opt[0, 1]
        # mpc_action = np.array([acceleration / 5, steering / (np.pi / 3)])
        mpc_action = np.array([acceleration, steering])
        return mpc_action

    def _prepare_obs(self):
        """
        Parse the observation, a collection of KinematicObservation for MPC modelling.

        Args:
            obs: np.ndarray, a collection of KinematicObservation directly received from the interacting environment.
        """
        if not isinstance(self.current_observation, np.ndarray):
            raise TypeError(
                f"Expect observation type np.ndarray, but got {type(self.current_observation)}.")
        if self.current_observation.shape != (self.config["observation"]['vehicles_count'], 8):
            raise ValueError(
                f"Expect observation's shape of ({(self.config["observation"]['vehicles_count'], 8)}), but got {self.current_observation.shape}")

        self.observed_vehicles_count = np.sum(
            self.current_observation[:, 0] == 1) - 1

        # Ego vehicle
        self.ego_vehicle = Vehicle(
            index=0,
            position=self.current_observation[0, 1:3],
            vectorized_speed=self.current_observation[0, 3:5],
            heading=self._normalize_angle(self.current_observation[0, 5]),
            sinh=self.current_observation[0,6], 
            cosh=self.current_observation[0, 7],
        )

        # Agent vehicles
        self.agent_vehicles = list()
        if self.observed_vehicles_count > 0:
            for i in range(self.observed_vehicles_count):
                self.agent_vehicles.append(Vehicle(
                    index=i+1,
                    position=self.current_observation[i+1, 1:3],
                    vectorized_speed=self.current_observation[i+1, 3:5],
                    heading=self._normalize_angle(self.current_observation[i+1, 5]),
                    sinh=self.current_observation[0,6], 
                    cosh=self.current_observation[0, 7],
                ))
            assert len(self.agent_vehicles) == self.observed_vehicles_count
        self.agent_vehicles_mpc = copy.deepcopy(self.agent_vehicles)

    def _normalize_angle(self, angle):
        """
        Normalize an angle to the range [-pi, pi].

        Parameters:
            angle (float): The angle to be normalized in radians.

        Returns:
            float: The normalized angle in the range [-pi, pi].
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    @property
    def reference_states(self):
        trajectory = []
        x, y, v, heading, v_ref = 2, 50, 10, -np.pi/2, 10  # Starting with 10 m/s speed
        turn_start_y = 20
        radius = 5  # Radius of the curve
        # Total angle to turn (90 degrees for a left turn)
        turn_angle = np.pi / 2

        # Go straight until reaching the turn start point
        for _ in range(40):
            # if collision_points is not None and (x, y) in collision_points:
            #     v = 0  # Set speed based on DRL agent's decision
            x += 0
            y += v * self.dt * np.sin(heading)
            trajectory.append((x, y, v, heading))

        # Compute the turn
        angle_increment = turn_angle / 20  # Divide the turn into 20 steps
        for _ in range(20):
            # if collision_points is not None and (x, y) in collision_points:
            #     v = 0  # Set speed based on DRL agent's decision
            heading -= angle_increment  # Decrease heading to turn left
            x += v * self.dt * np.cos(heading)
            y += v * self.dt * np.sin(heading)

            trajectory.append((x, y, v, heading))

        # Continue straight after the turn
        for _ in range(25):  # Continue for a bit after the turn
            # if collision_points is not None and (x, y) in collision_points:
            #     v = 0  # Set speed based on DRL agent's decisions
            x += v * self.dt * np.cos(heading)
            y += 0

            trajectory.append((x, y, v, heading))

        return np.array(trajectory)


# class IntersectionMpcrlEnv_v0(IntersectionMpcEnv):

#     def __init__(self, config: dict = None, render_mode: str | None = None):
#         super().__init__(config=config, render_mode=render_mode)

#     @classmethod
#     def default_config(cls) -> dict:
#         config = super().default_config()
#         config.update(
#             {
#                 "observation": {
#                     "type": "Kinematics",
#                     "vehicles_count": 10,
#                     "features": ["presence", "x", "y", "vx", "vy", "heading", "sin_h", "cos_h"],
#                     "features_range": {
#                         "x": [-100, 100],
#                         "y": [-100, 100],
#                         "vx": [-20, 20],
#                         "vy": [-20, 20],
#                         "heading": [-1 * np.pi, np.pi],
#                         "sin_h": [-1, 1],
#                         "cos_h": [-1, 1],
#                     },
#                     "absolute": True,
#                     "flatten": False,
#                     "observe_intentions": False,
#                 },
#                 "action": {
#                     "type": "ReferenceSpeedAction",
#                 },
#                 "vehicles_count": 10,
#                 "horizon": 16
#             }
#         )
#         return config

#     def __str__(self) -> str:
#         return f"<IntersectionMpcrlEnv-v0:Reference_speed instance>"

#     def __repr__(self) -> str:
#         return self.__str__()


class IntersectionMpcrlEnv_v1(IntersectionMpcEnv):
    """ 
    MPCRL: Dynamic weights
    """

    def __init__(self, config: dict = None, render_mode: str | None = None):
        super().__init__(config=config, render_mode=render_mode)

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 10,
                    "features": ["presence", "x", "y", "vx", "vy", "heading", "sin_h", "cos_h"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20],
                        "heading": [-1 * np.pi, np.pi],
                        "sin_h": [-1, 1],
                        "cos_h": [-1, 1],
                    },
                    "absolute": True,
                    "flatten": False,
                    "observe_intentions": False,
                },
                "action": {
                    "type": "DynamicWeightsAction",
                    "num_weights": 3,
                },
                "vehicles_count": 10,
                "horizon": 16
            }
        )
        return config

    def __str__(self) -> str:
        return f"<IntersectionMpcrlEnv-v1:Dynamic_weights instance>"

    def __repr__(self) -> str:
        return self.__str__()
