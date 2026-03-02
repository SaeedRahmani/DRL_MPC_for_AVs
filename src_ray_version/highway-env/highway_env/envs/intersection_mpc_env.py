import copy
import functools
import numpy as np
import highway_env
import casadi as ca
from shapely import LineString
from shapely.errors import GEOSException
import pandas as pd
from highway_env.envs import IntersectionEnv
from highway_env.envs.common.action import Action
from highway_env.envs.common.abstract import Observation
# from src_ray_version.utils.vehicle import Vehicle
from utils.vehicle import Vehicle


class IntersectionMpcEnv_noCA(IntersectionEnv):
    """ MPC: without collision avoidance. """

    def __init__(self, config: dict = None, render_mode: str | None = None):
        super().__init__(config=config, render_mode=render_mode)

        # MPC parameters
        self.horizon: int = 16
        self.dt: float = 0.1

        # Collision avoidance disable by default
        self.manual_collision_avoidance = False
        self.CA_mode = "noCA"
        self.is_collide = False
        # assert self.CA_mode == "noCA", "Expect CA mode to be `noCA`."
        self.agent_mode = "Pure_MPC"

        self.weight_components = [
            "state",
            "speed",
            "control",
            "input_diff",
            "final_state",
            # "distance",
            # "collision",
        ]
        self.all_default_weights = {
            "weight_speed": 1,
            "weight_control": 1,
            "weight_final_state": 1,
            "weight_input_diff": 1,
            "weight_distance": 10,
            "weight_collision": 1,
            "weight_state": 10,
        }
        self.default_weights = {f"weight_{weight_name}": self.all_default_weights[f"weight_{weight_name}"]
                                for weight_name in self.weight_components}

        self.reference_trajectory = self.reference_states[:, :2]
        self.last_acc = 0

        # Collision detection parameters
        self.collision_memory = 0           # Add collision memory counter
        self.collision_memory_steps = 10    # How many steps to remember collision
        self.memorized_conflict_points = None
        self.memorized_conflict_indices = None
        self.last_valid_stop_point = None
        # Load collision detection parameters from config
        # self.detection_dist = self.config.get("detection_distance", 100)
        self.ttc_threshold = self.config.get("ttc_threshold", 3)
        self.speed_override = self.config.get("speed_override", 0)

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
                "initial_vehicle_count": 10,
                "spawn_probability": 0.6,
                # time
                "duration": 13,            # [s]
                "policy_frequency": 10,      # 10,
                "simulation_frequency": 30, # 30,
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
        return f"<Intersection-Env {self.agent_mode} [{self.CA_mode}]>"

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
        mpc_action = self._simulate(action)
        self.time_index += 1

        obs = self.observation_type.observe()
        reward = self._reward(mpc_action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, mpc_action)
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

    # Maximum reference speed the RL agent can command (m/s)
    MAX_REF_SPEED = 15.0

    def _predict_mpc_action(self, action: Action) -> Action:
        """ Predict the action of ego vehicle using MPC. """
        self._prepare_obs()

        # Always run collision check so conflict info is available for obs/reward
        if self.CA_mode != "noCA":
            self._check_collision()
        elif self.agent_mode == "MPC-RL<Reference speed>":
            # For RL-based CA: detect conflicts (for observation) but don't act on them in MPC
            self._check_collision()

        if self.agent_mode == "Pure_MPC":
            mpc_action = self._solve_mpc(
                weights=None, ref_speed=None)
        elif self.agent_mode == "MPC-RL<Reference speed>":
            # Scale RL action from [-1, 1] → [0, MAX_REF_SPEED] m/s
            raw_action = float(action[0]) if hasattr(action, '__len__') else float(action)
            scaled_speed = (raw_action + 1.0) / 2.0 * self.MAX_REF_SPEED
            ref_speed = np.clip(scaled_speed, 0.0, self.MAX_REF_SPEED)
            mpc_action = self._solve_mpc(weights=None, ref_speed=np.array([[ref_speed]]))
        elif self.agent_mode == "MPC-RL<Dynamic weights>":
            weights = action
            mpc_action = self._solve_mpc(
                weights=weights, ref_speed=None)
        else:
            raise ValueError(
                f"Wrong agent mode received: `{self.agent_mode}`.")

        return mpc_action.astype(np.float32)

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
                key: weights[i]
                for i, key in enumerate(self.default_weights.keys())
            }

        # Get the index on the reference trajectory for ego vehicle
        self.ego_index = np.argmin(
            [np.linalg.norm(self.ego_vehicle.position - trajectory_point)
             for trajectory_point in self.reference_trajectory]
        )

        # if self.manual_collision_avoidance:
        if self.CA_mode == "manual":
            # Generate new reference states, given the result of collision detection
            # When RL provides ref_speed, update_reference_states now blends it
            # with the CA deceleration profile (mean of the two).
            ref = self.update_reference_states(
                speed_override=self.speed_override,
                speed_overide_from_RL=ref_speed)
            
        else:
            # Update reference speed from RL if provided
            ref = np.copy(self.reference_states)

            if ref_speed is not None:
                # Clip between 0 and max speed
                safe_speed = np.clip(ref_speed[0, 0], 0, 30.0)
                ref[:, 2] = safe_speed
        
        closest_index = self.ego_index

        # Define the cost function
        total_cost = 0
        state_cost = 0
        control_cost = 0
        input_diff_cost = 0
        final_state_cost = 0

        # if self.manual_collision_avoidance:
        if self.CA_mode == "manual" or self.CA_mode == "cost":
            distance_cost = 0
            collision_cost = 0

        for k in range(N):
            ref_traj_index = min(closest_index + k, ref.shape[0] - 1)

            dx = x[0, k] - ref[ref_traj_index, 0]
            dy = x[1, k] - ref[ref_traj_index, 1]

            ref_v = ref[ref_traj_index, 2]
            ref_heading = ref[ref_traj_index, 3]
            perp_deviation = dx * \
                ca.sin(ref_heading) - dy * ca.cos(ref_heading)
            para_deviation = dx * \
                ca.cos(ref_heading) + dy * ca.sin(ref_heading)

            speed_weight = weights_dict["weight_speed"]
            # Boost speed tracking when collision is detected.
            # In manual-CA mode the reference already contains the
            # safety-clamped speed, so the MPC must track it urgently.
            if self.is_collide and (self.agent_mode != "MPC-RL<Reference speed>" or self.CA_mode == "manual"):
                speed_weight = 100

            # State cost
            state_cost += (
                4 * perp_deviation**2 +
                2 * para_deviation**2 +
                speed_weight * (x[3, k] - ref_v)**2 +
                # weights_dict["weight_speed"] * (x[3, k] - ref_v)**2 +
                0.5 * (x[2, k] - ref_heading)**2
            )

            # Control cost
            control_cost += 0.01 * u[0, k]**2 + 0.01 * u[1, k]**2

            # Input difference cost
            if k > 0:
                input_diff_cost += 0.01 * \
                    ((u[0, k] - u[0, k-1])**2 + (u[1, k] - u[1, k-1])**2)

            # if not self.manual_collision_avoidance:
            if self.CA_mode == "cost":
                for other_vehicle in self.agent_vehicles_mpc:
                    dist = ca.norm_2(x[:2, k] - other_vehicle.position)
                    # in casadi, use ca.if_else to branch
                    distance_cost += ca.if_else(
                        dist < 1.0,  # if-statement
                        1000 / (dist + 1e-6)**2,  # if True
                        100 / (dist + 1e-6)**2    # if False
                    )

                collision_cost += ca.if_else(
                    self.is_collide,
                    3000 * x[3, k] ** 2,
                    0
                )
                for other_vehicle in self.agent_vehicles_mpc:
                    other_vehicle.position = self.other_vehicle_model(
                        other_vehicle, self.dt)
            elif self.CA_mode == "manual":
                collision_cost = 0
                distance_cost = 0

        # final state cost
        ref_traj_index = min(closest_index + N, ref.shape[0] - 1)
        desired_final_state = ref[ref_traj_index, :]
        final_state_cost += 100 * (
            (x[0, -1] - desired_final_state[0])**2 +
            (x[1, -1] - desired_final_state[1])**2 +
            20 * (x[3, -1] - desired_final_state[2])**2 +    # ref speed
            (x[2, -1] - desired_final_state[3])**2      # heading angle
        )

        total_cost = (
            state_cost * weights_dict["weight_state"] +
            control_cost * weights_dict["weight_control"] +
            input_diff_cost * weights_dict["weight_input_diff"] +
            final_state_cost * weights_dict["weight_final_state"]
        )            

        # if self.manual_collision_avoidance:
        if self.CA_mode == "cost":
            total_cost += (
                distance_cost * weights_dict["weight_distance"] +
                collision_cost * weights_dict["weight_collision"]
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
        # u0_controls = np.zeros(n_controls * N)
        # Instead of zeros, use previous solution as initial guess
       # Warm start the controls if available
        if hasattr(self, 'prev_solution') and self.prev_solution is not None:
            # Shift previous solution (drop first control, repeat last)
            u0_controls = np.vstack(
                [self.prev_solution[1:], self.prev_solution[-1]]).flatten()
        else:
            # Initialize with zeros if no previous solution
            u0_controls = np.zeros(n_controls * N)
        x0 = np.concatenate((x0_states, u0_controls))

        # Initial condition constraint
        g.append(x[:, 0] - state)

        # Collision avoidance constraints
        if self.CA_mode == "constraint":
            raise NotImplementedError(
                "Constraint-based collision avoidance is not yet implemented. "
                "Use CA_mode='cost' or CA_mode='manual' instead."
            )

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
            'ipopt.max_iter': 150,  # Bounded for real-time MPC; relies on warm-starting
            'ipopt.tol': 1e-6,
        }
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        if not solver.stats()['success']:
            print("WARNING: Optimization failed to find a solution")

        u_opt = sol['x'][-N * n_controls:].full().reshape(N, n_controls)
        self.last_acc = u_opt[0, 0]
        self.prev_solution = u_opt

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
            expected_shape = (self.config["observation"]['vehicles_count'], 8)
            raise ValueError(
                f"Expect observation's shape of {expected_shape}, but got {self.current_observation.shape}")

        self.observed_vehicles_count = np.sum(
            self.current_observation[:, 0] == 1) - 1

        # Ego vehicle
        self.ego_vehicle = Vehicle(
            index=0,
            position=self.current_observation[0, 1:3],
            vectorized_speed=self.current_observation[0, 3:5],
            heading=self._normalize_angle(self.current_observation[0, 5]),
            sinh=self.current_observation[0, 6],
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
                    heading=self._normalize_angle(
                        self.current_observation[i+1, 5]),
                    sinh=self.current_observation[i+1, 6],
                    cosh=self.current_observation[i+1, 7],
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

    @functools.cached_property
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

    def update_reference_states(self, speed_override=None, speed_overide_from_RL=None) -> np.ndarray:
        """Update reference states with stored collision information.

        When an RL agent provides a reference speed AND a collision is
        detected, we *blend* the two signals by averaging the RL speed
        and the CA-computed safe speed at every trajectory point.  This
        ensures the manual CA deceleration profile always participates
        instead of being completely overridden by the RL output.
        """
        DEFAULT_MAX_SPEED = 30.0
        SAFETY_BUFFER_POINTS = 5

        # ── Step 1: compute the CA-safe reference (decel profile if conflict) ──
        if not self.is_collide:
            new_reference_states = np.copy(self.reference_states)
        else:
            new_reference_states = np.copy(self.reference_states)

            # Use either current or memorized conflict indices
            conflict_indices = self.conflict_index
            if self.collision_memory > 0 and self.memorized_conflict_indices is not None:
                conflict_indices = self.memorized_conflict_indices

            valid_conflict_indices = [
                idx for idx in conflict_indices if idx is not None]

            if valid_conflict_indices:
                earliest_conflict_index = min(valid_conflict_indices)
                stop_index = max(self.ego_index + 1,
                                 earliest_conflict_index - SAFETY_BUFFER_POINTS)
                # Cap to max valid index
                stop_index = min(stop_index, len(self.reference_trajectory) - 1)
                points_to_stop = stop_index - self.ego_index

                if points_to_stop > 0:
                    current_speed = self.ego_vehicle.speed
                    deceleration_profile = np.linspace(
                        current_speed, 0, points_to_stop)
                    new_reference_states[self.ego_index:stop_index,
                                         2] = deceleration_profile
                    new_reference_states[stop_index:, 2] = 0.0
                    self.stop_point = self.reference_trajectory[stop_index]
                    self.last_valid_stop_point = self.stop_point
                elif self.last_valid_stop_point is not None:
                    self.stop_point = self.last_valid_stop_point

        # ── Step 2: clamp with RL speed if provided ──
        # Safety-filter approach (Wabersich & Zeilinger 2021): the CA
        # deceleration profile acts as a hard ceiling.  RL can request
        # any speed, but the MPC will never track more than the CA-safe
        # speed at each trajectory point.
        if speed_overide_from_RL is not None:
            rl_speed = np.clip(speed_overide_from_RL[0, 0], 0, DEFAULT_MAX_SPEED)
            ca_speeds = new_reference_states[:, 2]
            new_reference_states[:, 2] = np.minimum(rl_speed, ca_speeds)

        return new_reference_states

    def predict_ego_future_positions(self, current_position, speed, heading, max_acceleration, dt, prediction_horizon, reference_speed):
        """
        Predict ego vehicle future positions based strictly on reference trajectory points.
        Handles speed adjustments while following the reference path.

        Returns:
            list: List of predicted positions [(x1,y1), (x2,y2), ...]
        """
        future_positions = [current_position]
        current_speed = speed

        # Find starting index on reference trajectory
        start_index = np.argmin([
            np.linalg.norm(current_position - np.array(point[:2]))
            for point in self.reference_trajectory
        ])

        # Calculate cumulative distances along reference trajectory
        ref_points = self.reference_trajectory[start_index:, :2]
        if len(ref_points) < 2:
            return future_positions

        cumulative_distances = [0]
        for i in range(1, len(ref_points)):
            d = np.linalg.norm(ref_points[i] - ref_points[i-1])
            cumulative_distances.append(cumulative_distances[-1] + d)

        # For each prediction step
        current_distance = 0

        for _ in range(prediction_horizon):
            # Update speed based on reference speed
            if current_speed < reference_speed:
                current_speed = min(
                    current_speed + max_acceleration * dt, reference_speed)
            else:
                current_speed = reference_speed

            # Calculate distance traveled in this time step
            current_distance += current_speed * dt

            # Find the reference points we're between
            next_idx = np.searchsorted(cumulative_distances, current_distance)
            if next_idx >= len(ref_points):
                # If we've gone beyond the reference trajectory, stop here
                break

            if next_idx == 0:
                # We're still near the start
                next_position = ref_points[0]
            else:
                # Interpolate between reference points
                prev_idx = next_idx - 1
                prev_point = ref_points[prev_idx]
                next_point = ref_points[next_idx]

                # Calculate interpolation factor
                prev_dist = cumulative_distances[prev_idx]
                next_dist = cumulative_distances[next_idx]
                alpha = (current_distance - prev_dist) / (next_dist -
                                                          prev_dist) if next_dist != prev_dist else 1.0
                alpha = np.clip(alpha, 0, 1)

                # Interpolate position
                next_position = prev_point + alpha * (next_point - prev_point)

            future_positions.append(next_position)

        if len(future_positions) <= 1:
            return [current_position] * prediction_horizon  # Fallback
        return future_positions

    def predict_future_positions(self, current_position, speed, heading, dt, prediction_horizon):
        """
        Predict the future positions of a vehicle based on its current speed and heading.

        Args:
            current_position (np.ndarray): Current position [x, y] of the vehicle.
            speed (float): Current speed of the vehicle.
            heading (float): Heading angle of the vehicle in radians.
            dt (float): Time step for prediction.
            prediction_horizon (int): Number of steps to predict into the future.

        Returns:
            list: A list of future positions [x, y] at each time step.
        """
        future_positions = [current_position]
        for _ in range(prediction_horizon):
            next_position = future_positions[-1] + speed * dt * np.array([
                np.cos(heading),
                np.sin(heading)
            ])
            future_positions.append(next_position)
        return future_positions

    def _check_collision(self):
        """Modified collision detection to preserve collision state during memory period"""
        PREDICTION_HORIZON = 30
        TIME_THRESHOLD = 8

        # If we're in memory period and have stored collision points, use those
        if self.collision_memory > 0 and self.memorized_conflict_points is not None:
            self.conflict_points = self.memorized_conflict_points
            self.conflict_index = self.memorized_conflict_indices
            self.is_collide = True
            self.collision_memory -= 1
            return

        # Normal collision detection logic
        ego_location = np.array(self.ego_vehicle.position)
        self.ego_index = np.argmin([
            np.linalg.norm(ego_location - np.array(trajectory_point))
            for trajectory_point in self.reference_trajectory
        ])

        # Use the actual ego speed (not the static 10 m/s from
        # reference_states) so that collision detection is accurate
        # when the RL agent commands higher speeds.
        ego_future_positions = self.predict_ego_future_positions(
            current_position=self.ego_vehicle.position,
            speed=self.ego_vehicle.speed,
            heading=self.ego_vehicle.heading,
            max_acceleration=self.ego_vehicle.max_acceleration,
            dt=self.dt,
            prediction_horizon=PREDICTION_HORIZON,
            reference_speed=max(self.ego_vehicle.speed,
                                self.reference_states[self.ego_index, 2])
        )

        try:
            ego_path = LineString(ego_future_positions)
        except GEOSException as e:
            print(f"Warning: Invalid LineString input: {e}")
            # Handle the error (e.g., skip this step, use a default action)
            return  # Or take some other action

        self.agent_current_locations = []
        self.agent_future_locations = []
        self.conflict_points = []
        self.conflict_index = []
        self.agent_collide = []

        for agent_veh in self.agent_vehicles:
            agent_current_location = np.array(agent_veh.position)
            self.agent_current_locations.append(agent_current_location)

            agent_future_positions = self.predict_future_positions(
                current_position=agent_current_location,
                speed=agent_veh.speed,
                heading=agent_veh.heading,
                dt=self.dt,
                prediction_horizon=PREDICTION_HORIZON
            )
            self.agent_future_locations.append(agent_future_positions)

            agent_path = LineString(agent_future_positions)
            intersection = ego_path.intersection(agent_path)

            collision_detected = False
            intersection_point = None
            conflict_idx = None

            if not intersection.is_empty:
                intersection_points = []

                if intersection.geom_type == 'Point':
                    intersection_points.append(
                        (intersection.x, intersection.y))
                elif intersection.geom_type == 'LineString':
                    coords = list(intersection.coords)
                    if coords:
                        mid_idx = len(coords) // 2
                        intersection_points.append(coords[mid_idx])
                elif intersection.geom_type == 'MultiPoint':
                    for point in intersection.geoms:
                        intersection_points.append((point.x, point.y))
                elif intersection.geom_type == 'MultiLineString':
                    for line in intersection.geoms:
                        coords = list(line.coords)
                        if coords:
                            mid_idx = len(coords) // 2
                            intersection_points.append(coords[mid_idx])

                for int_point in intersection_points:
                    intersection_point = np.array(int_point)

                    ego_times = [i for i in range(len(ego_future_positions))]
                    agent_times = [i for i in range(
                        len(agent_future_positions))]

                    ego_dists = [np.linalg.norm(np.array(pos) - intersection_point)
                                 for pos in ego_future_positions]
                    agent_dists = [np.linalg.norm(np.array(pos) - intersection_point)
                                   for pos in agent_future_positions]

                    ego_time = ego_times[np.argmin(ego_dists)]
                    agent_time = agent_times[np.argmin(agent_dists)]

                    if abs(ego_time - agent_time) < TIME_THRESHOLD:
                        collision_detected = True
                        ref_traj_dists = [np.linalg.norm(np.array(pos) - intersection_point)
                                          for pos in self.reference_trajectory]
                        conflict_idx = np.argmin(ref_traj_dists)
                        break

            self.agent_collide.append(collision_detected)
            self.conflict_points.append(
                intersection_point if collision_detected else None)
            self.conflict_index.append(
                conflict_idx if collision_detected else None)

        # Check if there is any collision
        self.is_collide = np.any(self.agent_collide)

        # Update collision memory and store collision state
        if self.is_collide:
            self.collision_memory = self.collision_memory_steps
            # Store the current collision state
            self.memorized_conflict_points = self.conflict_points.copy()
            self.memorized_conflict_indices = self.conflict_index.copy()
        elif self.collision_memory > 0:
            # Use memorized values during memory period
            self.collision_memory -= 1
            self.is_collide = True
        else:
            # Clear memorized values when memory expires
            self.memorized_conflict_points = None
            self.memorized_conflict_indices = None

    def other_vehicle_model(self, other_vehicle, dt):
        new_position = other_vehicle.position + other_vehicle.speed * dt * \
            np.array([np.cos(other_vehicle.heading),
                     np.sin(other_vehicle.heading)])
        return new_position


class IntersectionMpcEnv_manual(IntersectionMpcEnv_noCA):
    """ MPC: with manual collision avoidance. """

    def __init__(self, config: dict = None, render_mode: str | None = None):
        super().__init__(config=config, render_mode=render_mode)

        # Used for manual collision avoidance checking:
        self.manual_collision_avoidance = True
        self.CA_mode = "manual"
        assert self.CA_mode == "manual", "Expect CA mode to be cost, but got {self.CA_mode}"
        self.agent_mode = "Pure_MPC"
        assert self.agent_mode == "Pure_MPC", "Expect agent mode to be `Pure_MPC`."

        self.weight_components = [
            "state",
            "speed",
            "control",
            "input_diff",
            "final_state",
            # "distance",
            # "collision",
        ]
        self.all_default_weights = {
            "weight_speed": 1,
            "weight_control": 1,
            "weight_final_state": 1,
            "weight_input_diff": 1,
            "weight_distance": 10,
            "weight_collision": 1,
            "weight_state": 10,
        }

        self.default_weights = {f"weight_{weight_name}": self.all_default_weights[f"weight_{weight_name}"]
                                for weight_name in self.weight_components}

class IntersectionMpcEnv_cost(IntersectionMpcEnv_noCA):
    """ MPC: with collision avoidance cost in MPC. """

    def __init__(self, config: dict = None, render_mode: str | None = None):
        super().__init__(config=config, render_mode=render_mode)

        # Used for manual collision avoidance checking:
        # self.manual_collision_avoidance = False
        self.CA_mode = "cost"
        assert self.CA_mode == "cost", "Expect CA mode to be cost, but got {self.CA_mode}"
        self.agent_mode = "Pure_MPC"
        assert self.agent_mode == "Pure_MPC", "Expect agent mode to be `Pure_MPC`."

        self.weight_components = [
            "state",
            "speed",
            "control",
            "input_diff",
            "distance",
            "collision",
            "final_state"
        ]
        self.all_default_weights = {
            "weight_speed": 1,
            "weight_control": 1,
            "weight_final_state": 1,
            "weight_input_diff": 1,
            "weight_distance": 10,
            "weight_collision": 1,
            "weight_state": 10,
        }
        self.default_weights = {f"weight_{weight_name}": self.all_default_weights[f"weight_{weight_name}"]
                                for weight_name in self.weight_components}

class IntersectionMpcEnv_constraint(IntersectionMpcEnv_noCA):
    """ MPC: with collision avoidance constraint in MPC. """

    def __init__(self, config: dict = None, render_mode: str | None = None):
        super().__init__(config=config, render_mode=render_mode)

        self.manual_collision_avoidance = False
        self.CA_mode = "constraint"
        assert self.CA_mode == "constraint", f"Expect CA mode to be cost, but got {self.CA_mode}"
        self.agent_mode = "Pure_MPC"
        assert self.agent_mode == "Pure_MPC", "Expect agent mode to be `Pure_MPC`."

        self.weight_components = [
            "state",
            "speed",
            "control",
            "input_diff",
            "distance",
            "collision",
            "final_state"
        ]
        self.all_default_weights = {
            "weight_speed": 1,
            "weight_control": 1,
            "weight_final_state": 1,
            "weight_input_diff": 1,
            "weight_distance": 10,
            "weight_collision": 1,
            "weight_state": 10,
        }
        self.default_weights = {f"weight_{weight_name}": self.all_default_weights[f"weight_{weight_name}"]
                                for weight_name in self.weight_components}
