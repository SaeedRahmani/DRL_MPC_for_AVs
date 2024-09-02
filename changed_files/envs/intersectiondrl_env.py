from typing import Dict, Tuple, Text
import numpy as np
from gymnasium.envs.registration import register,registry
from gymnasium import spaces
from highway_env import utils
from highway_env.envs.common.abstractmpcdrl import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import LineType, StraightLane, CircularLane, AbstractLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import ControlledVehicle
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math
from timeit import default_timer as timer
from highway_env.envs.mpc_controller import *
from highway_env.envs.common.observation import KinematicObservation


class intersectiondrl_env(AbstractEnv):
    
    
    def __init__(self, config: dict = None, render_mode: str = None):
        super().__init__(config)
        self.render_mode = render_mode
        self.solver_time = 0
        self.old_accel = 0
        self.steps = 0
        self.controlled_vehicles = []
        self.ego_vehicle = None
        self.current_state = np.zeros(4)
        self.reference_trajectory = None
        self.obstacles = []
        self.horizon = 6
        self.dt = 0.1
        self.LENGTH = 5.0
        self.WIDTH = 2.0
        self.WHEELBASE = 2.5
        self.MIN_SPEED = 0
        self.closest_index = 0
        self.original_reference_trajectory = generate_global_reference_trajectory()
        self.reference_trajectory = self.original_reference_trajectory.copy()
        self.ref_path = [(x, y) for x, y, v, psi in self.reference_trajectory]
        self.resume_original_trajectory = False
        self.collision_wait_time = 0.3  # 0.3 seconds
        self.collision_timer = 0
        self.real_path = []
        self.sim = 0
        self._observation_type = KinematicObservation(self)
        self.training_mode = False
        self.collision_detected = False

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 5,
                    
                    "normalize": False,
                    "features": [
                        "presence",
                        "x",
                        "y",
                        "vx",
                        "vy",
                        "long_off",
                        "lat_off",
                        "ang_off",
                    ],
                },
                "action": {
                    "type": "ContinuousAction",
                    "steering_range": [-np.pi / 3, np.pi / 3],
                    #"acceleration_range": [-10.0, 10.0],
                    "longitudinal": True,
                    "lateral": True,
                    "dynamical": True,
                },
                "duration": 13,  # [s]
                "destination": "o1",
                "controlled_vehicles": 1,
                "initial_vehicle_count": 7,
                "spawn_probability": 0.7,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.6],
                "scaling": 5.5 * 1.3,
                "collision_reward": -5,
                "arrived_reward": 1,
                "high_speed_reward": 0.3,
                "on_road_reward": 0.6,
                "policy_frequency":7,
                "reward_speed_range": [7.0, 9.0],
                "normalize_reward": False,
                "offroad_terminal": False
                
                
                
            }
        )
        return config

    def _reward(self, action: int) -> float:
        """Aggregated reward, for cooperative agents."""
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles
                   ) / len(self.controlled_vehicles)

    def _rewards(self, action: int) -> Dict[Text, float]:
        """Multi-objective rewards, for cooperative agents."""
        agents_rewards = [self._agent_rewards(action, vehicle) for vehicle in self.controlled_vehicles]
        return {
            name: sum(agent_rewards[name] for agent_rewards in agents_rewards) / len(agents_rewards)
            for name in agents_rewards[0].keys()
        }

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """Per-agent reward signal."""
        rewards = self._agent_rewards(action, vehicle)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        reward = self.config["arrived_reward"] if rewards["arrived_reward"] else reward
        reward *= rewards["on_road_reward"]
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward, [self.config["collision_reward"], self.config["arrived_reward"]], [0, 1])
        return reward

    def _agent_rewards(self, action: int, vehicle: Vehicle) -> Dict[Text, float]:
            # Collision reward: Higher penalty for collisions
        collision_reward = -10 if vehicle.crashed else 0

        # Speed alignment reward: Penalize deviations from desired speed
        desired_speed = utils.lmap(action[2], [-1, 1], [self.config["reward_speed_range"][0], self.config["reward_speed_range"][1]])
        speed_diff = abs(vehicle.speed - desired_speed)
        speed_alignment_reward = -speed_diff  # Penalize the difference from desired speed

        # High speed reward: Encourage maintaining speed within a specific range
        scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])
        high_speed_reward = np.clip(scaled_speed, 0, 1) * 0.5  # Weight this reward to emphasize speed control

        # On-road reward: Encourage staying on the road
        on_road_reward = 1.0 if vehicle.on_road else -5.0  # Adjust rewards for staying on the road

        # Arrived reward: Check if the vehicle has reached its destination
        arrived_reward = self.config["arrived_reward"] if self.has_arrived(vehicle) else 0

        # Aggregate rewards into a dictionary
        rewards = {
            "collision_reward": collision_reward,
            "speed_alignment_reward": speed_alignment_reward,
            "high_speed_reward": high_speed_reward,
            "on_road_reward": on_road_reward,
            "arrived_reward": arrived_reward
        }

        # Calculate total reward
        total_reward = sum(rewards.values())

        if self.config["normalize_reward"]:
            # Normalize the reward if required
            total_reward = utils.lmap(total_reward, [self.config["collision_reward"], self.config["arrived_reward"]], [0, 1])

        return rewards

    def _is_terminated(self) -> bool:
        return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
               or all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles) \
               or (self.config["offroad_terminal"] and not self.ego_vehicle.on_road)

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return (vehicle.crashed or
                self.has_arrived(vehicle))

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        info["agents_rewards"] = tuple(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles)
        info["agents_dones"] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        return info
    


    def local_to_global(self,x_local, y_local, vehicle_position, vehicle_heading):
    # Convert local coordinates (x_local, y_local) to global coordinates
        x_global = vehicle_position[0] + (x_local * np.cos(vehicle_heading) - y_local * np.sin(vehicle_heading))
        y_global = vehicle_position[1] + (x_local * np.sin(vehicle_heading) + y_local * np.cos(vehicle_heading))
        return -x_global, y_global

    def predict_vehicle_positions(self,vehicle, ego_heading, time_horizon=12.0):
        dt = 0.1
        positions = []
        current_position = np.array(vehicle.position) 
        current_heading = vehicle.heading
        current_speed = vehicle.speed
        directions = []

    # Determine the direction relative to the ego vehicle
        direction = determine_direction(ego_heading, -current_heading)
        directions.append(direction)
        


        for _ in np.arange(0, time_horizon, dt):
        # Predict the next position
            dx = current_speed * np.cos(current_heading) * dt
            dy = current_speed * np.sin(current_heading) * dt
            next_position = current_position + np.array([dx, dy])
        
            positions.append(next_position.copy())
            current_position = next_position  # Update to the new position
        

        return positions, directions



    def check_collisions(self, trajectory, other_vehicles, safety_distance=1.0, start_index=0, time_horizon=2.0):
        collision_points = []
        self.collision_detected = False
        predicted_positions = []  # Initialize this list to hold all vehicles' predicted positions
        all_directions = []  # This will hold directions for all vehicles
        buffer = 2.6
        half_width = buffer / 2
        dt = 0.1
        time_steps_window = int(1 / dt)

        ego_position = self.ego_vehicle.position
        ego_heading = self.ego_vehicle.heading
        length = 5.0
        ego_velocity = np.array(self.ego_vehicle.velocity)

        back_x = ego_position[0] - (length * np.cos(ego_heading))
        back_y = ego_position[1] - (length * np.sin(ego_heading))

    # Iterate over all other vehicles
        for vehicle in other_vehicles:
            if vehicle is self.ego_vehicle:
                continue  # Skip collision check with the ego vehicle itself

        # Predict future positions of the vehicle over the time horizon
            vehicle_predicted_positions, directions = self.predict_vehicle_positions(vehicle, self.ego_vehicle.heading, time_horizon)

            vehicle_velocity = np.array(vehicle.velocity)
            relative_velocity = ego_velocity - vehicle_velocity

            vehicle_position = np.array(vehicle.position)
            relative_position_to_ego = vehicle_position - np.array(ego_position)
            # Calculate the dot product with ego's heading to see if vehicle is in front or behind
            forward_vector = np.array([np.cos(ego_heading), np.sin(ego_heading)])
            dot_product = np.dot(relative_position_to_ego, forward_vector)

            if dot_product < 0:
                # Vehicle is behind the ego vehicle, skip collision check for this vehicle
                continue
        
        # Check if vehicle's predicted positions are within the safety distance behind the ego vehicle
            """for vx, vy in vehicle_predicted_positions:
                distance_to_back = np.sqrt((back_x - vx) ** 2 + (back_y - vy) ** 2)
                if distance_to_back < safety_distance:
                    should_consider_for_collision = False
                    break"""
        
            
            predicted_positions.append(vehicle_predicted_positions)  # Store positions for this vehicle 
            all_directions.append(directions)  # Store direction info for this vehicle

    # Check for collisions with the ego vehicle's trajectory
        for step, (x, y) in enumerate(trajectory[start_index:]):
        # Convert trajectory point to global coordinates relative to ego vehicle
            traj_x_global, traj_y_global = x, y
            ego_box = [(x - half_width, y - half_width), 
                   (x + half_width, y + half_width)]
        
            for vehicle_predicted_positions in predicted_positions:
                for obs_step in range(max(0, step - time_steps_window), min(len(vehicle_predicted_positions), step + time_steps_window + 1)):
                    ox, oy = vehicle_predicted_positions[obs_step]  # Get the predicted position
                    
                    relative_position = np.array([ox - traj_x_global, oy - traj_y_global])
                    distance_to_vehicle = np.linalg.norm(relative_position)
                    speed_towards_each_other = np.dot(relative_velocity, relative_position) / distance_to_vehicle if distance_to_vehicle > 0 else 0
                
                    if speed_towards_each_other > 0:  # Vehicles are moving towards each other
                        ttc = distance_to_vehicle / speed_towards_each_other
                    else:
                        ttc = np.inf 
                # Improved collision check using bounding box overlap
                    if (ego_box[0][0] <= ox <= ego_box[1][0]) and (ego_box[0][1] <= oy <= ego_box[1][1]) or (ttc < time_horizon and distance_to_vehicle < safety_distance):
                        collision_points.append((x, y))  # Record the point in local coordinates
                        self.collision_detected = True
                        print(f"Collision detected at: {(x, y)} with vehicle at {(ox, oy)}")  # Print collision details
                        return collision_points, self.collision_detected, predicted_positions, all_directions

    # Debugging print to show the result of the collision check

        return collision_points, self.collision_detected, predicted_positions, all_directions
    

    def simulate(self, action):
        for k in range(1):  # Adjust the number of simulation steps as needed
            start = timer()
            
            normalized_speed_override = action[2] 
            speed_override = utils.lmap(normalized_speed_override,[-1,1],[0,10.0])
           
            if speed_override < 0:
                speed_override = 0
            
            
            
            
            self.closest_index = find_closest_point(self.current_state, self.reference_trajectory)
            
            
            self.collision_points, self.collision_detected, self.predicted_positions, self.directions = self.check_collisions(self.ref_path, self.road.vehicles,safety_distance=1.0,start_index=self.closest_index)
            #print(self.collision_points,"collision****")
            if self.collision_detected:
                self.reference_trajectory = generate_global_reference_trajectory(self.collision_points,speed_override)
                self.ref_path = [(x, y) for x, y, v, psi in self.reference_trajectory]
                self.resume_original_trajectory = False
                self.collision_timer = 0
            elif not self.resume_original_trajectory and not self.collision_detected:
                self.collision_timer += self.dt
                if self.collision_timer >= self.collision_wait_time:
                    self.reference_trajectory = self.original_reference_trajectory
                    self.ref_path = [(x, y) for x, y, v, psi in self.reference_trajectory]
                    self.resume_original_trajectory = True
            
            current_reference = self.reference_trajectory[self.closest_index:self.closest_index+self.horizon]
            mpc_action = mpc_control(self.current_state, current_reference, self.obstacles, self.closest_index, self.collision_detected)
            end = timer()
            
            self.solver_time = end - start

            self.ego_vehicle.act({
                "acceleration": mpc_action[0],
                "steering": mpc_action[1],
            })

            
            self.road.act()
            self.road.step(self.dt)
           

            if self.collision_detected:
                speed = np.sqrt(self.ego_vehicle.velocity[0]**2 + self.ego_vehicle.velocity[1]**2)
            else:
                speed = np.sqrt(self.ego_vehicle.velocity[0]**2 + self.ego_vehicle.velocity[1]**2)
            
            self.current_state = np.array([float(self.ego_vehicle.position[0]),
                                           float(self.ego_vehicle.position[1]),
                                           speed,
                                           self.ego_vehicle.heading])

            self.old_accel = mpc_action[0]
            
            self.steps += 1
            self.solver_time = end - start 
        return speed_override

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.simulate(action)
        
        if not hasattr(self, 'real_path'):
            self.real_path = []
        self.real_path.append((self.current_state[0], self.current_state[1]))
        obs, reward, terminated, truncated, info = super().step(action)
        info = self._info(obs, action)
        
        
        # plot_trajectory(self.real_path, self.ref_path, self.predicted_obstacles, self.collision_points, directions)
        if not self.training_mode:
            plot_trajectory(self.real_path, self.ref_path, self.predicted_positions, self.collision_points, self.directions)
        
        

        return obs, reward, terminated, truncated, info

    def set_training_mode(self, is_training: bool):
        """Set whether the environment is in training mode or not."""
        self.training_mode = is_training
    
    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])
        speed = np.sqrt(self.ego_vehicle.velocity[0]**2 + self.ego_vehicle.velocity[1]**2)
        self.current_state = np.array([float(self.ego_vehicle.position[0]),
                                       float(self.ego_vehicle.position[1]),
                                       speed,
                                       self.ego_vehicle.heading])
        self.original_reference_trajectory = generate_global_reference_trajectory()
        self.reference_trajectory = self.original_reference_trajectory.copy()
        self.ref_path = [(x, y) for x, y, v, psi in self.reference_trajectory]
        self.real_path = []
        

    def _make_road(self) -> None:
        """
        Make an 4-way intersection.

        The horizontal road has the right of way. More precisely, the levels of priority are:
            - 3 for horizontal straight lanes and right-turns
            - 1 for vertical straight lanes and right-turns
            - 2 for horizontal left-turns
            - 0 for vertical left-turns

        The code for nodes in the road network is:
        (o:outer | i:inner + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)

        :return: the intersection road
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50 + 50  # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # Incoming
            start = rotation @ np.array([lane_width / 2, access_length + outer_distance])
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane("o" + str(corner), "ir" + str(corner),
                         StraightLane(start, end, line_types=[s, c], priority=priority, speed_limit=10))
            # Right turn
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane("ir" + str(corner), "il" + str((corner - 1) % 4),
                         CircularLane(r_center, right_turn_radius, angle + np.radians(180), angle + np.radians(270),
                                      line_types=[n, c], priority=priority, speed_limit=10))
            # Left turn
            l_center = rotation @ (np.array([-left_turn_radius + lane_width / 2, left_turn_radius - lane_width / 2]))
            net.add_lane("ir" + str(corner), "il" + str((corner + 1) % 4),
                         CircularLane(l_center, left_turn_radius, angle + np.radians(0), angle + np.radians(-90),
                                      clockwise=False, line_types=[n, n], priority=priority - 1, speed_limit=10))
            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane("ir" + str(corner), "il" + str((corner + 2) % 4),
                         StraightLane(start, end, line_types=[s, n], priority=priority, speed_limit=10))
            # Exit
            start = rotation @ np.flip([lane_width / 2, access_length + outer_distance], axis=0)
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane("il" + str((corner - 1) % 4), "o" + str((corner - 1) % 4),
                         StraightLane(end, start, line_types=[n, c], priority=priority, speed_limit=10))

        road = RegulatedRoad(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane

        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 3
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [
                (
                    self.road.act(),
                    self.road.step(1 / self.config["simulation_frequency"]),
                )
                for _ in range(self.config["simulation_frequency"])
            ]

        # Challenger vehicle
        self._spawn_vehicle(
            60,
            spawn_probability=1,
            go_straight=True,
            position_deviation=0.1,
            speed_deviation=0,
        )

        # Controlled vehicles
        self.controlled_vehicles = []
        for ego_id in range(0, self.config["controlled_vehicles"]):
            ego_lane = self.road.network.get_lane(
                ("o{}".format(ego_id % 4), "ir{}".format(ego_id % 4), 0)
            )
            destination = self.config["destination"] or "o" + str(
                self.np_random.integers(1, 4)
            )
            ego_vehicle = self.action_type.vehicle_class(
                self.road,
                ego_lane.position(60 + 5 * self.np_random.normal(1), 0),
                speed=ego_lane.speed_limit,
                heading=ego_lane.heading_at(60),
            )
            try:
                ego_vehicle.plan_route_to(destination)
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(
                    ego_lane.speed_limit
                )
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(
                    ego_vehicle.speed_index
                )
            except AttributeError:
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            for v in self.road.vehicles:  # Prevent early collisions
                if (
                    v is not ego_vehicle
                    and np.linalg.norm(v.position - ego_vehicle.position) < 20
                ):
                    self.road.vehicles.remove(v)
        self.ego_vehicle = ego_vehicle


    def _spawn_vehicle(self,
                       longitudinal: float = 0,
                       position_deviation: float = 1.,
                       speed_deviation: float = 1.,
                       spawn_probability: float = 0.6,
                       go_straight: bool = False) -> None:
        if self.np_random.uniform() > spawn_probability:
            return

        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(self.road, ("o" + str(route[0]), "ir" + str(route[0]), 0),
                                            longitudinal=(longitudinal + 5
                                                          + self.np_random.normal() * position_deviation),
                                            speed=8 + self.np_random.normal() * speed_deviation)
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("o" + str(route[1]))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _clear_vehicles(self) -> None:
        is_leaving = lambda vehicle: "il" in vehicle.lane_index[0] and "o" in vehicle.lane_index[1] \
                                     and vehicle.lane.local_coordinates(vehicle.position)[0] \
                                     >= vehicle.lane.length - 4 * vehicle.LENGTH
        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
                              vehicle in self.controlled_vehicles or not (is_leaving(vehicle) or vehicle.route is None)]

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
        return "il" in vehicle.lane_index[0] \
               and "o" in vehicle.lane_index[1] \
               and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance




env_id = 'intersectiondrl-v5'


if env_id not in registry:
    register(
        id=env_id,
        entry_point='highway_env.envs.intersectiondrl_env:intersectiondrl_env',  # Replace with actual module and class
        max_episode_steps=1000,
    )
    print("registered", env_id)