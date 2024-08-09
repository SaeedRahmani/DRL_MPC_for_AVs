from typing import Dict, Tuple, Text
import numpy as np
from gymnasium.envs.registration import register,registry
from gym import spaces
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
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




class intersectionmpc_env(AbstractEnv):

    
    def __init__(self, config: dict = None):
        super().__init__(config)
        
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
                "collision_reward": -5,
                "arrived_reward": 1,
                "high_speed_reward": 1,
                "on_road_reward": 1,
                "policy_frequency":7,
                "normalize_reward": False
                
                
                
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
        """Per-agent per-objective reward signal."""
        scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": vehicle.crashed,
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "arrived_reward": self.has_arrived(vehicle),
            "on_road_reward": vehicle.on_road
        }

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

    def simulate(self, action):
        for k in range(1):  # Adjust the number of simulation steps as needed
            start = timer()
            self.collision_points, self.collision_detected = check_collisions(self.ref_path, self.obstacles)
            
            self.closest_index = find_closest_point(self.current_state, self.reference_trajectory)
            current_reference = self.reference_trajectory[self.closest_index:self.closest_index+self.horizon]
            
            if self.collision_detected:
                self.reference_trajectory = generate_global_reference_trajectory(self.collision_points)
                self.ref_path = [(x, y) for x, y, v, psi in self.reference_trajectory]
                self.resume_original_trajectory = False
                self.collision_timer = 0
            elif not self.resume_original_trajectory and not self.collision_detected:
                self.collision_timer += self.dt
                if self.collision_timer >= self.collision_wait_time:
                    self.reference_trajectory = self.original_reference_trajectory
                    self.ref_path = [(x, y) for x, y, v, psi in self.reference_trajectory]
                    self.resume_original_trajectory = True
            
            mpc_action = mpc_control(self.current_state, current_reference, self.obstacles, self.closest_index, self.collision_detected)
            end = timer()
            self.solver_time = end - start

            self.ego_vehicle.act({
                "acceleration": mpc_action[0],
                "steering": mpc_action[1]
            })

            self.road.act()
            self.road.step(self.dt)
            speed = np.sqrt(self.ego_vehicle.velocity[0]**2 + self.ego_vehicle.velocity[1]**2)
            
            self.current_state = np.array([float(self.ego_vehicle.position[0]),
                                           float(self.ego_vehicle.position[1]),
                                           speed,
                                           self.ego_vehicle.heading])

            self.old_accel = mpc_action[0]
            self.steps += 1

        self.enable_auto_render = False

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.simulate(action)
        obs = self._observation_type.observe()
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)
        return obs, reward, terminated, truncated, info
    
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
        return self._observation_type.observe()
    

    

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




env_id = 'intersectionmpc-01'


if env_id not in registry:
    register(
        id=env_id,
        entry_point='highway_env.envs.intersectionmpc_env:intersectionmpc_env',  # Replace with actual module and class
        max_episode_steps=1000,
    )
    print("registered", env_id)
