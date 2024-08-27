from __future__ import division, print_function, absolute_import
import numpy as np
from gym.envs.registration import register
from gym import spaces

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.control import SteeringControlledVehicle
from highway_env.vehicle.dynamics import Obstacle
from mpcc_controller import mpcc
from timeit import default_timer as timer


class MergeEnv2(AbstractEnv):
    """
        A highway merge negotiation environment.

        The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
        It is rewarded for maintaining a high velocity and avoiding collisions, but also making room for merging
        vehicles.
    """
    EVALUATE = 1

    if EVALUATE == 0:
        K = 2
    else:
        K = 1

    only_mpc = False
    only_rl = False
    min_coop = 0.0
    cv = 'CV2'

    vref = 2.5
    n_test_episodes = 100
    offset = 1
    NUMBER_VEHICLES = 30
    aa = mpcc([100, 4, 0, 0])
    aa.ref_path()
    crash = 0
    lead_head_ = 0
    follow_head_ = 0
    lead_lateral_ = 0
    follow_lateral_ = 0
    # normalize_features = utils.RunningMeanStd(shape=(1, 10))
    # normalize_reward = utils.RunningStd(shape=())
    behavior = 6
    episode_no = 0
    episode_reward = 0
    step_num = 0
    n_timeouts = 0
    collisions = 0
    infeasible_solns = 0
    time_list = []
    time_goal_reached = []
    goal_x = 165
    goal_reached = 0
    dist_goal_not_reached = []
    results_list = []
    follower_merge = []
    merge_counter = 0
    follower_number = 0
    old_accel = 0
    solver_time = 0

    dce = np.Inf
    tce = np.Inf

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                "vehicles_count": 15,
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": False
            },
            "policy_frequency": 1,
            "screen_width": 1800,
            "screen_height": 500,
            "centering_position": [0.5, 0.5]

        })
        return config

    def define_spaces(self):
        super().define_spaces()
        self.observation_space = spaces.Box(-1, 1, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Box(-4, 4., shape=(1,), dtype=np.float32)

    def simulate(self, action=None):
        """
            Perform several steps of simulation with constant action, called every time a policy is executed
        """
        for k in range(self.K): 

            start = timer()
            self.aa.run_solver()
            end = timer()
            self.solver_time = end - start
            #print("Solver Time: " + str(self.solver_time))
            if self.EVALUATE:
                if self.aa.EXITFLAG != 1:
                    self.infeasible_solns += 1
            #print("SOlver flag: " + str(self.aa.EXITFLAG))
            # Save the future states calculated by MPC so they can be displayed
            self.vehicle.future = []
            for i in range(0, (self.aa.FORCES_N) * 9, 9):
                self.vehicle.future.append(self.aa.FORCES_x0[i + 3])
                self.vehicle.future.append(self.aa.FORCES_x0[i + 4])
                self.vehicle.future.append(self.aa.FORCES_x0[i + 5])

            if self.cv == 'CV':
                print("CV")
                s_i = self.aa.FORCES_x0[self.aa.FORCES_TOTAL_V -2] +0.1*(self.aa.FORCES_N-1)*self.aa.FORCES_x0[self.aa.FORCES_NU+3]
                self.vehicle.cooperative = self.aa.ref_path_y(s_i)
            elif self.cv == 'CV2':
                print("CV2")
                self.vehicle.cooperative = self.aa.FORCES_x0[self.aa.FORCES_NU + 1] + 0.1 * (self.aa.FORCES_N - 1) * \
                                       self.aa.FORCES_x0[self.aa.FORCES_NU + 3] * np.sin(self.aa.FORCES_x0[self.aa.FORCES_NU + 2])
            elif self.cv == 'CV3':
                print("CV3")
                self.vehicle.cooperative = self.aa.FORCES_x0[self.aa.FORCES_NU + 1] + 0.1 * (self.aa.FORCES_N - 1) * \
                                       self.aa.FORCES_x0[self.aa.FORCES_NU + 3] * np.sin(self.aa.FORCES_x0[self.aa.FORCES_NU + 2])
                self.vehicle.predicted_pose = np.array([self.aa.FORCES_x0[self.aa.FORCES_NU] + 0.1 * (self.aa.FORCES_N - 1) * \
                                                        self.aa.FORCES_x0[self.aa.FORCES_NU + 3] * np.cos(self.aa.FORCES_x0[self.aa.FORCES_NU + 2]),
                                                        self.aa.FORCES_x0[self.aa.FORCES_NU + 1] + 0.1 * (self.aa.FORCES_N - 1) * \
                                                        self.aa.FORCES_x0[self.aa.FORCES_NU + 3] * np.sin(self.aa.FORCES_x0[self.aa.FORCES_NU + 2])])
            elif self.cv == 'Reactive':
                print("Reactive Predictions")
                self.vehicle.cooperative = self.aa.FORCES_x0[4] *self.offset
            else:
                print("MPC")
                self.vehicle.cooperative = self.aa.FORCES_x0[((self.aa.FORCES_N - 1) * 9) + 4] *self.offset

            self.road.act()
            #print(self.aa.FORCES_x0[0])
            #if not self.only_rl:
            temp_accel = self.aa.FORCES_x0[0]
            #else:
            #    temp_accel = (self.aa.reference_velocity_-self.vehicle.velocity)/0.1

            self.vehicle.act({
                "acceleration": temp_accel
                , "steering": self.aa.FORCES_x0[1]*2
            })

            self.road.step(0.1)

            # update the current state for the solver
            self.aa.current_state_[0] = self.vehicle.position[0]
            self.aa.current_state_[1] = self.vehicle.position[1]
            self.aa.current_state_[2] = self.vehicle.heading
            self.aa.current_state_[3] = self.vehicle.velocity

            obs = self.observation.observe()

            if self.leader:
                self.aa.leader_state_[0] = self.leader.position[0]
                self.aa.leader_state_[1] = self.leader.position[1]
                self.aa.leader_state_[2] = self.leader.heading
                self.aa.leader_state_[3] = self.leader.velocity
            else:
                self.aa.leader_state_[0] = 0
                self.aa.leader_state_[1] = 0
                self.aa.leader_state_[2] = 0
                self.aa.leader_state_[3] = 0

            if self.follower:
                self.aa.follower_state_[0] = self.follower.position[0]
                self.aa.follower_state_[1] = self.follower.position[1]
                self.aa.follower_state_[2] = self.follower.heading
                self.aa.follower_state_[3] = self.follower.velocity
            else:
                self.aa.follower_state_[0] = 0
                self.aa.follower_state_[1] = 0
                self.aa.follower_state_[2] = 0
                self.aa.follower_state_[3] = 0

            self._automatic_rendering()
            self.old_accel = temp_accel

            self.steps += 1
        # Stop at terminal states

        self.enable_auto_render = False

    def step(self, action):

        # Take the action from the RL agent and feed it as input tp the solver
        self.aa.reference_velocity_ = action[0]
        self.aa.reference_velocity_ = round(self.aa.reference_velocity_, 1)
        if self.aa.reference_velocity_ < 0:
            self.aa.reference_velocity_ = 0.0

        if self.only_mpc:
            self.aa.reference_velocity_ = self.vref

        if self.only_rl:
            #self.aa.slack_weight_ = 0
            self.aa.only_rl = self.only_rl

        # Apply the same action for k number of times
        self.simulate(action)

        # Remove vehicles from the network that have left add add some
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        add_vehicle = True
        for v in self.road.vehicles:  # Prevent early collisions
            if v.position[0] > 230.0:
                self.road.vehicles.remove(v)

        crash = False
        for v in self.road.vehicles:
            if v.position[0] < 15:
                add_vehicle = False
            if v.crashed and v.cooperative!=2:
                print(v)
                self.crash += 1
            v.leader = 0
            v.follower = 0

        if not self.EVALUATE:
            beh = np.random.uniform(self.min_coop, 4)
        else:
            if self.behavior == 5:  # mixed
                beh = np.random.uniform(self.min_coop, 4)
            elif self.behavior == 6:  # cooperative
                beh = np.random.uniform(2.0, 4)
            elif self.behavior == 7:  # non-cooperative
                beh = np.random.uniform(self.min_coop, 2)

        if add_vehicle:
            self.road.vehicles.append(other_vehicles_type(self.road,
            self.road.network.get_lane(("a", "b", 0)).position(0, 0), beh, velocity=np.random.uniform(3,4)))

        obs = self.observation.observe()

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        self.step_num +=1

        info = {"is_success": False}
        terminal = self._is_terminal()
        reward = self.reward_()/ 2.0

        self.episode_reward += reward

        '''self.normalize_reward.update(reward)
        reward =  np.clip(reward /np.sqrt(self.normalize_reward.var + 1e-4), -10, 10)
        if terminal:
            print("reward std", self.normalize_reward.var)
            print("features mean", self.normalize_features.mean)
            print("features var", self.normalize_features.var)'''
        if self.EVALUATE:
            if terminal:
                self.episode_no += 1
                self.time_list.append(self.steps)
                if self.vehicle.crashed:
                    self.collisions += 1
                if self.vehicle.position[0] > self.goal_x:
                    self.time_goal_reached.append(self.steps)
                    self.goal_reached += 1
                else:
                    self.dist_goal_not_reached.append(self.vehicle.position[0])

        '''if self.vehicle.position[0]> 145 and self.merge_counter < 1:
            self.follower_merge.append(self.follower_number)
            self.merge_counter += 1'''

        # Compute risk metrics
        distance_to_follower = np.linalg.norm(self.aa.current_state_[:2] - self.aa.follower_state_[:2])
        if distance_to_follower < self.dce:
            self.dce = distance_to_follower
            self.tce = self.steps * 0.1

        return obs, reward, terminal, info

    def reward_(self):
        reward = 0
        '''
        for i in range(0, self.aa.FORCES_N*3, 3):
            x = self.vehicle.future[i]
            y = self.vehicle.future[i + 1]
            if not ((95 < x < 145 and 6 > y > -1.0) or (145 <= x < 250 and 1.0 > y > -1.0)):
                reward = reward - 0.3
        '''
        if self.aa.EXITFLAG != 1:
            reward = reward - 1#300

        if 2.0 > self.vehicle.position[1] > -2:
            if self.lead_head_ < 6.5:
                reward = reward - 3
            if self.follow_head_ < 6.5:
                reward = reward - 3

        if self.vehicle.crashed:
            reward = reward - 300
        else:
            reward = reward + self.vehicle.velocity

        #if self.vehicle.position[0] > self.goal_x:
        #    reward += 300

        #if self.steps > 560:
        #    reward += -30

        #if np.abs(self.vehicle.position[1]) < 0.1:
        #    reward += 30

        return reward

    def _is_terminal(self):
        """
            The episode is over when a collision occurs or when the access ramp has been passed.
        """

        if self.vehicle.crashed:
            self.collisions += 1
            return True
        elif self.vehicle.position[0] > self.goal_x:
            self.goal_reached += 1
            return True
        elif self.steps > 560:
            self.n_timeouts += 1
            return True

        if self.crash == 2:
            print("Change the margin value to prevent early collision of vehicles")

        return self.vehicle.crashed or self.vehicle.position[0] > self.goal_x or self.vehicle.velocity < 0 or self.crash == 2 or self.steps > 560# or self.aa.EXITFLAG != 1
        #return self.vehicle.crashed or np.abs(self.vehicle.position[1]) < 0.1 or self.steps > 560# or self.aa.EXITFLAG != 1


    def reset(self):
        if self.EVALUATE:
            print("Testing...")
        else:
            print("Training...")

        self.collisions = 0
        self.infeasible_solns = 0
        self.goal_reached = 0
        self.episode_reward = 0
        self.n_timeouts = 0

        self.episode_no += 1
        
        # Reset statistic flags
        self.dce = np.Inf
        self.tce = np.Inf

        if self.EVALUATE:
            self.offset = 1#np.random.uniform(0,1)
            if self.episode_no == 0:
                self.initialise_metrics()
                # Change the behavior to all cooperative
                #self.behavior = 3.9

            if self.episode_no < self.n_test_episodes:

                self.behavior = 6
            elif (self.episode_no <= self.n_test_episodes*2):

                self.behavior = 5

            elif (self.episode_no <= self.n_test_episodes*3):

                self.behavior = 7

        self._make_road()
        self._make_vehicles()
        self.aa.reset_solver()
        self.old_accel = 0
        self.crash = 0
        self.all_action_negative = True
        self.sum_reward = 0
        self.aa.traj_i = 0
        self.aa.current_state_[0] = self.vehicle.position[0]
        self.aa.current_state_[1] = self.vehicle.position[1]
        self.aa.current_state_[2] = self.vehicle.heading
        self.aa.current_state_[3] = self.vehicle.velocity
        self.aa.leader_state_[0] = 0
        self.aa.leader_state_[1] = 0
        self.aa.leader_state_[2] = 0
        self.aa.leader_state_[3] = 0
        self.aa.follower_state_[0] = 0
        self.aa.follower_state_[1] = 0
        self.aa.follower_state_[2] = 0
        self.aa.follower_state_[3] = 0
        self.lead_head_ = 0
        self.follow_head_ = 0
        self.lead_lateral_ = 0
        self.follow_lateral_ = 0
        self.merge_counter = 0
        '''if self.episode_no == 600:
            print(self.follower_merge)
        print(self.follower_merge)'''

        return super(MergeEnv2, self).reset()

    def initialise_metrics(self):
        self.collisions = 0
        self.infeasible_solns = 0
        self.goal_reached = 0
        self.time_list = []
        self.time_goal_reached = []
        self.dist_goal_not_reached = []

    def print_metrics(self):
        for i in range(len(self.results_list)):
            if i==0:
                print("ALL COOPERATIVE")
            elif i==1:
                print("MIXED")
            else:
                print("NON COOPERATIVE")
            print("Episode no. ", self.episode_no)
            print("No. of collisions: ", self.results_list[i][0]/4)
            print("No. of infeasible solutions: ", self.results_list[i][1])
            print("No. of times vehicle reached goal: ", self.results_list[i][2]/4)
            print("FRP: ", 100 - (self.results_list[i][0]/4 + self.results_list[i][2]/4))
            print("Average time for all episodes: ", self.results_list[i][3])
            print("Average time for episodes where goal was reached: ", self.results_list[i][4])
            print("Average distance covered for episodes where goal was not reached", self.results_list[i][5])
            print(" ")

    def _make_road(self):
        """
            Make a road composed of a straight highway and a merging lane.
        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [20, 80, 50, 80]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, 0]
        line_type = [[c, c], [n, c]]
        line_type_merge = [[c, s], [n, s]]

        net.add_lane("a", "b", StraightLane([0, 0], [sum(ends[:2]), 0], line_types=line_type[0]))
        net.add_lane("b", "c", StraightLane([sum(ends[:2]), 0], [sum(ends[:3]), 0], line_types=line_type_merge[0]))
        net.add_lane("c", "d", StraightLane([sum(ends[:3]), 0], [sum(ends), 0], line_types=line_type[0]))

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4], [ends[0], 6.5 + 4], line_types=[c, c], forbidden=True)
        ljk = StraightLane([0, 6.5 + 4], [ends[0], 6.5 + 4], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2*ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0],
                           line_types=[n, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        #road.vehicles.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self):
        """
            Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        road = self.road

        ego_vehicle = SteeringControlledVehicle(road, road.network.get_lane(("b", "c", 1)).position(20, 0), velocity=0)
        ego_vehicle.cooperative = 2
        road.vehicles.append(ego_vehicle)
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        MARGIN = 1
        position_gap = np.random.uniform(7,10)
        #MARGIN = 2
        #position_gap = np.random.uniform(10, 12)

        for i in range(0, self.NUMBER_VEHICLES):
            if self.behavior == 5:  # mixed
                beh = np.random.uniform(self.min_coop, 4)
            if self.behavior == 6:  # cooperative
                beh = np.random.uniform(2.0, 4)
            if self.behavior == 7:  # non-cooperative
                beh = np.random.uniform(self.min_coop, 2)

            if not self.EVALUATE:
                beh = np.random.uniform(0.0, 4.0)

            road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(i*position_gap +
            np.random.uniform(-1*MARGIN, 1*MARGIN), 0), beh, velocity=np.random.uniform(3,4)))

        self.vehicle = ego_vehicle


register(
    id='merge2-v0',
    entry_point='highway_env.envs:MergeEnv2',
)

