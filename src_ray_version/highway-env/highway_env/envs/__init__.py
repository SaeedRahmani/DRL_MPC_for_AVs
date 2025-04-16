from highway_env.envs.exit_env import ExitEnv
from highway_env.envs.highway_env import HighwayEnv, HighwayEnvFast
from highway_env.envs.intersection_env import (
    ContinuousIntersectionEnv,
    IntersectionEnv,
    MultiAgentIntersectionEnv,
)
from highway_env.envs.lane_keeping_env import LaneKeepingEnv
from highway_env.envs.merge_env import MergeEnv
from highway_env.envs.parking_env import (
    ParkingEnv,
    ParkingEnvActionRepeat,
    ParkingEnvParkedVehicles,
)
from highway_env.envs.racetrack_env import RacetrackEnv
from highway_env.envs.roundabout_env import RoundaboutEnv
from highway_env.envs.two_way_env import TwoWayEnv
from highway_env.envs.u_turn_env import UTurnEnv

from highway_env.envs.intersection_mpc_env import (
    IntersectionMpcEnv_noCA,
    IntersectionMpcEnv_manual,
    IntersectionMpcEnv_cost,
    IntersectionMpcEnv_constraint,
)
from highway_env.envs.intersection_mpcrl_speeds_env import (
    IntersectionMpcrlSpeedsEnv_noCA,
    IntersectionMpcrlSpeedsEnv_manual,
    IntersectionMpcrlSpeedsEnv_cost,
    IntersectionMpcrlSpeedsEnv_constraint,
)
from highway_env.envs.intersection_mpcrl_weights_env import (
    IntersectionMpcrlWeightsEnv_noCA,
    IntersectionMpcrlWeightsEnv_manual,
    IntersectionMpcrlWeightsEnv_cost,
)


__all__ = [
    "ExitEnv",
    "HighwayEnv",
    "HighwayEnvFast",
    "IntersectionEnv",
    "ContinuousIntersectionEnv",
    "MultiAgentIntersectionEnv",
    "LaneKeepingEnv",
    "MergeEnv",
    "ParkingEnv",
    "ParkingEnvActionRepeat",
    "ParkingEnvParkedVehicles",
    "RacetrackEnv",
    "RoundaboutEnv",
    "TwoWayEnv",
    "UTurnEnv",
    # ---------- PureMpc ---------- #
    "IntersectionMpcEnv_noCA",                  # w/o collision avoidance
    "IntersectionMpcEnv_manual",                # w/ manual collision avoidance
    "IntersectionMpcEnv_cost",                  # w/ collision cost in MPC
    "IntersectionMpcEnv_constraint",            # w/ collision avoidance constraint in MPC

    # ---------- MPCRL: Reference speeds ---------- #
    "IntersectionMpcrlSpeedsEnv_noCA",          # w/o collision avoidance
    "IntersectionMpcrlSpeedsEnv_manual",        # w/ manual collision avoidance
    "IntersectionMpcrlSpeedsEnv_cost",          # w/ collision cost in MPC
    "IntersectionMpcrlSpeedsEnv_constraint",    # w/ collision avoidance constraint in MPC
    
    # ---------- MPCRL: Dynamic weights ----------- #
    "IntersectionMpcrlWeightsEnv_noCA",         # w/o collision avoidance
    "IntersectionMpcrlWeightsEnv_manual",       # w/ manual collision avoidance
    "IntersectionMpcrlWeightsEnv_cost",         # w/o collision cost in MPC
]
