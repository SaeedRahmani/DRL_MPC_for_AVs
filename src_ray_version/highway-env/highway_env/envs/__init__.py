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
)
from highway_env.envs.intersection_mpcrl_speeds_env import (
    IntersectionMpcrlSpeedsEnv_noCA,
    IntersectionMpcrlSpeedsEnv_manual,
)
from highway_env.envs.intersection_mpcrl_weights_env import (
    IntersectionMpcrlWeightsEnv_noCA,
    IntersectionMpcrlWeightsEnv_manual,
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
    "IntersectionMpcEnv_noCA",                  # No collision avoidance
    "IntersectionMpcEnv_manual",              # With manual collision avoidance

    # ---------- MPCRL: Reference speeds ---------- #
    "IntersectionMpcrlSpeedsEnv_noCA",          # No collision avoidance
    "IntersectionMpcrlSpeedsEnv_manual",      # With manual collision avoidance

    # ---------- MPCRL: Dynamic weights ----------- #
    "IntersectionMpcrlWeightsEnv_noCA",         # No collision avoidance
    "IntersectionMpcrlWeightsEnv_manual",     # With manual collision avoidance
]
