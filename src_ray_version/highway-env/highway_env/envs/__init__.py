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
    IntersectionMpcEnv_v0,  # No collision avoidance
    IntersectionMpcEnv_v1,  # With manual external collision avoidance
    )
from highway_env.envs.intersection_mpcrl_speeds_env import (
    IntersectionMpcrlSpeedsEnv_v0)
from highway_env.envs.intersection_mpcrl_weights_env import (
    IntersectionMpcrlWeightsEnv_v0)


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
    "IntersectionMpcEnv_v0",    # No collision avoidance
    "IntersectionMpcEnv_v1",    # With manual external collision avoidance
    # ---------- MPCRL: Reference speeds ---------- #
    "IntersectionMpcrlSpeedsEnv_v0",    # No collision avoidance
    # ---------- MPCRL: Dynamic weights ----------- #
    "IntersectionMpcrlWeightsEnv_v0",   # No collision avoidance
]
