import os
import sys

from gymnasium.envs.registration import register


__version__ = "1.10.1"

try:
    from farama_notifications import notifications

    if "highway_env" in notifications and __version__ in notifications["gymnasium"]:
        print(notifications["highway_env"][__version__], file=sys.stderr)

except Exception:  # nosec
    pass

# Hide pygame support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


def _register_highway_envs():
    """Import the envs module so that envs register themselves."""

    from highway_env.envs.common.abstract import MultiAgentWrapper

    # exit_env.py
    register(
        id="exit-v0",
        entry_point="highway_env.envs.exit_env:ExitEnv",
    )

    # highway_env.py
    register(
        id="highway-v0",
        entry_point="highway_env.envs.highway_env:HighwayEnv",
    )

    register(
        id="highway-fast-v0",
        entry_point="highway_env.envs.highway_env:HighwayEnvFast",
    )

    # intersection_env.py
    register(
        id="intersection-v0",
        entry_point="highway_env.envs.intersection_env:IntersectionEnv",
    )

    register(
        id="intersection-v1",
        entry_point="highway_env.envs.intersection_env:ContinuousIntersectionEnv",
    )

    register(
        id="intersection-multi-agent-v0",
        entry_point="highway_env.envs.intersection_env:MultiAgentIntersectionEnv",
    )

    register(
        id="intersection-multi-agent-v1",
        entry_point="highway_env.envs.intersection_env:MultiAgentIntersectionEnv",
        additional_wrappers=(MultiAgentWrapper.wrapper_spec(),),
    )

    # lane_keeping_env.py
    register(
        id="lane-keeping-v0",
        entry_point="highway_env.envs.lane_keeping_env:LaneKeepingEnv",
        max_episode_steps=200,
    )

    # merge_env.py
    register(
        id="merge-v0",
        entry_point="highway_env.envs.merge_env:MergeEnv",
    )

    # parking_env.py
    register(
        id="parking-v0",
        entry_point="highway_env.envs.parking_env:ParkingEnv",
    )

    register(
        id="parking-ActionRepeat-v0",
        entry_point="highway_env.envs.parking_env:ParkingEnvActionRepeat",
    )

    register(
        id="parking-parked-v0",
        entry_point="highway_env.envs.parking_env:ParkingEnvParkedVehicles",
    )

    # racetrack_env.py
    register(
        id="racetrack-v0",
        entry_point="highway_env.envs.racetrack_env:RacetrackEnv",
    )

    # roundabout_env.py
    register(
        id="roundabout-v0",
        entry_point="highway_env.envs.roundabout_env:RoundaboutEnv",
    )

    # two_way_env.py
    register(
        id="two-way-v0",
        entry_point="highway_env.envs.two_way_env:TwoWayEnv",
        max_episode_steps=15,
    )

    # u_turn_env.py
    register(id="u-turn-v0", entry_point="highway_env.envs.u_turn_env:UTurnEnv")

    # --------------------- #
    # intersection_mpc_env.py
    # PureMpc

    # w/o collision avoidance
    register(
        id="intersection-mpc-noCA",
        entry_point="highway_env.envs.intersection_mpc_env:IntersectionMpcEnv_noCA",
    )
    # w/ manual collision avoidance
    register(
        id="intersection-mpc-manual",
        entry_point="highway_env.envs.intersection_mpc_env:IntersectionMpcEnv_manual",
    )
    # w/ collision cost in MPC
    register(
        id="intersection-mpc-cost",
        entry_point="highway_env.envs.intersection_mpc_env:IntersectionMpcEnv_cost",
    )
    # w/ collision avoidance in MPC
    register(
        id="intersection-mpc-constraint",
        entry_point="highway_env.envs.intersection_mpc_env:IntersectionMpcEnv_constraint",
    )
    # --------------------- #

    # ------------------------------ #
    # intersection_mpcrl_speeds_env.py
    # MPCRL: Reference speeds

    # w/o collision avoidance
    register(
        id="intersection-mpcrl-refspeed-noCA",
        entry_point="highway_env.envs.intersection_mpcrl_speeds_env:IntersectionMpcrlSpeedsEnv_noCA",
    )
    # w/ manual collision avoidance
    register(
        id="intersection-mpcrl-refspeed-manual",
        entry_point="highway_env.envs.intersection_mpcrl_speeds_env:IntersectionMpcrlSpeedsEnv_manual",
    )
    # w/ collision cost in MPC
    register(
        id="intersection-mpcrl-refspeed-cost",
        entry_point="highway_env.envs.intersection_mpcrl_speeds_env:IntersectionMpcrlSpeedsEnv_cost",
    )
    # w/ collision avoidance in MPC
    register(
        id="intersection-mpcrl-refspeed-constraint",
        entry_point="highway_env.envs.intersection_mpcrl_speeds_env:IntersectionMpcrlSpeedsEnv_constraint",
    )
    # ------------------------------ #

    # ------------------------------- #
    # intersection_mpcrl_weights_env.py
    # MPCRL: Dynamic weights

    # w/o collision avoidance
    register(
        id="intersection-mpcrl-dynamicweights-noCA",
        entry_point="highway_env.envs.intersection_mpcrl_weights_env:IntersectionMpcrlWeightsEnv_noCA",
    )
    # w/ manual collision avoidance
    register(
        id="intersection-mpcrl-dynamicweights-manual",
        entry_point="highway_env.envs.intersection_mpcrl_weights_env:IntersectionMpcrlWeightsEnv_manual",
    )
    # w/ collision cost in MPC
    register(    
        id="intersection-mpcrl-dynamicweights-cost",
        entry_point="highway_env.envs.intersection_mpcrl_weights_env:IntersectionMpcrlWeightsEnv_cost",
    )
    # ------------------------------- #


_register_highway_envs()
