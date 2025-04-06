from .intersection_mpc_env import (
    IntersectionMpcEnv_v0,
    IntersectionMpcEnv_v1,
)
from highway_env.envs.common.action import (
    Action,
    PureMpcAction,
    DynamicWeightsAction,
    ReferenceSpeedAction,
)
import numpy as np


class IntersectionMpcrlWeightsEnv_v0(IntersectionMpcEnv_v0):
    """ MPCRL: Dynamic weights without collision avoidance. """
    def __init__(self, config: dict = None, render_mode: str | None = None):
        super().__init__(config=config, render_mode=render_mode)

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "DynamicWeightsAction",
                    "num_weights": 6,
                },
            }
        )
        return config

    def __str__(self) -> str:
        return f"<Intersection-MpcRL-Env <Dynamic weights> [NO Collision Avoidance]>"

    def __repr__(self) -> str:
        return self.__str__()

    def _predict_mpc_action(self, action: Action) -> Action:
        """ Predict the action of ego vehicle using MPC. """
        self._prepare_obs()

        test_speeds = [5.0, 10.0, 15.0]  # Test three different speeds
        steps_per_speed = 33  # Change speed every 33 steps
        if self.time_index % steps_per_speed == 0 and self.current_speed_idx < len(test_speeds):
            self.ref_speed = test_speeds[self.current_speed_idx]
            self.current_speed_idx = (
                self.current_speed_idx + 1) % len(test_speeds)

        # the dynamic weights are used from RL agent.
        weights = action
        mpc_action = self._solve_mpc(
            weights=weights, ref_speed=np.array([[self.ref_speed]]))
        return mpc_action


class IntersectionMpcrlWeightsEnv_v1(IntersectionMpcEnv_v1):
    """ MPCRL: Dynamic weights with manual external collision avoidance. """
    def __init__(self, config: dict = None, render_mode: str | None = None):
        super().__init__(config=config, render_mode=render_mode)
    
    def __str__(self) -> str:
        return f"<Intersection-MpcRL-Env <Dynamic weights> [Manual Collision Avoidance]>"
    
    def __repr__(self) -> str:
        return self.__str__()