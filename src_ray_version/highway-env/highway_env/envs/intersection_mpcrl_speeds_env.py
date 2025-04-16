from .intersection_mpc_env import (
    IntersectionMpcEnv_noCA,
    IntersectionMpcEnv_manual,
    IntersectionMpcEnv_cost,
    IntersectionMpcEnv_constraint,
)
from highway_env.envs.common.action import (
    Action,
    PureMpcAction,
    DynamicWeightsAction,
    ReferenceSpeedAction,
)
import numpy as np


class IntersectionMpcrlSpeedsEnv_noCA(IntersectionMpcEnv_noCA):
    """ MPCRL: Reference speed without collision avoidance. """
    def __init__(self, config: dict = None, render_mode: str | None = None):
        super().__init__(config=config, render_mode=render_mode)

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "ReferenceSpeedAction",  # use 6 to predict 30, optimal: 16
                },
            }
        )
        return config

    def __str__(self) -> str:
        return f"<Intersection-MpcRL-Env <REF SPEEDS> [NO CA]>"

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
        ref_speed = action
        mpc_action = self._solve_mpc(weights=None, ref_speed=ref_speed)
        return mpc_action

class IntersectionMpcrlSpeedsEnv_manual(IntersectionMpcEnv_manual):
    """ MPCRL: Reference speed with manual collision avoidance. """
    def __init__(self, config: dict = None, render_mode: str | None = None):
        super().__init__(config=config, render_mode=render_mode)

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "ReferenceSpeedAction",  # use 6 to predict 30, optimal: 16
                },
            }
        )
        return config
    
    def __str__(self) -> str:
        return f"<Intersection-MpcRL-Env <REF SPEEDS> [MANUAL]>"
    
    def __repr__(self) -> str:
        return self.__str__()
    
class IntersectionMpcrlSpeedsEnv_cost(IntersectionMpcEnv_cost):
    """ MPCRL: Reference speed with collision cost in MPC. """
    def __init__(self, config: dict = None, render_mode: str | None = None):
        super().__init__(config=config, render_mode=render_mode)

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "ReferenceSpeedAction",  # use 6 to predict 30, optimal: 16
                },
            }
        )
        return config
    
    def __str__(self) -> str:
        return f"<Intersection-MpcRL-Env <REF SPEEDS> [COST]>"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    
class IntersectionMpcrlSpeedsEnv_constraint(IntersectionMpcEnv_constraint):
    """ MPCRL: Reference speed with  collision avoidance constraint in MPC. """
    def __init__(self, config: dict = None, render_mode: str | None = None):
        super().__init__(config=config, render_mode=render_mode)

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "ReferenceSpeedAction",  # use 6 to predict 30, optimal: 16
                },
            }
        )
        return config
    
    def __str__(self) -> str:
        return f"<Intersection-MpcRL-Env <REF SPEEDS> [CONSTRAINT]>"
    
    def __repr__(self) -> str:
        return self.__str__()