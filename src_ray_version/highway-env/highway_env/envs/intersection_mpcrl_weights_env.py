from .intersection_mpc_env import (
    IntersectionMpcEnv_noCA,
    IntersectionMpcEnv_manual,
    IntersectionMpcEnv_cost,
)
from highway_env.envs.common.action import Action
import numpy as np


class IntersectionMpcrlWeightsEnv_noCA(IntersectionMpcEnv_noCA):
    """ MPCRL: Dynamic weights without collision avoidance. """

    def __init__(self, config: dict = None, render_mode: str | None = None, action_dim: int = 7):
        super().__init__(config=config, render_mode=render_mode)
        self.action_dim = action_dim
        self.config["action"]["num_weights"] = self.action_dim
        self.define_spaces()

        self.CA_mode = "noCA"
        assert self.CA_mode == "noCA", "Expect CA mode to be `noCA`."
        self.agent_mode = "MPC-RL<Dynamic weights>"

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "DynamicWeightsAction",
                    "num_weights": 7,
                },
            }
        )
        return config


class IntersectionMpcrlWeightsEnv_manual(IntersectionMpcEnv_manual):
    """ MPCRL: Dynamic weights with manual external collision avoidance. """

    def __init__(self, config: dict = None, render_mode: str | None = None, action_dim: int = 7):
        super().__init__(config=config, render_mode=render_mode)
        self.action_dim = action_dim
        self.config["action"]["num_weights"] = self.action_dim
        self.define_spaces()
        self.CA_mode = "manual"
        assert self.CA_mode == "manual", "Expect CA mode to be `manual`."
        self.agent_mode = "MPC-RL<Dynamic weights>"

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "DynamicWeightsAction",
                    "num_weights": 7,
                },
            }
        )
        return config


class IntersectionMpcrlWeightsEnv_cost(IntersectionMpcEnv_cost):
    """ MPCRL: Dynamic weights with collision avoidance cost in MPC. """
    agent_mode = "MPC-RL<Dynamic weights>"

    def __init__(self, config: dict = None, render_mode: str | None = None, action_dim: int = 7):
        super().__init__(config=config, render_mode=render_mode)
        self.action_dim = action_dim
        self.config["action"]["num_weights"] = self.action_dim
        self.define_spaces()
        self.CA_mode = "cost"
        assert self.CA_mode == "cost", "Expect CA mode to be `cost`."
        self.agent_mode = "MPC-RL<Dynamic weights>"

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "DynamicWeightsAction",
                    "num_weights": 7,
                },
            }
        )
        return config
