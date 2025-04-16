from .intersection_mpc_env import (
    IntersectionMpcEnv_noCA,
    IntersectionMpcEnv_manual,
    IntersectionMpcEnv_cost,
    IntersectionMpcEnv_constraint,
)
from highway_env.envs.common.action import Action


class IntersectionMpcrlSpeedsEnv_noCA(IntersectionMpcEnv_noCA):
    """ MPCRL: Reference speed without collision avoidance. """

    def __init__(self, config: dict = None, render_mode: str | None = None, action_dim: int = 1):
        super().__init__(config=config, render_mode=render_mode)
        self.action_dim = action_dim
        self.config["action"]["num_reference_points"] = self.action_dim
        self.define_spaces()
        
        self.CA_mode = "noCA"
        assert self.CA_mode == "noCA", "Expect CA mode to be `noCA`."
        self.agent_mode = "MPC-RL<Reference speed>"
        assert self.agent_mode == "MPC-RL<Reference speed>", "Expect agent mode to be `MPC-RL<Reference speed>`."

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "ReferenceSpeedAction",
                    "num_reference_points": 1,
                },
            }
        )
        return config


class IntersectionMpcrlSpeedsEnv_manual(IntersectionMpcEnv_manual):
    """ MPCRL: Reference speed with manual collision avoidance. """

    def __init__(self, config: dict = None, render_mode: str | None = None, action_dim: int = 1):
        super().__init__(config=config, render_mode=render_mode)
        self.action_dim = action_dim
        self.config["action"]["num_reference_points"] = self.action_dim
        self.define_spaces()

        self.CA_mode = "manual"
        assert self.CA_mode == "manual", "Expect CA mode to be `manual`."
        self.agent_mode = "MPC-RL<Reference speed>"
        assert self.agent_mode == "MPC-RL<Reference speed>", "Expect agent mode to be `MPC-RL<Reference speed>`."

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "ReferenceSpeedAction",  # use 6 to predict 30, optimal: 16
                    "num_reference_points": 1,
                },
            }
        )
        return config


class IntersectionMpcrlSpeedsEnv_cost(IntersectionMpcEnv_cost):
    """ MPCRL: Reference speed with collision cost in MPC. """

    def __init__(self, config: dict = None, render_mode: str | None = None, action_dim: int = 1):
        super().__init__(config=config, render_mode=render_mode)
        self.action_dim = action_dim
        self.config["action"]["num_reference_points"] = self.action_dim
        self.define_spaces()
        
        self.CA_mode = "cost"
        assert self.CA_mode == "cost", "Expect CA mode to be `cost`."
        self.agent_mode = "MPC-RL<Reference speed>"

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "ReferenceSpeedAction",
                    "num_reference_points": 1,
                },
            }
        )
        return config


class IntersectionMpcrlSpeedsEnv_constraint(IntersectionMpcEnv_constraint):
    """ MPCRL: Reference speed with collision avoidance constraint in MPC. """

    def __init__(self, config: dict = None, render_mode: str | None = None):
        super().__init__(config=config, render_mode=render_mode)
        self.CA_mode = "constraint"
        assert self.CA_mode == "constraint", "Expect CA mode to be `constraint`."
        self.agent_mode = "MPC-RL<Reference speed>"
        assert self.agent_mode == "MPC-RL<Reference speed>", "Expect agent mode to be `MPC-RL<Reference speed>`."

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "ReferenceSpeedAction",
                    "num_reference_points": 1,
                },
            }
        )
        return config
