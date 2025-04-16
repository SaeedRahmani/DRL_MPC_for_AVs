import hydra
import gymnasium
import highway_env
from omegaconf import DictConfig


def test_mpc_envs(mpc_cfg: DictConfig):
    # List of environments to test
    envs = [
        "intersection-mpc-noCA",
        "intersection-mpc-manual",
        "intersection-mpc-cost",
        "intersection-mpc-constraint",
    ]

    for env_name in envs:
        try:
            env = gymnasium.make(env_name)
            assert env.action_space.shape == (
                mpc_cfg.action_dim,), f"{env_name}'s action shape is wrong."
            default_config = env.unwrapped.default_config()
            action_type = default_config["action"]["type"]
            assert action_type == "PureMpcAction", f"{env_name}'s action type is wrong."

            env.close()

            print(f"{env.unwrapped} is a valid environment.")
            print(f"{action_type} space: {env.action_space.shape}")

        except Exception as e:
            print(f"Failed to create {env_name}: {e}")


def test_mpcrl_v0_envs(mpcrl_cfg: DictConfig):
    # List of environments to test
    envs = [
        "intersection-mpcrl-refspeed-noCA",
        "intersection-mpcrl-refspeed-manual",
        "intersection-mpcrl-refspeed-cost",
        "intersection-mpcrl-refspeed-constraint",
    ]

    for env_name in envs:
        try:
            env = gymnasium.make(env_name)
            assert env.action_space.shape == (
                mpcrl_cfg.v0.action_dim,), f"{env_name}'s action shape is wrong."
            default_config = env.unwrapped.default_config()
            action_type = default_config["action"]["type"]
            assert action_type == "ReferenceSpeedAction", f"{env_name}'s action type is wrong."

            env.close()

            print(f"{env.unwrapped} is a valid environment.")
            print(f"{action_type} space: {env.action_space.shape}")

        except Exception as e:
            print(f"Failed to create {env_name}: {e}")


def test_mpcrl_v1_envs(mpcrl_cfg: DictConfig):
    # List of environments to test
    envs = [
        "intersection-mpcrl-dynamicweights-noCA",
        "intersection-mpcrl-dynamicweights-manual",
        "intersection-mpcrl-dynamicweights-cost",
    ]

    for env_name in envs:
        try:
            env = gymnasium.make(env_name)
            assert env.action_space.shape == (
                mpcrl_cfg.v1.action_dim,), f"{env_name}'s action shape is wrong."

            default_config = env.unwrapped.default_config()
            action_type = default_config["action"]["type"]
            assert action_type == "DynamicWeightsAction", f"{env_name}'s action type is wrong."

            env.close()

            print(f"{env.unwrapped} is a valid environment.")
            print(f"{action_type} space: {env.action_space.shape}")

        except Exception as e:
            print(f"Failed to create {env_name}: {e}")


@hydra.main(version_base=None, config_path=".", config_name="config")
def test(cfg: DictConfig):
    test_mpc_envs(mpc_cfg=cfg.mpc)
    test_mpcrl_v0_envs(mpcrl_cfg=cfg.env)
    test_mpcrl_v1_envs(mpcrl_cfg=cfg.env)


if __name__ == "__main__":
    test()
    print("All tests passed.")
