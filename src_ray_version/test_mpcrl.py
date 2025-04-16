import hydra
import gymnasium
import highway_env
from pprint import pp
from omegaconf import DictConfig
from highway_env.envs import (
    IntersectionMpcrlWeightsEnv_noCA, 
    IntersectionMpcrlWeightsEnv_manual,
    IntersectionMpcrlWeightsEnv_cost,
    IntersectionMpcrlSpeedsEnv_noCA,
    IntersectionMpcrlSpeedsEnv_manual,
    IntersectionMpcrlSpeedsEnv_cost,
    IntersectionMpcrlSpeedsEnv_constraint,
)


ENV_CLASS_MAPPING = {
    "intersection-mpcrl-dynamicweights-noCA": IntersectionMpcrlWeightsEnv_noCA,
    "intersection-mpcrl-dynamicweights-manual": IntersectionMpcrlWeightsEnv_manual,
    "intersection-mpcrl-dynamicweights-cost": IntersectionMpcrlWeightsEnv_cost,
    "intersection-mpcrl-refspeed-noCA": IntersectionMpcrlSpeedsEnv_noCA,
    "intersection-mpcrl-refspeed-manual": IntersectionMpcrlSpeedsEnv_manual,
    "intersection-mpcrl-refspeed-cost": IntersectionMpcrlSpeedsEnv_cost,
    "intersection-mpcrl-refspeed-constraint": IntersectionMpcrlSpeedsEnv_constraint,
}

@hydra.main(version_base=None, config_path=".", config_name="config")
def run_mpc_agent(cfg: DictConfig):
    env_cfg = cfg.env

    # Choose the gymnaisum env
    env_version: str = env_cfg.env_version
    subenv_version: str = env_cfg.subenv_version
    env_name: str = f"intersection-mpcrl-dynamicweights-{subenv_version}" \
                    if env_version == "v1" else f"intersection-mpcrl-refspeed-{subenv_version}"
    if env_version == "v1" and subenv_version == "constraint":
        raise ValueError("Do not implement constraint CA for dynamic weight RL agent.")
    action_dim = env_cfg[env_version]["action_dim"]

    # Create the environment with MPC controller built-in
    # The environment version is specified in the config file
    env = gymnasium.make(env_name, render_mode="rgb_array", action_dim=action_dim)
    default_config = env.unwrapped.default_config()
    action_type = default_config["action"]["type"]

    print(f"ENV: {env.unwrapped}")
    print(f"{action_type} space: {env.action_space.shape}")
    pp(env.unwrapped.default_weights)
    # Initialize the environment
    _, _ = env.reset()

    # Run simulation for up to 200 steps
    for i in range(200):
        # ! NOTICE !
        # For MPC environments, the action passed here is essentially ignored.
        # 
        # The actual control actions are computed internally by the MPC controller
        # during the environment's step function execution.
        # 
        # The action cannot be set to None, since the wrapper will replace a dummt value.
        _, reward, terminated, truncated, info = env.step(
            action=env.action_space.sample())

        env.render()
        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    run_mpc_agent()