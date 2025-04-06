import hydra
import numpy as np
import gymnasium
import highway_env
from pprint import pprint
from omegaconf import DictConfig
np.set_printoptions(suppress=True)


@hydra.main(version_base=None, config_path=".", config_name="config")
def run_mpc_agent(cfg: DictConfig):

    mpc_config = cfg.mpc

    # Create the MPC environment
    env = gymnasium.make(
        f"intersection-mpc-{mpc_config.env_version}", render_mode="rgb_array")
    print(f"ENV: {env.unwrapped}")

    _, _ = env.reset()
    for i in range(200):
        # For MPC environments, we pass an empty a dummy action. but
        # The MPC controller inside the environment will generate the actual control (inside _simulate)
        _, _, terminated, truncated, _ = env.step(
            action=env.action_space.sample())
        env.render()
        if terminated or truncated:
            break
    env.close()


if __name__ == "__main__":
    run_mpc_agent()
