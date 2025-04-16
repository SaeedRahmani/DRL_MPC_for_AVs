import hydra
import gymnasium
import highway_env
from pprint import pprint
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=".", config_name="config")
def run_mpc_agent(cfg: DictConfig):
    mpc_config = cfg.mpc

    # Choose the gymnaisum env
    assert mpc_config.CA_mode in [
        "noCA", "manual", "cost", "constraint"], f"Invalid collision mode: {mpc_config.CA_mode}."
    env_name = mpc_config.CA_mode

    # Create the environment with MPC controller built-in
    # The environment version is specified in the config file
    env = gymnasium.make(
        f"intersection-mpc-{env_name}", render_mode="rgb_array")
    print(f"ENV: {env.unwrapped}")

    # Initialize the environment
    _, _ = env.reset()

    # Run simulation for up to 200 steps
    for i in range(200):
        # For MPC environments, the action passed here is essentially ignored.
        # The actual control actions are computed internally by the MPC controller
        # during the environment's step function execution.
        # The action cannot be set to None, since the wrapper will replace a dummt value.
        _, reward, terminated, truncated, info = env.step(
            action=env.action_space.sample())

        # Render the current state
        env.render()

        # End the episode if terminated or truncated
        if terminated or truncated:
            break

    # Clean up environment resources
    env.close()


if __name__ == "__main__":
    run_mpc_agent()
