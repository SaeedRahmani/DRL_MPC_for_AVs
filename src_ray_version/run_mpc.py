import hydra
import numpy as np
import gymnasium
import highway_env
from pprint import pprint
from omegaconf import DictConfig

# Uncomment if you need to suppress scientific notation in numpy arrays
# np.set_printoptions(suppress=True)


@hydra.main(version_base=None, config_path=".", config_name="config")
def run_mpc_agent(cfg: DictConfig):
    """
    Run a Model Predictive Control (MPC) agent in the highway intersection environment.

    This function creates an environment with MPC controller built-in, then runs a
    simulation where the controller automatically handles vehicle' acceleration and steering
    through the intersection.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra, containing MPC parameters
                         in the 'mpc' section
    """
    # Extract MPC-specific configuration
    mpc_config = cfg.mpc

    # Create the environment with MPC controller built-in
    # The environment version is specified in the config file
    env = gymnasium.make(
        f"intersection-mpc-{mpc_config.env_version}", render_mode="rgb_array")
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
