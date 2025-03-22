import gymnasium
from highway_env.envs import IntersectionMpcEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.tune.registry import register_env
from pprint import pprint


def env_creator(config):
    # return gymnasium.make("intersection-mpc-v0", render_mode="rgb_array")
    return IntersectionMpcEnv()

register_env(name="intersection-mpc-v0", env_creator=env_creator)

# Configure the algorithm.
config = (
    PPOConfig()
    .environment("intersection-mpc-v0")
    .env_runners(
        num_env_runners=2,
        # Observations are discrete (ints) -> We need to flatten (one-hot) them.
        env_to_module_connector=lambda env: FlattenObservations(),
    )
    .evaluation(
        evaluation_num_env_runners=1,
        evaluation_interval=1)
)
config.api_stack(enable_rl_module_and_learner=False,enable_env_runner_and_connector_v2=False)

# Build the algorithm.
algo = config.build_algo()
algo.train()

# ... and evaluate it.
pprint(algo.evaluate())

# Release the algo's resources (remote actors, like EnvRunners and Learners).
algo.stop()