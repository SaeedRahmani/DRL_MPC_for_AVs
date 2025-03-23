import gymnasium
from highway_env.envs import IntersectionMpcrlEnv_v1
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.tune.registry import register_env
from pprint import pprint

env_name = "intersection-mpcrl-dynamicweights-v0" # "intersection-mpcrl-refspeed-v0"

def env_creator(config):
    # return gymnasium.make("intersection-mpc-v0", render_mode="rgb_array")
    return IntersectionMpcrlEnv_v1()

register_env(name=env_name, env_creator=env_creator)

# Configure the algorithm.
config = (
    PPOConfig()
    .framework("torch")
    .environment(env_name)
    .env_runners(
        num_env_runners=1,
        # env_to_module_connector=lambda env: FlattenObservations(), # NOTE: do we need it? Seems not.
    )
    .resources(num_gpus=1)
    .training(
        lr=1e-4,
        train_batch_size_per_learner=64,
        num_epochs=1,
    )
    .evaluation(
        evaluation_num_env_runners=1,
        evaluation_interval=1)
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )
)

# Build the algorithm.
algo = config.build_algo()
algo.train()

# ... and evaluate it.
pprint(algo.evaluate())

# Release the algo's resources (remote actors, like EnvRunners and Learners).
algo.stop()