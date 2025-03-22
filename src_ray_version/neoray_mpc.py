import ray
import gymnasium
from highway_env.envs import IntersectionMpcEnv
from ray.tune.registry import register_env
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModule
from pprint import pprint
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.connectors.env_to_module import FlattenObservations


ray.init(local_mode=True) 

def env_creator(config):
    # return gymnasium.make("intersection-mpc-v0", render_mode="rgb_array")
    return IntersectionMpcEnv()


env = gymnasium.make("intersection-mpc-v0", render_mode="rgb_array")

register_env(name="intersection-mpc-v0", env_creator=env_creator)


config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    .framework("torch")
    .environment("intersection-mpc-v0")
    .learners(
        num_learners=1,
        num_gpus_per_learner=0,
    )
    .env_runners(
        enable_connectors=True,
        env_to_module_connector=lambda env: FlattenObservations(),
    )
    .training(
        lr=1e-4,
        train_batch_size_per_learner=32,
        num_epochs=1,
        minibatch_size=16,
    )
    .rl_module(
        rl_module_spec=RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
        observation_space=gymnasium.make("intersection-mpc-v0").observation_space,
        action_space=gymnasium.make("intersection-mpc-v0").action_space,
            catalog_class=PPOCatalog,
            inference_only=False,
            model_config=DefaultModelConfig(
                fcnet_hiddens=[32, 32],
                fcnet_activation="relu"
            )
        )
    )
)

# Build the algorithm.
algo = config.build_algo()
algo.train()

# ... and evaluate it.
pprint(algo.evaluate())

# Release the algo's resources (remote actors, like EnvRunners and Learners).
algo.stop()
