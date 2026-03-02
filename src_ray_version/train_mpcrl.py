import logging
import ray
import ray.tune as tune
import hydra
# import gymnasium
from gymnasium.envs.registration import VectorizeMode
import ray.tune
from highway_env.envs import (
    IntersectionMpcrlWeightsEnv_noCA, 
    IntersectionMpcrlWeightsEnv_manual,
    IntersectionMpcrlWeightsEnv_cost,
    IntersectionMpcrlSpeedsEnv_noCA,
    IntersectionMpcrlSpeedsEnv_manual,
    IntersectionMpcrlSpeedsEnv_cost,
    IntersectionMpcrlSpeedsEnv_constraint,
)
from omegaconf import DictConfig, OmegaConf
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.logger import (
    JsonLoggerCallback,
    CSVLoggerCallback,
    TBXLoggerCallback
)
from ray.tune import TuneConfig, RunConfig, CLIReporter
from ray.train import CheckpointConfig
from ray.tune.schedulers import ASHAScheduler
from pprint import pprint


ALGO_CONFIG_MAPPING = {
    "PPO": PPOConfig,
    "SAC": SACConfig,
}
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
def train_mpcrl_agent(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    ray.shutdown()

    ray.init(
        num_cpus=16,  
        num_gpus=1,
        logging_level=logging.INFO,
        log_to_driver=False,    # disable the pid loggings.
        include_dashboard=False,
    )

    pprint(ray.available_resources())

    framework: str = cfg.rllib.framework # torch
    use_rllib_new_API_stack: bool = cfg.rllib.use_new_API_stack
    
    env_version: str = cfg.env.env_version
    subenv_version: str = cfg.env.subenv_version
    env_class_name: str = f"intersection-mpcrl-dynamicweights-{subenv_version}" \
                    if env_version == "v1" else f"intersection-mpcrl-refspeed-{subenv_version}"
    if env_version == "v1" and subenv_version == "constraint":
        raise ValueError("Do not implement constraint CA for dynamic weight RL agent.")
                     
    algo_name: str = cfg.agent.version
    algo_parameters = cfg.agent[algo_name]
    algo_config_class = ALGO_CONFIG_MAPPING[algo_name]
              
    def env_creator(config):
        return ENV_CLASS_MAPPING[env_class_name](config=config, render_mode="rgb_array")
                    
    register_env(
        name=env_class_name,
        env_creator=env_creator,
    )

    # Create a Env instance first to register the environment
    env = ENV_CLASS_MAPPING[env_class_name](config=None, render_mode="rgb_array", action_dim=cfg.env[env_version]["action_dim"])
    default_config = env.default_config()
    action_type = default_config["action"]["type"]

    print(f"ENV: {env}")
    print(f"{action_type} space: {env.action_space.shape}")
    pprint(env.default_weights)

    config = (
        algo_config_class()
        .api_stack(
            enable_rl_module_and_learner=use_rllib_new_API_stack,
            enable_env_runner_and_connector_v2=use_rllib_new_API_stack,
        )
        .framework(framework)
        # https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.
        # environment.html#ray.rllib.algorithms.algorithm_config.AlgorithmConfig.environment
        .environment(
            env=env_class_name,
            render_env=False, # FIXME: enable visualization later for debugging
            is_atari=False,
            disable_env_checking=True,
        )
        # https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.
        # env_runners.html#ray.rllib.algorithms.algorithm_config.AlgorithmConfig.env_runners
        .env_runners(
            num_env_runners=cfg.rllib.num_env_runners,
            num_cpus_per_env_runner=cfg.rllib.num_cpus_per_env_runner,
            num_gpus_per_env_runner=0,  # EnvRunners run MPC on CPU; they don't need GPU
            num_envs_per_env_runner=cfg.rllib.num_envs_per_env_runner,
            gym_env_vectorize_mode=VectorizeMode.ASYNC, 
            # ASYNC parallelizes sub-environments within each runner.
            # Requires num_envs_per_env_runner > 1 to have any effect.
        )
        # Give the GPU to the learner (local Algorithm on old API stack)
        .resources(
            num_gpus=cfg.rllib.num_gpus,
        )
        # https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.
        # training.html#ray.rllib.algorithms.algorithm_config.AlgorithmConfig.training
        .training(
            gamma=cfg.agent.gamma,
            lr=cfg.agent.lr,
            num_epochs=cfg.agent.num_epochs,
            train_batch_size=cfg.agent.train_batch_size,
            minibatch_size=cfg.agent.minibatch_size,
            shuffle_batch_per_epoch=cfg.agent.shuffle_batch_per_epoch,
            grad_clip=0.5,           # prevent NaN explosion from large reward gradients
            grad_clip_by="global_norm",
            torch_skip_nan_gradients=True,  # skip update if NaN detected instead of corrupting model
            model={
                "fcnet_hiddens": [512, 256],
            },
        )
        # https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.
        # evaluation.html#ray.rllib.algorithms.algorithm_config.AlgorithmConfig.evaluation
        .evaluation(
            evaluation_num_env_runners=cfg.agent.evaluation_num_env_runners,
            evaluation_interval=cfg.agent.evaluation_interval
        )
        .callbacks()
        .reporting(
            keep_per_episode_custom_metrics=True,
        #     metrics_episode_collection_timeout_s=60,
        #     metrics_num_episodes_for_smoothing=100    
        )
    )
    
    if algo_name == "PPO":
        config.training(
            use_critic=algo_parameters.use_critic,
            use_gae=algo_parameters.use_gae,
            lambda_=algo_parameters.lambda_,
            use_kl_loss=algo_parameters.use_kl_loss,
            kl_coeff=algo_parameters.kl_coeff,
            kl_target=algo_parameters.kl_target,
        )
    elif algo_name == "SAC":
        config.training(
            # target_network_update_freq=algo_parameters.target_network_update_freq,
            replay_buffer_config={
                "_enable_replay_buffer_api": True, 
                "type": "ReplayBuffer",  # Single-agent env; use ReplayBuffer, not MultiAgent
                "capacity": 50000, 
                "replay_sequence_length": 1,
                # "MultiAgentPrioritizedReplayBuffer"
                # "prioritized_replay_alpha": 0.6, 
                # "prioritized_replay_beta": 0.4, 
                # "prioritized_replay_eps": 1e-6,
            },
            # tau=algo_parameters.tau,
        )
    else:
        raise ValueError("Using unexpected algorithm name")
    
    ################################################
    ## This way of training the agent is deprecated.
    ## Now, use ray.tune.run(...) or Tuner().fit() 
    ## to train the agent.
    ################################################
    # from ray.tune.logger import UnifiedLogger
    # def logger_creator(config):
    #     return UnifiedLogger(config, "./loggings/", loggers=None)
    # algo = config.build_algo(logger_creator=logger_creator)   
    # results = algo.train()
    # pp(results)

    experiment_name = f"{cfg.agent.version}_{env_version}_{subenv_version}"

    if cfg.rllib.enable_tuner == False:
        tuner = ray.tune.Tuner(
        cfg.agent.version,
        param_space=config.to_dict(),
        tune_config=TuneConfig(
            ),
        run_config=RunConfig(
            storage_path=cfg.rllib.storage_path,
            name=experiment_name,
            callbacks=[TBXLoggerCallback(), CSVLoggerCallback(), JsonLoggerCallback()],
            stop={
                "timesteps_total": 5_000_000  # ~1220 iters @ 4096 steps/iter; change to 1_000_000, 2_000_000, etc.
            },
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=20,   # save every 20 iterations
                checkpoint_at_end=True,    # always save final weights
                num_to_keep=5,             # keep last 5 checkpoints
            ),
            # 0 = silent, 
            # 1 = default (display result table for the last iteration), 
            # 2 = verbose (display result table for each iteration).
            verbose=1,
            # progress_reporter=reporter,
            ),
        )
        result = tuner.fit()
        pprint(result[-1].metrics)
    else:
        # define param space
        param_space = config.to_dict()
        param_space["lr"] = tune.loguniform(1e-4, 1e-1)
        param_space["gamma"] = tune.choice([0.95, 0.98, 0.99])
        param_space["train_batch_size"] = tune.choice([2000, 4000, 6000])
        # set tune config
        tune_config = TuneConfig(
                metric="env_runners/episode_reward_mean",
                mode="max",
                num_samples=10,
                scheduler=ASHAScheduler()
            )
        # set run config
        run_config = RunConfig(
            name=f"{cfg.agent.version}_tuning",
            storage_path="./ray_results",
            stop={"env_runners/episode_reward_mean": 200},  # stop condition
            verbose=1,
        )
        # paramenter tuning
        result = tune.Tuner(
            trainable=cfg.agent.version,
            param_space=param_space,
            tune_config=tune_config,
            run_config=run_config,
        ).fit()
        pprint(result[-1].metrics)

    ray.shutdown()

if __name__ == "__main__":
    train_mpcrl_agent()