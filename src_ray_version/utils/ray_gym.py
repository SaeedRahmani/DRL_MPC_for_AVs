import ray
import warnings
from ray.tune import Tuner, RunConfig, TuneConfig, CLIReporter
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.logger import TBXLoggerCallback, CSVLoggerCallback, JsonLoggerCallback


def main():
    # disable the useless warning from gymnasium
    warnings.filterwarnings("ignore", module="gymnasium")

    ray.init(
        num_cpus=22,
        num_gpus=1,
        log_to_driver=False,        # disable ray workers from logging the output.
        include_dashboard=False,
    )

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        .environment(env="CartPole-v1")
        .framework("torch")
        .env_runners(
            num_env_runners=10,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=1 / 10,
        )
        .training(gamma=0.99, lr=1e-3, train_batch_size=4000)
        .reporting(log_gradients=True)
    )

    reporter = CLIReporter(
        metric_columns=["env_runners/episode_len_mean", "env_runners/episode_return_mean"]
    )
    tuner = Tuner(
        "PPO",
        param_space=config.to_dict(),
        tune_config=TuneConfig(
            ),
        run_config=RunConfig(
            storage_path="~/DEV/",
            name="XZL",
            callbacks=[TBXLoggerCallback(), CSVLoggerCallback(), JsonLoggerCallback()],
            stop={
                "training_iteration": 2
            },
            # 0 = silent, 
            # 1 = default (display result table for the last iteration), 
            # 2 = verbose (display result table for each iteration).
            verbose=1,
            # progress_reporter=reporter,
            ),
        )
    results = tuner.fit()
    from pprint import pp
    pp(results._results[0])

    ray.shutdown()


if __name__ == "__main__":
    main()
