"""
Train pure PPO (no MPC) on the intersection environment using Ray/RLlib.

This uses the same Ray infrastructure, observation config, and comparable
reward scale as the MPCRL training, so results can be directly compared.

The environment is `intersection-v1` (ContinuousIntersectionEnv) where the
RL agent directly outputs [acceleration, steering] — no MPC in the loop.

Usage:
    cd /users/saeani/src/mpcrl/MPC-RL_for_AVs/src_ray_version
    source .venv/bin/activate
    PYTHONPATH="$PWD:$PWD/highway-env:$PYTHONPATH" \
    CUDA_VISIBLE_DEVICES=<gpu_id> \
    python train_pure_ppo.py
"""

import logging
import ray
import ray.tune as tune
import numpy as np
import gymnasium
from gymnasium.envs.registration import VectorizeMode

from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import (
    JsonLoggerCallback,
    CSVLoggerCallback,
    TBXLoggerCallback,
)
from ray.tune import TuneConfig, RunConfig
from ray.train import CheckpointConfig
from pprint import pprint

import highway_env  # noqa: F401 — registers gymnasium envs


# ──────────────────────────────────────────────────────────────────────
#  Custom wrapper: align obs/reward with MPCRL for fair comparison
# ──────────────────────────────────────────────────────────────────────

class IntersectionPurePPOEnv(gymnasium.Wrapper):
    """
    Wraps `intersection-v1` (ContinuousIntersectionEnv) with:
      - Same observation config as MPCRL (10 vehicles, 8 features, absolute, unnormalized)
      - Flattened observation (80-D) — same layout as MPCRL minus the 6 MPC features
        (which don't exist without MPC)
      - Reward function aligned with MPCRL manual-CA for fair comparison
      - Same duration (13s), policy_frequency (10), simulation_frequency (30)

    Action space: ContinuousAction → [acceleration, steering] (2D)
    """

    MAX_SPEED = 15.0  # same as MPCRL MAX_REF_SPEED for reward normalization

    def __init__(self, config=None, render_mode="rgb_array"):
        # MUST import here — Ray workers don't inherit the main module's imports,
        # so gymnasium won't know about "intersection-v1" without this.
        import highway_env  # noqa: F401 — registers intersection-v1 in gymnasium

        env_config = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 10,
                "features": ["presence", "x", "y", "vx", "vy", "heading", "sin_h", "cos_h"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                    "heading": [-np.pi, np.pi],
                    "sin_h": [-1, 1],
                    "cos_h": [-1, 1],
                },
                "absolute": True,
                "normalize": False,
                "flatten": False,      # keep (V, 8) shape so we can normalize per-column
                "order": "sorted",
            },
            "action": {
                "type": "ContinuousAction",
                "steering_range": [-np.pi / 4, np.pi / 4],
                "acceleration_range": [-5.0, 5.0],
                "longitudinal": True,
                "lateral": True,
                "dynamical": True,
            },
            "duration": 13,  # same as MPCRL
            "initial_vehicle_count": 10,
            "spawn_probability": 0.6,
            "screen_width": 600,
            "screen_height": 600,
            "scaling": 3,
            "policy_frequency": 10,
            "simulation_frequency": 30,
            # Disable the default reward normalization — we compute our own
            "normalize_reward": False,
            "offroad_terminal": True,
        }

        # Allow overrides from Ray config
        if config:
            env_config.update(config)

        env = gymnasium.make("intersection-v1", render_mode=render_mode, config=env_config)
        super().__init__(env)

        # Override observation space to be flat.
        # Use -inf/inf bounds: normalization can produce values slightly outside
        # [-1, 1] when raw features exceed declared ranges, and RLlib's
        # preprocessor strictly validates obs ∈ [low, high].
        vehicles_count = 10
        n_features = 8
        flat_size = vehicles_count * n_features
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_size,), dtype=np.float32,
        )

        # Feature normalization ranges (same as MPCRL)
        self._FEATURE_RANGES = {
            0: (0.0, 1.0),          # presence
            1: (-100.0, 100.0),     # x
            2: (-100.0, 100.0),     # y
            3: (-20.0, 20.0),       # vx
            4: (-20.0, 20.0),       # vy
            5: (-np.pi, np.pi),     # heading
            6: (-1.0, 1.0),         # sin_h
            7: (-1.0, 1.0),         # cos_h
        }

        self._prev_acc = 0.0
        self._last_min_dist = 1.0

    def _flatten_obs(self, obs):
        """Flatten & normalize observation, same as MPCRL _augment_obs (minus MPC features)."""
        if obs.ndim == 1:
            return np.clip(obs.astype(np.float32), -1.0, 1.0)
        norm = obs.copy().astype(np.float32)
        for col, (lo, hi) in self._FEATURE_RANGES.items():
            if hi > lo:
                norm[:, col] = 2.0 * (norm[:, col] - lo) / (hi - lo) - 1.0
        # Clip to [-1, 1]: raw values may slightly exceed feature_ranges
        return np.clip(norm.flatten(), -1.0, 1.0)

    def _compute_min_dist(self, obs):
        """Compute min distance to other vehicles, same as MPCRL."""
        PROXIMITY_RANGE = 30.0
        min_dist_norm = 1.0
        for i in range(1, obs.shape[0]):
            if obs[i, 0] > 0.5:  # presence flag
                d = np.sqrt((obs[0, 1] - obs[i, 1])**2 + (obs[0, 2] - obs[i, 2])**2)
                min_dist_norm = min(min_dist_norm, d / PROXIMITY_RANGE)
        return float(np.clip(min_dist_norm, 0.0, 1.0))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_acc = 0.0
        self._last_min_dist = 1.0
        if obs.ndim == 2:
            self._last_min_dist = self._compute_min_dist(obs)
        return self._flatten_obs(obs), info

    def step(self, action):
        obs, _original_reward, terminated, truncated, info = self.env.step(action)

        # Update proximity BEFORE computing reward so the penalty uses current state
        if obs.ndim == 2:
            self._last_min_dist = self._compute_min_dist(obs)

        # Compute our own reward aligned with MPCRL
        reward = self._compute_reward(obs, action, terminated, truncated, info)

        return self._flatten_obs(obs), reward, terminated, truncated, info

    def _compute_reward(self, obs, action, terminated, truncated, info):
        """
        Reward aligned with MPCRL manual-CA variant for fair comparison.

        Same components and scale:
        - Crash: -50 (same as MPCRL noCA)
        - Off-road: -10
        - Arrived: +20
        - Alive: +0.2
        - Speed efficiency: up to +1.5
        - Proximity: up to -2.0
        - Smoothness: -0.3 * |delta_acc|/max_acc
        - Centering: up to +0.3
        """
        # Guard against empty vehicle list (edge case during reset/terminal)
        cvs = self.unwrapped.controlled_vehicles
        if not cvs:
            return 0.0
        vehicle = cvs[0]

        # Terminal events
        if vehicle.crashed:
            return -50.0
        if not vehicle.on_road:
            return -10.0
        try:
            arrived = self.unwrapped.has_arrived(vehicle)
        except Exception:
            arrived = False
        if arrived:
            return 20.0

        reward = 0.0

        # 1. Alive bonus
        reward += 0.2

        # 2. Speed efficiency
        speed_ratio = np.clip(vehicle.speed / self.MAX_SPEED, 0.0, 1.5)
        reward += 1.5 * speed_ratio

        # 3. Proximity warning
        min_dist = self._last_min_dist
        if min_dist < 0.5:
            reward -= 2.0 * (1.0 - 2.0 * min_dist) ** 2

        # 4. Smoothness — penalize abrupt acceleration changes
        current_acc = float(action[0]) if hasattr(action, '__len__') else float(action)
        acc_change = abs(current_acc - self._prev_acc)
        reward -= 0.3 * (acc_change / 5.0)
        self._prev_acc = current_acc

        # 5. Lane centering (guard: vehicle.lane can be None at road edges)
        if vehicle.on_road and vehicle.lane is not None:
            try:
                lateral = vehicle.lane.local_coordinates(vehicle.position)[1]
                centering = 1.0 - min(1.0, abs(lateral) / (vehicle.lane.width / 2))
                reward += 0.3 * centering
            except Exception:
                pass  # skip centering if lane geometry unavailable

        return reward


# ──────────────────────────────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────────────────────────────

def main():
    ray.shutdown()
    ray.init(
        num_cpus=16,
        num_gpus=1,
        logging_level=logging.INFO,
        log_to_driver=False,
        include_dashboard=False,
    )
    pprint(ray.available_resources())

    env_name = "intersection-pure-ppo"

    def env_creator(config):
        return IntersectionPurePPOEnv(config=config, render_mode="rgb_array")

    register_env(name=env_name, env_creator=env_creator)

    # Create a test instance to verify spaces
    test_env = IntersectionPurePPOEnv(render_mode="rgb_array")
    print(f"Observation space: {test_env.observation_space}")
    print(f"Action space: {test_env.action_space}")
    test_env.close()

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .framework("torch")
        .environment(
            env=env_name,
            render_env=False,
            is_atari=False,
            disable_env_checking=True,
        )
        .env_runners(
            num_env_runners=10,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0,
            num_envs_per_env_runner=2,
            gym_env_vectorize_mode=VectorizeMode.ASYNC,
        )
        .resources(
            num_gpus=1,
        )
        .training(
            gamma=0.99,
            lr=1e-4,
            num_epochs=10,
            train_batch_size=4096,
            minibatch_size=64,
            shuffle_batch_per_epoch=True,
            grad_clip=0.5,
            grad_clip_by="global_norm",
            model={
                "fcnet_hiddens": [512, 256],
            },
            use_critic=True,
            use_gae=True,
            lambda_=0.95,
            use_kl_loss=True,
            kl_coeff=0.2,
            kl_target=0.01,
        )
    )

    # Set directly — not accepted as a .training() kwarg in Ray 2.43
    config.torch_skip_nan_gradients = True

    config = (
        config
        .evaluation(
            evaluation_num_env_runners=0,
            evaluation_interval=1,
        )
        .reporting(
            keep_per_episode_custom_metrics=True,
        )
    )

    experiment_name = "PPO_pure_RL"

    tuner = ray.tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        tune_config=TuneConfig(),
        run_config=RunConfig(
            storage_path="~/DEV/ray_results",
            name=experiment_name,
            callbacks=[TBXLoggerCallback(), CSVLoggerCallback(), JsonLoggerCallback()],
            stop={
                "timesteps_total": 4_000_000,
            },
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=20,   # save every 20 iters (~80k steps)
                checkpoint_at_end=True,
                num_to_keep=10,            # keep 10 checkpoints for evaluation
            ),
            verbose=1,
        ),
    )

    result = tuner.fit()
    pprint(result[-1].metrics)
    ray.shutdown()


if __name__ == "__main__":
    main()
