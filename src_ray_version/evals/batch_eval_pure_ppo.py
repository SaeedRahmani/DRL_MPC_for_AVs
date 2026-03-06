"""
Batch-evaluate Pure PPO (no MPC) on N episodes.
The RL agent directly outputs [acceleration, steering] — no MPC in the loop.

Usage:
    cd /users/saeani/src/mpcrl/MPC-RL_for_AVs/src_ray_version && \
    PYTHONPATH="$PWD:$PWD/highway-env:$PYTHONPATH" \
    python /users/saeani/src/mpcrl/DRL_MPC_for_AVs/src_ray_version/evals/batch_eval_pure_ppo.py \
      --checkpoint /users/saeani/DEV/ray_results/PPO_pure_RL_v2/.../checkpoint_000048 \
      --episodes 400 --difficulty very_easy --no-video
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import gymnasium

# ── Headless rendering ──
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# ── Ensure highway_env envs are importable ──
_HW_ENV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "MPC-RL_for_AVs", "src_ray_version", "highway-env",
)
if os.path.isdir(_HW_ENV_PATH):
    sys.path.insert(0, os.path.abspath(_HW_ENV_PATH))

import highway_env  # noqa: F401 — registers gymnasium envs

from ray.rllib.policy.policy import Policy


# ── Difficulty presets (same as MPC-RL and Pure MPC evals) ──
DIFFICULTY_PRESETS = {
    "very_easy": {"initial_vehicle_count": 2,  "spawn_probability": 0.1},
    "easy":      {"initial_vehicle_count": 5,  "spawn_probability": 0.3},
    "moderate":  {"initial_vehicle_count": 10, "spawn_probability": 0.6},
    "hard":      {"initial_vehicle_count": 15, "spawn_probability": 0.9},
}


# ──────────────────────────────────────────────────────────────────────
#  IntersectionPurePPOEnv — copied from train_pure_ppo.py so the eval
#  script is self-contained (no import dependency on training code).
# ──────────────────────────────────────────────────────────────────────

class IntersectionPurePPOEnv(gymnasium.Wrapper):
    """
    Wraps `intersection-v1` (ContinuousIntersectionEnv) with:
      - Same observation config as MPCRL (10 vehicles, 8 features, absolute, unnormalized)
      - Flattened observation (80-D)
      - Reward function aligned with MPCRL manual-CA for fair comparison
      - Same duration (13s), policy_frequency (10), simulation_frequency (30)

    Action space: ContinuousAction → [acceleration, steering] (2D)
    """

    MAX_SPEED = 15.0

    def __init__(self, config=None, render_mode="rgb_array"):
        import highway_env  # noqa: F401

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
                "flatten": False,
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
            "duration": 13,
            "initial_vehicle_count": 10,
            "spawn_probability": 0.6,
            "screen_width": 600,
            "screen_height": 600,
            "scaling": 3,
            "policy_frequency": 10,
            "simulation_frequency": 30,
            "normalize_reward": False,
            "offroad_terminal": True,
        }

        if config:
            env_config.update(config)

        env = gymnasium.make("intersection-v1", render_mode=render_mode, config=env_config)
        super().__init__(env)

        vehicles_count = 10
        n_features = 8
        flat_size = vehicles_count * n_features
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_size,), dtype=np.float32,
        )

        self._FEATURE_RANGES = {
            0: (0.0, 1.0),
            1: (-100.0, 100.0),
            2: (-100.0, 100.0),
            3: (-20.0, 20.0),
            4: (-20.0, 20.0),
            5: (-np.pi, np.pi),
            6: (-1.0, 1.0),
            7: (-1.0, 1.0),
        }

        self._prev_acc = 0.0
        self._last_min_dist = 1.0

    def _flatten_obs(self, obs):
        if obs.ndim == 1:
            return np.clip(obs.astype(np.float32), -1.0, 1.0)
        norm = obs.copy().astype(np.float32)
        for col, (lo, hi) in self._FEATURE_RANGES.items():
            if hi > lo:
                norm[:, col] = 2.0 * (norm[:, col] - lo) / (hi - lo) - 1.0
        return np.clip(norm.flatten(), -1.0, 1.0)

    def _compute_min_dist(self, obs):
        PROXIMITY_RANGE = 30.0
        min_dist_norm = 1.0
        for i in range(1, obs.shape[0]):
            if obs[i, 0] > 0.5:
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
        if obs.ndim == 2:
            self._last_min_dist = self._compute_min_dist(obs)
        reward = self._compute_reward(obs, action, terminated, truncated, info)
        return self._flatten_obs(obs), reward, terminated, truncated, info

    def _compute_reward(self, obs, action, terminated, truncated, info):
        cvs = self.unwrapped.controlled_vehicles
        if not cvs:
            return 0.0
        vehicle = cvs[0]

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
        reward += 0.2
        speed_ratio = np.clip(vehicle.speed / self.MAX_SPEED, 0.0, 1.5)
        reward += 1.5 * speed_ratio

        min_dist = self._last_min_dist
        if min_dist < 0.5:
            reward -= 2.0 * (1.0 - 2.0 * min_dist) ** 2

        current_acc = float(action[0]) if hasattr(action, '__len__') else float(action)
        acc_change = abs(current_acc - self._prev_acc)
        reward -= 0.3 * (acc_change / 5.0)
        self._prev_acc = current_acc

        if vehicle.on_road and vehicle.lane is not None:
            try:
                lateral = vehicle.lane.local_coordinates(vehicle.position)[1]
                centering = 1.0 - min(1.0, abs(lateral) / (vehicle.lane.width / 2))
                reward += 0.3 * centering
            except Exception:
                pass

        return reward


# ──────────────────────────────────────────────────────────────────────
#  Evaluation loop
# ──────────────────────────────────────────────────────────────────────

def run_evaluation(
    checkpoint_path: str,
    n_episodes: int = 100,
    output_dir: str = ".",
    record_video: bool = True,
    difficulty: str = "moderate",
):
    """Run Pure PPO on N episodes and return summary dict."""
    model_name = "pure_ppo"
    video_dir = os.path.join(output_dir, f"videos_{model_name}")
    os.makedirs(output_dir, exist_ok=True)

    diff_cfg = DIFFICULTY_PRESETS[difficulty]
    print(f"\n{'='*60}")
    print(f"  Model:      Pure PPO (no MPC)")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Difficulty: {difficulty} (vehicles={diff_cfg['initial_vehicle_count']}, spawn={diff_cfg['spawn_probability']})")
    print(f"  Episodes:   {n_episodes}")
    print(f"  Output:     {output_dir}")
    print(f"{'='*60}\n")

    # ── Load policy ──
    t0 = time.time()
    policy = Policy.from_checkpoint(checkpoint_path)
    if isinstance(policy, dict):
        policy = policy["default_policy"]
    print(f"[eval] Policy loaded in {time.time()-t0:.1f}s")

    # ── Create env with difficulty config ──
    env = IntersectionPurePPOEnv(config=diff_cfg, render_mode="rgb_array")

    if record_video:
        os.makedirs(video_dir, exist_ok=True)
        env = gymnasium.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda ep: True,
            name_prefix=f"eval_{model_name}",
        )

    # ── Run episodes ──
    all_rewards = []
    all_lengths = []
    all_crashed = []
    all_arrived = []
    all_offroad = []

    for ep in range(n_episodes):
        obs, info_dict = env.reset()
        ep_reward = 0.0
        ep_len = 0
        done = False

        while not done:
            action, _, _ = policy.compute_single_action(obs, explore=False)
            obs, reward, terminated, truncated, step_info = env.step(action)
            ep_reward += reward
            ep_len += 1
            done = terminated or truncated

        # Determine outcome
        vehicle = env.unwrapped.controlled_vehicles[0]
        crashed = bool(vehicle.crashed)
        arrived = bool(env.unwrapped.has_arrived(vehicle))
        on_road = bool(vehicle.on_road)

        all_rewards.append(ep_reward)
        all_lengths.append(ep_len)
        all_crashed.append(crashed)
        all_arrived.append(arrived)
        all_offroad.append(not on_road and not crashed)

        status = "CRASH" if crashed else ("ARRIVE" if arrived else ("OFF-ROAD" if not on_road else "TIMEOUT"))
        print(f"  Ep {ep+1:3d}/{n_episodes}: reward={ep_reward:8.2f}  len={ep_len:3d}  {status}")

    env.close()

    # ── Compute metrics ──
    n_crash = sum(all_crashed)
    n_arrive = sum(all_arrived)
    n_offroad = sum(all_offroad)
    n_timeout = n_episodes - n_crash - n_arrive - n_offroad

    summary = {
        "model": model_name,
        "difficulty": difficulty,
        "difficulty_config": diff_cfg,
        "checkpoint": checkpoint_path,
        "n_episodes": n_episodes,
        "collision_rate": n_crash / n_episodes,
        "arrival_rate": n_arrive / n_episodes,
        "offroad_rate": n_offroad / n_episodes,
        "timeout_rate": n_timeout / n_episodes,
        "n_crashed": n_crash,
        "n_arrived": n_arrive,
        "n_offroad": n_offroad,
        "n_timeout": n_timeout,
        "reward_mean": float(np.mean(all_rewards)),
        "reward_std": float(np.std(all_rewards)),
        "reward_min": float(np.min(all_rewards)),
        "reward_max": float(np.max(all_rewards)),
        "length_mean": float(np.mean(all_lengths)),
        "per_episode": [
            {
                "episode": i + 1,
                "reward": all_rewards[i],
                "length": all_lengths[i],
                "crashed": all_crashed[i],
                "arrived": all_arrived[i],
            }
            for i in range(n_episodes)
        ],
    }

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"  RESULTS: Pure PPO (no MPC)  ({n_episodes} episodes)")
    print(f"{'='*60}")
    print(f"  Collision rate : {summary['collision_rate']*100:5.1f}%  ({n_crash}/{n_episodes})")
    print(f"  Arrival rate   : {summary['arrival_rate']*100:5.1f}%  ({n_arrive}/{n_episodes})")
    print(f"  Off-road rate  : {summary['offroad_rate']*100:5.1f}%  ({n_offroad}/{n_episodes})")
    print(f"  Timeout rate   : {summary['timeout_rate']*100:5.1f}%  ({n_timeout}/{n_episodes})")
    print(f"  Mean reward    : {summary['reward_mean']:8.2f} ± {summary['reward_std']:.2f}")
    print(f"  Min/Max reward : {summary['reward_min']:.2f} / {summary['reward_max']:.2f}")
    print(f"  Mean length    : {summary['length_mean']:.1f} steps")
    if record_video:
        print(f"  Videos         : {os.path.abspath(video_dir)}")
    print(f"{'='*60}\n")

    # ── Save JSON ──
    json_path = os.path.join(output_dir, f"eval_results_{model_name}_{difficulty}_{n_episodes}ep.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] Results saved to: {json_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate Pure PPO (no MPC)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the RLlib checkpoint directory")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of evaluation episodes (default: 100)")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help="Directory to save results and videos")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip video recording (faster)")
    parser.add_argument("--difficulty", type=str, default="moderate",
                        choices=["very_easy", "easy", "moderate", "hard"],
                        help="Scenario difficulty (default: moderate)")
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        n_episodes=args.episodes,
        output_dir=args.output_dir,
        record_video=not args.no_video,
        difficulty=args.difficulty,
    )


if __name__ == "__main__":
    main()
