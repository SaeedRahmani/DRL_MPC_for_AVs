"""
Zero-shot evaluate Pure PPO (intersection-trained) on merge-v0 with ego on ramp.

The PPO was trained on intersection-v1 with:
  - Observation: 80D = 10 vehicles × 8 features (flattened, normalized)
  - Action: 2D [acceleration, steering]

For zero-shot transfer to merging we:
  1. Put ego on the merge ramp (ego_on_ramp=True)
  2. Construct the same 80D observation (same normalization / clipping)
  3. Feed it to the trained PPO policy which directly outputs [acc, steer]
  4. Apply the action to the merge env

Difficulty presets (controls number of highway traffic vehicles):
  very_easy  : 1 vehicle
  easy       : 2 vehicles
  moderate   : 4 vehicles (default)

Usage:
    cd /users/saeani/src/mpcrl/MPC-RL_for_AVs/src_ray_version && \\
    PYTHONPATH="$PWD:$PWD/highway-env:$PYTHONPATH" CUDA_VISIBLE_DEVICES="" \\
    python /users/saeani/src/mpcrl/DRL_MPC_for_AVs/src_ray_version/evals/batch_eval_merge_pure_ppo.py \\
      --checkpoint <path> --episodes 1000 --difficulty easy --no-video
"""

import os
import sys
import json
import argparse
import numpy as np
import gymnasium

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import highway_env  # noqa: F401

from ray.rllib.policy.policy import Policy


# ──────────────────────────────────────────────────────────────────────
#  Difficulty presets (number of highway traffic vehicles)
# ──────────────────────────────────────────────────────────────────────
MERGE_DIFFICULTY = {
    "very_easy": {"other_vehicles_count": 1},
    "easy":      {"other_vehicles_count": 2},
    "moderate":  {"other_vehicles_count": 4},
}


# ──────────────────────────────────────────────────────────────────────
#  Merge wrapper that produces the same 80D obs as intersection training
# ──────────────────────────────────────────────────────────────────────

class MergePurePPOEnv(gymnasium.Wrapper):
    """Wraps merge-v0 (ego_on_ramp) to produce 80D obs matching the
    intersection PPO training wrapper."""

    MAX_SPEED = 15.0

    def __init__(self, render_mode="rgb_array", other_vehicles_count=None):
        env_config = {
            "ego_on_ramp": True,
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 10,
                "features": ["presence", "x", "y", "vx", "vy",
                             "heading", "sin_h", "cos_h"],
                "features_range": {
                    "x": [-100, 100], "y": [-100, 100],
                    "vx": [-20, 20], "vy": [-20, 20],
                    "heading": [-np.pi, np.pi],
                    "sin_h": [-1, 1], "cos_h": [-1, 1],
                },
                "absolute": True,
                "normalize": False,
                "flatten": False,
                "order": "sorted",
            },
            "action": {
                "type": "ContinuousAction",
                "acceleration_range": [-5.0, 5.0],
                "steering_range": [-np.pi / 4, np.pi / 4],
                "longitudinal": True,
                "lateral": True,
                "dynamical": True,
            },
            "duration": 20,
            "policy_frequency": 10,
            "simulation_frequency": 30,
            "normalize_reward": False,
        }
        if other_vehicles_count is not None:
            env_config["other_vehicles_count"] = other_vehicles_count

        env = gymnasium.make("merge-v0", render_mode=render_mode,
                             config=env_config)
        super().__init__(env)

        vehicles_count = 10
        n_features = 8
        flat_size = vehicles_count * n_features
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_size,), dtype=np.float32,
        )

        # Same normalization ranges as intersection training
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
        self._step_count = 0
        self._max_steps = 200  # 20s at 10 Hz policy_frequency

    def _flatten_obs(self, obs):
        if obs.ndim == 1:
            return np.nan_to_num(
                np.clip(obs.astype(np.float32), -1.0, 1.0), nan=0.0)
        norm = obs.copy().astype(np.float32)
        norm = np.nan_to_num(norm, nan=0.0)
        for col, (lo, hi) in self._FEATURE_RANGES.items():
            if hi > lo:
                norm[:, col] = 2.0 * (norm[:, col] - lo) / (hi - lo) - 1.0
        return np.nan_to_num(
            np.clip(norm.flatten(), -1.0, 1.0), nan=0.0)

    def _compute_min_dist(self, obs):
        PROXIMITY_RANGE = 30.0
        min_dist_norm = 1.0
        for i in range(1, obs.shape[0]):
            if obs[i, 0] > 0.5:
                d = np.sqrt((obs[0, 1] - obs[i, 1])**2 +
                            (obs[0, 2] - obs[i, 2])**2)
                min_dist_norm = min(min_dist_norm, d / PROXIMITY_RANGE)
        return float(np.clip(min_dist_norm, 0.0, 1.0))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_acc = 0.0
        self._last_min_dist = 1.0
        self._step_count = 0
        if obs.ndim == 2:
            self._last_min_dist = self._compute_min_dist(obs)
        return self._flatten_obs(obs), info

    def step(self, action):
        obs, _orig_reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        if obs.ndim == 2:
            self._last_min_dist = self._compute_min_dist(obs)
        # Force truncation after max_steps (MergeEnv._is_truncated returns False)
        if self._step_count >= self._max_steps:
            truncated = True
        # Also truncate if vehicle left the road (off-road)
        vehicle = self.unwrapped.controlled_vehicles[0]
        if not vehicle.on_road and not vehicle.crashed:
            truncated = True
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

        # Check arrival (merge env may not have has_arrived)
        try:
            arrived = self.unwrapped.has_arrived(vehicle)
        except Exception:
            arrived = False
        if arrived:
            return 20.0

        reward = 0.0
        reward += 0.2  # alive bonus
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
#  Evaluation
# ──────────────────────────────────────────────────────────────────────

def run_evaluation(checkpoint_path, n_episodes=100, output_dir=".",
                   record_video=True, difficulty="moderate"):
    diff_cfg = MERGE_DIFFICULTY[difficulty]
    model_name = "pure_ppo_merge"
    out_tag = f"{model_name}_{difficulty}_{n_episodes}ep"
    video_dir = os.path.join(output_dir, f"videos_{out_tag}")

    print(f"\n{'='*60}")
    print(f"  Model:      Pure PPO (zero-shot, ego merging)")
    print(f"  Difficulty: {difficulty} ({diff_cfg})")
    print(f"  Env:        merge-v0 (ego_on_ramp=True)")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Episodes:   {n_episodes}")
    print(f"  Output:     {output_dir}")
    print(f"{'='*60}\n")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    policy = Policy.from_checkpoint(checkpoint_path)
    if isinstance(policy, dict):
        policy = policy.get("default_policy", list(policy.values())[0])
    # Move model to CPU explicitly
    policy.model.cpu()
    policy.model.eval()

    render_mode = "rgb_array" if record_video else None
    env = MergePurePPOEnv(render_mode=render_mode,
                          other_vehicles_count=diff_cfg["other_vehicles_count"])
    if record_video:
        os.makedirs(video_dir, exist_ok=True)
        env = gymnasium.wrappers.RecordVideo(
            env, video_folder=video_dir,
            episode_trigger=lambda ep: ep < 10,
            name_prefix=f"eval_{model_name}",
        )

    all_rewards, all_lengths = [], []
    all_crashed, all_arrived, all_offroad = [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward, ep_len, done = 0.0, 0, False

        while not done:
            try:
                action, _, _ = policy.compute_single_action(obs, explore=False)
                action = np.nan_to_num(np.asarray(action, dtype=np.float32),
                                       nan=0.0)
            except (ValueError, RuntimeError):
                # NaN from policy — send zero action (coast)
                action = np.zeros(2, dtype=np.float32)
            obs, reward, terminated, truncated, step_info = env.step(action)
            ep_reward += reward
            ep_len += 1
            done = terminated or truncated

        vehicle = env.unwrapped.controlled_vehicles[0]
        crashed = bool(vehicle.crashed)
        on_road = bool(vehicle.on_road)
        # MergeEnv has no has_arrived(); use same criterion as _is_terminated
        arrived = bool(vehicle.position[0] > 370 and not crashed)

        all_rewards.append(ep_reward)
        all_lengths.append(ep_len)
        all_crashed.append(crashed)
        all_arrived.append(arrived)
        all_offroad.append(not on_road and not crashed and not arrived)

        status = ("CRASH" if crashed else
                  ("ARRIVE" if arrived else
                   ("OFF-ROAD" if not on_road else "TIMEOUT")))
        print(f"  Ep {ep+1:3d}/{n_episodes}: reward={ep_reward:8.2f}  "
              f"len={ep_len:3d}  {status}")

    env.close()

    n_crash = sum(all_crashed)
    n_arrive = sum(all_arrived)
    n_offroad = sum(all_offroad)
    n_timeout = n_episodes - n_crash - n_arrive - n_offroad

    summary = {
        "model": model_name,
        "difficulty": difficulty,
        "difficulty_config": diff_cfg,
        "env": "merge-v0 (ego_on_ramp)",
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
            {"episode": i+1, "reward": all_rewards[i], "length": all_lengths[i],
             "crashed": all_crashed[i], "arrived": all_arrived[i]}
            for i in range(n_episodes)
        ],
    }

    print(f"\n{'='*60}")
    print(f"  RESULTS: Pure PPO on Merge [{difficulty}] ({n_episodes} episodes)")
    print(f"{'='*60}")
    print(f"  Collision rate : {summary['collision_rate']*100:5.1f}%  ({n_crash}/{n_episodes})")
    print(f"  Arrival rate   : {summary['arrival_rate']*100:5.1f}%  ({n_arrive}/{n_episodes})")
    print(f"  Off-road rate  : {summary['offroad_rate']*100:5.1f}%  ({n_offroad}/{n_episodes})")
    print(f"  Timeout rate   : {summary['timeout_rate']*100:5.1f}%  ({n_timeout}/{n_episodes})")
    print(f"  Mean reward    : {summary['reward_mean']:8.2f} +/- {summary['reward_std']:.2f}")
    print(f"  Min/Max reward : {summary['reward_min']:.2f} / {summary['reward_max']:.2f}")
    print(f"  Mean length    : {summary['length_mean']:.1f} steps")
    if record_video:
        print(f"  Videos         : {os.path.abspath(video_dir)}")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"eval_results_{out_tag}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] Saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch eval Pure PPO on merge-v0 (zero-shot)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--difficulty", type=str, default="moderate",
                        choices=["very_easy", "easy", "moderate"])
    parser.add_argument("--output-dir", type=str,
                        default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    run_evaluation(args.checkpoint, args.episodes, args.output_dir,
                   not args.no_video, args.difficulty)


if __name__ == "__main__":
    main()
