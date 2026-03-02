"""
Batch-evaluate RLlib PPO checkpoints: run N episodes, record videos,
compute collision rate, and save a summary JSON.

Usage:
    # Evaluate noCA best checkpoint (100 episodes)
    python batch_eval.py --model noCA \
        --checkpoint /users/saeani/DEV/ray_results/PPO_v0_noCA/PPO_intersection-mpcrl-refspeed-noCA_b1f0b_00000_0_2026-03-01_00-43-41/checkpoint_000023

    # Evaluate manual CA best checkpoint (100 episodes)
    python batch_eval.py --model manual \
        --checkpoint /users/saeani/DEV/ray_results/PPO_v0_manual/PPO_intersection-mpcrl-refspeed-manual_c129f_00000_0_2026-03-01_00-44-07/checkpoint_000016
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
# Add the highway-env source from whichever repo has the latest code.
_HW_ENV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "MPC-RL_for_AVs", "src_ray_version", "highway-env",
)
if os.path.isdir(_HW_ENV_PATH):
    sys.path.insert(0, os.path.abspath(_HW_ENV_PATH))

import highway_env  # noqa: F401 — registers gymnasium envs
from highway_env.envs import (
    IntersectionMpcrlSpeedsEnv_noCA,
    IntersectionMpcrlSpeedsEnv_manual,
)

from ray.rllib.policy.policy import Policy


# ── Registry ──
MODEL_REGISTRY = {
    "noCA": {
        "env_name": "intersection-mpcrl-refspeed-noCA",
        "env_cls": IntersectionMpcrlSpeedsEnv_noCA,
    },
    "manual": {
        "env_name": "intersection-mpcrl-refspeed-manual",
        "env_cls": IntersectionMpcrlSpeedsEnv_manual,
    },
}


def run_evaluation(
    checkpoint_path: str,
    model_name: str,
    n_episodes: int = 100,
    output_dir: str = ".",
    record_video: bool = True,
):
    """Run batch evaluation and return summary dict."""
    info = MODEL_REGISTRY[model_name]
    env_name = info["env_name"]
    video_dir = os.path.join(output_dir, f"videos_{model_name}")

    print(f"\n{'='*60}")
    print(f"  Model:      {model_name}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Episodes:   {n_episodes}")
    print(f"  Output:     {output_dir}")
    print(f"{'='*60}\n")

    # ── Load policy ──
    t0 = time.time()
    policy = Policy.from_checkpoint(checkpoint_path)
    if isinstance(policy, dict):
        policy = policy["default_policy"]
    print(f"[eval] Policy loaded in {time.time()-t0:.1f}s")

    # ── Create env ──
    env = gymnasium.make(env_name, render_mode="rgb_array")
    if record_video:
        os.makedirs(video_dir, exist_ok=True)
        env = gymnasium.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda ep: True,  # record every episode
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
    print(f"  RESULTS: {model_name}  ({n_episodes} episodes)")
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
    json_path = os.path.join(output_dir, f"eval_results_{model_name}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] Results saved to: {json_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate MPCRL checkpoint")
    parser.add_argument("--model", type=str, required=True, choices=["noCA", "manual"],
                        help="Model variant to evaluate")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the RLlib checkpoint directory")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of evaluation episodes (default: 100)")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help="Directory to save results and videos")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip video recording (faster)")
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        n_episodes=args.episodes,
        output_dir=args.output_dir,
        record_video=not args.no_video,
    )


if __name__ == "__main__":
    main()
