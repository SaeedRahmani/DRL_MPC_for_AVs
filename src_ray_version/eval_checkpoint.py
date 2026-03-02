"""
Evaluate a saved RLlib PPO checkpoint and record a video.

Usage:
    python eval_checkpoint.py
    python eval_checkpoint.py --checkpoint <path>
"""

import os
import argparse
import gymnasium
import numpy as np
from ray.rllib.policy.policy import Policy
import highway_env  # noqa: F401 — registers custom gymnasium envs
from highway_env.envs import IntersectionMpcrlSpeedsEnv_noCA  # noqa: F401

# ── default checkpoint (best from the Feb-26 run, iter ~480, ~1.97M steps) ──
DEFAULT_CHECKPOINT = (
    "/users/saeani/DEV/ray_results/PPO/"
    "PPO_intersection-mpcrl-refspeed-noCA_3839d_00000_0_2026-02-26_02-52-42/"
    "checkpoint_000024"
)

ENV_NAME  = "intersection-mpcrl-refspeed-noCA"
N_EPISODES = 5
VIDEO_DIR  = "./eval_videos"
os.makedirs(VIDEO_DIR, exist_ok=True)


def main(checkpoint_path: str):
    print(f"\n[eval] Loading checkpoint: {checkpoint_path}")

    # Load ONLY the policy weights — no Ray workers, no distributed setup.
    # This is instant and uses ~1 CPU.
    policy = Policy.from_checkpoint(checkpoint_path)
    if isinstance(policy, dict):
        policy = policy["default_policy"]
    print("[eval] Policy loaded successfully.")

    # Create env directly in this process for video recording
    env = gymnasium.make(ENV_NAME, render_mode="rgb_array")
    env = gymnasium.wrappers.RecordVideo(
        env,
        video_folder=VIDEO_DIR,
        episode_trigger=lambda ep: True,
        name_prefix="mpcrl_v0_noCA",
    )

    all_rewards = []
    all_lengths = []

    for ep in range(N_EPISODES):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_len    = 0
        done = False

        while not done:
            action, _, _ = policy.compute_single_action(obs, explore=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            ep_len    += 1
            done = terminated or truncated

        all_rewards.append(ep_reward)
        all_lengths.append(ep_len)
        print(f"  Episode {ep+1:2d}: reward={ep_reward:7.2f}  length={ep_len} steps")

    env.close()

    print(f"\n[eval] Mean reward : {np.mean(all_rewards):.2f}")
    print(f"[eval] Mean length : {np.mean(all_lengths):.1f} steps")
    print(f"[eval] Videos saved to: {os.path.abspath(VIDEO_DIR)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
        help="Path to the RLlib checkpoint directory to evaluate."
    )
    args = parser.parse_args()
    main(args.checkpoint)
