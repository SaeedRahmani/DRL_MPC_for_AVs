"""
Batch-evaluate a pure PPO (no MPC) checkpoint on N scenarios.

Usage:
    cd /users/saeani/src/mpcrl/MPC-RL_for_AVs/src_ray_version && \
    PYTHONPATH="$PWD:$PWD/highway-env:$PYTHONPATH" \
    python /users/saeani/src/mpcrl/DRL_MPC_for_AVs/src_ray_version/evals/batch_eval_pure_ppo.py \
      --checkpoint <path_to_checkpoint> \
      --episodes 100 \
      --output-dir <output_dir>
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

# We need IntersectionPurePPOEnv so the checkpoint can restore
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../MPC-RL_for_AVs/src_ray_version"))


def make_env(render_mode="rgb_array"):
    """Create the same env used during training."""
    from train_pure_ppo import IntersectionPurePPOEnv
    return IntersectionPurePPOEnv(render_mode=render_mode)


def run_evaluation(checkpoint_path, n_episodes=100, output_dir=".", record_video=True):
    from ray.rllib.policy.policy import Policy

    model_name = "pure_ppo"
    video_dir = os.path.join(output_dir, f"videos_{model_name}")

    print(f"\n{'='*60}")
    print(f"  Model:      Pure PPO (no MPC)")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Episodes:   {n_episodes}")
    print(f"  Output:     {output_dir}")
    print(f"{'='*60}\n")

    # Load policy
    policy = Policy.from_checkpoint(checkpoint_path)["default_policy"]

    # Create env
    env = make_env()
    if record_video:
        os.makedirs(video_dir, exist_ok=True)
        env = gymnasium.wrappers.RecordVideo(
            env, video_folder=video_dir,
            episode_trigger=lambda ep: True,
            name_prefix=f"eval_{model_name}",
        )

    all_rewards, all_lengths, all_crashed, all_arrived, all_offroad = [], [], [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward, ep_len, done = 0.0, 0, False

        while not done:
            action = policy.compute_single_action(obs, explore=False)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1
            done = terminated or truncated

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
            {"episode": i+1, "reward": all_rewards[i], "length": all_lengths[i],
             "crashed": all_crashed[i], "arrived": all_arrived[i]}
            for i in range(n_episodes)
        ],
    }

    print(f"\n{'='*60}")
    print(f"  RESULTS: Pure PPO  ({n_episodes} episodes)")
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

    json_path = os.path.join(output_dir, f"eval_results_{model_name}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] Saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch eval pure PPO")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=str,
                        default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    run_evaluation(args.checkpoint, args.episodes, args.output_dir, not args.no_video)


if __name__ == "__main__":
    main()
