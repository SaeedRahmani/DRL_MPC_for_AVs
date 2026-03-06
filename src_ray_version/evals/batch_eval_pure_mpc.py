"""
Batch-evaluate Pure MPC (with manual CA) on N scenarios.
No RL policy — just the MPC solver with collision avoidance heuristics.

Usage:
    cd /users/saeani/src/mpcrl/MPC-RL_for_AVs/src_ray_version && \
    PYTHONPATH="$PWD:$PWD/highway-env:$PYTHONPATH" \
    python /users/saeani/src/mpcrl/DRL_MPC_for_AVs/src_ray_version/evals/batch_eval_pure_mpc.py \
      --episodes 100 \
      --output-dir /users/saeani/src/mpcrl/DRL_MPC_for_AVs/src_ray_version/evals
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

import highway_env  # noqa: F401 — registers gymnasium envs


# ── Difficulty presets ──
DIFFICULTY_PRESETS = {
    "very_easy": {"initial_vehicle_count": 2,  "spawn_probability": 0.1},
    "easy":      {"initial_vehicle_count": 5,  "spawn_probability": 0.3},
    "moderate":  {"initial_vehicle_count": 10, "spawn_probability": 0.6},
    "hard":      {"initial_vehicle_count": 15, "spawn_probability": 0.9},
}


def run_evaluation(
    n_episodes: int = 100,
    output_dir: str = ".",
    record_video: bool = True,
    difficulty: str = "moderate",
):
    """Run pure MPC with manual CA on N episodes."""
    env_name = "intersection-mpc-manual"
    model_name = "pure_mpc_manual"
    video_dir = os.path.join(output_dir, f"videos_{model_name}")
    os.makedirs(output_dir, exist_ok=True)

    diff_cfg = DIFFICULTY_PRESETS[difficulty]
    print(f"\n{'='*60}")
    print(f"  Model:      Pure MPC (manual CA)")
    print(f"  Env:        {env_name}")
    print(f"  Difficulty: {difficulty} (vehicles={diff_cfg['initial_vehicle_count']}, spawn={diff_cfg['spawn_probability']})")
    print(f"  Episodes:   {n_episodes}")
    print(f"  Output:     {output_dir}")
    print(f"{'='*60}\n")

    # ── Create env ──
    env = gymnasium.make(env_name, render_mode="rgb_array")
    env.unwrapped.configure(diff_cfg)
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
            # Pure MPC ignores the action — pass a dummy action.
            # The env's _predict_mpc_action() handles everything internally
            # when agent_mode == "Pure_MPC".
            dummy_action = env.action_space.sample()
            obs, reward, terminated, truncated, step_info = env.step(dummy_action)
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
        "checkpoint": "N/A (pure MPC)",
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
    print(f"  RESULTS: Pure MPC (manual CA)  ({n_episodes} episodes)")
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

    # ── Save JSON ──
    json_path = os.path.join(output_dir, f"eval_results_{model_name}_{difficulty}_{n_episodes}ep.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] Results saved to: {json_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate Pure MPC with manual CA")
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
        n_episodes=args.episodes,
        output_dir=args.output_dir,
        record_video=not args.no_video,
        difficulty=args.difficulty,
    )


if __name__ == "__main__":
    main()
