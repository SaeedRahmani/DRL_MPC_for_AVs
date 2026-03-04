"""
Experiment 1: Scenario Difficulty Sweep
=======================================
Evaluate Pure RL, Pure MPC, and MPC-RL across three traffic densities:

    Easy   : initial_vehicle_count=5,  spawn_probability=0.3
    Medium : initial_vehicle_count=10, spawn_probability=0.6  (training default)
    Hard   : initial_vehicle_count=15, spawn_probability=0.9

For each (method × difficulty), run N episodes and report:
  - Arrival rate (%)
  - Crash rate (%)
  - Mean episode return
  - Mean episode length (steps)

Outputs (in experiment1_results/):
  - difficulty_sweep_results.json   Raw per-episode data
  - difficulty_arrival_rate.pdf     Grouped bar chart: arrival rate by difficulty
  - difficulty_crash_rate.pdf       Grouped bar chart: crash rate by difficulty
  - difficulty_summary_table.pdf    Summary table

Usage:
    cd /users/saeani/src/mpcrl/MPC-RL_for_AVs/src_ray_version && \
    PYTHONPATH="$PWD:$PWD/highway-env:$PYTHONPATH" CUDA_VISIBLE_DEVICES="" \
    python /users/saeani/src/mpcrl/DRL_MPC_for_AVs/src_ray_version/evals/experiment1_difficulty_sweep.py
"""

import os, sys, json, argparse
import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import gymnasium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import highway_env  # noqa: F401

# ── paths ──
EVALS_DIR  = os.path.dirname(os.path.abspath(__file__))
PPO_CKPT   = "/users/saeani/DEV/ray_results/PPO_pure_RL/PPO_intersection-pure-ppo_03a31_00000_0_2026-03-01_23-40-22/checkpoint_000025"
MPCRL_CKPT = "/users/saeani/DEV/ray_results/PPO_v0_manual/PPO_intersection-mpcrl-refspeed-manual_c129f_00000_0_2026-03-01_00-44-07/checkpoint_000020"

sys.path.insert(0, os.path.join(EVALS_DIR, "../../../MPC-RL_for_AVs/src_ray_version"))

# ── difficulty levels ──
DIFFICULTY_LEVELS = {
    "Easy":   {"initial_vehicle_count": 5,  "spawn_probability": 0.3},
    "Medium": {"initial_vehicle_count": 10, "spawn_probability": 0.6},
    "Hard":   {"initial_vehicle_count": 15, "spawn_probability": 0.9},
}

COLORS = {"Pure RL": "#2196F3", "Pure MPC": "#FF9800", "MPC-RL": "#4CAF50"}


# =====================================================================
#  Helpers
# =====================================================================
def _get_ego(env):
    inner = env.unwrapped if hasattr(env, "unwrapped") else env
    if hasattr(inner, "controlled_vehicles") and inner.controlled_vehicles:
        return inner.controlled_vehicles[0]
    if hasattr(inner, "vehicle"):
        return inner.vehicle
    return None


def _has_arrived(env, ego):
    inner = env.unwrapped if hasattr(env, "unwrapped") else env
    try:
        return bool(inner.has_arrived(ego))
    except Exception:
        return False


# =====================================================================
#  1. Pure RL
# =====================================================================
def eval_pure_rl(n_episodes, ppo_ckpt, difficulty_cfg):
    from ray.rllib.policy.policy import Policy
    from train_pure_ppo import IntersectionPurePPOEnv

    policy = Policy.from_checkpoint(ppo_ckpt)
    if isinstance(policy, dict):
        policy = policy["default_policy"]

    env = IntersectionPurePPOEnv(config=difficulty_cfg, render_mode="rgb_array")
    results = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            action = policy.compute_single_action(obs, explore=False)[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated

        ego = _get_ego(env)
        arrived = _has_arrived(env, ego) if ego else False
        crashed = bool(ego.crashed) if ego else True
        status = "ARRIVE" if arrived else ("CRASH" if crashed else "OTHER")

        results.append({
            "episode": ep + 1,
            "status": status,
            "reward": float(ep_reward),
            "steps": steps,
        })
        if (ep + 1) % 10 == 0:
            n_arr = sum(1 for r in results if r["status"] == "ARRIVE")
            print(f"    [Pure RL] ep {ep+1}/{n_episodes}  arrived={n_arr}  "
                  f"last={status}  reward={ep_reward:.1f}")

    env.close()
    return results


# =====================================================================
#  2. Pure MPC
# =====================================================================
def eval_pure_mpc(n_episodes, difficulty_cfg):
    env = gymnasium.make("intersection-mpc-manual", render_mode="rgb_array",
                         config=difficulty_cfg)
    results = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            dummy = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(dummy)
            ep_reward += reward
            steps += 1
            done = terminated or truncated

        ego = _get_ego(env)
        arrived = _has_arrived(env, ego) if ego else False
        crashed = bool(ego.crashed) if ego else True
        status = "ARRIVE" if arrived else ("CRASH" if crashed else "OTHER")

        results.append({
            "episode": ep + 1,
            "status": status,
            "reward": float(ep_reward),
            "steps": steps,
        })
        if (ep + 1) % 10 == 0:
            n_arr = sum(1 for r in results if r["status"] == "ARRIVE")
            print(f"    [Pure MPC] ep {ep+1}/{n_episodes}  arrived={n_arr}  "
                  f"last={status}  reward={ep_reward:.1f}")

    env.close()
    return results


# =====================================================================
#  3. MPC-RL
# =====================================================================
def eval_mpcrl(n_episodes, mpcrl_ckpt, difficulty_cfg):
    from ray.rllib.policy.policy import Policy

    policy = Policy.from_checkpoint(mpcrl_ckpt)
    if isinstance(policy, dict):
        policy = policy["default_policy"]

    env = gymnasium.make("intersection-mpcrl-refspeed-manual", render_mode="rgb_array",
                         config=difficulty_cfg)
    results = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            action = policy.compute_single_action(obs, explore=False)[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated

        ego = _get_ego(env)
        arrived = _has_arrived(env, ego) if ego else False
        crashed = bool(ego.crashed) if ego else True
        status = "ARRIVE" if arrived else ("CRASH" if crashed else "OTHER")

        results.append({
            "episode": ep + 1,
            "status": status,
            "reward": float(ep_reward),
            "steps": steps,
        })
        if (ep + 1) % 10 == 0:
            n_arr = sum(1 for r in results if r["status"] == "ARRIVE")
            print(f"    [MPC-RL] ep {ep+1}/{n_episodes}  arrived={n_arr}  "
                  f"last={status}  reward={ep_reward:.1f}")

    env.close()
    return results


# =====================================================================
#  Plotting
# =====================================================================
def _compute_stats(results):
    """Compute summary statistics from a list of episode results."""
    n = len(results)
    if n == 0:
        return {"arrival_rate": 0, "crash_rate": 0, "mean_reward": 0, "mean_steps": 0}
    n_arrived = sum(1 for r in results if r["status"] == "ARRIVE")
    n_crashed = sum(1 for r in results if r["status"] == "CRASH")
    return {
        "arrival_rate": 100.0 * n_arrived / n,
        "crash_rate": 100.0 * n_crashed / n,
        "mean_reward": float(np.mean([r["reward"] for r in results])),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "n_arrived": n_arrived,
        "n_crashed": n_crashed,
        "n_other": n - n_arrived - n_crashed,
        "n_total": n,
    }


def plot_grouped_bars(all_stats, metric_key, ylabel, title, filename, output_dir):
    """Grouped bar chart: x=difficulty, bars=methods."""
    difficulties = list(DIFFICULTY_LEVELS.keys())
    methods = ["Pure RL", "Pure MPC", "MPC-RL"]
    n_methods = len(methods)
    x = np.arange(len(difficulties))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method in enumerate(methods):
        vals = [all_stats[diff][method][metric_key] for diff in difficulties]
        bars = ax.bar(x + i * width, vals, width, label=method,
                      color=COLORS[method], alpha=0.85)
        # Value labels
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Traffic Difficulty", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(difficulties, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"{filename}.{ext}")
        fig.savefig(path, dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {filename}.pdf/png")


def plot_summary_table(all_stats, output_dir):
    """Render a summary table as a figure."""
    difficulties = list(DIFFICULTY_LEVELS.keys())
    methods = ["Pure RL", "Pure MPC", "MPC-RL"]

    col_labels = []
    for d in difficulties:
        for m in methods:
            col_labels.append(f"{d}\n{m}")

    row_labels = ["Arrival %", "Crash %", "Mean Reward", "Mean Steps"]
    cell_data = []
    for metric in ["arrival_rate", "crash_rate", "mean_reward", "mean_steps"]:
        row = []
        for d in difficulties:
            for m in methods:
                val = all_stats[d][m][metric]
                if metric in ("arrival_rate", "crash_rate"):
                    row.append(f"{val:.1f}%")
                elif metric == "mean_reward":
                    row.append(f"{val:.1f}")
                else:
                    row.append(f"{val:.0f}")
        cell_data.append(row)

    fig, ax = plt.subplots(figsize=(16, 3))
    ax.axis("off")

    table = ax.table(cellText=cell_data, rowLabels=row_labels,
                     colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # Color headers
    for j, col_label in enumerate(col_labels):
        for m_name, color in COLORS.items():
            if m_name in col_label:
                table[0, j].set_facecolor(color)
                table[0, j].set_text_props(color="white", fontweight="bold")
                break

    fig.suptitle("Scenario Difficulty Sweep — Summary", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"difficulty_summary_table.{ext}")
        fig.savefig(path, dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] difficulty_summary_table.pdf/png")


# =====================================================================
#  Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Difficulty Sweep")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per (method × difficulty) (default: 50)")
    parser.add_argument("--ppo-ckpt", type=str, default=PPO_CKPT)
    parser.add_argument("--mpcrl-ckpt", type=str, default=MPCRL_CKPT)
    parser.add_argument("--output-dir", type=str, default=EVALS_DIR)
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, "experiment1_results")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  Experiment 1: Scenario Difficulty Sweep")
    print(f"  Episodes per condition: {args.episodes}")
    print(f"  Output: {output_dir}")
    print(f"{'#'*60}")

    all_raw = {}      # diff -> method -> [episode results]
    all_stats = {}    # diff -> method -> {arrival_rate, ...}

    for diff_name, diff_cfg in DIFFICULTY_LEVELS.items():
        print(f"\n{'='*60}")
        print(f"  Difficulty: {diff_name}  "
              f"(vehicles={diff_cfg['initial_vehicle_count']}, "
              f"spawn_p={diff_cfg['spawn_probability']})")
        print(f"{'='*60}")

        all_raw[diff_name] = {}
        all_stats[diff_name] = {}

        # Pure RL
        print(f"\n  --- Pure RL ---")
        rl_results = eval_pure_rl(args.episodes, args.ppo_ckpt, diff_cfg)
        all_raw[diff_name]["Pure RL"] = rl_results
        all_stats[diff_name]["Pure RL"] = _compute_stats(rl_results)

        # Pure MPC
        print(f"\n  --- Pure MPC ---")
        mpc_results = eval_pure_mpc(args.episodes, diff_cfg)
        all_raw[diff_name]["Pure MPC"] = mpc_results
        all_stats[diff_name]["Pure MPC"] = _compute_stats(mpc_results)

        # MPC-RL
        print(f"\n  --- MPC-RL ---")
        mpcrl_results = eval_mpcrl(args.episodes, args.mpcrl_ckpt, diff_cfg)
        all_raw[diff_name]["MPC-RL"] = mpcrl_results
        all_stats[diff_name]["MPC-RL"] = _compute_stats(mpcrl_results)

        # Print summary for this difficulty
        print(f"\n  Summary for {diff_name}:")
        for m in ["Pure RL", "Pure MPC", "MPC-RL"]:
            s = all_stats[diff_name][m]
            print(f"    {m:10s}  Arrive={s['arrival_rate']:5.1f}%  "
                  f"Crash={s['crash_rate']:5.1f}%  "
                  f"Reward={s['mean_reward']:7.1f}  Steps={s['mean_steps']:5.0f}")

    # ── Save JSON ──
    json_path = os.path.join(output_dir, "difficulty_sweep_results.json")
    # Convert raw results for JSON serialization
    json_out = {
        "summary": all_stats,
        "raw": {
            diff: {method: episodes for method, episodes in methods.items()}
            for diff, methods in all_raw.items()
        },
    }
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\n[json] Saved: {json_path}")

    # ── Generate plots ──
    print("\nGenerating plots ...")

    plot_grouped_bars(all_stats, "arrival_rate", "Arrival Rate (%)",
                      "Arrival Rate by Traffic Difficulty",
                      "difficulty_arrival_rate", output_dir)

    plot_grouped_bars(all_stats, "crash_rate", "Crash Rate (%)",
                      "Crash Rate by Traffic Difficulty",
                      "difficulty_crash_rate", output_dir)

    plot_grouped_bars(all_stats, "mean_reward", "Mean Episode Reward",
                      "Mean Reward by Traffic Difficulty",
                      "difficulty_mean_reward", output_dir)

    plot_summary_table(all_stats, output_dir)

    print("\n[DONE] Experiment 1 complete!")


if __name__ == "__main__":
    main()
