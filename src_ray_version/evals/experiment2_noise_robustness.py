"""
Experiment 2: Robustness to Observation Noise
==============================================
Inject Gaussian noise into observations and measure how each method
degrades.  Pure MPC uses ground-truth state internally, so its performance
should be unaffected — demonstrating inherent robustness.  Pure RL and
MPC-RL rely on (possibly noisy) observations, so they should degrade.

Noise levels (σ applied to normalized obs ∈ [-1,1]):
    0.00  (clean — baseline)
    0.02
    0.05
    0.10
    0.20

For each (method × noise), run N episodes and report arrival rate.

Outputs (in experiment2_results/):
  - noise_robustness_results.json     Raw data
  - noise_arrival_rate.pdf            Line plot: arrival rate vs σ
  - noise_crash_rate.pdf              Line plot: crash rate vs σ
  - noise_summary_table.pdf           Table

Usage:
    cd /users/saeani/src/mpcrl/MPC-RL_for_AVs/src_ray_version && \
    PYTHONPATH="$PWD:$PWD/highway-env:$PYTHONPATH" CUDA_VISIBLE_DEVICES="" \
    python /users/saeani/src/mpcrl/DRL_MPC_for_AVs/src_ray_version/evals/experiment2_noise_robustness.py
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

EVALS_DIR  = os.path.dirname(os.path.abspath(__file__))
PPO_CKPT   = "/users/saeani/DEV/ray_results/PPO_pure_RL/PPO_intersection-pure-ppo_03a31_00000_0_2026-03-01_23-40-22/checkpoint_000025"
MPCRL_CKPT = "/users/saeani/DEV/ray_results/PPO_v0_manual/PPO_intersection-mpcrl-refspeed-manual_c129f_00000_0_2026-03-01_00-44-07/checkpoint_000020"

sys.path.insert(0, os.path.join(EVALS_DIR, "../../../MPC-RL_for_AVs/src_ray_version"))

NOISE_LEVELS = [0.00, 0.02, 0.05, 0.10, 0.20]
COLORS = {"Pure RL": "#2196F3", "Pure MPC": "#FF9800", "MPC-RL": "#4CAF50"}
MARKERS = {"Pure RL": "o", "Pure MPC": "s", "MPC-RL": "D"}


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


def _add_noise(obs, sigma, rng):
    """Add Gaussian noise to observation."""
    if sigma <= 0:
        return obs
    noise = rng.normal(0, sigma, size=obs.shape).astype(obs.dtype)
    return obs + noise


# =====================================================================
#  1. Pure RL with noise
# =====================================================================
def eval_pure_rl_noisy(n_episodes, ppo_ckpt, sigma, seed=42):
    from ray.rllib.policy.policy import Policy
    from train_pure_ppo import IntersectionPurePPOEnv

    policy = Policy.from_checkpoint(ppo_ckpt)
    if isinstance(policy, dict):
        policy = policy["default_policy"]

    env = IntersectionPurePPOEnv(render_mode="rgb_array")
    rng = np.random.default_rng(seed)
    results = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            # Add noise to observation before policy inference
            noisy_obs = _add_noise(obs, sigma, rng)
            action = policy.compute_single_action(noisy_obs, explore=False)[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated

        ego = _get_ego(env)
        arrived = _has_arrived(env, ego) if ego else False
        crashed = bool(ego.crashed) if ego else True
        status = "ARRIVE" if arrived else ("CRASH" if crashed else "OTHER")
        results.append({"episode": ep + 1, "status": status,
                        "reward": float(ep_reward), "steps": steps})

    env.close()
    return results


# =====================================================================
#  2. Pure MPC (noise does not affect MPC — uses true state internally)
# =====================================================================
def eval_pure_mpc(n_episodes, seed=42):
    """Pure MPC is immune to observation noise since the MPC controller
    reads ground-truth vehicle state directly.  We run once as baseline."""
    env = gymnasium.make("intersection-mpc-manual", render_mode="rgb_array")
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
        results.append({"episode": ep + 1, "status": status,
                        "reward": float(ep_reward), "steps": steps})

    env.close()
    return results


# =====================================================================
#  3. MPC-RL with noise
# =====================================================================
def eval_mpcrl_noisy(n_episodes, mpcrl_ckpt, sigma, seed=42):
    """Noise is added to the observation before the RL policy infers
    the reference speed.  The MPC controller still uses true state, so
    the impact is through the RL decision (ref speed) only."""
    from ray.rllib.policy.policy import Policy

    policy = Policy.from_checkpoint(mpcrl_ckpt)
    if isinstance(policy, dict):
        policy = policy["default_policy"]

    env = gymnasium.make("intersection-mpcrl-refspeed-manual", render_mode="rgb_array")
    rng = np.random.default_rng(seed)
    results = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            noisy_obs = _add_noise(obs, sigma, rng)
            action = policy.compute_single_action(noisy_obs, explore=False)[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated

        ego = _get_ego(env)
        arrived = _has_arrived(env, ego) if ego else False
        crashed = bool(ego.crashed) if ego else True
        status = "ARRIVE" if arrived else ("CRASH" if crashed else "OTHER")
        results.append({"episode": ep + 1, "status": status,
                        "reward": float(ep_reward), "steps": steps})

    env.close()
    return results


# =====================================================================
#  Stats
# =====================================================================
def _compute_stats(results):
    n = len(results)
    if n == 0:
        return {"arrival_rate": 0, "crash_rate": 0, "mean_reward": 0,
                "mean_steps": 0, "n_total": 0}
    n_arrived = sum(1 for r in results if r["status"] == "ARRIVE")
    n_crashed = sum(1 for r in results if r["status"] == "CRASH")
    return {
        "arrival_rate": 100.0 * n_arrived / n,
        "crash_rate": 100.0 * n_crashed / n,
        "mean_reward": float(np.mean([r["reward"] for r in results])),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "n_arrived": n_arrived,
        "n_crashed": n_crashed,
        "n_total": n,
    }


# =====================================================================
#  Plotting
# =====================================================================
def plot_metric_vs_noise(all_stats, metric_key, ylabel, title, filename, output_dir):
    """Line plot: metric vs noise level for each method."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for method in ["Pure RL", "Pure MPC", "MPC-RL"]:
        xs = []
        ys = []
        for sigma in NOISE_LEVELS:
            key = f"{sigma:.2f}"
            if key in all_stats and method in all_stats[key]:
                xs.append(sigma)
                ys.append(all_stats[key][method][metric_key])
        ax.plot(xs, ys, marker=MARKERS[method], color=COLORS[method],
                linewidth=2.5, markersize=8, label=method)

    ax.set_xlabel("Observation Noise σ", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"{filename}.{ext}")
        fig.savefig(path, dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {filename}.pdf/png")


def plot_summary_table(all_stats, output_dir):
    """Render summary table."""
    methods = ["Pure RL", "Pure MPC", "MPC-RL"]
    col_labels = [f"σ={s:.2f}" for s in NOISE_LEVELS]

    fig, axes = plt.subplots(len(methods), 1, figsize=(12, 2.5 * len(methods)))
    if len(methods) == 1:
        axes = [axes]

    for idx, method in enumerate(methods):
        ax = axes[idx]
        ax.axis("off")

        row_labels = ["Arrival %", "Crash %", "Mean Reward"]
        cell_data = []
        for metric in ["arrival_rate", "crash_rate", "mean_reward"]:
            row = []
            for sigma in NOISE_LEVELS:
                key = f"{sigma:.2f}"
                val = all_stats.get(key, {}).get(method, {}).get(metric, 0)
                if metric in ("arrival_rate", "crash_rate"):
                    row.append(f"{val:.1f}%")
                else:
                    row.append(f"{val:.1f}")
            cell_data.append(row)

        table = ax.table(cellText=cell_data, rowLabels=row_labels,
                         colLabels=col_labels, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.5)
        ax.set_title(method, fontsize=12, fontweight="bold",
                     color=COLORS[method], pad=10)

    fig.suptitle("Noise Robustness — Summary", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"noise_summary_table.{ext}")
        fig.savefig(path, dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] noise_summary_table.pdf/png")


# =====================================================================
#  Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Noise Robustness")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per (method × noise) (default: 50)")
    parser.add_argument("--ppo-ckpt", type=str, default=PPO_CKPT)
    parser.add_argument("--mpcrl-ckpt", type=str, default=MPCRL_CKPT)
    parser.add_argument("--output-dir", type=str, default=EVALS_DIR)
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, "experiment2_results")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  Experiment 2: Robustness to Observation Noise")
    print(f"  Episodes per condition: {args.episodes}")
    print(f"  Noise levels: {NOISE_LEVELS}")
    print(f"  Output: {output_dir}")
    print(f"{'#'*60}")

    all_raw = {}
    all_stats = {}

    # ── Pure MPC baseline (noise-immune, run once) ──
    print(f"\n{'='*60}")
    print(f"  Pure MPC baseline (noise-immune)")
    print(f"{'='*60}")
    mpc_results = eval_pure_mpc(args.episodes)
    mpc_stats = _compute_stats(mpc_results)
    print(f"  Pure MPC:  Arrive={mpc_stats['arrival_rate']:.1f}%  "
          f"Crash={mpc_stats['crash_rate']:.1f}%")

    for sigma in NOISE_LEVELS:
        key = f"{sigma:.2f}"
        print(f"\n{'='*60}")
        print(f"  Noise σ = {sigma:.2f}")
        print(f"{'='*60}")

        all_raw[key] = {}
        all_stats[key] = {}

        # Pure RL
        print(f"  --- Pure RL (σ={sigma:.2f}) ---")
        rl_results = eval_pure_rl_noisy(args.episodes, args.ppo_ckpt, sigma)
        all_raw[key]["Pure RL"] = rl_results
        all_stats[key]["Pure RL"] = _compute_stats(rl_results)

        # Pure MPC — same results for all noise levels
        all_raw[key]["Pure MPC"] = mpc_results
        all_stats[key]["Pure MPC"] = mpc_stats

        # MPC-RL
        print(f"  --- MPC-RL (σ={sigma:.2f}) ---")
        mpcrl_results = eval_mpcrl_noisy(args.episodes, args.mpcrl_ckpt, sigma)
        all_raw[key]["MPC-RL"] = mpcrl_results
        all_stats[key]["MPC-RL"] = _compute_stats(mpcrl_results)

        # Print summary
        for m in ["Pure RL", "Pure MPC", "MPC-RL"]:
            s = all_stats[key][m]
            print(f"    {m:10s}  Arrive={s['arrival_rate']:5.1f}%  "
                  f"Crash={s['crash_rate']:5.1f}%")

    # ── Save JSON ──
    json_path = os.path.join(output_dir, "noise_robustness_results.json")
    with open(json_path, "w") as f:
        json.dump({"summary": all_stats, "raw": all_raw}, f, indent=2)
    print(f"\n[json] Saved: {json_path}")

    # ── Plots ──
    print("\nGenerating plots ...")

    plot_metric_vs_noise(all_stats, "arrival_rate", "Arrival Rate (%)",
                         "Arrival Rate vs Observation Noise",
                         "noise_arrival_rate", output_dir)

    plot_metric_vs_noise(all_stats, "crash_rate", "Crash Rate (%)",
                         "Crash Rate vs Observation Noise",
                         "noise_crash_rate", output_dir)

    plot_metric_vs_noise(all_stats, "mean_reward", "Mean Episode Reward",
                         "Mean Reward vs Observation Noise",
                         "noise_mean_reward", output_dir)

    plot_summary_table(all_stats, output_dir)

    print("\n[DONE] Experiment 2 complete!")


if __name__ == "__main__":
    main()
