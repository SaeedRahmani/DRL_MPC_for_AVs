"""
Experiment 7: Statistical Rigor
================================
Run multiple independent evaluation batches (different env random seeds)
for Pure RL, Pure MPC, and MPC-RL.  For each batch compute arrival rate,
crash rate, and mean episode return, then report:

  - Mean ± std across batches
  - 95 % confidence intervals  (t-distribution for small n)
  - Pair-wise Welch's t-tests for significance between methods
  - Bootstrap 95 % CIs as a robustness check

Design:
  K independent batches  (default K=10)  ×  M episodes per batch  (default M=30)
  Each batch uses a distinct seed → different initial traffic spawns.

Outputs (in experiment7_results/):
  - statistical_rigor_results.json      Full per-batch data + CIs
  - stat_summary_table.pdf/png          Publication-ready summary table
  - stat_ci_bars.pdf/png                Bar chart with 95 % CI whiskers
  - stat_pairwise_tests.pdf/png         Significance matrix heatmap

Usage:
    cd /users/saeani/src/mpcrl/MPC-RL_for_AVs/src_ray_version && \\
    PYTHONPATH="$PWD:$PWD/highway-env:$PYTHONPATH" CUDA_VISIBLE_DEVICES="" \\
    python /users/saeani/src/mpcrl/DRL_MPC_for_AVs/src_ray_version/evals/experiment7_statistical_rigor.py
"""

import os, sys, json, argparse, time
import numpy as np
from scipy import stats as sp_stats

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

COLORS = {"Pure RL": "#2196F3", "Pure MPC": "#FF9800", "MPC-RL": "#4CAF50"}
METHODS = ["Pure RL", "Pure MPC", "MPC-RL"]

DIFFICULTY_PRESETS = {
    "easy":   {"initial_vehicle_count": 5,  "spawn_probability": 0.3},
    "medium": {"initial_vehicle_count": 10, "spawn_probability": 0.6},
    "hard":   {"initial_vehicle_count": 15, "spawn_probability": 0.9},
}


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
#  Evaluation functions  (one batch = M episodes with a given seed)
# =====================================================================
def run_pure_rl_batch(n_episodes, ppo_ckpt, seed, env_cfg=None):
    from ray.rllib.policy.policy import Policy
    from train_pure_ppo import IntersectionPurePPOEnv

    policy = Policy.from_checkpoint(ppo_ckpt)
    if isinstance(policy, dict):
        policy = policy["default_policy"]

    env = IntersectionPurePPOEnv(config=env_cfg or {}, render_mode="rgb_array")
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ep_reward, steps = False, 0.0, 0
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
        results.append({"status": status, "reward": float(ep_reward), "steps": steps})
    env.close()
    return results


def run_pure_mpc_batch(n_episodes, seed, env_cfg=None):
    env = gymnasium.make("intersection-mpc-manual", render_mode="rgb_array",
                         config=env_cfg or {})
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ep_reward, steps = False, 0.0, 0
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
        results.append({"status": status, "reward": float(ep_reward), "steps": steps})
    env.close()
    return results


def run_mpcrl_batch(n_episodes, mpcrl_ckpt, seed, env_cfg=None):
    from ray.rllib.policy.policy import Policy

    policy = Policy.from_checkpoint(mpcrl_ckpt)
    if isinstance(policy, dict):
        policy = policy["default_policy"]

    env = gymnasium.make("intersection-mpcrl-refspeed-manual", render_mode="rgb_array",
                         config=env_cfg or {})
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ep_reward, steps = False, 0.0, 0
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
        results.append({"status": status, "reward": float(ep_reward), "steps": steps})
    env.close()
    return results


# =====================================================================
#  Statistics
# =====================================================================
def batch_stats(episodes):
    """Compute per-batch aggregate metrics."""
    n = len(episodes)
    n_arr = sum(1 for e in episodes if e["status"] == "ARRIVE")
    n_crash = sum(1 for e in episodes if e["status"] == "CRASH")
    rewards = [e["reward"] for e in episodes]
    steps = [e["steps"] for e in episodes]
    return {
        "arrival_rate": 100.0 * n_arr / n,
        "crash_rate": 100.0 * n_crash / n,
        "mean_reward": float(np.mean(rewards)),
        "mean_steps": float(np.mean(steps)),
    }


def compute_ci(values, confidence=0.95):
    """Mean, std, and t-based CI for a list of values."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    se = std / np.sqrt(n)
    t_crit = sp_stats.t.ppf((1 + confidence) / 2, df=n - 1)
    ci_lo = mean - t_crit * se
    ci_hi = mean + t_crit * se
    return {"mean": mean, "std": std, "se": se,
            "ci_lo": ci_lo, "ci_hi": ci_hi, "n": n}


def bootstrap_ci(values, n_boot=10000, confidence=0.95, rng=None):
    """Bootstrap percentile CI."""
    if rng is None:
        rng = np.random.default_rng(42)
    arr = np.array(values, dtype=float)
    n = len(arr)
    boot_means = np.array([
        np.mean(rng.choice(arr, size=n, replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - confidence) / 2
    lo = float(np.percentile(boot_means, 100 * alpha))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return {"boot_ci_lo": lo, "boot_ci_hi": hi}


def welch_t_test(vals_a, vals_b):
    """Two-sided Welch's t-test. Returns t-stat and p-value."""
    t_stat, p_val = sp_stats.ttest_ind(vals_a, vals_b, equal_var=False)
    return {"t_stat": float(t_stat), "p_value": float(p_val)}


# =====================================================================
#  Plotting
# =====================================================================
def plot_ci_bars(summary, output_dir):
    """Bar chart with 95% CI error bars for each metric × method."""
    metrics = [
        ("arrival_rate", "Arrival Rate (%)"),
        ("crash_rate", "Crash Rate (%)"),
        ("mean_reward", "Mean Episode Reward"),
        ("mean_steps", "Mean Episode Steps"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    x = np.arange(len(METHODS))
    width = 0.5

    for ax, (metric, label) in zip(axes, metrics):
        means = [summary[m][metric]["mean"] for m in METHODS]
        ci_los = [summary[m][metric]["ci_lo"] for m in METHODS]
        ci_his = [summary[m][metric]["ci_hi"] for m in METHODS]
        err_lo = [means[i] - ci_los[i] for i in range(len(METHODS))]
        err_hi = [ci_his[i] - means[i] for i in range(len(METHODS))]

        bars = ax.bar(x, means, width,
                      color=[COLORS[m] for m in METHODS],
                      alpha=0.85,
                      yerr=[err_lo, err_hi],
                      capsize=6, error_kw={"linewidth": 1.5})
        # Value labels
        for i, (bar, m, lo, hi) in enumerate(zip(bars, means, ci_los, ci_his)):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + err_hi[i] + 0.5,
                    f"{m:.1f}\n[{lo:.1f}, {hi:.1f}]",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, fontsize=10)
        ax.set_ylabel(label, fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Method Comparison with 95% Confidence Intervals",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"stat_ci_bars.{ext}")
        fig.savefig(path, dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] stat_ci_bars.pdf/png")


def plot_summary_table(summary, output_dir):
    """Render a publication-style summary table as a figure."""
    metrics = [
        ("arrival_rate", "Arrival Rate (%)"),
        ("crash_rate", "Crash Rate (%)"),
        ("mean_reward", "Mean Reward"),
        ("mean_steps", "Mean Steps"),
    ]

    col_labels = METHODS
    row_labels = [label for _, label in metrics]
    cell_data = []
    for metric, _ in metrics:
        row = []
        for m in METHODS:
            s = summary[m][metric]
            row.append(f"{s['mean']:.1f} ± {s['std']:.1f}\n"
                       f"[{s['ci_lo']:.1f}, {s['ci_hi']:.1f}]")
        cell_data.append(row)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    table = ax.table(cellText=cell_data, rowLabels=row_labels,
                     colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.2)

    # Color column headers
    for j, m in enumerate(METHODS):
        table[0, j].set_facecolor(COLORS[m])
        table[0, j].set_text_props(color="white", fontweight="bold")

    fig.suptitle("Statistical Summary: Mean ± Std  [95% CI]",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"stat_summary_table.{ext}")
        fig.savefig(path, dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] stat_summary_table.pdf/png")


def plot_pairwise_tests(pairwise, output_dir):
    """Heatmap of p-values from pair-wise Welch's t-tests."""
    metrics = [
        ("arrival_rate", "Arrival Rate"),
        ("crash_rate", "Crash Rate"),
        ("mean_reward", "Mean Reward"),
        ("mean_steps", "Mean Steps"),
    ]
    pairs = [(a, b) for i, a in enumerate(METHODS) for b in METHODS[i+1:]]
    pair_labels = [f"{a}\nvs\n{b}" for a, b in pairs]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Build matrix: rows = metrics, cols = pairs
    matrix = []
    annot = []
    for metric, _ in metrics:
        row_vals = []
        row_annot = []
        for a, b in pairs:
            key = f"{a} vs {b}"
            p = pairwise[metric][key]["p_value"]
            row_vals.append(p)
            sig = ""
            if p < 0.001:
                sig = "***"
            elif p < 0.01:
                sig = "**"
            elif p < 0.05:
                sig = "*"
            else:
                sig = "ns"
            row_annot.append(f"p={p:.3f}\n{sig}")
        matrix.append(row_vals)
        annot.append(row_annot)

    matrix = np.array(matrix)

    # Custom colormap: green (significant) → red (not significant)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("sig",
        [(0, "#4CAF50"), (0.05, "#FFC107"), (0.2, "#FF5722"), (1.0, "#B71C1C")])

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] < 0.1 else "black"
            ax.text(j, i, annot[i][j], ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(range(len(pair_labels)))
    ax.set_xticklabels(pair_labels, fontsize=10)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([label for _, label in metrics], fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("p-value", fontsize=11)

    ax.set_title("Pair-wise Welch's t-test  (* p<0.05  ** p<0.01  *** p<0.001)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"stat_pairwise_tests.{ext}")
        fig.savefig(path, dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] stat_pairwise_tests.pdf/png")


def plot_batch_distributions(batch_data, output_dir):
    """Box + swarm style plot showing per-batch distributions for each metric."""
    metrics = [
        ("arrival_rate", "Arrival Rate (%)"),
        ("crash_rate", "Crash Rate (%)"),
        ("mean_reward", "Mean Episode Reward"),
        ("mean_steps", "Mean Episode Steps"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, (metric, label) in zip(axes, metrics):
        data = []
        positions = []
        for i, m in enumerate(METHODS):
            vals = [b[metric] for b in batch_data[m]]
            data.append(vals)
            positions.append(i)

        bp = ax.boxplot(data, positions=positions, widths=0.4,
                        patch_artist=True, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="white",
                                       markeredgecolor="black", markersize=6))

        for patch, m in zip(bp["boxes"], METHODS):
            patch.set_facecolor(COLORS[m])
            patch.set_alpha(0.6)

        # Scatter individual batch points
        rng = np.random.default_rng(0)
        for i, (m, vals) in enumerate(zip(METHODS, data)):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter([i + j for j in jitter], vals,
                      color=COLORS[m], alpha=0.7, s=30, zorder=5,
                      edgecolors="black", linewidths=0.5)

        ax.set_xticks(positions)
        ax.set_xticklabels(METHODS, fontsize=10)
        ax.set_ylabel(label, fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-Batch Distribution of Metrics (each dot = 1 batch)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"stat_batch_distributions.{ext}")
        fig.savefig(path, dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] stat_batch_distributions.pdf/png")


# =====================================================================
#  Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Experiment 7: Statistical Rigor")
    parser.add_argument("--batches", type=int, default=10,
                        help="Number of independent batches K (default: 10)")
    parser.add_argument("--episodes", type=int, default=30,
                        help="Episodes per batch M (default: 30)")
    parser.add_argument("--ppo-ckpt", type=str, default=PPO_CKPT)
    parser.add_argument("--mpcrl-ckpt", type=str, default=MPCRL_CKPT)
    parser.add_argument("--output-dir", type=str, default=EVALS_DIR)
    parser.add_argument("--difficulty", type=str, default="medium",
                        choices=["easy", "medium", "hard"],
                        help="Traffic difficulty preset (default: medium)")
    args = parser.parse_args()

    K = args.batches
    M = args.episodes
    diff = args.difficulty
    env_cfg = DIFFICULTY_PRESETS[diff]

    dir_suffix = f"__{diff}" if diff != "medium" else ""
    output_dir = os.path.join(args.output_dir, f"experiment7_results{dir_suffix}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  Experiment 7: Statistical Rigor")
    print(f"  Difficulty: {diff}  {env_cfg}")
    print(f"  Batches K = {K},  Episodes per batch M = {M}")
    print(f"  Total episodes per method: {K * M}")
    print(f"  Output: {output_dir}")
    print(f"{'#'*60}")

    # ── Seed schedule: each batch gets a unique base seed ──
    base_seeds = [1000 * (b + 1) for b in range(K)]

    # ── Run all batches ──
    # batch_data[method] = list of K dicts  {arrival_rate, crash_rate, ...}
    # raw_data[method]   = list of K lists of episode results
    batch_data = {m: [] for m in METHODS}
    raw_data   = {m: [] for m in METHODS}

    for b_idx, seed in enumerate(base_seeds):
        print(f"\n{'='*60}")
        print(f"  Batch {b_idx+1}/{K}  (seed={seed})")
        print(f"{'='*60}")

        # Pure RL
        t0 = time.time()
        rl_eps = run_pure_rl_batch(M, args.ppo_ckpt, seed, env_cfg)
        rl_stats = batch_stats(rl_eps)
        raw_data["Pure RL"].append(rl_eps)
        batch_data["Pure RL"].append(rl_stats)
        rl_time = time.time() - t0
        print(f"  Pure RL   : arrive={rl_stats['arrival_rate']:5.1f}%  "
              f"crash={rl_stats['crash_rate']:5.1f}%  "
              f"reward={rl_stats['mean_reward']:7.1f}  ({rl_time:.1f}s)")

        # Pure MPC
        t0 = time.time()
        mpc_eps = run_pure_mpc_batch(M, seed, env_cfg)
        mpc_stats = batch_stats(mpc_eps)
        raw_data["Pure MPC"].append(mpc_eps)
        batch_data["Pure MPC"].append(mpc_stats)
        mpc_time = time.time() - t0
        print(f"  Pure MPC  : arrive={mpc_stats['arrival_rate']:5.1f}%  "
              f"crash={mpc_stats['crash_rate']:5.1f}%  "
              f"reward={mpc_stats['mean_reward']:7.1f}  ({mpc_time:.1f}s)")

        # MPC-RL
        t0 = time.time()
        mpcrl_eps = run_mpcrl_batch(M, args.mpcrl_ckpt, seed, env_cfg)
        mpcrl_stats = batch_stats(mpcrl_eps)
        raw_data["MPC-RL"].append(mpcrl_eps)
        batch_data["MPC-RL"].append(mpcrl_stats)
        mpcrl_time = time.time() - t0
        print(f"  MPC-RL    : arrive={mpcrl_stats['arrival_rate']:5.1f}%  "
              f"crash={mpcrl_stats['crash_rate']:5.1f}%  "
              f"reward={mpcrl_stats['mean_reward']:7.1f}  ({mpcrl_time:.1f}s)")

    # ================================================================
    #  Compute aggregate statistics
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  Computing Statistics ...")
    print(f"{'='*60}")

    metric_keys = ["arrival_rate", "crash_rate", "mean_reward", "mean_steps"]
    summary = {}   # method -> metric -> {mean, std, ci_lo, ci_hi, boot_ci_lo, ...}

    for m in METHODS:
        summary[m] = {}
        for metric in metric_keys:
            values = [b[metric] for b in batch_data[m]]
            ci = compute_ci(values)
            bci = bootstrap_ci(values)
            summary[m][metric] = {**ci, **bci}

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"  STATISTICAL SUMMARY  (K={K} batches × M={M} episodes)")
    print(f"{'='*60}")
    header = f"  {'Metric':<18s}"
    for m in METHODS:
        header += f"  {m:>28s}"
    print(header)
    print("  " + "-" * (18 + 30 * len(METHODS)))

    for metric in metric_keys:
        row = f"  {metric:<18s}"
        for m in METHODS:
            s = summary[m][metric]
            row += f"  {s['mean']:6.1f} ± {s['std']:5.1f}  [{s['ci_lo']:6.1f}, {s['ci_hi']:6.1f}]"
        print(row)

    # ── Pair-wise tests ──
    print(f"\n  Pair-wise Welch's t-tests:")
    pairwise = {}  # metric -> "A vs B" -> {t_stat, p_value}
    for metric in metric_keys:
        pairwise[metric] = {}
        for i, a in enumerate(METHODS):
            for b in METHODS[i+1:]:
                vals_a = [bd[metric] for bd in batch_data[a]]
                vals_b = [bd[metric] for bd in batch_data[b]]
                result = welch_t_test(vals_a, vals_b)
                key = f"{a} vs {b}"
                pairwise[metric][key] = result
                sig = "***" if result["p_value"] < 0.001 else \
                      "**" if result["p_value"] < 0.01 else \
                      "*" if result["p_value"] < 0.05 else "ns"
                print(f"    {metric:<18s}  {key:<25s}  "
                      f"t={result['t_stat']:+6.2f}  p={result['p_value']:.4f}  {sig}")

    # ================================================================
    #  Save JSON
    # ================================================================
    json_out = {
        "config": {"K": K, "M": M, "total_per_method": K * M,
                   "difficulty": diff, "env_cfg": env_cfg,
                   "base_seeds": base_seeds},
        "summary": {},
        "pairwise_tests": pairwise,
        "batch_data": {},
    }
    for m in METHODS:
        json_out["summary"][m] = summary[m]
        json_out["batch_data"][m] = batch_data[m]

    json_path = os.path.join(output_dir, "statistical_rigor_results.json")
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\n[json] Saved: {json_path}")

    # ================================================================
    #  Generate plots
    # ================================================================
    print("\nGenerating plots ...")
    plot_ci_bars(summary, output_dir)
    plot_summary_table(summary, output_dir)
    plot_pairwise_tests(pairwise, output_dir)
    plot_batch_distributions(batch_data, output_dir)

    print("\n[DONE] Experiment 7 complete!")


if __name__ == "__main__":
    main()
