"""
Intersection 1000-Episode Boxplots — Experiment-7 Style
========================================================
Splits each 1000-episode evaluation into K batches (default 20 × 50 ep).
For each batch computes arrival rate, crash rate, mean reward, mean steps.
Produces per-difficulty + combined 4-panel boxplots matching the Experiment 7
visual style (box + individual-dot overlay).

Outputs (in evals/intersection_boxplots/):
  - intersection_boxplot_very_easy.pdf/png
  - intersection_boxplot_easy.pdf/png
  - intersection_boxplot_moderate.pdf/png
  - intersection_boxplot_combined.pdf/png     (3000 ep → 60 batches)
  - intersection_reward_mpcrl_vs_ppo.pdf/png  (comparable reward scale only)

Usage:
    cd /users/saeani/src/mpcrl/MPC-RL_for_AVs/src_ray_version && \\
    PYTHONPATH="$PWD:$PWD/highway-env:$PYTHONPATH" CUDA_VISIBLE_DEVICES="" \\
    python /users/saeani/src/mpcrl/DRL_MPC_for_AVs/src_ray_version/evals/plot_intersection_boxplots.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ──
EVALS_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(EVALS_DIR, "intersection_boxplots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

K = 20  # number of batches per 1000 episodes → 50 ep/batch

DIFFICULTIES = ["very_easy", "easy", "moderate"]
DIFF_LABELS  = {"very_easy": "Very Easy", "easy": "Easy", "moderate": "Moderate"}

METHODS = ["Pure PPO", "Pure MPC", "MPC-RL"]
COLORS  = {"Pure PPO": "#2196F3", "Pure MPC": "#FF9800", "MPC-RL": "#4CAF50"}

# Map (method_key, difficulty) → file path relative to EVALS_DIR
FILE_MAP = {
    ("MPC-RL", "very_easy"):    "eval1k_mpcrl_very_easy/eval_results_manual_very_easy_1000ep.json",
    ("MPC-RL", "easy"):         "eval1k_mpcrl_easy/eval_results_manual_easy_1000ep.json",
    ("MPC-RL", "moderate"):     "eval1k_mpcrl_moderate/eval_results_manual_moderate_1000ep.json",
    ("Pure MPC", "very_easy"):  "eval1k_purempc_very_easy/eval_results_pure_mpc_manual_very_easy_1000ep.json",
    ("Pure MPC", "easy"):       "eval1k_purempc_easy/eval_results_pure_mpc_manual_easy_1000ep.json",
    ("Pure MPC", "moderate"):   "eval1k_purempc_moderate/eval_results_pure_mpc_manual_moderate_1000ep.json",
    ("Pure PPO", "very_easy"):  "eval1k_pureppo_very_easy/eval_results_pure_ppo_very_easy_1000ep.json",
    ("Pure PPO", "easy"):       "eval1k_pureppo_easy/eval_results_pure_ppo_easy_1000ep.json",
    ("Pure PPO", "moderate"):   "eval1k_pureppo_moderate/eval_results_pure_ppo_moderate_1000ep.json",
}


# ── Data Loading ──
def load_per_episode():
    """Return nested dict: episodes[method][difficulty] = list of ep dicts."""
    episodes = {}
    for method in METHODS:
        episodes[method] = {}
        for diff in DIFFICULTIES:
            fpath = os.path.join(EVALS_DIR, FILE_MAP[(method, diff)])
            with open(fpath) as f:
                d = json.load(f)
            episodes[method][diff] = d["per_episode"]
            print(f"  Loaded {method}/{diff}: {len(d['per_episode'])} episodes")
    return episodes


def compute_batch_stats(ep_list, k):
    """Split ep_list into k equal batches, return list of dicts with per-batch metrics."""
    n = len(ep_list)
    batch_size = n // k
    batches = []
    for b in range(k):
        chunk = ep_list[b * batch_size : (b + 1) * batch_size]
        n_arrived = sum(1 for e in chunk if e.get("arrived", False))
        n_crashed = sum(1 for e in chunk if e.get("crashed", False))
        batches.append({
            "arrival_rate": 100.0 * n_arrived / len(chunk),
            "crash_rate":   100.0 * n_crashed / len(chunk),
            "mean_reward":  float(np.mean([e["reward"] for e in chunk])),
            "mean_steps":   float(np.mean([e["length"] for e in chunk])),
        })
    return batches


# ── Plot Function ──
def make_boxplot(batch_data, title, filename, subtitle=""):
    """
    batch_data: dict  method → list of batch-stat dicts
    Produces a 4-panel figure matching Experiment-7 style.
    """
    metrics = [
        ("arrival_rate", "Arrival Rate (%)"),
        ("crash_rate",   "Crash Rate (%)"),
        ("mean_reward",  "Mean Episode Reward"),
        ("mean_steps",   "Mean Episode Steps"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, (metric, ylabel) in zip(axes, metrics):
        data = []
        for m in METHODS:
            vals = [b[metric] for b in batch_data[m]]
            data.append(vals)

        positions = np.arange(len(METHODS))
        bp = ax.boxplot(
            data, positions=positions, widths=0.4,
            patch_artist=True, showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="white",
                           markeredgecolor="black", markersize=6),
            medianprops=dict(color="orange", linewidth=1.5),
        )

        for patch, m in zip(bp["boxes"], METHODS):
            patch.set_facecolor(COLORS[m])
            patch.set_alpha(0.6)

        # Scatter individual batch points with jitter
        rng = np.random.default_rng(42)
        for i, (m, vals) in enumerate(zip(METHODS, data)):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(
                [i + j for j in jitter], vals,
                color=COLORS[m], alpha=0.7, s=30, zorder=5,
                edgecolors="black", linewidths=0.5,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(METHODS, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    n_batches = len(batch_data[METHODS[0]])
    batch_size = 1000 // K
    fig.suptitle(f"{title}\n(each dot = 1 batch of {batch_size} episodes, "
                 f"{n_batches} batches per method)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.91])

    for ext in ("pdf", "png"):
        path = os.path.join(OUTPUT_DIR, f"{filename}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {filename}.pdf/png")


def make_reward_only_comparable(batch_data_per_diff, filename):
    """
    Separate reward boxplot excluding Pure MPC (different reward scale).
    One panel per difficulty + combined.
    """
    rl_methods = ["Pure PPO", "MPC-RL"]
    panels = DIFFICULTIES + ["combined"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for ax, panel in zip(axes, panels):
        if panel == "combined":
            title = "Combined (3000 ep)"
        else:
            title = DIFF_LABELS[panel]

        data = []
        for m in rl_methods:
            vals = [b["mean_reward"] for b in batch_data_per_diff[panel][m]]
            data.append(vals)

        positions = np.arange(len(rl_methods))
        bp = ax.boxplot(
            data, positions=positions, widths=0.35,
            patch_artist=True, showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="white",
                           markeredgecolor="black", markersize=6),
            medianprops=dict(color="orange", linewidth=1.5),
        )

        for patch, m in zip(bp["boxes"], rl_methods):
            patch.set_facecolor(COLORS[m])
            patch.set_alpha(0.6)

        rng = np.random.default_rng(42)
        for i, (m, vals) in enumerate(zip(rl_methods, data)):
            jitter = rng.uniform(-0.10, 0.10, size=len(vals))
            ax.scatter(
                [i + j for j in jitter], vals,
                color=COLORS[m], alpha=0.7, s=30, zorder=5,
                edgecolors="black", linewidths=0.5,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(rl_methods, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean Episode Reward" if ax == axes[0] else "")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Episode Reward — MPC-RL vs Pure PPO (comparable reward scale)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    for ext in ("pdf", "png"):
        path = os.path.join(OUTPUT_DIR, f"{filename}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {filename}.pdf/png")


# ── Main ──
if __name__ == "__main__":
    print("Loading intersection 1000-ep evaluation data …")
    episodes = load_per_episode()

    # Per-difficulty boxplots
    batch_data_per_diff = {}

    for diff in DIFFICULTIES:
        print(f"\n── {DIFF_LABELS[diff]} ──")
        batch_data = {}
        for method in METHODS:
            batch_data[method] = compute_batch_stats(episodes[method][diff], K)
        batch_data_per_diff[diff] = batch_data

        make_boxplot(
            batch_data,
            title=f"Intersection — {DIFF_LABELS[diff]}",
            filename=f"intersection_boxplot_{diff}",
        )

    # Combined (pool all difficulties → 3000 ep → 3K batches)
    print(f"\n── Combined (all difficulties) ──")
    combined_batch = {}
    for method in METHODS:
        all_eps = []
        for diff in DIFFICULTIES:
            all_eps.extend(episodes[method][diff])
        combined_batch[method] = compute_batch_stats(all_eps, K * 3)  # 60 batches
    batch_data_per_diff["combined"] = combined_batch

    make_boxplot(
        combined_batch,
        title="Intersection — All Difficulties Combined",
        filename="intersection_boxplot_combined",
        subtitle="3000 episodes pooled (very_easy + easy + moderate)",
    )

    # Comparable reward plot (excluding Pure MPC)
    print(f"\n── Reward-only (MPC-RL vs Pure PPO) ──")
    make_reward_only_comparable(batch_data_per_diff, "intersection_reward_mpcrl_vs_ppo")

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
