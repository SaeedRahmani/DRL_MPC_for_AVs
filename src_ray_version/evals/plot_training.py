"""
Plot key training curves from Ray/RLlib progress.csv.

Generates SEPARATE plots for each metric with clean, non-stretched dimensions.

Usage:
    python plot_training.py --csv <path_to_progress.csv> --output <output_dir>
"""

import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np


# ── Style defaults ────────────────────────────────────────────────────
FIG_W, FIG_H = 8, 4          # width x height for each individual plot
DPI = 150
SMOOTH_WINDOW = 10
RAW_ALPHA = 0.25
RAW_LW = 0.7
SMOOTH_LW = 2.0


def smooth(y, window=SMOOTH_WINDOW):
    """Simple centred moving average."""
    y = np.asarray(y, dtype=float)
    if len(y) < window:
        return y
    cumsum = np.cumsum(np.insert(y, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _single_plot(x, y, xlabel, ylabel, title, color, path,
                 fill_lo=None, fill_hi=None, hlines=None):
    """Create one standalone figure with raw + smoothed line."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    y = np.asarray(y, dtype=float)
    ax.plot(x, y, alpha=RAW_ALPHA, color=color, linewidth=RAW_LW)
    sy = smooth(y)
    ax.plot(x[:len(sy)], sy, color=color, linewidth=SMOOTH_LW, label="Smoothed")
    if fill_lo is not None and fill_hi is not None:
        ax.fill_between(x, fill_lo, fill_hi, alpha=0.10, color=color)
    if hlines:
        for hl in hlines:
            ax.axhline(y=hl, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _save(fig, path)


def plot_training(csv_path: str, output_dir: str):
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # X-axis
    x_col = "timesteps_total" if "timesteps_total" in df.columns else "num_env_steps_sampled"
    x = df[x_col].values
    x_label = "Environment Steps"

    prefix = "info/learner/default_policy/learner_stats/"

    # ── Helper to fetch a column (tries env_runners/ variants too) ──
    def col(name, fallback=None):
        if name in df.columns:
            return df[name].values
        if fallback and fallback in df.columns:
            return df[fallback].values
        return None

    # ────────────────────────────────────────────────────────────────
    # 1. Episode Reward Mean
    # ────────────────────────────────────────────────────────────────
    rw_mean = col("env_runners/episode_reward_mean",
                  "env_runners/episode_return_mean")
    rw_max  = col("env_runners/episode_reward_max",
                  "env_runners/episode_return_max")
    rw_min  = col("env_runners/episode_reward_min",
                  "env_runners/episode_return_min")
    if rw_mean is not None:
        _single_plot(x, rw_mean, x_label, "Episode Reward",
                     "Episode Reward (Mean)", "C0",
                     os.path.join(output_dir, "01_episode_reward.png"),
                     fill_lo=rw_min, fill_hi=rw_max)

    # ────────────────────────────────────────────────────────────────
    # 2. Episode Length
    # ────────────────────────────────────────────────────────────────
    ep_len = col("env_runners/episode_len_mean")
    if ep_len is not None:
        _single_plot(x, ep_len, x_label, "Episode Length (steps)",
                     "Episode Length (Mean)", "C1",
                     os.path.join(output_dir, "02_episode_length.png"))

    # ────────────────────────────────────────────────────────────────
    # 3. Total Loss
    # ────────────────────────────────────────────────────────────────
    total_loss = col(prefix + "total_loss")
    if total_loss is not None:
        _single_plot(x, total_loss, x_label, "Total Loss",
                     "PPO Total Loss", "C3",
                     os.path.join(output_dir, "03_total_loss.png"))

    # ────────────────────────────────────────────────────────────────
    # 4. Policy (Surrogate) Loss
    # ────────────────────────────────────────────────────────────────
    policy_loss = col(prefix + "policy_loss")
    if policy_loss is not None:
        _single_plot(x, policy_loss, x_label, "Policy Loss",
                     "PPO Policy (Surrogate) Loss", "C4",
                     os.path.join(output_dir, "04_policy_loss.png"))

    # ────────────────────────────────────────────────────────────────
    # 5. Value Function Loss
    # ────────────────────────────────────────────────────────────────
    vf_loss = col(prefix + "vf_loss")
    if vf_loss is not None:
        _single_plot(x, vf_loss, x_label, "VF Loss",
                     "PPO Value Function Loss", "C5",
                     os.path.join(output_dir, "05_vf_loss.png"))

    # ────────────────────────────────────────────────────────────────
    # 6. Entropy
    # ────────────────────────────────────────────────────────────────
    entropy = col(prefix + "entropy")
    if entropy is not None:
        _single_plot(x, entropy, x_label, "Entropy",
                     "Policy Entropy", "C2",
                     os.path.join(output_dir, "06_entropy.png"))

    # ────────────────────────────────────────────────────────────────
    # 7. KL Divergence
    # ────────────────────────────────────────────────────────────────
    kl = col(prefix + "kl")
    if kl is not None:
        _single_plot(x, kl, x_label, "KL Divergence",
                     "KL Divergence", "C6",
                     os.path.join(output_dir, "07_kl_divergence.png"))

    # ────────────────────────────────────────────────────────────────
    # 8. KL Coefficient (adaptive)
    # ────────────────────────────────────────────────────────────────
    kl_coeff = col(prefix + "cur_kl_coeff")
    if kl_coeff is not None:
        _single_plot(x, kl_coeff, x_label, "KL Coefficient",
                     "Adaptive KL Coefficient", "C7",
                     os.path.join(output_dir, "08_kl_coeff.png"))

    # ────────────────────────────────────────────────────────────────
    # 9. Value Function Explained Variance
    # ────────────────────────────────────────────────────────────────
    vf_expl = col(prefix + "vf_explained_var")
    if vf_expl is not None:
        _single_plot(x, vf_expl, x_label, "Explained Variance",
                     "Value Function Explained Variance", "C8",
                     os.path.join(output_dir, "09_vf_explained_var.png"),
                     hlines=[0.0, 1.0])

    # ────────────────────────────────────────────────────────────────
    # 10. Learning Rate
    # ────────────────────────────────────────────────────────────────
    lr = col(prefix + "cur_lr")
    if lr is not None:
        _single_plot(x, lr, x_label, "Learning Rate",
                     "Current Learning Rate", "C9",
                     os.path.join(output_dir, "10_learning_rate.png"))

    # ────────────────────────────────────────────────────────────────
    # 11. Eval Reward (if available)
    # ────────────────────────────────────────────────────────────────
    eval_rw = col("evaluation/env_runners/episode_reward_mean",
                  "evaluation/env_runners/episode_return_mean")
    if eval_rw is not None:
        valid = ~np.isnan(eval_rw.astype(float))
        if valid.any():
            _single_plot(x[valid], eval_rw[valid], x_label,
                         "Eval Episode Reward",
                         "Evaluation Reward (Mean)", "tab:green",
                         os.path.join(output_dir, "11_eval_reward.png"))

    # ── Print summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training Summary  ({len(df)} iterations)")
    print(f"{'='*60}")
    print(f"  Total steps:      {x[-1]:,.0f}")
    if rw_mean is not None:
        print(f"  Final reward:     {rw_mean[-1]:.2f}  (max: {np.nanmax(rw_mean):.2f})")
    if total_loss is not None:
        print(f"  Final total_loss: {total_loss[-1]:.4f}")
    if entropy is not None:
        print(f"  Final entropy:    {entropy[-1]:.4f}  (start: {entropy[0]:.4f})")
    if kl is not None:
        print(f"  Final KL:         {kl[-1]:.6f}")
    if vf_expl is not None:
        print(f"  Final VF expl:    {vf_expl[-1]:.4f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Plot RLlib training curves")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to progress.csv")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for plots")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.csv), "training_plots")

    plot_training(args.csv, args.output)


if __name__ == "__main__":
    main()
