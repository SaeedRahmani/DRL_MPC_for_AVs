"""
Compare MPCRL vs Pure PPO training curves side-by-side on the same plots.

Usage:
    python plot_training_comparison.py \
      --mpcrl-csv <path> --ppo-csv <path> --output <dir>
"""

import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def smooth(y, window=10):
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    y_clean = np.where(mask, y, 0)
    if len(y_clean) < window:
        return y_clean
    cumsum = np.cumsum(np.insert(y_clean, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def get_col(df, *names):
    for n in names:
        if n in df.columns:
            return df[n].values
    return None


def plot_comparison(mpcrl_csv: str, ppo_csv: str, output_dir: str):
    mpcrl = pd.read_csv(mpcrl_csv)
    ppo = pd.read_csv(ppo_csv)
    os.makedirs(output_dir, exist_ok=True)

    # X-axes: timesteps
    mx = get_col(mpcrl, "timesteps_total", "num_env_steps_sampled")
    px = get_col(ppo, "timesteps_total", "num_env_steps_sampled")

    MPCRL_COLOR = "C0"
    PPO_COLOR = "C3"
    MPCRL_LABEL = "MPC-RL (CA)"
    PPO_LABEL = "Pure PPO"

    # ── Figure 1: Reward & Episode Length ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # Reward
    ax = axes[0]
    for x, df, color, label in [
        (mx, mpcrl, MPCRL_COLOR, MPCRL_LABEL),
        (px, ppo, PPO_COLOR, PPO_LABEL),
    ]:
        rm = get_col(df, "env_runners/episode_reward_mean", "env_runners/episode_return_mean")
        rmax = get_col(df, "env_runners/episode_reward_max", "env_runners/episode_return_max")
        rmin = get_col(df, "env_runners/episode_reward_min", "env_runners/episode_return_min")
        if rm is not None:
            ax.plot(x, rm, alpha=0.15, color=color, linewidth=0.6)
            s = smooth(rm)
            ax.plot(x[:len(s)], s, color=color, linewidth=2, label=f"{label} (smoothed)")
            if rmax is not None and rmin is not None:
                ax.fill_between(x, rmin, rmax, alpha=0.07, color=color)
    ax.set_ylabel("Episode Reward")
    ax.set_title("Episode Reward — MPC-RL vs Pure PPO")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Episode length
    ax = axes[1]
    for x, df, color, label in [
        (mx, mpcrl, MPCRL_COLOR, MPCRL_LABEL),
        (px, ppo, PPO_COLOR, PPO_LABEL),
    ]:
        el = get_col(df, "env_runners/episode_len_mean")
        if el is not None:
            ax.plot(x, el, alpha=0.15, color=color, linewidth=0.6)
            s = smooth(el)
            ax.plot(x[:len(s)], s, color=color, linewidth=2, label=f"{label} (smoothed)")
    ax.set_ylabel("Episode Length (steps)")
    ax.set_xlabel("Environment Steps")
    ax.set_title("Episode Length — MPC-RL vs Pure PPO")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_reward_length.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Figure 2: Eval Reward ──
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    for x, df, color, label in [
        (mx, mpcrl, MPCRL_COLOR, MPCRL_LABEL),
        (px, ppo, PPO_COLOR, PPO_LABEL),
    ]:
        ev = get_col(df, "evaluation/env_runners/episode_reward_mean")
        if ev is not None:
            ax.plot(x, ev, alpha=0.15, color=color, linewidth=0.6)
            s = smooth(ev)
            ax.plot(x[:len(s)], s, color=color, linewidth=2, label=f"{label} (smoothed)")
    ax.set_ylabel("Eval Episode Reward")
    ax.set_xlabel("Environment Steps")
    ax.set_title("Evaluation Reward — MPC-RL vs Pure PPO")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_eval_reward.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Figure 3: PPO Losses ──
    loss_keys = [
        ("info/learner/default_policy/learner_stats/total_loss", "Total Loss"),
        ("info/learner/default_policy/learner_stats/policy_loss", "Policy Loss"),
        ("info/learner/default_policy/learner_stats/vf_loss", "VF Loss"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    for i, (key, title) in enumerate(loss_keys):
        ax = axes[i]
        for x, df, color, label in [
            (mx, mpcrl, MPCRL_COLOR, MPCRL_LABEL),
            (px, ppo, PPO_COLOR, PPO_LABEL),
        ]:
            vals = get_col(df, key)
            if vals is not None:
                ax.plot(x, vals, alpha=0.15, color=color, linewidth=0.6)
                s = smooth(vals)
                ax.plot(x[:len(s)], s, color=color, linewidth=2, label=f"{label}")
        ax.set_ylabel(title)
        ax.set_title(f"{title} — MPC-RL vs Pure PPO")
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Environment Steps")
    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_losses.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Figure 4: KL, Entropy, VF Explained Variance ──
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Entropy
    ax = axes[0]
    for x, df, color, label in [
        (mx, mpcrl, MPCRL_COLOR, MPCRL_LABEL),
        (px, ppo, PPO_COLOR, PPO_LABEL),
    ]:
        vals = get_col(df, "info/learner/default_policy/learner_stats/entropy")
        if vals is not None:
            ax.plot(x, vals, alpha=0.15, color=color, linewidth=0.6)
            s = smooth(vals)
            ax.plot(x[:len(s)], s, color=color, linewidth=2, label=f"{label}")
    ax.set_ylabel("Entropy")
    ax.set_title("Policy Entropy — MPC-RL vs Pure PPO")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # KL
    ax = axes[1]
    for x, df, color, label in [
        (mx, mpcrl, MPCRL_COLOR, MPCRL_LABEL),
        (px, ppo, PPO_COLOR, PPO_LABEL),
    ]:
        vals = get_col(df, "info/learner/default_policy/learner_stats/kl")
        if vals is not None:
            ax.plot(x, vals, alpha=0.15, color=color, linewidth=0.6)
            s = smooth(vals)
            ax.plot(x[:len(s)], s, color=color, linewidth=2, label=f"{label}")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Divergence — MPC-RL vs Pure PPO")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # VF explained var
    ax = axes[2]
    for x, df, color, label in [
        (mx, mpcrl, MPCRL_COLOR, MPCRL_LABEL),
        (px, ppo, PPO_COLOR, PPO_LABEL),
    ]:
        vals = get_col(df, "info/learner/default_policy/learner_stats/vf_explained_var")
        if vals is not None:
            ax.plot(x, vals, alpha=0.15, color=color, linewidth=0.6)
            s = smooth(vals)
            ax.plot(x[:len(s)], s, color=color, linewidth=2, label=f"{label}")
    ax.set_ylabel("VF Explained Variance")
    ax.set_xlabel("Environment Steps")
    ax.set_title("VF Explained Variance — MPC-RL vs Pure PPO")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_kl_entropy_vf.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  Comparison Summary")
    print(f"{'='*60}")
    for name, df, x in [("MPC-RL", mpcrl, mx), ("Pure PPO", ppo, px)]:
        rm = get_col(df, "env_runners/episode_reward_mean", "env_runners/episode_return_mean")
        ev = get_col(df, "evaluation/env_runners/episode_reward_mean")
        print(f"\n  {name}:")
        print(f"    Iterations:  {len(df)}")
        print(f"    Total steps: {x[-1]:,.0f}")
        if rm is not None:
            print(f"    Train reward (final): {rm[-1]:.2f}  (max: {np.nanmax(rm):.2f})")
        if ev is not None:
            print(f"    Eval reward  (final): {ev[-1]:.2f}  (max: {np.nanmax(ev):.2f})")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Compare MPCRL vs PPO training")
    parser.add_argument("--mpcrl-csv", type=str, required=True)
    parser.add_argument("--ppo-csv", type=str, required=True)
    parser.add_argument("--output", type=str, default="evals/comparison_plots")
    args = parser.parse_args()
    plot_comparison(args.mpcrl_csv, args.ppo_csv, args.output)


if __name__ == "__main__":
    main()
