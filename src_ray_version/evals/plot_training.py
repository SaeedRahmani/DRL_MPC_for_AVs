"""
Plot key training curves from Ray/RLlib progress.csv.

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


def smooth(y, window=10):
    """Simple moving average."""
    if len(y) < window:
        return y
    cumsum = np.cumsum(np.insert(y, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def plot_training(csv_path: str, output_dir: str):
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # X-axis: timesteps
    x_col = "timesteps_total"
    if x_col not in df.columns:
        x_col = "num_env_steps_sampled"
    x = df[x_col].values
    x_label = "Environment Steps"

    # Also make an iteration x-axis
    iters = df["training_iteration"].values if "training_iteration" in df.columns else np.arange(len(df))

    # ── Figure 1: Episode Reward ──
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    reward_mean = df.get("env_runners/episode_reward_mean", df.get("env_runners/episode_return_mean"))
    reward_max = df.get("env_runners/episode_reward_max", df.get("env_runners/episode_return_max"))
    reward_min = df.get("env_runners/episode_reward_min", df.get("env_runners/episode_return_min"))

    if reward_mean is not None:
        ax = axes[0]
        ax.plot(x, reward_mean, alpha=0.3, color="C0", linewidth=0.8)
        ax.plot(x[:len(smooth(reward_mean))], smooth(reward_mean), color="C0", linewidth=2, label="Mean (smoothed)")
        if reward_max is not None and reward_min is not None:
            ax.fill_between(x, reward_min, reward_max, alpha=0.1, color="C0", label="Min–Max range")
        ax.set_ylabel("Episode Reward")
        ax.set_title("Episode Reward over Training")
        ax.legend()
        ax.grid(True, alpha=0.3)

    ep_len = df.get("env_runners/episode_len_mean")
    if ep_len is not None:
        ax = axes[1]
        ax.plot(x, ep_len, alpha=0.3, color="C1", linewidth=0.8)
        ax.plot(x[:len(smooth(ep_len))], smooth(ep_len), color="C1", linewidth=2, label="Mean (smoothed)")
        ax.set_ylabel("Episode Length (steps)")
        ax.set_xlabel(x_label)
        ax.set_title("Episode Length over Training")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path1 = os.path.join(output_dir, "reward_and_length.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"Saved: {path1}")

    # ── Figure 2: PPO Losses ──
    total_loss = df.get("info/learner/default_policy/learner_stats/total_loss")
    policy_loss = df.get("info/learner/default_policy/learner_stats/policy_loss")
    vf_loss = df.get("info/learner/default_policy/learner_stats/vf_loss")

    if total_loss is not None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        ax = axes[0]
        ax.plot(x, total_loss, alpha=0.3, color="C3", linewidth=0.8)
        ax.plot(x[:len(smooth(total_loss))], smooth(total_loss), color="C3", linewidth=2)
        ax.set_ylabel("Total Loss")
        ax.set_title("PPO Total Loss")
        ax.grid(True, alpha=0.3)

        if policy_loss is not None:
            ax = axes[1]
            ax.plot(x, policy_loss, alpha=0.3, color="C4", linewidth=0.8)
            ax.plot(x[:len(smooth(policy_loss))], smooth(policy_loss), color="C4", linewidth=2)
            ax.set_ylabel("Policy Loss")
            ax.set_title("PPO Policy (Surrogate) Loss")
            ax.grid(True, alpha=0.3)

        if vf_loss is not None:
            ax = axes[2]
            ax.plot(x, vf_loss, alpha=0.3, color="C5", linewidth=0.8)
            ax.plot(x[:len(smooth(vf_loss))], smooth(vf_loss), color="C5", linewidth=2)
            ax.set_ylabel("Value Function Loss")
            ax.set_xlabel(x_label)
            ax.set_title("PPO Value Function Loss")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path2 = os.path.join(output_dir, "ppo_losses.png")
        fig.savefig(path2, dpi=150)
        plt.close(fig)
        print(f"Saved: {path2}")

    # ── Figure 3: KL, Entropy, LR ──
    kl = df.get("info/learner/default_policy/learner_stats/kl")
    entropy = df.get("info/learner/default_policy/learner_stats/entropy")
    vf_expl = df.get("info/learner/default_policy/learner_stats/vf_explained_var")
    lr = df.get("info/learner/default_policy/learner_stats/cur_lr")
    kl_coeff = df.get("info/learner/default_policy/learner_stats/cur_kl_coeff")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    if entropy is not None:
        ax = axes[0]
        ax.plot(x, entropy, alpha=0.3, color="C2", linewidth=0.8)
        ax.plot(x[:len(smooth(entropy))], smooth(entropy), color="C2", linewidth=2)
        ax.set_ylabel("Entropy")
        ax.set_title("Policy Entropy (exploration)")
        ax.grid(True, alpha=0.3)

    if kl is not None:
        ax = axes[1]
        ax.plot(x, kl, alpha=0.3, color="C6", linewidth=0.8)
        ax.plot(x[:len(smooth(kl))], smooth(kl), color="C6", linewidth=2, label="KL divergence")
        if kl_coeff is not None:
            ax2 = ax.twinx()
            ax2.plot(x, kl_coeff, color="C7", linewidth=1.5, linestyle="--", label="KL coeff")
            ax2.set_ylabel("KL Coeff", color="C7")
            ax2.legend(loc="upper left")
        ax.set_ylabel("KL Divergence")
        ax.set_title("KL Divergence & Adaptive Coefficient")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    if vf_expl is not None:
        ax = axes[2]
        ax.plot(x, vf_expl, alpha=0.3, color="C8", linewidth=0.8)
        ax.plot(x[:len(smooth(vf_expl))], smooth(vf_expl), color="C8", linewidth=2)
        ax.set_ylabel("VF Explained Variance")
        ax.set_xlabel(x_label)
        ax.set_title("Value Function Explained Variance")
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
        ax.set_ylim(-0.5, 1.1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path3 = os.path.join(output_dir, "kl_entropy_vf.png")
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    print(f"Saved: {path3}")

    # ── Figure 4: Throughput ──
    throughput = df.get("num_env_steps_sampled_throughput_per_sec")
    time_per_iter = df.get("time_this_iter_s")

    if throughput is not None or time_per_iter is not None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        if throughput is not None:
            ax = axes[0]
            ax.plot(x, throughput, alpha=0.3, color="C9", linewidth=0.8)
            ax.plot(x[:len(smooth(throughput))], smooth(throughput), color="C9", linewidth=2)
            ax.set_ylabel("Steps/sec")
            ax.set_title("Sampling Throughput")
            ax.grid(True, alpha=0.3)

        if time_per_iter is not None:
            ax = axes[1]
            ax.plot(x, time_per_iter, alpha=0.3, color="C0", linewidth=0.8)
            ax.plot(x[:len(smooth(time_per_iter))], smooth(time_per_iter), color="C0", linewidth=2)
            ax.set_ylabel("Seconds")
            ax.set_xlabel(x_label)
            ax.set_title("Time per Training Iteration")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path4 = os.path.join(output_dir, "throughput.png")
        fig.savefig(path4, dpi=150)
        plt.close(fig)
        print(f"Saved: {path4}")

    # ── Print summary stats ──
    print(f"\n{'='*60}")
    print(f"  Training Summary  ({len(df)} iterations)")
    print(f"{'='*60}")
    print(f"  Total steps:      {x[-1]:,.0f}")
    if reward_mean is not None:
        print(f"  Final reward:     {reward_mean.iloc[-1]:.2f}  (max: {reward_mean.max():.2f} at iter {reward_mean.idxmax()})")
    if total_loss is not None:
        print(f"  Final total_loss: {total_loss.iloc[-1]:.4f}")
    if entropy is not None:
        print(f"  Final entropy:    {entropy.iloc[-1]:.4f}  (start: {entropy.iloc[0]:.4f})")
    if kl is not None:
        print(f"  Final KL:         {kl.iloc[-1]:.6f}")
    if vf_expl is not None:
        print(f"  Final VF expl:    {vf_expl.iloc[-1]:.4f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Plot RLlib training curves")
    parser.add_argument("--csv", type=str, required=True, help="Path to progress.csv")
    parser.add_argument("--output", type=str, default=None, help="Output directory for plots")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.csv), "training_plots")

    plot_training(args.csv, args.output)


if __name__ == "__main__":
    main()
