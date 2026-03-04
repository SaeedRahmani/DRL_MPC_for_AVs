"""
Experiment 6: Ablation Studies
================================
Systematically disable / modify MPC-RL components to measure their
individual contribution to performance.

Ablation variants
-----------------
1. Full MPC-RL           – learned RL policy + manual CA safety filter  (baseline)
2. MPC-RL w/o CA         – learned RL policy, NO manual CA  (noCA env variant)
3. Constant ref-speed    – manual CA but ref speed fixed at 7.5 m/s  (no RL)
4. Random ref-speed      – manual CA but ref speed is random each step
5. Horizon N=4           – full MPC-RL but MPC horizon shortened to 4
6. Horizon N=8           – full MPC-RL but MPC horizon shortened to 8
7. Horizon N=24          – full MPC-RL but MPC horizon extended to 24

Metrics per variant (over N_EPISODES):
  – Arrival rate, crash rate, avg episode reward, avg steps (arrived)

Outputs (in experiment6_results/):
  – ablation_results.json              Raw data
  – ablation_performance_bars.pdf      Bar chart of arrival / crash rates
  – ablation_reward_bars.pdf           Bar chart of mean episode reward
  – ablation_horizon_sweep.pdf         Line plot: horizon vs metrics

Usage:
    cd /users/saeani/src/mpcrl/MPC-RL_for_AVs/src_ray_version && \\
    PYTHONPATH="$PWD:$PWD/highway-env:$PYTHONPATH" CUDA_VISIBLE_DEVICES="" \\
    python /users/saeani/src/mpcrl/DRL_MPC_for_AVs/src_ray_version/evals/experiment6_ablation.py
"""

import os, sys, json, argparse, time
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
MPCRL_CKPT = "/users/saeani/DEV/ray_results/PPO_v0_manual/PPO_intersection-mpcrl-refspeed-manual_c129f_00000_0_2026-03-01_00-44-07/checkpoint_000020"

sys.path.insert(0, os.path.join(EVALS_DIR, "../../../MPC-RL_for_AVs/src_ray_version"))

COLORS = {
    "Full MPC-RL":       "#4CAF50",
    "MPC-RL w/o CA":     "#F44336",
    "Constant RefSpeed":  "#FF9800",
    "Random RefSpeed":    "#9C27B0",
    "Horizon N=4":       "#00BCD4",
    "Horizon N=8":       "#3F51B5",
    "Horizon N=24":      "#795548",
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
#  Run episodes for a variant
# =====================================================================
def run_variant(variant_name, n_episodes, mpcrl_ckpt, horizon_override=None):
    """
    Run *n_episodes* for a specified ablation variant.

    Returns list of episode dicts: {status, reward, n_steps}.
    """
    from ray.rllib.policy.policy import Policy

    # ── Choose env and action source ──
    use_policy = True
    env_id = "intersection-mpcrl-refspeed-manual"   # default
    constant_action = None
    random_action = False

    if variant_name == "Full MPC-RL":
        pass  # defaults
    elif variant_name == "MPC-RL w/o CA":
        env_id = "intersection-mpcrl-refspeed-noCA"
    elif variant_name == "Constant RefSpeed":
        constant_action = np.array([0.0], dtype=np.float32)  # maps to 7.5 m/s
        use_policy = False
    elif variant_name == "Random RefSpeed":
        random_action = True
        use_policy = False
    elif variant_name.startswith("Horizon"):
        pass  # will override horizon below

    # ── Create env ──
    env = gymnasium.make(env_id, render_mode="rgb_array")

    # ── Override MPC horizon if requested ──
    if horizon_override is not None:
        inner = env.unwrapped if hasattr(env, "unwrapped") else env
        inner.horizon = horizon_override

    # ── Load policy ──
    policy = None
    if use_policy:
        policy = Policy.from_checkpoint(mpcrl_ckpt)
        if isinstance(policy, dict):
            policy = policy["default_policy"]

    episodes = []

    for ep in range(n_episodes):
        obs, _ = env.reset()

        # Re-apply horizon override after reset (in case reset re-inits it)
        if horizon_override is not None:
            inner = env.unwrapped if hasattr(env, "unwrapped") else env
            inner.horizon = horizon_override

        done = False
        ep_reward = 0.0
        n_steps = 0

        while not done:
            if use_policy and policy is not None:
                action = policy.compute_single_action(obs, explore=False)[0]
            elif constant_action is not None:
                action = constant_action.copy()
            elif random_action:
                action = env.action_space.sample()
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            n_steps += 1
            done = terminated or truncated

        ego = _get_ego(env)
        arrived = _has_arrived(env, ego) if ego else False
        crashed = bool(ego.crashed) if ego else True
        status = "ARRIVE" if arrived else ("CRASH" if crashed else "OTHER")

        episodes.append({
            "status": status,
            "reward": float(ep_reward),
            "n_steps": n_steps,
        })
        print(f"  [{variant_name}] ep {ep+1}/{n_episodes}  "
              f"steps={n_steps}  reward={ep_reward:.1f}  {status}")

    env.close()
    return episodes


# =====================================================================
#  Plotting
# =====================================================================
def plot_performance_bars(results, output_dir):
    """Bar chart: arrival rate & crash rate for each variant."""
    variants = list(results.keys())
    arrival_rates = []
    crash_rates = []
    for v in variants:
        eps = results[v]["episodes"]
        n = len(eps)
        arrival_rates.append(sum(1 for e in eps if e["status"] == "ARRIVE") / n * 100)
        crash_rates.append(sum(1 for e in eps if e["status"] == "CRASH") / n * 100)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(variants))
    width = 0.35

    bars1 = ax.bar(x - width / 2, arrival_rates, width, label="Arrival Rate (%)",
                   color="#4CAF50", alpha=0.85)
    bars2 = ax.bar(x + width / 2, crash_rates, width, label="Crash Rate (%)",
                   color="#F44336", alpha=0.85)

    for bar, val in zip(bars1, arrival_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, val in zip(bars2, crash_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_title("Ablation Study: Arrival & Crash Rates", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=9, rotation=15, ha="right")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"ablation_performance_bars.{ext}"),
                    dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    plt.close(fig)
    print("  [plot] ablation_performance_bars.pdf/png")


def plot_reward_bars(results, output_dir):
    """Bar chart: mean episode reward for each variant."""
    variants = list(results.keys())
    means = []
    stds = []
    for v in variants:
        rews = [e["reward"] for e in results[v]["episodes"]]
        means.append(np.mean(rews))
        stds.append(np.std(rews))

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(variants))
    colors = [COLORS.get(v, "#607D8B") for v in variants]

    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
    for bar, m in zip(bars, means):
        va = "bottom" if m >= 0 else "top"
        offset = 1 if m >= 0 else -1
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f"{m:.1f}", ha="center", va=va, fontsize=9, fontweight="bold")

    ax.set_ylabel("Mean Episode Reward", fontsize=12)
    ax.set_title("Ablation Study: Mean Episode Reward", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=9, rotation=15, ha="right")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"ablation_reward_bars.{ext}"),
                    dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    plt.close(fig)
    print("  [plot] ablation_reward_bars.pdf/png")


def plot_horizon_sweep(results, output_dir):
    """Line plot: MPC horizon vs arrival rate, crash rate, mean reward."""
    horizon_variants = {
        4:  "Horizon N=4",
        8:  "Horizon N=8",
        16: "Full MPC-RL",    # default horizon
        24: "Horizon N=24",
    }

    horizons = []
    arrival_rates = []
    crash_rates = []
    mean_rews = []
    mean_steps_arrived = []

    for h, v in sorted(horizon_variants.items()):
        if v not in results:
            continue
        eps = results[v]["episodes"]
        n = len(eps)
        horizons.append(h)
        arrival_rates.append(sum(1 for e in eps if e["status"] == "ARRIVE") / n * 100)
        crash_rates.append(sum(1 for e in eps if e["status"] == "CRASH") / n * 100)
        mean_rews.append(np.mean([e["reward"] for e in eps]))
        arrived_steps = [e["n_steps"] for e in eps if e["status"] == "ARRIVE"]
        mean_steps_arrived.append(np.mean(arrived_steps) if arrived_steps else 0)

    if len(horizons) < 2:
        print("  [skip] horizon sweep plot — not enough data points")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Arrival & crash rate
    ax = axes[0]
    ax.plot(horizons, arrival_rates, "o-", color="#4CAF50", linewidth=2, markersize=8, label="Arrival %")
    ax.plot(horizons, crash_rates, "s--", color="#F44336", linewidth=2, markersize=8, label="Crash %")
    ax.set_xlabel("MPC Horizon (N)", fontsize=11)
    ax.set_ylabel("Rate (%)", fontsize=11)
    ax.set_title("Safety vs Horizon", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(horizons)

    # Mean reward
    ax = axes[1]
    ax.plot(horizons, mean_rews, "D-", color="#3F51B5", linewidth=2, markersize=8)
    ax.set_xlabel("MPC Horizon (N)", fontsize=11)
    ax.set_ylabel("Mean Episode Reward", fontsize=11)
    ax.set_title("Reward vs Horizon", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.set_xticks(horizons)

    # Mean steps (arrived)
    ax = axes[2]
    ax.plot(horizons, mean_steps_arrived, "^-", color="#FF9800", linewidth=2, markersize=8)
    ax.set_xlabel("MPC Horizon (N)", fontsize=11)
    ax.set_ylabel("Mean Steps (Arrived Episodes)", fontsize=11)
    ax.set_title("Efficiency vs Horizon", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.set_xticks(horizons)

    fig.suptitle("MPC Horizon Sweep (Ablation)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"ablation_horizon_sweep.{ext}"),
                    dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    plt.close(fig)
    print("  [plot] ablation_horizon_sweep.pdf/png")


def plot_component_summary(results, output_dir):
    """
    Grouped bar chart summarising all ablation variants together:
    arrival rate, mean reward, mean steps.
    """
    variants = list(results.keys())
    n = len(variants)

    arrival = []
    mean_rew = []
    mean_steps = []
    for v in variants:
        eps = results[v]["episodes"]
        arrival.append(sum(1 for e in eps if e["status"] == "ARRIVE") / len(eps) * 100)
        mean_rew.append(np.mean([e["reward"] for e in eps]))
        arrived_steps = [e["n_steps"] for e in eps if e["status"] == "ARRIVE"]
        mean_steps.append(np.mean(arrived_steps) if arrived_steps else 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(n)
    colors = [COLORS.get(v, "#607D8B") for v in variants]

    # Arrival rate
    ax = axes[0]
    bars = ax.bar(x, arrival, color=colors, alpha=0.85)
    for bar, val in zip(bars, arrival):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylabel("Arrival Rate (%)", fontsize=11)
    ax.set_title("Arrival Rate", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=8, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)

    # Mean reward
    ax = axes[1]
    bars = ax.bar(x, mean_rew, color=colors, alpha=0.85)
    for bar, val in zip(bars, mean_rew):
        va = "bottom" if val >= 0 else "top"
        offset = 0.5 if val >= 0 else -0.5
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f"{val:.1f}", ha="center", va=va, fontsize=8, fontweight="bold")
    ax.set_ylabel("Mean Episode Reward", fontsize=11)
    ax.set_title("Mean Reward", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=8, rotation=20, ha="right")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    # Mean steps (arrived)
    ax = axes[2]
    bars = ax.bar(x, mean_steps, color=colors, alpha=0.85)
    for bar, val in zip(bars, mean_steps):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylabel("Mean Steps (Arrived)", fontsize=11)
    ax.set_title("Efficiency (Arrived Episodes)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=8, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Ablation Study — Component Contribution Summary",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"ablation_component_summary.{ext}"),
                    dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    plt.close(fig)
    print("  [plot] ablation_component_summary.pdf/png")


# =====================================================================
#  Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Experiment 6: Ablation Studies")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per variant (default: 50)")
    parser.add_argument("--mpcrl-ckpt", type=str, default=MPCRL_CKPT)
    parser.add_argument("--output-dir", type=str, default=EVALS_DIR)
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, "experiment6_results")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  Experiment 6: Ablation Studies")
    print(f"  Episodes per variant: {args.episodes}")
    print(f"  Output: {output_dir}")
    print(f"{'#'*60}")

    # ── Define variants ──
    # (variant_name, horizon_override or None)
    VARIANTS = [
        ("Full MPC-RL",       None),
        ("MPC-RL w/o CA",     None),
        ("Constant RefSpeed", None),
        ("Random RefSpeed",   None),
        ("Horizon N=4",       4),
        ("Horizon N=8",       8),
        ("Horizon N=24",      24),
    ]

    results = {}
    for variant_name, horizon_override in VARIANTS:
        print(f"\n{'='*60}")
        print(f"  [{variant_name}]  horizon={horizon_override or 16}")
        print(f"{'='*60}")

        t0 = time.time()
        episodes = run_variant(
            variant_name, args.episodes, args.mpcrl_ckpt,
            horizon_override=horizon_override,
        )
        elapsed = time.time() - t0

        n_arr = sum(1 for e in episodes if e["status"] == "ARRIVE")
        n_crash = sum(1 for e in episodes if e["status"] == "CRASH")
        mean_rew = np.mean([e["reward"] for e in episodes])
        arrived_steps = [e["n_steps"] for e in episodes if e["status"] == "ARRIVE"]
        mean_steps = np.mean(arrived_steps) if arrived_steps else float("nan")

        results[variant_name] = {
            "episodes": episodes,
            "n_episodes": len(episodes),
            "n_arrived": n_arr,
            "n_crashed": n_crash,
            "arrival_rate": n_arr / len(episodes) * 100,
            "crash_rate": n_crash / len(episodes) * 100,
            "mean_reward": float(mean_rew),
            "mean_steps_arrived": float(mean_steps),
            "elapsed_s": elapsed,
        }

        print(f"\n  → {variant_name}: arrived={n_arr}/{len(episodes)} "
              f"({n_arr/len(episodes)*100:.0f}%), "
              f"crashed={n_crash}/{len(episodes)} "
              f"({n_crash/len(episodes)*100:.0f}%), "
              f"mean_reward={mean_rew:.1f}, "
              f"mean_steps_arrived={mean_steps:.0f}, "
              f"time={elapsed:.1f}s")

    # ── Print summary table ──
    print(f"\n{'='*60}")
    print("  ABLATION STUDY — SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Variant':<22s} {'Arrive%':>8s} {'Crash%':>8s} "
          f"{'Reward':>8s} {'Steps':>8s}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for v, data in results.items():
        print(f"  {v:<22s} {data['arrival_rate']:>7.0f}% {data['crash_rate']:>7.0f}% "
              f"{data['mean_reward']:>8.1f} {data['mean_steps_arrived']:>8.0f}")

    # ── Save JSON ──
    json_path = os.path.join(output_dir, "ablation_results.json")
    json_out = {}
    for v, data in results.items():
        json_out[v] = {
            "n_episodes": data["n_episodes"],
            "n_arrived": data["n_arrived"],
            "n_crashed": data["n_crashed"],
            "arrival_rate": data["arrival_rate"],
            "crash_rate": data["crash_rate"],
            "mean_reward": data["mean_reward"],
            "mean_steps_arrived": data["mean_steps_arrived"],
            "elapsed_s": data["elapsed_s"],
            "episodes": data["episodes"],
        }
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\n[json] Saved: {json_path}")

    # ── Plots ──
    print("\nGenerating plots ...")
    plot_performance_bars(results, output_dir)
    plot_reward_bars(results, output_dir)
    plot_horizon_sweep(results, output_dir)
    plot_component_summary(results, output_dir)

    print("\n[DONE] Experiment 6 complete!")


if __name__ == "__main__":
    main()
