"""
Experiment 4 v2: Trajectory Quality Analysis — All Difficulties
================================================================
Run Pure RL, Pure MPC, and MPC-RL on the intersection scenario at
very_easy, easy, and moderate difficulty.  For each (method, difficulty),
run 100 episodes and collect per-step trajectory data for *arrived* episodes.

Plots:
  Per difficulty (3 plots × 4 metrics = 12 panels):
    - Speed profile, lateral accel, longitudinal jerk, min distance
    - Individual episode traces (thin, transparent) + bold average line
  Combined comfort bar chart across difficulties

Usage:
    cd /users/saeani/src/mpcrl/MPC-RL_for_AVs/src_ray_version && \
    PYTHONPATH="$PWD:$PWD/highway-env:$PYTHONPATH" CUDA_VISIBLE_DEVICES="" \
    python .../experiment4_trajectory_quality_v2.py --episodes 100
"""

import os, sys, json, time, argparse, warnings
import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import gymnasium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import highway_env  # noqa: F401

warnings.filterwarnings("ignore", message="invalid value")

EVALS_DIR = os.path.dirname(os.path.abspath(__file__))
PPO_CKPT = "/users/saeani/DEV/ray_results/PPO_pure_RL/PPO_intersection-pure-ppo_03a31_00000_0_2026-03-01_23-40-22/checkpoint_000025"
MPCRL_CKPT = "/users/saeani/DEV/ray_results/PPO_v0_manual_v2/PPO_intersection-mpcrl-refspeed-manual_450ad_00000_0_2026-03-04_11-35-39/checkpoint_000050"

DIFFICULTY_PRESETS = {
    "very_easy": {"initial_vehicle_count": 2, "spawn_probability": 0.1},
    "easy":      {"initial_vehicle_count": 5, "spawn_probability": 0.3},
    "moderate":  {"initial_vehicle_count": 10, "spawn_probability": 0.6},
}

sys.path.insert(0, os.path.join(EVALS_DIR, "../../../MPC-RL_for_AVs/src_ray_version"))

POLICY_FREQ = 10
DT = 1.0 / POLICY_FREQ

# ── Colors ──
COLORS = {
    "Pure PPO":  "#2196F3",  # blue
    "Pure MPC":  "#FF9800",  # orange
    "MPC-RL":    "#4CAF50",  # green
}
METHOD_ORDER = ["Pure PPO", "Pure MPC", "MPC-RL"]
DIFF_ORDER = ["very_easy", "easy", "moderate"]
DIFF_LABELS = {"very_easy": "Very Easy", "easy": "Easy", "moderate": "Moderate"}


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


def _min_dist_to_others(env, ego, max_val=200.0):
    inner = env.unwrapped if hasattr(env, "unwrapped") else env
    min_d = max_val  # finite cap instead of inf
    for v in inner.road.vehicles:
        if v is ego:
            continue
        d = np.linalg.norm(ego.position - v.position)
        min_d = min(min_d, d)
    return min_d


def _has_arrived(env, ego):
    inner = env.unwrapped if hasattr(env, "unwrapped") else env
    try:
        return bool(inner.has_arrived(ego))
    except Exception:
        return False


def record_step(env, ego):
    """Capture one step's trajectory data (speed-based acc computed later)."""
    return {
        "x": float(ego.position[0]),
        "y": float(ego.position[1]),
        "speed": float(ego.speed),
        "heading": float(ego.heading),
        "min_dist": float(_min_dist_to_others(env, ego)),
    }


def derive_metrics(steps, dt):
    """Compute derived metrics from step records."""
    speed = np.array([s["speed"] for s in steps])
    heading = np.array([s["heading"] for s in steps])
    min_dist = np.array([s["min_dist"] for s in steps])
    t = np.arange(len(steps)) * dt

    # Acceleration from finite differences of speed
    acc = np.gradient(speed, dt)

    # Lateral acceleration: a_lat = v * d(heading)/dt
    dheading = np.gradient(heading, dt)
    lat_accel = speed * dheading

    # Longitudinal jerk: d(acc)/dt
    jerk = np.gradient(acc, dt)

    return {
        "time": t,
        "speed": speed,
        "acc": acc,
        "lat_accel": lat_accel,
        "jerk": jerk,
        "min_dist": min_dist,
        "mean_speed": float(np.mean(speed)),
        "mean_abs_lat_accel": float(np.mean(np.abs(lat_accel))),
        "mean_abs_jerk": float(np.mean(np.abs(jerk))),
        "mean_min_dist": float(np.mean(min_dist)),
        "min_min_dist": float(np.min(min_dist)) if len(min_dist) > 0 else 0.0,
    }


# =====================================================================
#  Data collection
# =====================================================================
def collect_pure_ppo(n_episodes, ppo_ckpt, difficulty):
    from ray.rllib.policy.policy import Policy
    from train_pure_ppo import IntersectionPurePPOEnv

    print(f"\n  [Pure PPO / {difficulty}] Loading policy ...")
    policy = Policy.from_checkpoint(ppo_ckpt)
    if isinstance(policy, dict):
        policy = policy["default_policy"]

    env = IntersectionPurePPOEnv(render_mode="rgb_array")
    if difficulty in DIFFICULTY_PRESETS:
        env.unwrapped.configure(DIFFICULTY_PRESETS[difficulty])

    episodes = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = []
        while not done:
            action = policy.compute_single_action(obs, explore=False)[0]
            obs, _, terminated, truncated, _ = env.step(action)
            ego = _get_ego(env)
            if ego is not None:
                steps.append(record_step(env, ego))
            done = terminated or truncated

        ego = _get_ego(env)
        arrived = _has_arrived(env, ego) if ego else False
        crashed = bool(ego.crashed) if ego else True
        status = "ARRIVE" if arrived else ("CRASH" if crashed else "OTHER")
        print(f"    ep {ep+1:3d}/{n_episodes}  steps={len(steps):3d}  {status}")

        if arrived and len(steps) > 5:
            episodes.append(steps)

    env.close()
    print(f"  [Pure PPO / {difficulty}] {len(episodes)} arrived out of {n_episodes}")
    return episodes


def collect_pure_mpc(n_episodes, difficulty):
    print(f"\n  [Pure MPC / {difficulty}] Running ...")
    env = gymnasium.make("intersection-mpc-manual", render_mode="rgb_array")
    if difficulty in DIFFICULTY_PRESETS:
        env.unwrapped.configure(DIFFICULTY_PRESETS[difficulty])

    episodes = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = []
        while not done:
            dummy = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(dummy)
            ego = _get_ego(env)
            if ego is not None:
                steps.append(record_step(env, ego))
            done = terminated or truncated

        ego = _get_ego(env)
        arrived = _has_arrived(env, ego) if ego else False
        crashed = bool(ego.crashed) if ego else True
        status = "ARRIVE" if arrived else ("CRASH" if crashed else "OTHER")
        print(f"    ep {ep+1:3d}/{n_episodes}  steps={len(steps):3d}  {status}")

        if arrived and len(steps) > 5:
            episodes.append(steps)

    env.close()
    print(f"  [Pure MPC / {difficulty}] {len(episodes)} arrived out of {n_episodes}")
    return episodes


def collect_mpcrl(n_episodes, mpcrl_ckpt, difficulty):
    from ray.rllib.policy.policy import Policy

    print(f"\n  [MPC-RL / {difficulty}] Loading policy ...")
    policy = Policy.from_checkpoint(mpcrl_ckpt)
    if isinstance(policy, dict):
        policy = policy["default_policy"]

    env = gymnasium.make("intersection-mpcrl-refspeed-manual", render_mode="rgb_array")
    if difficulty in DIFFICULTY_PRESETS:
        env.unwrapped.configure(DIFFICULTY_PRESETS[difficulty])

    episodes = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = []
        while not done:
            action = policy.compute_single_action(obs, explore=False)[0]
            obs, _, terminated, truncated, _ = env.step(action)
            ego = _get_ego(env)
            if ego is not None:
                steps.append(record_step(env, ego))
            done = terminated or truncated

        ego = _get_ego(env)
        arrived = _has_arrived(env, ego) if ego else False
        crashed = bool(ego.crashed) if ego else True
        status = "ARRIVE" if arrived else ("CRASH" if crashed else "OTHER")
        print(f"    ep {ep+1:3d}/{n_episodes}  steps={len(steps):3d}  {status}")

        if arrived and len(steps) > 5:
            episodes.append(steps)

    env.close()
    print(f"  [MPC-RL / {difficulty}] {len(episodes)} arrived out of {n_episodes}")
    return episodes


# =====================================================================
#  Plotting
# =====================================================================
def _compute_mean_trace(all_series, min_coverage=0.10):
    """Pad variable-length arrays with NaN, compute nanmean.
    Truncate the mean where fewer than min_coverage fraction of episodes
    contribute, to avoid noisy tail artifacts."""
    if not all_series:
        return np.array([]), np.array([])
    n_eps = len(all_series)
    max_len = max(len(s) for s in all_series)
    padded = np.full((n_eps, max_len), np.nan)
    for i, s in enumerate(all_series):
        padded[i, :len(s)] = s
    # Replace inf with NaN so nanmean ignores them
    padded[~np.isfinite(padded)] = np.nan
    mean_vals = np.nanmean(padded, axis=0)
    # Count how many episodes contribute at each timestep
    coverage = np.sum(np.isfinite(padded), axis=0)
    # Mask timesteps where too few episodes contribute
    min_count = max(3, int(n_eps * min_coverage))
    mean_vals[coverage < min_count] = np.nan
    t = np.arange(max_len) * DT
    return t, mean_vals


def plot_per_difficulty(all_data, difficulty, output_dir):
    """
    4-panel figure for one difficulty level.
    Each panel: individual episode traces (thin, transparent) + bold average.
    all_data: {method_name: [list of derive_metrics dicts]}
    """
    metrics_info = [
        ("speed",     "Speed (m/s)",                   "Speed Profiles",                  None),
        ("lat_accel", "Lateral Acceleration (m/s²)",   "Lateral Acceleration",            (-10, 10)),
        ("jerk",      "Longitudinal Jerk (m/s³)",      "Longitudinal Jerk",               (-80, 80)),
        ("min_dist",  "Min Distance to Other Veh. (m)","Distance to Nearest Vehicle",     None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax_idx, (key, ylabel, title, clip) in enumerate(metrics_info):
        ax = axes[ax_idx]

        for method in METHOD_ORDER:
            if method not in all_data or not all_data[method]:
                continue
            color = COLORS[method]
            series_list = []

            # Plot individual traces
            for ep_metrics in all_data[method]:
                vals = ep_metrics[key]
                t = ep_metrics["time"]
                if clip:
                    vals = np.clip(vals, clip[0], clip[1])
                ax.plot(t, vals, color=color, alpha=0.08, linewidth=0.6)
                series_list.append(vals)

            # Compute and plot mean trace (bold, clear)
            t_mean, mean_vals = _compute_mean_trace(series_list)
            if clip and len(mean_vals) > 0:
                mean_vals = np.clip(mean_vals, clip[0], clip[1])
            if len(mean_vals) > 0:
                ax.plot(t_mean, mean_vals, color=color, linewidth=2.8,
                        label=f"{method} (n={len(series_list)})", zorder=10)

        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="best")
        ax.grid(alpha=0.3)

    diff_label = DIFF_LABELS.get(difficulty, difficulty)
    fig.suptitle(f"Experiment 4: Trajectory Quality — {diff_label}",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"exp4_traj_{difficulty}.{ext}")
        fig.savefig(path, dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] exp4_traj_{difficulty}.pdf/png")


def plot_combined_all_difficulties(results, output_dir):
    """
    Combined figure: 4 metrics × 3 difficulties = 12 panels (3 cols × 4 rows).
    Each panel has 3 methods overlaid.
    """
    metrics_info = [
        ("speed",     "Speed (m/s)",                   "Speed Profiles",                  None),
        ("lat_accel", "Lateral Accel. (m/s²)",         "Lateral Acceleration",            (-10, 10)),
        ("jerk",      "Jerk (m/s³)",                   "Longitudinal Jerk",               (-80, 80)),
        ("min_dist",  "Min Distance (m)",              "Min Distance to Nearest Veh.",    None),
    ]

    n_rows = len(metrics_info)
    n_cols = len(DIFF_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))

    for col_idx, diff in enumerate(DIFF_ORDER):
        diff_label = DIFF_LABELS[diff]
        if diff not in results:
            continue
        all_data = results[diff]

        for row_idx, (key, ylabel, title, clip) in enumerate(metrics_info):
            ax = axes[row_idx, col_idx]

            for method in METHOD_ORDER:
                if method not in all_data or not all_data[method]:
                    continue
                color = COLORS[method]
                series_list = []

                for ep_metrics in all_data[method]:
                    vals = ep_metrics[key]
                    t = ep_metrics["time"]
                    if clip:
                        vals = np.clip(vals, clip[0], clip[1])
                    ax.plot(t, vals, color=color, alpha=0.06, linewidth=0.4)
                    series_list.append(vals)

                t_mean, mean_vals = _compute_mean_trace(series_list)
                if clip and len(mean_vals) > 0:
                    mean_vals = np.clip(mean_vals, clip[0], clip[1])
                if len(mean_vals) > 0:
                    ax.plot(t_mean, mean_vals, color=color, linewidth=2.5,
                            label=f"{method} (n={len(series_list)})", zorder=10)

            ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=8)

            if row_idx == 0:
                ax.set_title(f"{diff_label}\n{title}", fontsize=10, fontweight="bold")
            else:
                ax.set_title(title, fontsize=10)

            if col_idx == n_cols - 1 and row_idx == 0:
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Experiment 4: Trajectory Quality — All Difficulties",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"exp4_traj_combined.{ext}")
        fig.savefig(path, dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] exp4_traj_combined.pdf/png")


def plot_comfort_bars(results, output_dir):
    """
    Grouped bar chart: 3 comfort metrics × 3 difficulties, grouped by method.
    """
    comfort_keys = ["mean_abs_lat_accel", "mean_abs_jerk", "mean_min_dist"]
    comfort_labels = ["Mean |Lat. Accel| (m/s²)", "Mean |Jerk| (m/s³)", "Mean Min Dist (m)"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    x = np.arange(len(DIFF_ORDER))
    bar_w = 0.25

    for panel_idx, (ck, cl) in enumerate(zip(comfort_keys, comfort_labels)):
        ax = axes[panel_idx]

        for m_idx, method in enumerate(METHOD_ORDER):
            vals = []
            stds = []
            for diff in DIFF_ORDER:
                if diff in results and method in results[diff] and results[diff][method]:
                    ep_vals = [ep[ck] for ep in results[diff][method]]
                    vals.append(np.mean(ep_vals))
                    stds.append(np.std(ep_vals))
                else:
                    vals.append(0)
                    stds.append(0)

            offset = (m_idx - 1) * bar_w
            bars = ax.bar(x + offset, vals, bar_w, yerr=stds, capsize=4,
                          label=method, color=COLORS[method], alpha=0.85)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height(), f"{v:.2f}",
                            ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([DIFF_LABELS[d] for d in DIFF_ORDER], fontsize=10)
        ax.set_ylabel(cl, fontsize=10)
        ax.set_title(cl.split("(")[0].strip(), fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Experiment 4: Comfort & Safety Metrics by Difficulty",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"exp4_comfort_summary.{ext}")
        fig.savefig(path, dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] exp4_comfort_summary.pdf/png")


def plot_arrival_count_bars(arrival_counts, n_episodes, output_dir):
    """Bar chart showing how many episodes arrived per method per difficulty."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(DIFF_ORDER))
    bar_w = 0.25

    for m_idx, method in enumerate(METHOD_ORDER):
        vals = [arrival_counts.get(diff, {}).get(method, 0) for diff in DIFF_ORDER]
        offset = (m_idx - 1) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=method,
                      color=COLORS[method], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v}/{n_episodes}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([DIFF_LABELS[d] for d in DIFF_ORDER], fontsize=11)
    ax.set_ylabel("# Arrived Episodes", fontsize=11)
    ax.set_title(f"Arrived Episodes (out of {n_episodes} runs each)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"exp4_arrival_counts.{ext}")
        fig.savefig(path, dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] exp4_arrival_counts.pdf/png")


# =====================================================================
#  Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Experiment 4 v2: All Difficulties")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Episodes per method per difficulty (default: 100)")
    parser.add_argument("--ppo-ckpt", type=str, default=PPO_CKPT)
    parser.add_argument("--mpcrl-ckpt", type=str, default=MPCRL_CKPT)
    parser.add_argument("--output-dir", type=str, default=EVALS_DIR)
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, "experiment4_results")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  Experiment 4 v2: Trajectory Quality — All Difficulties")
    print(f"  Episodes per method per difficulty: {args.episodes}")
    print(f"  Difficulties: {', '.join(DIFF_ORDER)}")
    print(f"  Methods: {', '.join(METHOD_ORDER)}")
    print(f"  Output: {output_dir}")
    print(f"{'#'*60}")

    # results[difficulty][method] = [list of derive_metrics dicts]
    results = {}
    arrival_counts = {}  # arrival_counts[difficulty][method] = int
    json_output = {}

    total_combinations = len(DIFF_ORDER) * 3  # 3 methods
    combo = 0

    for diff in DIFF_ORDER:
        print(f"\n{'='*60}")
        print(f"  DIFFICULTY: {DIFF_LABELS[diff]}")
        print(f"{'='*60}")

        results[diff] = {}
        arrival_counts[diff] = {}
        json_output[diff] = {}

        # ── Pure PPO ──
        combo += 1
        print(f"\n  [{combo}/{total_combinations}] Pure PPO / {DIFF_LABELS[diff]}")
        ppo_eps = collect_pure_ppo(args.episodes, args.ppo_ckpt, diff)
        ppo_metrics = [derive_metrics(ep, DT) for ep in ppo_eps]
        results[diff]["Pure PPO"] = ppo_metrics
        arrival_counts[diff]["Pure PPO"] = len(ppo_eps)
        json_output[diff]["pure_ppo"] = {
            "n_arrived": len(ppo_eps),
            "n_episodes": args.episodes,
            "comfort": {
                "mean_speed": float(np.mean([m["mean_speed"] for m in ppo_metrics])) if ppo_metrics else 0,
                "mean_abs_lat_accel": float(np.mean([m["mean_abs_lat_accel"] for m in ppo_metrics])) if ppo_metrics else 0,
                "mean_abs_jerk": float(np.mean([m["mean_abs_jerk"] for m in ppo_metrics])) if ppo_metrics else 0,
                "mean_min_dist": float(np.mean([m["mean_min_dist"] for m in ppo_metrics])) if ppo_metrics else 0,
            },
        }

        # ── Pure MPC ──
        combo += 1
        print(f"\n  [{combo}/{total_combinations}] Pure MPC / {DIFF_LABELS[diff]}")
        mpc_eps = collect_pure_mpc(args.episodes, diff)
        mpc_metrics = [derive_metrics(ep, DT) for ep in mpc_eps]
        results[diff]["Pure MPC"] = mpc_metrics
        arrival_counts[diff]["Pure MPC"] = len(mpc_eps)
        json_output[diff]["pure_mpc"] = {
            "n_arrived": len(mpc_eps),
            "n_episodes": args.episodes,
            "comfort": {
                "mean_speed": float(np.mean([m["mean_speed"] for m in mpc_metrics])) if mpc_metrics else 0,
                "mean_abs_lat_accel": float(np.mean([m["mean_abs_lat_accel"] for m in mpc_metrics])) if mpc_metrics else 0,
                "mean_abs_jerk": float(np.mean([m["mean_abs_jerk"] for m in mpc_metrics])) if mpc_metrics else 0,
                "mean_min_dist": float(np.mean([m["mean_min_dist"] for m in mpc_metrics])) if mpc_metrics else 0,
            },
        }

        # ── MPC-RL ──
        combo += 1
        print(f"\n  [{combo}/{total_combinations}] MPC-RL / {DIFF_LABELS[diff]}")
        mpcrl_eps = collect_mpcrl(args.episodes, args.mpcrl_ckpt, diff)
        mpcrl_metrics = [derive_metrics(ep, DT) for ep in mpcrl_eps]
        results[diff]["MPC-RL"] = mpcrl_metrics
        arrival_counts[diff]["MPC-RL"] = len(mpcrl_eps)
        json_output[diff]["mpcrl"] = {
            "n_arrived": len(mpcrl_eps),
            "n_episodes": args.episodes,
            "comfort": {
                "mean_speed": float(np.mean([m["mean_speed"] for m in mpcrl_metrics])) if mpcrl_metrics else 0,
                "mean_abs_lat_accel": float(np.mean([m["mean_abs_lat_accel"] for m in mpcrl_metrics])) if mpcrl_metrics else 0,
                "mean_abs_jerk": float(np.mean([m["mean_abs_jerk"] for m in mpcrl_metrics])) if mpcrl_metrics else 0,
                "mean_min_dist": float(np.mean([m["mean_min_dist"] for m in mpcrl_metrics])) if mpcrl_metrics else 0,
            },
        }

        # ── Per-difficulty plot ──
        plot_per_difficulty(results[diff], diff, output_dir)

    # ── Save JSON summary ──
    json_path = os.path.join(output_dir, "exp4_summary.json")
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"\n[json] Saved: {json_path}")

    # ── Combined plots ──
    print("\nGenerating combined plots ...")
    plot_combined_all_difficulties(results, output_dir)
    plot_comfort_bars(results, output_dir)
    plot_arrival_count_bars(arrival_counts, args.episodes, output_dir)

    # ── Print summary table ──
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 4 SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Difficulty':<12} {'Method':<12} {'Arrived':>8} {'MeanSpd':>8} {'|LatAcc|':>8} {'|Jerk|':>8} {'MinDist':>8}")
    print(f"  {'-'*12} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for diff in DIFF_ORDER:
        for method in METHOD_ORDER:
            mets = results[diff].get(method, [])
            n_arr = len(mets)
            if mets:
                ms = np.mean([m["mean_speed"] for m in mets])
                la = np.mean([m["mean_abs_lat_accel"] for m in mets])
                jk = np.mean([m["mean_abs_jerk"] for m in mets])
                md = np.mean([m["mean_min_dist"] for m in mets])
                print(f"  {DIFF_LABELS[diff]:<12} {method:<12} {n_arr:>5}/{args.episodes:<2} {ms:>8.2f} {la:>8.3f} {jk:>8.3f} {md:>8.2f}")
            else:
                print(f"  {DIFF_LABELS[diff]:<12} {method:<12} {n_arr:>5}/{args.episodes:<2} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
    print(f"{'='*70}")

    print(f"\n[DONE] Experiment 4 v2 complete!")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
