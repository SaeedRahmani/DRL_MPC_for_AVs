"""
Experiment 4: Trajectory Quality Analysis
==========================================
Run Pure RL, Pure MPC, and MPC-RL on the intersection scenario,
collect per-step trajectory data for *successful* (arrived) episodes,
then generate side-by-side comparison plots:

  1. Speed profile over time
  2. Lateral acceleration (smoothness) over time
  3. Longitudinal jerk over time
  4. Distance to nearest vehicle over time
  5. Summary bar chart: mean comfort metrics

Outputs (in experiment4_results/):
  - trajectory_data.json         Raw per-step data
  - traj_speed_profiles.pdf      Speed over time
  - traj_lateral_accel.pdf       Lateral accel
  - traj_jerk.pdf                Longitudinal jerk
  - traj_min_distance.pdf        Distance to nearest vehicle
  - traj_comfort_summary.pdf     Bar chart of comfort metrics

Usage:
    cd /users/saeani/src/mpcrl/MPC-RL_for_AVs/src_ray_version && \\
    PYTHONPATH="$PWD:$PWD/highway-env:$PYTHONPATH" CUDA_VISIBLE_DEVICES="" \\
    python /users/saeani/src/mpcrl/DRL_MPC_for_AVs/src_ray_version/evals/experiment4_trajectory_quality.py
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

import highway_env  # noqa: F401

warnings.filterwarnings("ignore", message="invalid value")

EVALS_DIR  = os.path.dirname(os.path.abspath(__file__))
PPO_CKPT   = "/users/saeani/DEV/ray_results/PPO_pure_RL/PPO_intersection-pure-ppo_03a31_00000_0_2026-03-01_23-40-22/checkpoint_000025"
MPCRL_CKPT = "/users/saeani/DEV/ray_results/PPO_v0_manual/PPO_intersection-mpcrl-refspeed-manual_c129f_00000_0_2026-03-01_00-44-07/checkpoint_000020"

sys.path.insert(0, os.path.join(EVALS_DIR, "../../../MPC-RL_for_AVs/src_ray_version"))

# Policy frequency and dt
POLICY_FREQ = 10  # Hz
DT = 1.0 / POLICY_FREQ  # seconds between decision steps


# =====================================================================
#  Data collection helpers
# =====================================================================
def _get_ego(env):
    """Return the ego vehicle from the unwrapped env."""
    inner = env.unwrapped if hasattr(env, "unwrapped") else env
    # Try controlled_vehicles first (MPCRL / Pure MPC envs)
    if hasattr(inner, "controlled_vehicles") and inner.controlled_vehicles:
        return inner.controlled_vehicles[0]
    if hasattr(inner, "vehicle"):
        return inner.vehicle
    return None


def _min_dist_to_others(env, ego):
    """Compute min Euclidean distance from ego to any other vehicle."""
    inner = env.unwrapped if hasattr(env, "unwrapped") else env
    min_d = float("inf")
    for v in inner.road.vehicles:
        if v is ego:
            continue
        d = np.linalg.norm(ego.position - v.position)
        min_d = min(min_d, d)
    return min_d


def _has_arrived(env, ego):
    """Check if the ego has arrived."""
    inner = env.unwrapped if hasattr(env, "unwrapped") else env
    try:
        return bool(inner.has_arrived(ego))
    except Exception:
        return False


StepRecord = dict  # type alias


def record_step(env, ego, action_applied) -> StepRecord:
    """Capture one step's trajectory data."""
    return {
        "x": float(ego.position[0]),
        "y": float(ego.position[1]),
        "speed": float(ego.speed),
        "heading": float(ego.heading),
        "acc": float(action_applied[0]) if hasattr(action_applied, "__len__") else float(action_applied),
        "steer": float(action_applied[1]) if (hasattr(action_applied, "__len__") and len(action_applied) > 1) else 0.0,
        "min_dist": float(_min_dist_to_others(env, ego)),
    }


def derive_metrics(steps: list[StepRecord], dt: float) -> dict:
    """Compute derived metrics from a sequence of step records."""
    speed = np.array([s["speed"] for s in steps])
    heading = np.array([s["heading"] for s in steps])
    acc = np.array([s["acc"] for s in steps])
    min_dist = np.array([s["min_dist"] for s in steps])
    t = np.arange(len(steps)) * dt

    # Lateral acceleration: a_lat = v * d(heading)/dt
    dheading = np.gradient(heading, dt)
    lat_accel = speed * dheading

    # Longitudinal jerk: d(acc)/dt
    jerk = np.gradient(acc, dt)

    return {
        "time": t.tolist(),
        "speed": speed.tolist(),
        "lat_accel": lat_accel.tolist(),
        "jerk": jerk.tolist(),
        "min_dist": min_dist.tolist(),
        "mean_speed": float(np.mean(speed)),
        "mean_abs_lat_accel": float(np.mean(np.abs(lat_accel))),
        "mean_abs_jerk": float(np.mean(np.abs(jerk))),
        "mean_min_dist": float(np.mean(min_dist)),
        "min_min_dist": float(np.min(min_dist)) if len(min_dist) > 0 else 0.0,
    }


# =====================================================================
#  1. PURE RL
# =====================================================================
def collect_pure_rl(n_episodes, ppo_ckpt, max_arrived=5):
    """Run Pure RL episodes, collect trajectory data for arrived episodes."""
    from ray.rllib.policy.policy import Policy
    from train_pure_ppo import IntersectionPurePPOEnv

    print("\n[Pure RL] Loading policy ...")
    policy = Policy.from_checkpoint(ppo_ckpt)
    if isinstance(policy, dict):
        policy = policy["default_policy"]

    env = IntersectionPurePPOEnv(render_mode="rgb_array")
    episodes_data = []
    n_arrived = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = []

        while not done:
            action = policy.compute_single_action(obs, explore=False)[0]
            obs, _, terminated, truncated, _ = env.step(action)

            ego = _get_ego(env)
            if ego is not None:
                steps.append(record_step(env, ego, action))

            done = terminated or truncated

        ego = _get_ego(env)
        arrived = _has_arrived(env, ego) if ego else False
        crashed = bool(ego.crashed) if ego else True
        status = "ARRIVE" if arrived else ("CRASH" if crashed else "OTHER")
        print(f"  [Pure RL] ep {ep+1}/{n_episodes}  steps={len(steps)}  {status}")

        if arrived and len(steps) > 5:
            episodes_data.append({"steps": steps, "outcome": "arrived"})
            n_arrived += 1
            if n_arrived >= max_arrived:
                break

    env.close()
    print(f"  [Pure RL] Collected {len(episodes_data)} arrived episodes")
    return episodes_data


# =====================================================================
#  2. PURE MPC
# =====================================================================
def collect_pure_mpc(n_episodes, max_arrived=5):
    """Run Pure MPC episodes, collect trajectory data for arrived episodes."""
    env = gymnasium.make("intersection-mpc-manual", render_mode="rgb_array")
    episodes_data = []
    n_arrived = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = []

        while not done:
            dummy = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(dummy)

            ego = _get_ego(env)
            if ego is not None:
                # For Pure MPC, the actual action is computed internally;
                # record the ego state and use speed delta as proxy for acc
                rec = record_step(env, ego, np.array([0.0, 0.0]))
                # Override acc with speed-based estimate
                if len(steps) > 0:
                    rec["acc"] = (rec["speed"] - steps[-1]["speed"]) / DT
                steps.append(rec)

            done = terminated or truncated

        ego = _get_ego(env)
        arrived = _has_arrived(env, ego) if ego else False
        crashed = bool(ego.crashed) if ego else True
        status = "ARRIVE" if arrived else ("CRASH" if crashed else "OTHER")
        print(f"  [Pure MPC] ep {ep+1}/{n_episodes}  steps={len(steps)}  {status}")

        if arrived and len(steps) > 5:
            episodes_data.append({"steps": steps, "outcome": "arrived"})
            n_arrived += 1
            if n_arrived >= max_arrived:
                break

    env.close()
    print(f"  [Pure MPC] Collected {len(episodes_data)} arrived episodes")
    return episodes_data


# =====================================================================
#  3. MPC-RL
# =====================================================================
def collect_mpcrl(n_episodes, mpcrl_ckpt, max_arrived=5):
    """Run MPC-RL episodes, collect trajectory data for arrived episodes."""
    from ray.rllib.policy.policy import Policy

    print("\n[MPC-RL] Loading policy ...")
    policy = Policy.from_checkpoint(mpcrl_ckpt)
    if isinstance(policy, dict):
        policy = policy["default_policy"]

    env = gymnasium.make("intersection-mpcrl-refspeed-manual", render_mode="rgb_array")
    episodes_data = []
    n_arrived = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = []

        while not done:
            action = policy.compute_single_action(obs, explore=False)[0]
            obs, _, terminated, truncated, _ = env.step(action)

            ego = _get_ego(env)
            if ego is not None:
                rec = record_step(env, ego, np.array([0.0, 0.0]))
                if len(steps) > 0:
                    rec["acc"] = (rec["speed"] - steps[-1]["speed"]) / DT
                steps.append(rec)

            done = terminated or truncated

        ego = _get_ego(env)
        arrived = _has_arrived(env, ego) if ego else False
        crashed = bool(ego.crashed) if ego else True
        status = "ARRIVE" if arrived else ("CRASH" if crashed else "OTHER")
        print(f"  [MPC-RL] ep {ep+1}/{n_episodes}  steps={len(steps)}  {status}")

        if arrived and len(steps) > 5:
            episodes_data.append({"steps": steps, "outcome": "arrived"})
            n_arrived += 1
            if n_arrived >= max_arrived:
                break

    env.close()
    print(f"  [MPC-RL] Collected {len(episodes_data)} arrived episodes")
    return episodes_data


# =====================================================================
#  Plotting
# =====================================================================
COLORS = {"Pure RL": "#2196F3", "Pure MPC": "#FF9800", "MPC-RL": "#4CAF50"}


def _plot_metric(all_data, metric_key, ylabel, title, filename, output_dir,
                 clip_range=None):
    """Plot a single metric for all methods, overlaying individual episodes
    as thin lines and the mean as a thick line."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for method_name, eps_list in all_data.items():
        color = COLORS[method_name]
        all_vals = []
        max_len = 0
        for ep_metrics in eps_list:
            vals = np.array(ep_metrics[metric_key])
            if clip_range:
                vals = np.clip(vals, clip_range[0], clip_range[1])
            t = np.array(ep_metrics["time"])
            ax.plot(t, vals, color=color, alpha=0.15, linewidth=0.8)
            if len(vals) > max_len:
                max_len = len(vals)
            all_vals.append(vals)

        # Compute mean (pad shorter episodes with NaN)
        if all_vals:
            padded = np.full((len(all_vals), max_len), np.nan)
            for i, v in enumerate(all_vals):
                padded[i, :len(v)] = v
            mean_vals = np.nanmean(padded, axis=0)
            t_mean = np.arange(max_len) * DT
            ax.plot(t_mean, mean_vals, color=color, linewidth=2.5,
                    label=f"{method_name} (mean)")

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"{filename}.{ext}")
        fig.savefig(path, dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {filename}.pdf/png")


def make_comfort_bar_chart(all_metrics, output_dir):
    """Bar chart comparing mean |lateral accel|, mean |jerk|, mean min-dist."""
    methods = list(all_metrics.keys())
    metrics = ["mean_abs_lat_accel", "mean_abs_jerk", "mean_min_dist"]
    labels  = ["Mean |Lat. Accel|\n(m/s²)", "Mean |Jerk|\n(m/s³)", "Mean Min Dist\n(m)"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[idx]
        vals = []
        stds = []
        for m in methods:
            ep_vals = [ep[metric] for ep in all_metrics[m]]
            vals.append(np.mean(ep_vals))
            stds.append(np.std(ep_vals))

        bars = ax.bar(methods, vals, yerr=stds, capsize=5,
                      color=[COLORS[m] for m in methods], alpha=0.8)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label.split("\n")[0], fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Trajectory Comfort & Safety Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"traj_comfort_summary.{ext}")
        fig.savefig(path, dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] traj_comfort_summary.pdf/png")


# =====================================================================
#  Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Experiment 4: Trajectory Quality")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Max episodes to try per method (default: 50)")
    parser.add_argument("--max-arrived", type=int, default=5,
                        help="Stop after this many arrived episodes per method (default: 5)")
    parser.add_argument("--ppo-ckpt", type=str, default=PPO_CKPT)
    parser.add_argument("--mpcrl-ckpt", type=str, default=MPCRL_CKPT)
    parser.add_argument("--output-dir", type=str, default=EVALS_DIR)
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, "experiment4_results")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  Experiment 4: Trajectory Quality Analysis")
    print(f"  Max episodes per method: {args.episodes}")
    print(f"  Target arrived episodes: {args.max_arrived}")
    print(f"  Output: {output_dir}")
    print(f"{'#'*60}")

    # ── Collect data ──
    print("\n" + "="*60)
    print("  [1/3] Collecting Pure RL trajectories ...")
    print("="*60)
    pure_rl_data = collect_pure_rl(args.episodes, args.ppo_ckpt, args.max_arrived)

    print("\n" + "="*60)
    print("  [2/3] Collecting Pure MPC trajectories ...")
    print("="*60)
    pure_mpc_data = collect_pure_mpc(args.episodes, args.max_arrived)

    print("\n" + "="*60)
    print("  [3/3] Collecting MPC-RL trajectories ...")
    print("="*60)
    mpcrl_data = collect_mpcrl(args.episodes, args.mpcrl_ckpt, args.max_arrived)

    # ── Derive metrics per episode ──
    all_metrics = {}
    all_plot_data = {}
    for method_name, eps_list in [
        ("Pure RL", pure_rl_data),
        ("Pure MPC", pure_mpc_data),
        ("MPC-RL", mpcrl_data),
    ]:
        metrics_list = []
        for ep in eps_list:
            m = derive_metrics(ep["steps"], DT)
            metrics_list.append(m)
        all_metrics[method_name] = metrics_list
        all_plot_data[method_name] = metrics_list

        # Print summary
        if metrics_list:
            print(f"\n  {method_name} ({len(metrics_list)} arrived episodes):")
            print(f"    Mean speed:         {np.mean([m['mean_speed'] for m in metrics_list]):.2f} m/s")
            print(f"    Mean |lat accel|:   {np.mean([m['mean_abs_lat_accel'] for m in metrics_list]):.3f} m/s²")
            print(f"    Mean |jerk|:        {np.mean([m['mean_abs_jerk'] for m in metrics_list]):.3f} m/s³")
            print(f"    Mean min-dist:      {np.mean([m['mean_min_dist'] for m in metrics_list]):.2f} m")
            print(f"    Min min-dist:       {np.min([m['min_min_dist'] for m in metrics_list]):.2f} m")

    # ── Save raw data as JSON ──
    json_path = os.path.join(output_dir, "trajectory_data.json")
    json_out = {}
    for method_name, eps_list in [
        ("pure_rl", pure_rl_data),
        ("pure_mpc", pure_mpc_data),
        ("mpcrl", mpcrl_data),
    ]:
        json_out[method_name] = {
            "n_arrived": len(eps_list),
            "episodes": [
                {
                    "n_steps": len(ep["steps"]),
                    "outcome": ep["outcome"],
                    "steps": ep["steps"],
                }
                for ep in eps_list
            ],
        }
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\n[json] Saved: {json_path}")

    # ── Generate plots ──
    print("\nGenerating plots ...")

    _plot_metric(all_plot_data, "speed", "Speed (m/s)",
                 "Speed Profile Over Time", "traj_speed_profiles", output_dir)

    _plot_metric(all_plot_data, "lat_accel", "Lateral Acceleration (m/s²)",
                 "Lateral Acceleration Over Time", "traj_lateral_accel", output_dir,
                 clip_range=(-15, 15))

    _plot_metric(all_plot_data, "jerk", "Longitudinal Jerk (m/s³)",
                 "Longitudinal Jerk Over Time", "traj_jerk", output_dir,
                 clip_range=(-100, 100))

    _plot_metric(all_plot_data, "min_dist", "Distance to Nearest Vehicle (m)",
                 "Minimum Distance to Other Vehicles", "traj_min_distance", output_dir)

    make_comfort_bar_chart(all_metrics, output_dir)

    print("\n[DONE] Experiment 4 complete!")


if __name__ == "__main__":
    main()
