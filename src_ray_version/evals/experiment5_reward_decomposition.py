"""
Experiment 5: Reward Decomposition
===================================
For each method (Pure RL, Pure MPC, MPC-RL), run episodes and compute
the MPCRL-style reward components at every step — even for Pure MPC
(computed externally from vehicle state).

Components tracked:
  1. Alive bonus       (+0.2 / step)
  2. Speed efficiency  (up to +1.5)
  3. Proximity penalty (up to -2.0)
  4. Smoothness        (-0.3 * |Δacc|/5)
  5. Centering         (up to +0.3)
  6. Terminal          (crash -50, off-road -10, arrived +20)

Outputs (in experiment5_results/):
  - reward_decomposition_results.json     Raw per-step data
  - reward_cumulative_breakdown.pdf       Stacked area: cumulative reward by component
  - reward_component_bars.pdf             Bar chart of mean per-step contribution
  - reward_pie_chart.pdf                  Pie chart of positive/negative contributions

Usage:
    cd /users/saeani/src/mpcrl/MPC-RL_for_AVs/src_ray_version && \
    PYTHONPATH="$PWD:$PWD/highway-env:$PYTHONPATH" CUDA_VISIBLE_DEVICES="" \
    python /users/saeani/src/mpcrl/DRL_MPC_for_AVs/src_ray_version/evals/experiment5_reward_decomposition.py
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

MAX_REF_SPEED = 15.0
PROXIMITY_RANGE = 30.0
COLORS = {"Pure RL": "#2196F3", "Pure MPC": "#FF9800", "MPC-RL": "#4CAF50"}

# Component colors for stacked charts
COMP_COLORS = {
    "alive":     "#66BB6A",   # green
    "speed":     "#42A5F5",   # blue
    "proximity": "#EF5350",   # red
    "smoothness":"#FFA726",   # orange
    "centering": "#AB47BC",   # purple
    "terminal":  "#78909C",   # grey
}


# =====================================================================
#  Reward decomposition (external, from vehicle state)
# =====================================================================
def compute_reward_components(vehicle, min_dist_norm, prev_acc, on_road,
                              crashed, arrived):
    """
    Compute individual reward components exactly as in the MPCRL reward
    function, but externally from vehicle state.

    Returns dict of component values and total.
    """
    components = {
        "alive": 0.0,
        "speed": 0.0,
        "proximity": 0.0,
        "smoothness": 0.0,
        "centering": 0.0,
        "terminal": 0.0,
    }

    # Terminal events
    if crashed:
        components["terminal"] = -50.0
        return components
    if not on_road:
        components["terminal"] = -10.0
        return components
    if arrived:
        components["terminal"] = 20.0
        return components

    # 1. Alive bonus
    components["alive"] = 0.2

    # 2. Speed efficiency
    speed_ratio = np.clip(vehicle.speed / MAX_REF_SPEED, 0.0, 1.5)
    components["speed"] = 1.5 * speed_ratio

    # 3. Proximity penalty
    if min_dist_norm < 0.5:
        components["proximity"] = -2.0 * (1.0 - 2.0 * min_dist_norm) ** 2

    # 4. Smoothness
    current_acc = getattr(vehicle, "action", {}).get("acceleration", 0.0) \
        if isinstance(getattr(vehicle, "action", None), dict) else 0.0
    # Use speed-derived acc if action not available
    acc_change = abs(current_acc - prev_acc)
    components["smoothness"] = -0.3 * (acc_change / 5.0)

    # 5. Centering
    if on_road and vehicle.lane is not None:
        try:
            lateral = vehicle.lane.local_coordinates(vehicle.position)[1]
            centering = 1.0 - min(1.0, abs(lateral) / (vehicle.lane.width / 2))
            components["centering"] = 0.3 * centering
        except Exception:
            pass

    return components


def _compute_min_dist_norm(env, ego):
    """Compute normalized min distance (0=touching, 1=far away)."""
    inner = env.unwrapped if hasattr(env, "unwrapped") else env
    min_d = float("inf")
    for v in inner.road.vehicles:
        if v is ego:
            continue
        d = np.linalg.norm(ego.position - v.position)
        min_d = min(min_d, d)
    return float(np.clip(min_d / PROXIMITY_RANGE, 0.0, 1.0))


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
#  Data collection
# =====================================================================
def collect_episodes(method_name, n_episodes, ppo_ckpt=None, mpcrl_ckpt=None,
                     max_arrived=10):
    """Collect episodes with per-step reward decomposition."""
    from ray.rllib.policy.policy import Policy

    policy = None
    env = None

    if method_name == "Pure RL":
        from train_pure_ppo import IntersectionPurePPOEnv
        policy = Policy.from_checkpoint(ppo_ckpt)
        if isinstance(policy, dict):
            policy = policy["default_policy"]
        env = IntersectionPurePPOEnv(render_mode="rgb_array")

    elif method_name == "Pure MPC":
        env = gymnasium.make("intersection-mpc-manual", render_mode="rgb_array")

    elif method_name == "MPC-RL":
        policy = Policy.from_checkpoint(mpcrl_ckpt)
        if isinstance(policy, dict):
            policy = policy["default_policy"]
        env = gymnasium.make("intersection-mpcrl-refspeed-manual", render_mode="rgb_array")

    episodes = []
    n_arrived = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step_components = []
        prev_acc = 0.0
        prev_speed = 0.0

        while not done:
            # Get action
            if policy is not None:
                action = policy.compute_single_action(obs, explore=False)[0]
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            ego = _get_ego(env)
            if ego is None:
                continue

            arrived = _has_arrived(env, ego)
            min_dist_norm = _compute_min_dist_norm(env, ego)

            # Estimate acceleration from speed delta
            dt = 0.1  # 1/policy_freq
            current_acc = (ego.speed - prev_speed) / dt if len(step_components) > 0 else 0.0

            comp = compute_reward_components(
                vehicle=ego,
                min_dist_norm=min_dist_norm,
                prev_acc=prev_acc,
                on_road=ego.on_road,
                crashed=ego.crashed,
                arrived=arrived,
            )
            # Override smoothness with speed-derived acc
            acc_change = abs(current_acc - prev_acc)
            comp["smoothness"] = -0.3 * (acc_change / 5.0)

            step_components.append(comp)
            prev_acc = current_acc
            prev_speed = ego.speed

        ego = _get_ego(env)
        arrived_final = _has_arrived(env, ego) if ego else False
        crashed_final = bool(ego.crashed) if ego else True
        status = "ARRIVE" if arrived_final else ("CRASH" if crashed_final else "OTHER")

        ep_data = {
            "status": status,
            "n_steps": len(step_components),
            "components": step_components,
        }
        episodes.append(ep_data)
        print(f"  [{method_name}] ep {ep+1}/{n_episodes}  steps={len(step_components)}  {status}")

        if arrived_final:
            n_arrived += 1
            if n_arrived >= max_arrived:
                break

    env.close()
    n_arr = sum(1 for e in episodes if e["status"] == "ARRIVE")
    print(f"  [{method_name}] Collected {len(episodes)} episodes ({n_arr} arrived)")
    return episodes


# =====================================================================
#  Plotting
# =====================================================================
def _aggregate_components(episodes):
    """Aggregate reward components across all episodes.
    Returns dict of component_name -> list of per-step values (all episodes concat)."""
    comp_names = ["alive", "speed", "proximity", "smoothness", "centering", "terminal"]
    agg = {c: [] for c in comp_names}

    for ep in episodes:
        for step in ep["components"]:
            for c in comp_names:
                agg[c].append(step.get(c, 0.0))
    return agg


def plot_cumulative_breakdown(all_data, output_dir):
    """Stacked area chart of cumulative reward over time for each method."""
    comp_names = ["alive", "speed", "centering", "proximity", "smoothness", "terminal"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    methods = ["Pure RL", "Pure MPC", "MPC-RL"]

    for idx, method in enumerate(methods):
        ax = axes[idx]
        episodes = all_data[method]

        # Use the longest arrived episode (or first episode if none arrived)
        arrived_eps = [e for e in episodes if e["status"] == "ARRIVE"]
        ep = arrived_eps[0] if arrived_eps else episodes[0]

        t = np.arange(len(ep["components"])) * 0.1  # seconds

        # Build cumulative sum per component
        cum_data = {}
        for c in comp_names:
            vals = np.array([s.get(c, 0.0) for s in ep["components"]])
            cum_data[c] = np.cumsum(vals)

        # Stack positive and negative separately for clarity
        pos_comps = ["alive", "speed", "centering", "terminal"]
        neg_comps = ["proximity", "smoothness"]

        # Plot positive cumulative
        bottom_pos = np.zeros(len(t))
        for c in pos_comps:
            vals = np.maximum(cum_data[c], 0)
            if np.any(vals > 0):
                ax.fill_between(t, bottom_pos, bottom_pos + vals,
                                alpha=0.7, color=COMP_COLORS[c], label=c.capitalize())
                bottom_pos += vals

        # Plot negative cumulative
        bottom_neg = np.zeros(len(t))
        for c in neg_comps:
            vals = np.minimum(cum_data[c], 0)
            if np.any(vals < 0):
                ax.fill_between(t, bottom_neg + vals, bottom_neg,
                                alpha=0.7, color=COMP_COLORS[c], label=c.capitalize())
                bottom_neg += vals

        # Total reward line
        total = sum(cum_data[c] for c in comp_names)
        ax.plot(t, total, "k-", linewidth=2, label="Total")

        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Cumulative Reward", fontsize=11)
        ax.set_title(method, fontsize=13, fontweight="bold", color=COLORS[method])
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Cumulative Reward Breakdown (Single Arrived Episode)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"reward_cumulative_breakdown.{ext}")
        fig.savefig(path, dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] reward_cumulative_breakdown.pdf/png")


def plot_component_bars(all_data, output_dir):
    """Bar chart of mean per-step reward component for each method."""
    comp_names = ["alive", "speed", "proximity", "smoothness", "centering"]
    methods = ["Pure RL", "Pure MPC", "MPC-RL"]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(comp_names))
    width = 0.25

    for i, method in enumerate(methods):
        agg = _aggregate_components(all_data[method])
        means = [np.mean(agg[c]) if agg[c] else 0.0 for c in comp_names]
        bars = ax.bar(x + i * width, means, width, label=method,
                      color=COLORS[method], alpha=0.85)
        for bar, v in zip(bars, means):
            va = "bottom" if v >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.01 if v >= 0 else -0.01),
                    f"{v:.3f}", ha="center", va=va, fontsize=8)

    ax.set_xlabel("Reward Component", fontsize=12)
    ax.set_ylabel("Mean Per-Step Value", fontsize=12)
    ax.set_title("Mean Per-Step Reward Components", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([c.capitalize() for c in comp_names], fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"reward_component_bars.{ext}")
        fig.savefig(path, dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] reward_component_bars.pdf/png")


def plot_episode_return_breakdown(all_data, output_dir):
    """Horizontal stacked bar: total episode return broken into components for arrived episodes."""
    comp_names = ["alive", "speed", "centering", "proximity", "smoothness", "terminal"]
    methods = ["Pure RL", "Pure MPC", "MPC-RL"]

    fig, ax = plt.subplots(figsize=(12, 5))

    y_pos = np.arange(len(methods))
    bar_height = 0.5

    for method_idx, method in enumerate(methods):
        arrived_eps = [e for e in all_data[method] if e["status"] == "ARRIVE"]
        if not arrived_eps:
            continue

        # Average across arrived episodes
        mean_totals = {}
        for c in comp_names:
            ep_sums = []
            for ep in arrived_eps:
                total = sum(s.get(c, 0.0) for s in ep["components"])
                ep_sums.append(total)
            mean_totals[c] = np.mean(ep_sums)

        # Plot positive going right, negative going left
        left_pos = 0
        left_neg = 0
        for c in comp_names:
            val = mean_totals[c]
            if val >= 0:
                ax.barh(y_pos[method_idx], val, bar_height, left=left_pos,
                        color=COMP_COLORS[c], alpha=0.85,
                        label=c.capitalize() if method_idx == 0 else "")
                if abs(val) > 1:
                    ax.text(left_pos + val / 2, y_pos[method_idx],
                            f"{val:.1f}", ha="center", va="center", fontsize=8)
                left_pos += val
            else:
                ax.barh(y_pos[method_idx], val, bar_height, left=left_neg,
                        color=COMP_COLORS[c], alpha=0.85,
                        label=c.capitalize() if method_idx == 0 else "")
                if abs(val) > 1:
                    ax.text(left_neg + val / 2, y_pos[method_idx],
                            f"{val:.1f}", ha="center", va="center", fontsize=8)
                left_neg += val

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=12)
    ax.set_xlabel("Total Episode Reward", fontsize=12)
    ax.set_title("Episode Return Breakdown (Mean of Arrived Episodes)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"reward_episode_breakdown.{ext}")
        fig.savefig(path, dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] reward_episode_breakdown.pdf/png")


# =====================================================================
#  Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Experiment 5: Reward Decomposition")
    parser.add_argument("--episodes", type=int, default=30,
                        help="Max episodes per method (default: 30)")
    parser.add_argument("--max-arrived", type=int, default=10,
                        help="Stop after N arrived episodes (default: 10)")
    parser.add_argument("--ppo-ckpt", type=str, default=PPO_CKPT)
    parser.add_argument("--mpcrl-ckpt", type=str, default=MPCRL_CKPT)
    parser.add_argument("--output-dir", type=str, default=EVALS_DIR)
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, "experiment5_results")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  Experiment 5: Reward Decomposition")
    print(f"  Max episodes per method: {args.episodes}")
    print(f"  Target arrived: {args.max_arrived}")
    print(f"  Output: {output_dir}")
    print(f"{'#'*60}")

    all_data = {}

    for method_name, ckpt_kwarg in [
        ("Pure RL", {"ppo_ckpt": args.ppo_ckpt}),
        ("Pure MPC", {}),
        ("MPC-RL", {"mpcrl_ckpt": args.mpcrl_ckpt}),
    ]:
        print(f"\n{'='*60}")
        print(f"  {method_name}")
        print(f"{'='*60}")

        episodes = collect_episodes(
            method_name, args.episodes,
            max_arrived=args.max_arrived,
            **ckpt_kwarg,
        )
        all_data[method_name] = episodes

    # ── Print summary ──
    print(f"\n{'='*60}")
    print("  Summary: Mean per-step reward components (all episodes)")
    print(f"{'='*60}")
    comp_names = ["alive", "speed", "proximity", "smoothness", "centering"]
    for method in ["Pure RL", "Pure MPC", "MPC-RL"]:
        agg = _aggregate_components(all_data[method])
        n_steps = len(agg["alive"])
        print(f"\n  {method} ({n_steps} total steps):")
        for c in comp_names:
            mean_val = np.mean(agg[c]) if agg[c] else 0.0
            print(f"    {c:12s}: {mean_val:+.4f}")
        total = sum(np.mean(agg[c]) for c in comp_names if agg[c])
        print(f"    {'TOTAL':12s}: {total:+.4f}")

    # ── Save JSON ──
    json_path = os.path.join(output_dir, "reward_decomposition_results.json")
    json_out = {}
    for method, eps in all_data.items():
        json_out[method] = {
            "n_episodes": len(eps),
            "n_arrived": sum(1 for e in eps if e["status"] == "ARRIVE"),
            "episodes": [
                {
                    "status": e["status"],
                    "n_steps": e["n_steps"],
                    "components": e["components"],
                }
                for e in eps
            ],
        }
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\n[json] Saved: {json_path}")

    # ── Plots ──
    print("\nGenerating plots ...")
    plot_cumulative_breakdown(all_data, output_dir)
    plot_component_bars(all_data, output_dir)
    plot_episode_return_breakdown(all_data, output_dir)

    print("\n[DONE] Experiment 5 complete!")


if __name__ == "__main__":
    main()
