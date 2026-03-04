"""
Experiment 3: Inference-Time Benchmark
======================================
Measure per-step decision time for Pure RL, Pure MPC, and MPC-RL
on the intersection scenario.

For each method we run N episodes and record wall-clock time of
the *decision computation* at every step:
  - Pure RL:  policy.compute_single_action(obs)
  - Pure MPC: env._predict_mpc_action(dummy)  (monkey-patched for timing)
  - MPC-RL:   policy.compute_single_action(obs) + env._predict_mpc_action(rl_action)

Outputs:
  - inference_time_results.json   (all raw per-step times + summary stats)
  - inference_time_boxplot.pdf    (box plot comparing methods)
  - inference_time_table.pdf      (LaTeX-style summary table)

Usage:
    cd /users/saeani/src/mpcrl/MPC-RL_for_AVs/src_ray_version && \
    PYTHONPATH="$PWD:$PWD/highway-env:$PYTHONPATH" CUDA_VISIBLE_DEVICES="" \
    python /users/saeani/src/mpcrl/DRL_MPC_for_AVs/src_ray_version/evals/experiment3_inference_time.py
"""

import os, sys, json, time, argparse
import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import gymnasium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import highway_env  # noqa: F401

# ── paths (defaults) ──
EVALS_DIR  = os.path.dirname(os.path.abspath(__file__))
PPO_CKPT   = "/users/saeani/DEV/ray_results/PPO_pure_RL/PPO_intersection-pure-ppo_03a31_00000_0_2026-03-01_23-40-22/checkpoint_000025"
MPCRL_CKPT = "/users/saeani/DEV/ray_results/PPO_v0_manual/PPO_intersection-mpcrl-refspeed-manual_c129f_00000_0_2026-03-01_00-44-07/checkpoint_000020"

# Add training repo so we can import IntersectionPurePPOEnv
sys.path.insert(0, os.path.join(EVALS_DIR, "../../../MPC-RL_for_AVs/src_ray_version"))


# =====================================================================
#  1. PURE RL — policy forward pass only
# =====================================================================
def benchmark_pure_rl(n_episodes, ppo_ckpt):
    """Time policy.compute_single_action() per step."""
    from ray.rllib.policy.policy import Policy
    from train_pure_ppo import IntersectionPurePPOEnv

    print("\n[Pure RL] Loading policy ...")
    policy = Policy.from_checkpoint(ppo_ckpt)
    if isinstance(policy, dict):
        policy = policy["default_policy"]

    env = IntersectionPurePPOEnv(render_mode="rgb_array")
    step_times = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            t0 = time.perf_counter()
            action = policy.compute_single_action(obs, explore=False)[0]
            dt = time.perf_counter() - t0
            step_times.append(dt)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        print(f"  [Pure RL] ep {ep+1}/{n_episodes}  steps_so_far={len(step_times)}")

    env.close()
    return step_times


# =====================================================================
#  2. PURE MPC — MPC solve inside env
# =====================================================================
def benchmark_pure_mpc(n_episodes):
    """Monkey-patch _predict_mpc_action to measure MPC decision time."""
    env = gymnasium.make("intersection-mpc-manual", render_mode="rgb_array")
    inner = env.unwrapped
    step_times = []

    # Monkey-patch _predict_mpc_action
    _orig_predict = inner._predict_mpc_action

    def _timed_predict(action):
        t0 = time.perf_counter()
        result = _orig_predict(action)
        dt = time.perf_counter() - t0
        step_times.append(dt)
        return result

    inner._predict_mpc_action = _timed_predict

    for ep in range(n_episodes):
        n_before = len(step_times)
        obs, _ = env.reset()
        done = False
        while not done:
            dummy = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(dummy)
            done = terminated or truncated
        n_this = len(step_times) - n_before
        print(f"  [Pure MPC] ep {ep+1}/{n_episodes}  steps_this_ep={n_this}")

    env.close()
    return step_times


# =====================================================================
#  3. MPC-RL — RL forward pass + MPC solve
# =====================================================================
def benchmark_mpcrl(n_episodes, mpcrl_ckpt):
    """Time RL forward pass + MPC solve per step."""
    from ray.rllib.policy.policy import Policy

    print("\n[MPC-RL] Loading policy ...")
    policy = Policy.from_checkpoint(mpcrl_ckpt)
    if isinstance(policy, dict):
        policy = policy["default_policy"]

    env = gymnasium.make("intersection-mpcrl-refspeed-manual", render_mode="rgb_array")
    inner = env.unwrapped

    rl_times  = []   # just the RL forward pass
    mpc_times = []   # just the MPC solve
    total_times = [] # RL + MPC combined

    # Monkey-patch _predict_mpc_action
    _orig_predict = inner._predict_mpc_action
    _mpc_dt = [0.0]  # mutable container for the latest MPC time

    def _timed_predict(action):
        t0 = time.perf_counter()
        result = _orig_predict(action)
        _mpc_dt[0] = time.perf_counter() - t0
        return result

    inner._predict_mpc_action = _timed_predict

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_steps = 0
        while not done:
            # RL forward pass
            t0_rl = time.perf_counter()
            action = policy.compute_single_action(obs, explore=False)[0]
            dt_rl = time.perf_counter() - t0_rl
            rl_times.append(dt_rl)

            # env.step triggers _predict_mpc_action internally
            _mpc_dt[0] = 0.0
            obs, _, terminated, truncated, _ = env.step(action)
            mpc_times.append(_mpc_dt[0])
            total_times.append(dt_rl + _mpc_dt[0])

            done = terminated or truncated
            ep_steps += 1
        print(f"  [MPC-RL] ep {ep+1}/{n_episodes}  steps={ep_steps}")

    env.close()
    return rl_times, mpc_times, total_times


# =====================================================================
#  Plotting & Reporting
# =====================================================================
def to_ms(times):
    """Convert list of seconds to numpy array of milliseconds."""
    return np.array(times) * 1000.0


def make_box_plot(results, output_dir):
    """Generate a publication-quality box plot."""
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = []
    data = []

    # Pure RL
    labels.append("Pure RL")
    data.append(to_ms(results["pure_rl"]["step_times"]))

    # Pure MPC
    labels.append("Pure MPC")
    data.append(to_ms(results["pure_mpc"]["step_times"]))

    # MPC-RL (total)
    labels.append("MPC-RL\n(total)")
    data.append(to_ms(results["mpcrl"]["total_times"]))

    # MPC-RL (RL only)
    labels.append("MPC-RL\n(RL only)")
    data.append(to_ms(results["mpcrl"]["rl_times"]))

    # MPC-RL (MPC only)
    labels.append("MPC-RL\n(MPC only)")
    data.append(to_ms(results["mpcrl"]["mpc_times"]))

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#81C784", "#A5D6A7"]

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    showfliers=False, widths=0.55,
                    medianprops=dict(color="black", linewidth=1.5))

    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    ax.set_ylabel("Decision Time (ms)", fontsize=13)
    ax.set_title("Per-Step Inference Time Comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Add mean markers
    for i, d in enumerate(data):
        ax.scatter(i + 1, np.mean(d), marker="D", color="red",
                   s=40, zorder=5, label="Mean" if i == 0 else None)

    ax.legend(loc="upper left", fontsize=10)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"inference_time_boxplot.{ext}")
        fig.savefig(path, dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
        print(f"[plot] Saved: {path}")
    plt.close(fig)


def make_summary_table(results, output_dir):
    """Print and save a formatted summary table."""
    rows = []
    for name, key, time_key in [
        ("Pure RL",  "pure_rl",  "step_times"),
        ("Pure MPC", "pure_mpc", "step_times"),
        ("MPC-RL (total)", "mpcrl", "total_times"),
        ("MPC-RL (RL only)", "mpcrl", "rl_times"),
        ("MPC-RL (MPC only)", "mpcrl", "mpc_times"),
    ]:
        t = to_ms(results[key][time_key])
        rows.append({
            "method": name,
            "mean_ms": float(np.mean(t)),
            "std_ms": float(np.std(t)),
            "median_ms": float(np.median(t)),
            "p5_ms": float(np.percentile(t, 5)),
            "p95_ms": float(np.percentile(t, 95)),
            "min_ms": float(np.min(t)),
            "max_ms": float(np.max(t)),
            "n_samples": len(t),
        })

    # Print table
    print(f"\n{'='*90}")
    print(f"  INFERENCE TIME SUMMARY (milliseconds)")
    print(f"{'='*90}")
    print(f"  {'Method':<22s} {'Mean':>8s} {'Std':>8s} {'Median':>8s} "
          f"{'P5':>8s} {'P95':>8s} {'N':>6s}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for r in rows:
        print(f"  {r['method']:<22s} {r['mean_ms']:8.3f} {r['std_ms']:8.3f} "
              f"{r['median_ms']:8.3f} {r['p5_ms']:8.3f} {r['p95_ms']:8.3f} "
              f"{r['n_samples']:6d}")
    print(f"{'='*90}\n")

    # Save as table figure
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    col_labels = ["Method", "Mean (ms)", "Std (ms)", "Median (ms)",
                  "P5 (ms)", "P95 (ms)", "N"]
    cell_text = []
    for r in rows:
        cell_text.append([
            r["method"],
            f"{r['mean_ms']:.3f}",
            f"{r['std_ms']:.3f}",
            f"{r['median_ms']:.3f}",
            f"{r['p5_ms']:.3f}",
            f"{r['p95_ms']:.3f}",
            f"{r['n_samples']}",
        ])
    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for j in range(len(col_labels)):
        table[0, j].set_text_props(fontweight="bold")
    ax.set_title("Per-Step Inference Time", fontsize=12, fontweight="bold", pad=20)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"inference_time_table.{ext}")
        fig.savefig(path, dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
        print(f"[table] Saved: {path}")
    plt.close(fig)

    return rows


# =====================================================================
#  Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Inference Time Benchmark")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Episodes per method (default: 20)")
    parser.add_argument("--ppo-ckpt", type=str, default=PPO_CKPT)
    parser.add_argument("--mpcrl-ckpt", type=str, default=MPCRL_CKPT)
    parser.add_argument("--output-dir", type=str, default=EVALS_DIR)
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, "experiment3_results")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  Experiment 3: Inference Time Benchmark")
    print(f"  Episodes per method: {args.episodes}")
    print(f"  Output: {output_dir}")
    print(f"{'#'*60}")

    # ── Run benchmarks ──
    print("\n" + "="*60)
    print("  [1/3] Benchmarking Pure RL ...")
    print("="*60)
    pure_rl_times = benchmark_pure_rl(args.episodes, args.ppo_ckpt)

    print("\n" + "="*60)
    print("  [2/3] Benchmarking Pure MPC ...")
    print("="*60)
    pure_mpc_times = benchmark_pure_mpc(args.episodes)

    print("\n" + "="*60)
    print("  [3/3] Benchmarking MPC-RL ...")
    print("="*60)
    mpcrl_rl, mpcrl_mpc, mpcrl_total = benchmark_mpcrl(args.episodes, args.mpcrl_ckpt)

    # ── Collect results ──
    results = {
        "pure_rl": {"step_times": pure_rl_times},
        "pure_mpc": {"step_times": pure_mpc_times},
        "mpcrl": {
            "rl_times": mpcrl_rl,
            "mpc_times": mpcrl_mpc,
            "total_times": mpcrl_total,
        },
    }

    # ── Save raw JSON ──
    json_path = os.path.join(output_dir, "inference_time_results.json")
    json_out = {
        "n_episodes": args.episodes,
        "pure_rl": {
            "n_steps": len(pure_rl_times),
            "mean_ms": float(np.mean(pure_rl_times)) * 1000,
            "std_ms": float(np.std(pure_rl_times)) * 1000,
            "median_ms": float(np.median(pure_rl_times)) * 1000,
            "step_times_ms": [t * 1000 for t in pure_rl_times],
        },
        "pure_mpc": {
            "n_steps": len(pure_mpc_times),
            "mean_ms": float(np.mean(pure_mpc_times)) * 1000,
            "std_ms": float(np.std(pure_mpc_times)) * 1000,
            "median_ms": float(np.median(pure_mpc_times)) * 1000,
            "step_times_ms": [t * 1000 for t in pure_mpc_times],
        },
        "mpcrl": {
            "n_steps": len(mpcrl_total),
            "total_mean_ms": float(np.mean(mpcrl_total)) * 1000,
            "rl_mean_ms": float(np.mean(mpcrl_rl)) * 1000,
            "mpc_mean_ms": float(np.mean(mpcrl_mpc)) * 1000,
            "total_times_ms": [t * 1000 for t in mpcrl_total],
            "rl_times_ms": [t * 1000 for t in mpcrl_rl],
            "mpc_times_ms": [t * 1000 for t in mpcrl_mpc],
        },
    }
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\n[json] Saved: {json_path}")

    # ── Generate plots ──
    make_box_plot(results, output_dir)
    table_rows = make_summary_table(results, output_dir)

    print("\n[DONE] Experiment 3 complete!")


if __name__ == "__main__":
    main()
