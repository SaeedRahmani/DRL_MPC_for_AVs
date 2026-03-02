"""
Generate and visualize the reference trajectory for the merge environment
with **ego on the merge ramp**.

The merge env road layout:
  - Highway lane 0: y=0, x=[0..460], heading=0
  - Highway lane 1: y=4, x=[0..460], heading=0
  - Merge ramp straight: j->k  (0,14.5)->(150,14.5), heading=0
  - Merge ramp curve:    k->b  SineLane (150,14.5)->(230,8)
  - Merge parallel lane:  b->c[2]  (230,8)->(310,8), heading=0
  - Obstacle at x~310, y=8 (end of merge lane)
  - Episode terminates at x > 370 or crash

Ego starts on the ramp at (~110, 14.5), speed=20 m/s, heading=0.

Reference trajectory phases:
  1. j->k  straight ramp
  2. k->b  SineLane curve  (accelerate 20 -> 30)
  3. b->c lane 2  merge parallel  (~50 m)
  4. Lane change  y~8 -> y=4  (smooth S-curve, ~40 m)
  5. Highway lane 1  (y=4, heading=0)

Usage:
    cd /users/saeani/src/mpcrl/MPC-RL_for_AVs/src_ray_version && \\
    PYTHONPATH="$PWD:$PWD/highway-env:$PYTHONPATH" \\
    python /users/saeani/src/mpcrl/DRL_MPC_for_AVs/src_ray_version/evals/merge_reference_trajectory.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gymnasium
import highway_env  # noqa: F401


# ──────────────────────────────────────────────────────────────────────
#  Reference trajectory generator (ramp -> highway)
# ──────────────────────────────────────────────────────────────────────
def generate_merge_ramp_reference(
    road_network,
    ego_position: np.ndarray,
    ego_speed: float = 20.0,
    target_speed: float = 30.0,
    dt: float = 0.1,
) -> np.ndarray:
    """
    Generate a reference trajectory for an ego vehicle merging from the ramp.

    Samples lane centerlines for the ramp, then appends a smooth lane-change
    from y~8 to y=4 and continues straight on the highway.

    Args:
        road_network: The RoadNetwork object from the merge env.
        ego_position: Ego (x, y) at episode start.
        ego_speed:    Ego speed at start (m/s).
        target_speed: Desired highway speed after merge (m/s).
        dt:           MPC time-step (s).

    Returns:
        np.ndarray of shape (N, 4) with columns [x, y, v, heading].
    """
    jk = road_network.get_lane(("j", "k", 0))
    kb = road_network.get_lane(("k", "b", 0))
    bc2 = road_network.get_lane(("b", "c", 2))

    # Find starting arc-length on j->k
    best_s, best_d = 0.0, np.inf
    for s_cand in np.linspace(0, jk.length, 500):
        d = np.linalg.norm(jk.position(s_cand, 0) - ego_position)
        if d < best_d:
            best_s, best_d = s_cand, d

    waypoints = []

    # Phase 1: j->k straight ramp
    s, speed = best_s, ego_speed
    while s < jk.length - 1e-3:
        pos = jk.position(s, 0)
        hdg = jk.heading_at(s)
        waypoints.append([pos[0], pos[1], speed, hdg])
        s += speed * dt

    # Phase 2: k->b SineLane (accelerate ego_speed -> target_speed)
    s = 0.0
    while s < kb.length - 1e-3:
        frac = min(s / kb.length, 1.0)
        speed = ego_speed + (target_speed - ego_speed) * frac
        pos = kb.position(s, 0)
        hdg = kb.heading_at(s)
        waypoints.append([pos[0], pos[1], speed, hdg])
        s += speed * dt

    # Phase 3: b->c lane 2 (merge parallel), first ~50 m
    speed = target_speed
    s = 0.0
    merge_use = 50.0
    while s < merge_use:
        pos = bc2.position(s, 0)
        hdg = bc2.heading_at(s)
        waypoints.append([pos[0], pos[1], speed, hdg])
        s += speed * dt

    n_lane_sampled = len(waypoints)  # exact headings up to here

    # Phase 4: Smooth lane change y~8 -> y=4 over ~40 m
    lc_x0 = waypoints[-1][0]
    lc_y0 = waypoints[-1][1]
    target_y = 4.0
    lc_dist = 40.0
    dx = speed * dt
    n_lc = max(1, int(lc_dist / dx))
    for i in range(1, n_lc + 1):
        t = i / n_lc
        x = lc_x0 + dx * i
        y = lc_y0 + (target_y - lc_y0) * (3 * t**2 - 2 * t**3)
        waypoints.append([x, y, speed, 0.0])  # heading corrected below

    # Phase 5: Continue straight on highway lane 1 (y=4)
    last_x = waypoints[-1][0]
    for i in range(1, 30):
        waypoints.append([last_x + dx * i, target_y, speed, 0.0])

    traj = np.array(waypoints)

    # Fix headings for phases 4-5 via finite differences
    for i in range(max(0, n_lane_sampled - 1), len(traj) - 1):
        ddx = traj[i + 1, 0] - traj[i, 0]
        ddy = traj[i + 1, 1] - traj[i, 1]
        if abs(ddx) > 1e-6 or abs(ddy) > 1e-6:
            traj[i, 3] = np.arctan2(ddy, ddx)
    if len(traj) > 1:
        traj[-1, 3] = traj[-2, 3]

    return traj


# ──────────────────────────────────────────────────────────────────────
#  Visualization
# ──────────────────────────────────────────────────────────────────────
def visualize_reference_trajectory(output_dir: str = "."):
    """Generate plots showing the merge env layout + ramp reference trajectory."""

    # Create env with ego on the merge ramp
    env_config = {
        "ego_on_ramp": True,
        "action": {
            "type": "ContinuousAction",
            "acceleration_range": [-5.0, 5.0],
            "steering_range": [-np.pi / 4, np.pi / 4],
        },
    }
    env = gymnasium.make("merge-v0", render_mode="rgb_array", config=env_config)
    obs, info = env.reset()
    road = env.unwrapped.road
    net = road.network
    ego = env.unwrapped.vehicle

    # Generate reference trajectory
    ref_traj = generate_merge_ramp_reference(
        road_network=net,
        ego_position=ego.position,
        ego_speed=ego.speed,
        target_speed=30.0,
    )

    # Collect road lane centerlines
    lane_lines = {}
    for from_node in net.graph:
        for to_node in net.graph[from_node]:
            for lane_idx, lane_obj in enumerate(net.graph[from_node][to_node]):
                key = f"{from_node}->{to_node}[{lane_idx}]"
                n_pts = max(10, int(lane_obj.length / 2))
                pts = []
                for i in range(n_pts + 1):
                    s_val = lane_obj.length * i / n_pts
                    pts.append(lane_obj.position(s_val, 0))
                lane_lines[key] = np.array(pts)

    # Collect vehicle positions
    vehicles = []
    for v in road.vehicles:
        vehicles.append({
            "pos": v.position.copy(),
            "speed": v.speed,
            "heading": v.heading,
            "lane": v.lane_index,
            "is_ego": v is ego,
        })

    env.close()

    # ── Plot 1: Full road layout + reference trajectory ──
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    ax = axes[0]
    ax.set_title(
        "Merge Environment — Road Layout & Ramp Reference Trajectory (ego merging)",
        fontsize=14, fontweight="bold")

    highway_color = "#aaaaaa"
    ramp_color = "#cc8800"
    for key, pts in lane_lines.items():
        color = ramp_color if "j->" in key or "k->" in key else highway_color
        ls = "--" if "j->" in key or "k->" in key else "-"
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.5,
                linestyle=ls, alpha=0.7, label=key)

    ax.plot(ref_traj[:, 0], ref_traj[:, 1], "b-", linewidth=2.5,
            label="MPC Reference Trajectory", zorder=5)
    for i in range(0, len(ref_traj), 10):
        j = min(i + 1, len(ref_traj) - 1)
        ddx = ref_traj[j, 0] - ref_traj[i, 0]
        ddy = ref_traj[j, 1] - ref_traj[i, 1]
        ax.annotate("", xy=(ref_traj[i, 0] + ddx * 2, ref_traj[i, 1] + ddy * 2),
                     xytext=(ref_traj[i, 0], ref_traj[i, 1]),
                     arrowprops=dict(arrowstyle="->", color="blue", lw=1.5))

    for v in vehicles:
        color = ("red" if v["is_ego"]
                 else ("orange" if "j" in str(v["lane"]) or "k" in str(v["lane"])
                       else "green"))
        marker = "s" if v["is_ego"] else "o"
        label = (f"Ego (v={v['speed']:.0f})" if v["is_ego"]
                 else f"Other (v={v['speed']:.0f})")
        ax.plot(v["pos"][0], v["pos"][1], marker=marker, color=color,
                markersize=10, zorder=10, label=label)

    ax.axvspan(150, 230, alpha=0.05, color="red", label="Converging zone")
    ax.axvspan(230, 310, alpha=0.1, color="orange", label="Merge zone")
    ax.axvline(370, color="gray", linestyle=":", alpha=0.5, label="Terminal (x>370)")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.set_xlim(-10, 480)
    ax.set_ylim(-5, 20)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Reference trajectory speed & heading ──
    ax2 = axes[1]
    color_v, color_h = "tab:blue", "tab:red"
    ax2_v = ax2
    ax2_h = ax2.twinx()

    ax2_v.plot(ref_traj[:, 0], ref_traj[:, 2], color=color_v, linewidth=2,
               label="Reference Speed")
    ax2_v.set_xlabel("x position (m)")
    ax2_v.set_ylabel("Speed (m/s)", color=color_v)
    ax2_v.tick_params(axis="y", labelcolor=color_v)
    ax2_v.set_ylim(0, 35)

    ax2_h.plot(ref_traj[:, 0], np.degrees(ref_traj[:, 3]), color=color_h,
               linewidth=2, linestyle="--", label="Reference Heading")
    ax2_h.set_ylabel("Heading (deg)", color=color_h)
    ax2_h.tick_params(axis="y", labelcolor=color_h)
    ax2_h.set_ylim(-20, 10)

    ax2.set_title("Reference Trajectory — Speed & Heading Profile",
                   fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    lines1, labels1 = ax2_v.get_legend_handles_labels()
    lines2, labels2 = ax2_h.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "merge_reference_trajectory.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[traj] Saved: {out_path}")

    # Demo video
    env = gymnasium.make("merge-v0", render_mode="rgb_array", config=env_config)
    video_dir = os.path.join(output_dir, "merge_env_demo")
    os.makedirs(video_dir, exist_ok=True)
    env = gymnasium.wrappers.RecordVideo(
        env, video_folder=video_dir,
        episode_trigger=lambda ep: True,
        name_prefix="merge_ramp_demo",
    )
    obs, info = env.reset()
    for _ in range(150):
        action = env.action_space.sample() * 0
        obs, rew, term, trunc, info = env.step(action)
        if term or trunc:
            break
    env.close()
    print(f"[traj] Demo video saved to: {video_dir}")

    # Print trajectory stats
    s_vals = np.cumsum(np.sqrt(
        np.diff(ref_traj[:, 0])**2 + np.diff(ref_traj[:, 1])**2))
    s_vals = np.insert(s_vals, 0, 0)
    print(f"\n{'='*50}")
    print(f"  Reference Trajectory Summary (ego merging)")
    print(f"{'='*50}")
    print(f"  Points:     {len(ref_traj)}")
    print(f"  Start:      ({ref_traj[0, 0]:.1f}, {ref_traj[0, 1]:.1f})")
    print(f"  End:        ({ref_traj[-1, 0]:.1f}, {ref_traj[-1, 1]:.1f})")
    print(f"  Speed:      {ref_traj[0, 2]:.1f} -> {ref_traj[-1, 2]:.1f} m/s")
    print(f"  Heading:    {np.degrees(ref_traj[0, 3]):.1f} -> "
          f"{np.degrees(ref_traj[-1, 3]):.1f} deg")
    print(f"  Total dist: {s_vals[-1]:.1f} m")
    print(f"{'='*50}\n")
    return ref_traj


if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    ref = visualize_reference_trajectory(output_dir)
