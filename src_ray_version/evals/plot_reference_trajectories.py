"""
Generate publication-quality reference-trajectory plots for:
  1. Intersection (4-arm, three possible ego manoeuvres)
  2. Merge (ego starting on ramp, merging onto highway)

Uses the actual highway-env lane objects to draw the road network
and overlays the MPC reference trajectories.

Usage:
    cd /users/saeani/src/mpcrl/MPC-RL_for_AVs/src_ray_version && \
    source .venv/bin/activate && \
    PYTHONPATH="$PWD:$PWD/highway-env:$PYTHONPATH" \
    python /users/saeani/src/mpcrl/DRL_MPC_for_AVs/src_ray_version/evals/plot_reference_trajectories.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.patheffects as pe

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gymnasium
import highway_env  # noqa: F401

# ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DPI = 200

# ── Palette ───────────────────────────────────────────────────────────
ROAD_FC     = "#e8e8e8"
ROAD_EC     = "#c0c0c0"
RAMP_FC     = "#fff3d4"
RAMP_EC     = "#d4b870"
TRAJ_BLUE   = "#1a5fb4"
TRAJ_ORANGE = "#c64600"
TRAJ_GREEN  = "#1a8a3e"
EGO_COLOR   = "#cc0000"
BG_COLOR    = "#fafafa"

# Vehicle display size (exaggerated for visibility)
VEH_LENGTH = 24.0
VEH_WIDTH  = 7.5

# Intersection uses equal aspect — smaller vehicles look right
VEH_LENGTH_INT = 10.0
VEH_WIDTH_INT  = 4.0


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════
def deoverlap_vehicles(veh_data, min_gap=2.0):
    """Shift non-ego vehicles along their heading so no pair overlaps.

    veh_data: list of (pos, heading, length, width, is_ego) tuples.
    min_gap:  minimum centre-to-centre distance along heading to enforce.
    Returns a new list with adjusted positions.
    """
    result = list(veh_data)  # shallow copy of tuples
    n = len(result)
    for i in range(n):
        pos_i, hdg_i, l_i, w_i, ego_i = result[i]
        for j in range(i + 1, n):
            pos_j, hdg_j, l_j, w_j, ego_j = result[j]
            dx = pos_j[0] - pos_i[0]
            dy = pos_j[1] - pos_i[1]
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = (l_i + l_j) / 2 + min_gap
            if dist < min_dist:
                # Shift the non-ego vehicle (or j if both non-ego)
                shift = min_dist - dist + 1.0
                if ego_j and not ego_i:
                    target = i
                else:
                    target = j
                pt, ht, lt, wt, et = result[target]
                # Shift along its heading direction
                new_pos = pt.copy()
                new_pos[0] += shift * np.cos(ht)
                new_pos[1] += shift * np.sin(ht)
                result[target] = (new_pos, ht, lt, wt, et)
    return result


def sample_lane(lane, n_pts=200):
    """Return (N, 2) centre-line points."""
    ss = np.linspace(0, lane.length, n_pts)
    return np.array([lane.position(s, 0) for s in ss])


def lane_polygon(lane, half_width=2.0, n_pts=200):
    """Return a closed polygon that traces the lane band (left → right reversed)."""
    ss = np.linspace(0, lane.length, n_pts)
    left  = np.array([lane.position(s, -half_width) for s in ss])
    right = np.array([lane.position(s,  half_width) for s in ss])
    return np.vstack([left, right[::-1]])


def add_arrows(ax, traj, colour, fracs=(0.25, 0.55, 0.80),
               scale=4, lw=1.8, zorder=6):
    """Place direction arrow-heads along a trajectory at given fractions."""
    for f in fracs:
        i = min(int(f * len(traj)), len(traj) - 4)
        dx = traj[i + 3, 0] - traj[i, 0]
        dy = traj[i + 3, 1] - traj[i, 1]
        ax.annotate(
            "", xy=(traj[i, 0] + dx * scale, traj[i, 1] + dy * scale),
            xytext=(traj[i, 0], traj[i, 1]),
            arrowprops=dict(arrowstyle="-|>", color=colour, lw=lw),
            zorder=zorder,
        )


def draw_vehicle(ax, x, y, heading, length=5.0, width=2.0,
                 fc="#5599dd", ec="#336699", alpha=0.85, zorder=8,
                 label=None, aspect_correction=1.0):
    """Draw a single vehicle as a rotated rectangle.

    aspect_correction: ratio  (data_per_pixel_x / data_per_pixel_y).
        When the axes do NOT have equal aspect, y-data units are visually
        larger than x-data units.  Pass the ratio so the rectangle looks
        correct on screen.  For equal-aspect axes pass 1.0.
    """
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    hl, hw = length / 2, width / 2
    corners_local = np.array([
        [-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]
    ])
    # Stretch y-coords so the rectangle *looks* right on a non-equal-aspect plot
    scale = np.array([1.0, 1.0 / aspect_correction])
    R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners = (corners_local * scale) @ R.T + np.array([x, y])
    patch = Polygon(corners, closed=True, fc=fc, ec=ec, lw=0.6,
                    alpha=alpha, zorder=zorder, label=label)
    ax.add_patch(patch)


def draw_vehicles(ax, vehicles, ego_vehicle, aspect_equal=True):
    """Draw ego + traffic vehicles.  ego_vehicle is highlighted in red."""
    labelled_traffic = False
    for v in vehicles:
        if v is ego_vehicle:
            draw_vehicle(ax, v.position[0], v.position[1], v.heading,
                         v.LENGTH, v.WIDTH,
                         fc=EGO_COLOR, ec="#880000", alpha=0.9, zorder=9,
                         label="Ego vehicle")
        else:
            lbl = "Traffic vehicles" if not labelled_traffic else None
            draw_vehicle(ax, v.position[0], v.position[1], v.heading,
                         v.LENGTH, v.WIDTH,
                         fc="#5599dd", ec="#336699", alpha=0.75, zorder=8,
                         label=lbl)
            labelled_traffic = True


# ══════════════════════════════════════════════════════════════════════
#  1.  INTERSECTION
# ══════════════════════════════════════════════════════════════════════
def plot_intersection_reference(output_dir):
    env = gymnasium.make("intersection-v1", render_mode=None)
    obs, _ = env.reset()
    net = env.unwrapped.road.network
    ego = env.unwrapped.vehicle

    # Collect lanes
    approach, turning, exit_l, all_objs = {}, {}, {}, {}
    for fn in net.graph:
        for tn in net.graph[fn]:
            for li, lane in enumerate(net.graph[fn][tn]):
                key = f"{fn}->{tn}[{li}]"
                pts = sample_lane(lane)
                all_objs[key] = lane
                if fn.startswith("o"):
                    approach[key] = pts
                elif fn.startswith("ir"):
                    turning[key] = pts
                elif fn.startswith("il"):
                    exit_l[key] = pts

    # Three reference trajectories
    ea  = net.get_lane(("o0", "ir0", 0))
    eth = net.get_lane(("ir0", "il2", 0))
    exi = net.get_lane(("il2", "o2", 0))
    ref_straight = np.vstack([sample_lane(ea, 100), sample_lane(eth, 60),
                               sample_lane(exi, 100)])

    rt = net.get_lane(("ir0", "il3", 0))
    lt = net.get_lane(("ir0", "il1", 0))
    ref_right = np.vstack([sample_lane(ea, 100), sample_lane(rt, 40),
                            sample_lane(net.get_lane(("il3", "o3", 0)), 100)])
    ref_left  = np.vstack([sample_lane(ea, 100), sample_lane(lt, 40),
                            sample_lane(net.get_lane(("il1", "o1", 0)), 100)])

    # Capture vehicle states before closing
    all_vehicles = list(env.unwrapped.road.vehicles)
    veh_data = [(v.position.copy(), v.heading, VEH_LENGTH_INT, VEH_WIDTH_INT, v is ego)
                for v in all_vehicles]

    # ── Hand-place extra traffic ──
    extra_lane_positions = [
        # West arm
        (("o1", "ir1", 0), 0.45),
        # North arm
        (("o2", "ir2", 0), 0.40),
        # Inside intersection – one turning
        (("ir1", "il3", 0), 0.50),
    ]
    for lane_idx, frac in extra_lane_positions:
        try:
            lane = net.get_lane(lane_idx)
            s = frac * lane.length
            pos = lane.position(s, 0)
            hdg = lane.heading_at(s)
            veh_data.append((pos.copy(), hdg, VEH_LENGTH_INT, VEH_WIDTH_INT, False))
        except Exception:
            pass  # lane might not exist in some configs

    veh_data = deoverlap_vehicles(veh_data)
    env.close()

    # ── Draw ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")
    ax.set_facecolor(BG_COLOR)

    # Road bands
    for key, lane in all_objs.items():
        is_turn = key.split("->")[0].startswith("ir")
        fc = "#f0f0f0" if is_turn else ROAD_FC
        ec = "#dddddd" if is_turn else ROAD_EC
        p = Polygon(lane_polygon(lane, 2.0, 100), closed=True,
                    fc=fc, ec=ec, lw=0.5, zorder=1)
        ax.add_patch(p)

    # Centre dashes
    for pts in {**approach, **exit_l}.values():
        ax.plot(pts[:, 0], pts[:, 1], color="#d0d0d0", lw=0.6, ls="--", zorder=2)

    # Intersection box
    ax.add_patch(Rectangle((-12, -12), 24, 24, lw=1.0, ec="#aaaaaa",
                            fc="#f2f2f2", zorder=1, alpha=0.7))

    # Trajectories
    white_stroke = [pe.Stroke(linewidth=3.5, foreground="white"), pe.Normal()]
    ax.plot(ref_left[:, 0], ref_left[:, 1], color=TRAJ_ORANGE, lw=2.2,
            ls="--", alpha=0.8, label="Left turn", zorder=4,
            path_effects=white_stroke)

    add_arrows(ax, ref_left,     TRAJ_ORANGE, fracs=(0.62,))

    # Draw vehicles
    labelled_traffic = False
    for pos, hdg, length, width, is_ego in veh_data:
        if is_ego:
            draw_vehicle(ax, pos[0], pos[1], hdg, length, width,
                         fc=EGO_COLOR, ec="#880000", alpha=0.9, zorder=9,
                         label="Ego vehicle")
        else:
            lbl = "Traffic vehicles" if not labelled_traffic else None
            draw_vehicle(ax, pos[0], pos[1], hdg, length, width,
                         fc="#5599dd", ec="#336699", alpha=0.75, zorder=8,
                         label=lbl)
            labelled_traffic = True

    # Arm labels  (y-axis inverted: high y = bottom = South)
    bx = dict(boxstyle="round,pad=0.25", fc="white", ec="#bbbbbb",
              alpha=0.9, lw=0.6)
    ax.text(2,    128,  "South (Ego)", fontsize=9.5, fontweight="bold",
            ha="center", va="center", bbox=bx)
    ax.text(-128, 0,    "West",        fontsize=9.5, fontweight="bold",
            ha="center", va="center", bbox=bx)
    ax.text(2,   -128,  "North",       fontsize=9.5, fontweight="bold",
            ha="center", va="center", bbox=bx)
    ax.text(128,  0,    "East",        fontsize=9.5, fontweight="bold",
            ha="center", va="center", bbox=bx)

    ax.set_xlim(-145, 145)
    ax.set_ylim(145, -145)   # INVERTED: high y at bottom (South), low y at top (North)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=11)
    ax.set_ylabel("y (m)", fontsize=11)
    ax.set_title("Intersection — MPC Reference Trajectories",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95,
              edgecolor="#cccccc", fancybox=True)
    ax.grid(True, alpha=0.15, lw=0.5)
    ax.tick_params(labelsize=9)

    out = os.path.join(output_dir, "intersection_reference_trajectory.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════
#  2.  MERGE  (ego on ramp)
# ══════════════════════════════════════════════════════════════════════
def generate_merge_ref(net, ego_pos, ego_speed=20.0, target_speed=30.0, dt=0.1):
    """Return (N,4) array  [x, y, speed, heading]."""
    jk  = net.get_lane(("j", "k", 0))
    kb  = net.get_lane(("k", "b", 0))
    bc2 = net.get_lane(("b", "c", 2))

    # Find starting arc-length on j→k
    best_s, best_d = 0.0, np.inf
    for s in np.linspace(0, jk.length, 500):
        d = np.linalg.norm(jk.position(s, 0) - ego_pos)
        if d < best_d:
            best_s, best_d = s, d

    wps = []
    # Phase 1 – ramp straight (j→k)
    s, speed = best_s, ego_speed
    while s < jk.length - 0.5:
        p = jk.position(s, 0)
        wps.append([p[0], p[1], speed, jk.heading_at(s)])
        s += speed * dt

    # Phase 2 – SineLane (k→b), accelerate
    s = 0.0
    while s < kb.length - 0.5:
        frac = min(s / kb.length, 1.0)
        speed = ego_speed + (target_speed - ego_speed) * frac
        p = kb.position(s, 0)
        wps.append([p[0], p[1], speed, kb.heading_at(s)])
        s += speed * dt

    # Phase 3 – merge lane (b→c[2]) ~50 m
    speed = target_speed
    s = 0.0
    while s < 50:
        p = bc2.position(s, 0)
        wps.append([p[0], p[1], speed, bc2.heading_at(s)])
        s += speed * dt

    # Phase 4 – lane-change y=8 → y=4
    x0, y0 = wps[-1][0], wps[-1][1]
    lc_dist, dx = 40.0, speed * dt
    n_lc = max(1, int(lc_dist / dx))
    for i in range(1, n_lc + 1):
        t = i / n_lc
        wps.append([x0 + dx * i,
                     y0 + (4.0 - y0) * (3 * t**2 - 2 * t**3), speed, 0.0])

    # Phase 5 – cruise on highway lane 1
    for i in range(1, 20):
        wps.append([wps[-1][0] + dx * i, 4.0, speed, 0.0])

    traj = np.array(wps)
    for i in range(len(traj) - 1):
        ddx = traj[i + 1, 0] - traj[i, 0]
        ddy = traj[i + 1, 1] - traj[i, 1]
        if abs(ddx) > 1e-6 or abs(ddy) > 1e-6:
            traj[i, 3] = np.arctan2(ddy, ddx)
    traj[-1, 3] = traj[-2, 3]
    return traj


def plot_merge_reference(output_dir):
    env = gymnasium.make("merge-v0", render_mode=None,
                         config={"ego_on_ramp": True})
    obs, _ = env.reset()
    net = env.unwrapped.road.network
    ego = env.unwrapped.vehicle
    ref_traj = generate_merge_ref(net, ego.position, ego.speed)

    # Collect lane objects
    lane_objs = {}
    for fn in net.graph:
        for tn in net.graph[fn]:
            for li, lane in enumerate(net.graph[fn][tn]):
                lane_objs[f"{fn}->{tn}[{li}]"] = lane

    # Capture vehicle states before closing
    all_vehicles = list(env.unwrapped.road.vehicles)
    veh_data_merge = [(v.position.copy(), v.heading, VEH_LENGTH, VEH_WIDTH, v is ego)
                      for v in all_vehicles]

    # ── Hand-place extra traffic, focused near converging/merge ──
    extra_merge_positions = [
        # Highway lane 0 (y=0)
        (("b", "c", 0), 0.40),
        # Highway lane 1 (y=4) – near merge area
        (("b", "c", 1), 0.50),   # converging zone
        (("c", "d", 1), 0.25),   # just past merge
        # On the SineLane (converging)
        (("k", "b", 0), 0.55),
    ]
    for lane_idx, frac in extra_merge_positions:
        try:
            lane = net.get_lane(lane_idx)
            s = frac * lane.length
            pos = lane.position(s, 0)
            hdg = lane.heading_at(s)
            veh_data_merge.append((pos.copy(), hdg, VEH_LENGTH, VEH_WIDTH, False))
        except Exception:
            pass

    veh_data_merge = deoverlap_vehicles(veh_data_merge)
    env.close()

    highway_keys = [k for k in lane_objs if k[0] in "abcd"]
    ramp_keys    = [k for k in lane_objs if k[0] in "jk"]

    # ── Draw ──────────────────────────────────────────────────────
    # 14×5  (wide but tall enough; NO equal aspect)
    fig, ax = plt.subplots(figsize=(14, 5), facecolor="white")
    ax.set_facecolor(BG_COLOR)

    # Road bands – filled polygons
    for key in highway_keys:
        poly = lane_polygon(lane_objs[key], half_width=2.0, n_pts=200)
        ax.add_patch(Polygon(poly, closed=True, fc=ROAD_FC, ec=ROAD_EC,
                             lw=0.4, zorder=1))
    for key in ramp_keys:
        poly = lane_polygon(lane_objs[key], half_width=2.0, n_pts=200)
        ax.add_patch(Polygon(poly, closed=True, fc=RAMP_FC, ec=RAMP_EC,
                             lw=0.4, zorder=1))

    # Centre dashes
    for lane in lane_objs.values():
        pts = sample_lane(lane, 200)
        ax.plot(pts[:, 0], pts[:, 1], color="#c8c8c8", lw=0.5, ls="--",
                zorder=2)

    # ── Zone shading ──────────────────────────────────────────────
    ax.axvspan(150, 230, alpha=0.05, color="#cc4400", zorder=0)
    ax.axvspan(230, 310, alpha=0.07, color="#cc6600", zorder=0)

    # Zone labels – INSIDE the figure (near top)
    zbox = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
    ax.text(190, 19, "Converging zone", fontsize=8.5, ha="center",
            va="center", color="#cc4400", fontstyle="italic", bbox=zbox)
    ax.text(270, 19, "Merge zone", fontsize=8.5, ha="center",
            va="center", color="#cc6600", fontstyle="italic", bbox=zbox)

    # Terminal line
    ax.axvline(370, color="#888888", ls=":", alpha=0.5, lw=1.0, zorder=2)
    ax.text(373, 19, "Terminal", fontsize=7.5, color="#888888",
            va="center", fontstyle="italic")

    # ── Lane labels (left-aligned, well-separated) ────────────────
    lbl = dict(fontsize=7.5, ha="left", va="center", color="#555555",
               bbox=dict(boxstyle="round,pad=0.15", fc="white",
                         ec="none", alpha=0.8))
    ax.text(15, 0,    "Lane 0  (y = 0)",        **lbl)
    ax.text(15, 4,    "Lane 1  (y = 4)",        **lbl)
    ax.text(235, 8,   "Merge lane  (y = 8)",    **lbl)
    ax.text(15, 14.5, "On-ramp  (y = 14.5)",    **lbl)

    # ── Reference trajectory ──────────────────────────────────────
    ax.plot(ref_traj[:, 0], ref_traj[:, 1], color=TRAJ_BLUE, lw=2.8,
            label="MPC Reference Trajectory", zorder=5,
            path_effects=[pe.Stroke(linewidth=4.5, foreground="white"),
                          pe.Normal()])
    add_arrows(ax, ref_traj, TRAJ_BLUE,
               fracs=(0.10, 0.30, 0.50, 0.72, 0.90),
               scale=3, lw=1.6)

    # Compute aspect correction BEFORE drawing vehicles
    # (x-range / fig-width) / (y-range / fig-height)
    ax.set_xlim(-15, 460)
    ax.set_ylim(-4, 22)
    fig.canvas.draw()  # force layout so get_position works
    bbox = ax.get_position()
    fig_w, fig_h = fig.get_size_inches()
    ax_w = bbox.width * fig_w    # axes width in inches
    ax_h = bbox.height * fig_h   # axes height in inches
    x_range = 460 - (-15)        # data units
    y_range = 22 - (-4)
    asp_corr = (x_range / ax_w) / (y_range / ax_h)

    # Draw vehicles
    labelled_traffic_m = False
    for pos, hdg, length, width, is_ego in veh_data_merge:
        if is_ego:
            draw_vehicle(ax, pos[0], pos[1], hdg, length, width,
                         fc=EGO_COLOR, ec="#880000", alpha=0.9, zorder=9,
                         label="Ego vehicle", aspect_correction=asp_corr)
        else:
            lbl = "Traffic vehicles" if not labelled_traffic_m else None
            draw_vehicle(ax, pos[0], pos[1], hdg, length, width,
                         fc="#5599dd", ec="#336699", alpha=0.75, zorder=8,
                         label=lbl, aspect_correction=asp_corr)
            labelled_traffic_m = True

    # ── Axis setup ────────────────────────────────────────────────
    # xlim/ylim already set above for aspect correction
    # Deliberately NOT equal aspect — the road is ~475 m long but only
    # ~18 m wide; equal aspect would crush it into an unreadable strip.
    ax.set_xlabel("x (m)", fontsize=11)
    ax.set_ylabel("y (m)", fontsize=11)
    ax.set_title("Merge — MPC Reference Trajectory (Ego on Ramp)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95,
              edgecolor="#cccccc", fancybox=True)
    ax.grid(True, alpha=0.15, lw=0.5)
    ax.tick_params(labelsize=9)

    out = os.path.join(output_dir, "merge_reference_trajectory.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    plot_intersection_reference(OUTPUT_DIR)
    plot_merge_reference(OUTPUT_DIR)
    print("\nDone — both plots saved.")
