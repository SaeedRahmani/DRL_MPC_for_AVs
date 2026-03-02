"""
Zero-shot evaluate MPCRL (intersection-trained) on merge-v0 with ego on ramp.

The MPCRL agent was trained on the intersection env where:
  - Observation: 86D = 80 (kinematics) + 6 (MPC context features)
  - Action: 1D scalar in [-1, 1] mapped to [0, MAX_REF_SPEED=15] m/s
  - The MPC solver tracks the RL-recommended speed along a reference trajectory

For zero-shot transfer to merging we:
  1. Put ego on the merge ramp (ego_on_ramp=True)
  2. Construct the same 86D observation (kinematics + MPC features)
  3. Feed it to the trained policy to get a reference speed
  4. Run the same CasADi MPC solver with a merge-ramp reference trajectory
  5. Apply the MPC output as [acc, steer] to the merge env

Usage:
    cd /users/saeani/src/mpcrl/MPC-RL_for_AVs/src_ray_version && \\
    PYTHONPATH="$PWD:$PWD/highway-env:$PYTHONPATH" \\
    python /users/saeani/src/mpcrl/DRL_MPC_for_AVs/src_ray_version/evals/batch_eval_merge_mpcrl.py \\
      --checkpoint <path> --model manual --episodes 100
"""

import os
import sys
import json
import argparse
import time
import warnings
import numpy as np
import gymnasium
from shapely import LineString
from shapely.errors import GEOSException

warnings.filterwarnings("ignore", message="invalid value encountered in intersection")

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import highway_env  # noqa: F401

try:
    import casadi as ca
except ImportError:
    raise ImportError("casadi required: pip install casadi")


# ──────────────────────────────────────────────────────────────────────
#  Constants (must match intersection MPC env)
# ──────────────────────────────────────────────────────────────────────
VEHICLE_LENGTH = 5.0
VEHICLE_LENGTH_REAR = 2.5
MAX_REF_SPEED = 15.0
PROXIMITY_RANGE = 30.0

# Normalization ranges for kinematic features (same as training env)
FEATURE_RANGES = {
    0: (0.0, 1.0),          # presence
    1: (-100.0, 100.0),     # x
    2: (-100.0, 100.0),     # y
    3: (-20.0, 20.0),       # vx
    4: (-20.0, 20.0),       # vy
    5: (-np.pi, np.pi),     # heading
    6: (-1.0, 1.0),         # sin_h
    7: (-1.0, 1.0),         # cos_h
}


# ──────────────────────────────────────────────────────────────────────
#  Reference trajectory generator (ramp -> highway)
# ──────────────────────────────────────────────────────────────────────
def generate_merge_ramp_reference(
    road_network, ego_position, ego_speed=20.0, target_speed=30.0, dt=0.1,
) -> np.ndarray:
    """Reference trajectory following merge ramp -> lane change -> highway."""
    jk = road_network.get_lane(("j", "k", 0))
    kb = road_network.get_lane(("k", "b", 0))
    bc2 = road_network.get_lane(("b", "c", 2))

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

    # Phase 2: k->b SineLane (ego_speed -> target_speed)
    s = 0.0
    while s < kb.length - 1e-3:
        frac = min(s / kb.length, 1.0)
        speed = ego_speed + (target_speed - ego_speed) * frac
        pos = kb.position(s, 0)
        hdg = kb.heading_at(s)
        waypoints.append([pos[0], pos[1], speed, hdg])
        s += speed * dt

    # Phase 3: b->c lane 2, first ~50 m
    speed = target_speed
    s = 0.0
    while s < 50.0:
        pos = bc2.position(s, 0)
        hdg = bc2.heading_at(s)
        waypoints.append([pos[0], pos[1], speed, hdg])
        s += speed * dt

    n_lane_sampled = len(waypoints)

    # Phase 4: Lane change y~8 -> y=4 over ~40 m
    lc_x0 = waypoints[-1][0]
    lc_y0 = waypoints[-1][1]
    target_y = 4.0
    dx = speed * dt
    n_lc = max(1, int(40.0 / dx))
    for i in range(1, n_lc + 1):
        t = i / n_lc
        x = lc_x0 + dx * i
        y = lc_y0 + (target_y - lc_y0) * (3 * t**2 - 2 * t**3)
        waypoints.append([x, y, speed, 0.0])

    # Phase 5: Continue straight y=4
    last_x = waypoints[-1][0]
    for i in range(1, 30):
        waypoints.append([last_x + dx * i, target_y, speed, 0.0])

    traj = np.array(waypoints)

    for i in range(max(0, n_lane_sampled - 1), len(traj) - 1):
        ddx = traj[i + 1, 0] - traj[i, 0]
        ddy = traj[i + 1, 1] - traj[i, 1]
        if abs(ddx) > 1e-6 or abs(ddy) > 1e-6:
            traj[i, 3] = np.arctan2(ddy, ddx)
    if len(traj) > 1:
        traj[-1, 3] = traj[-2, 3]

    return traj


# ──────────────────────────────────────────────────────────────────────
#  86-D Observation builder  (matches _augment_obs in training env)
# ──────────────────────────────────────────────────────────────────────
def build_mpcrl_obs(
    raw_obs: np.ndarray,
    ego_speed: float,
    ego_index: int,
    n_ref_points: int,
    is_collide: bool,
    conflict_indices: list,
    n_vehicles_count: int = 10,
    time_remaining: float = 1.0,
) -> np.ndarray:
    """
    Build the 86-D observation vector matching the MPCRL training format.

    The 6 MPC-context features (matching intersection_mpcrl_speeds_env._augment_obs):
      1. ego_speed / MAX_REF_SPEED  (clipped to [0, 2])
      2. ego_progress  = ego_index / (n_ref_points - 1)
      3. min_dist_norm  (0 = touching, 1 = far)
      4. min_conflict_dist  (normalized distance to nearest conflict point)
      5. n_vehicles_norm  (visible vehicles / max)
      6. time_remaining  (1 - elapsed / duration)
    """
    # Normalize kinematics (same formula as training env).
    # In training (intersection), absolute coords stay within features_range so
    # normalized values are in [-1, 1].  In merge, absolute x reaches 110-370+
    # which normalizes to 1.1-3.7 — far out of distribution.  Clip to [-1, 1]
    # to prevent neural-network overflow / NaN.
    norm = raw_obs.copy().astype(np.float32)
    norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
    for col, (lo, hi) in FEATURE_RANGES.items():
        if hi > lo:
            norm[:, col] = 2.0 * (norm[:, col] - lo) / (hi - lo) - 1.0
    flat_kin = np.clip(norm.flatten(), -1.0, 1.0)  # (80,) — clip for OOD safety

    # Feature 1: ego speed / MAX_REF_SPEED
    ego_speed_norm = np.clip(ego_speed / MAX_REF_SPEED, 0.0, 2.0)

    # Feature 2: progress along reference
    ego_progress = float(ego_index) / max(n_ref_points - 1, 1)

    # Feature 3: min distance to nearest other vehicle
    min_dist_norm = 1.0
    for i in range(1, raw_obs.shape[0]):
        if raw_obs[i, 0] > 0.5:
            d = np.sqrt((raw_obs[0, 1] - raw_obs[i, 1])**2 +
                        (raw_obs[0, 2] - raw_obs[i, 2])**2)
            min_dist_norm = min(min_dist_norm, d / PROXIMITY_RANGE)
    min_dist_norm = float(np.clip(min_dist_norm, 0.0, 1.0))

    # Feature 4: min conflict distance along trajectory
    min_conflict_dist = 1.0
    if is_collide and conflict_indices:
        valid = [idx for idx in conflict_indices if idx is not None]
        if valid:
            dists_along = [abs(idx - ego_index) for idx in valid]
            min_conflict_dist = float(min(dists_along)) / max(n_ref_points, 1)
            min_conflict_dist = np.clip(min_conflict_dist, 0.0, 1.0)

    # Feature 5: number of visible vehicles (normalized)
    n_visible = int(np.sum(raw_obs[1:, 0]))
    n_vehicles_norm = n_visible / max(n_vehicles_count - 1, 1)

    # Feature 6: time remaining
    time_remaining = float(np.clip(time_remaining, 0.0, 1.0))

    mpc_features = np.array([
        ego_speed_norm,
        ego_progress,
        min_dist_norm,
        min_conflict_dist,
        n_vehicles_norm,
        time_remaining,
    ], dtype=np.float32)

    obs_86 = np.concatenate([flat_kin, mpc_features])
    return np.nan_to_num(obs_86, nan=0.0, posinf=1.0, neginf=-1.0)


# ──────────────────────────────────────────────────────────────────────
#  Collision detection (trajectory-based, from intersection MPC env)
# ──────────────────────────────────────────────────────────────────────
def check_collision_simple(ego_pos, ego_speed, ego_heading,
                           other_vehicles, ref_traj, dt=0.1):
    """Simplified trajectory-based collision prediction."""
    PREDICTION_HORIZON = 30
    ref_xy = ref_traj[:, :2]

    ego_future = []
    pos = ego_pos.copy()
    for _ in range(PREDICTION_HORIZON):
        pos = pos + ego_speed * dt * np.array([
            np.cos(ego_heading), np.sin(ego_heading)])
        ego_future.append(pos.copy())

    if len(ego_future) < 2:
        return False, []

    try:
        ego_path = LineString([ego_pos] + ego_future)
    except GEOSException:
        return False, []

    is_collide = False
    conflict_indices = []

    for other in other_vehicles:
        other_future = []
        opos = other["pos"].copy()
        for _ in range(PREDICTION_HORIZON):
            opos = opos + other["speed"] * dt * np.array([
                np.cos(other["heading"]), np.sin(other["heading"])])
            other_future.append(opos.copy())

        if len(other_future) < 2:
            continue
        try:
            other_path = LineString([other["pos"]] + other_future)
            intersection = ego_path.intersection(other_path)
            if not intersection.is_empty:
                is_collide = True
                if intersection.geom_type == "Point":
                    pt = np.array([intersection.x, intersection.y])
                else:
                    centroid = intersection.centroid
                    pt = np.array([centroid.x, centroid.y])
                idx = int(np.argmin(np.linalg.norm(ref_xy - pt, axis=1)))
                conflict_indices.append(idx)
        except GEOSException:
            continue

    return is_collide, conflict_indices


# ──────────────────────────────────────────────────────────────────────
#  MPC solver (same as pure MPC script + collision-aware speed weight)
# ──────────────────────────────────────────────────────────────────────
class StandaloneMPC:
    def __init__(self, horizon=16, dt=0.1):
        self.horizon = horizon
        self.dt = dt
        self.weights = {
            "weight_state": 10, "weight_speed": 1,
            "weight_control": 1, "weight_input_diff": 1,
            "weight_final_state": 1,
        }
        self.prev_solution = None

    def solve(self, ego_x, ego_y, ego_heading, ego_speed,
              reference, ego_index, is_collide=False, ca_mode="manual"):
        N = self.horizon
        n_states, n_controls = 4, 2

        x = ca.SX.sym("x", n_states, N + 1)
        u = ca.SX.sym("u", n_controls, N)

        def vehicle_model(state, ctrl):
            beta = ca.atan(VEHICLE_LENGTH_REAR / VEHICLE_LENGTH * ca.tan(ctrl[1]))
            return ca.vertcat(
                state[3] * ca.cos(state[2] + beta),
                state[3] * ca.sin(state[2] + beta),
                (state[3] / VEHICLE_LENGTH) * ca.sin(beta),
                ctrl[0],
            )

        state_cost, control_cost, input_diff_cost = 0, 0, 0
        closest = ego_index

        for k in range(N):
            ref_idx = min(closest + k, reference.shape[0] - 1)
            dx = x[0, k] - reference[ref_idx, 0]
            dy = x[1, k] - reference[ref_idx, 1]
            ref_v = reference[ref_idx, 2]
            ref_h = reference[ref_idx, 3]

            perp = dx * ca.sin(ref_h) - dy * ca.cos(ref_h)
            para = dx * ca.cos(ref_h) + dy * ca.sin(ref_h)

            speed_w = 100 if (is_collide and ca_mode == "manual") else self.weights["weight_speed"]

            state_cost += (
                4 * perp**2 + 2 * para**2 +
                speed_w * (x[3, k] - ref_v)**2 +
                0.5 * (x[2, k] - ref_h)**2)
            control_cost += 0.01 * u[0, k]**2 + 0.01 * u[1, k]**2
            if k > 0:
                input_diff_cost += 0.01 * (
                    (u[0, k] - u[0, k-1])**2 + (u[1, k] - u[1, k-1])**2)

        ref_final = min(closest + N, reference.shape[0] - 1)
        desired = reference[ref_final]
        final_state_cost = 100 * (
            (x[0, -1] - desired[0])**2 + (x[1, -1] - desired[1])**2 +
            20 * (x[3, -1] - desired[2])**2 + (x[2, -1] - desired[3])**2)

        total_cost = (
            state_cost * self.weights["weight_state"] +
            control_cost * self.weights["weight_control"] +
            input_diff_cost * self.weights["weight_input_diff"] +
            final_state_cost * self.weights["weight_final_state"])

        g = []
        state0 = np.array([ego_x, ego_y, ego_heading, ego_speed])
        g.append(x[:, 0] - state0)
        for k in range(N):
            x_next = x[:, k] + vehicle_model(x[:, k], u[:, k]) * self.dt
            g.append(x[:, k + 1] - x_next)
        g = ca.vertcat(*g)

        x0_states = np.tile(state0, (N + 1, 1)).flatten()
        if self.prev_solution is not None:
            u0 = np.vstack([self.prev_solution[1:],
                            self.prev_solution[-1]]).flatten()
        else:
            u0 = np.zeros(n_controls * N)
        x0 = np.concatenate((x0_states, u0))

        lbg = [0] * g.size1()
        ubg = [0] * g.size1()
        lbx, ubx = [], []
        for _ in range(N + 1):
            lbx += [-500, -500, -ca.pi, 0]
            ubx += [500, 500, ca.pi, 30]
        for _ in range(N):
            lbx += [-5, -ca.pi / 3]
            ubx += [5, ca.pi / 3]

        opt_vars = ca.vertcat(ca.reshape(x, -1, 1), ca.reshape(u, -1, 1))
        nlp = {"x": opt_vars, "f": total_cost, "g": g}
        opts = {"ipopt.print_level": 0, "print_time": 0,
                "ipopt.max_iter": 150, "ipopt.tol": 1e-6}
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        u_opt = sol["x"][-N * n_controls:].full().reshape(N, n_controls)
        self.prev_solution = u_opt
        return np.array([u_opt[0, 0], u_opt[0, 1]])


# ──────────────────────────────────────────────────────────────────────
#  Update reference with CA  (mirrors update_reference_states)
# ──────────────────────────────────────────────────────────────────────
def update_reference_with_ca(ref, ego_index, ego_speed,
                             is_collide, conflict_indices,
                             rl_speed=None):
    """Apply CA deceleration profile, then clamp with RL speed."""
    new_ref = ref.copy()
    SAFETY_BUFFER = 5

    if is_collide and conflict_indices:
        valid = [idx for idx in conflict_indices if idx is not None]
        if valid:
            earliest = min(valid)
            stop_idx = max(ego_index + 1, earliest - SAFETY_BUFFER)
            stop_idx = min(stop_idx, len(ref) - 1)
            pts = stop_idx - ego_index
            if pts > 0:
                decel = np.linspace(ego_speed, 0, pts)
                new_ref[ego_index:stop_idx, 2] = decel
                new_ref[stop_idx:, 2] = 0.0

    if rl_speed is not None:
        ca_speeds = new_ref[:, 2]
        new_ref[:, 2] = np.minimum(rl_speed, ca_speeds)

    return new_ref


# ──────────────────────────────────────────────────────────────────────
#  Evaluation
# ──────────────────────────────────────────────────────────────────────
def run_evaluation(checkpoint_path, model_type="manual",
                   n_episodes=100, output_dir=".", record_video=True):
    from ray.rllib.policy.policy import Policy
    import torch

    model_name = f"mpcrl_{model_type}_merge"
    video_dir = os.path.join(output_dir, f"videos_{model_name}")

    print(f"\n{'='*60}")
    print(f"  Model:      MPCRL {model_type} (zero-shot, ego merging)")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Episodes:   {n_episodes}")
    print(f"  Output:     {output_dir}")
    print(f"{'='*60}\n")

    # Load trained policy on CPU to avoid GPU conflicts with training
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    policy = Policy.from_checkpoint(checkpoint_path)
    if isinstance(policy, dict):
        policy = policy["default_policy"]

    env_config = {
        "ego_on_ramp": True,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy",
                         "heading", "sin_h", "cos_h"],
            "features_range": {
                "x": [-100, 100], "y": [-100, 100],
                "vx": [-20, 20], "vy": [-20, 20],
                "heading": [-np.pi, np.pi],
                "sin_h": [-1, 1], "cos_h": [-1, 1],
            },
            "absolute": True,
            "normalize": False,
            "flatten": False,
            "order": "sorted",
        },
        "action": {
            "type": "ContinuousAction",
            "acceleration_range": [-5.0, 5.0],
            "steering_range": [-np.pi / 4, np.pi / 4],
            "longitudinal": True,
            "lateral": True,
            "dynamical": True,
        },
        "duration": 20,
        "policy_frequency": 10,
        "simulation_frequency": 30,
        "normalize_reward": False,
    }

    env = gymnasium.make("merge-v0", render_mode="rgb_array", config=env_config)
    if record_video:
        os.makedirs(video_dir, exist_ok=True)
        env = gymnasium.wrappers.RecordVideo(
            env, video_folder=video_dir,
            episode_trigger=lambda ep: ep < 10,
            name_prefix=f"eval_{model_name}",
        )

    mpc = StandaloneMPC(horizon=16, dt=0.1)
    use_ca = (model_type == "manual")

    all_rewards, all_lengths, all_crashed, all_arrived = [], [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset()
        mpc.prev_solution = None

        ego = env.unwrapped.vehicle
        ref_traj = generate_merge_ramp_reference(
            road_network=env.unwrapped.road.network,
            ego_position=ego.position,
            ego_speed=ego.speed,
            target_speed=30.0,
        )
        ref_xy = ref_traj[:, :2]

        ep_reward, ep_len, done = 0.0, 0, False
        duration = env_config.get("duration", 20)
        step_count = 0
        policy_freq = env_config.get("policy_frequency", 10)

        while not done:
            ego = env.unwrapped.vehicle

            # Raw observation: ensure (10, 8) shape
            if obs.ndim == 1:
                raw_obs = obs.reshape(10, 8) if len(obs) == 80 else obs.reshape(-1, 8)
            else:
                raw_obs = obs

            ego_idx = int(np.argmin(
                np.linalg.norm(ref_xy - ego.position, axis=1)))

            # Collision detection
            is_collide, conflict_indices = False, []
            if use_ca:
                other_vehicles = []
                for i in range(1, raw_obs.shape[0]):
                    if raw_obs[i, 0] > 0.5:
                        other_vehicles.append({
                            "pos": raw_obs[i, 1:3].copy(),
                            "speed": np.sqrt(raw_obs[i, 3]**2 + raw_obs[i, 4]**2),
                            "heading": raw_obs[i, 5],
                        })
                is_collide, conflict_indices = check_collision_simple(
                    ego.position, ego.speed, ego.heading,
                    other_vehicles, ref_traj)

            # Time remaining
            elapsed = step_count / policy_freq
            time_remaining = np.clip(1.0 - elapsed / duration, 0.0, 1.0)

            # Build 86-D observation
            mpcrl_obs = build_mpcrl_obs(
                raw_obs=raw_obs,
                ego_speed=ego.speed,
                ego_index=ego_idx,
                n_ref_points=len(ref_traj),
                is_collide=is_collide,
                conflict_indices=conflict_indices,
                n_vehicles_count=10,
                time_remaining=time_remaining,
            )

            # Get RL action (reference speed)
            try:
                rl_action = policy.compute_single_action(
                    mpcrl_obs, explore=False)[0]
                raw_rl = (float(rl_action[0])
                          if hasattr(rl_action, '__len__')
                          else float(rl_action))
                if not np.isfinite(raw_rl):
                    raw_rl = 0.0
            except Exception:
                raw_rl = 0.0  # fallback: mid-range speed
            rl_ref_speed = np.clip((raw_rl + 1.0) / 2.0 * MAX_REF_SPEED,
                                    0.0, MAX_REF_SPEED)

            # Update reference trajectory with RL speed + CA
            if use_ca:
                ref_for_mpc = update_reference_with_ca(
                    ref_traj, ego_idx, ego.speed,
                    is_collide, conflict_indices,
                    rl_speed=rl_ref_speed)
            else:
                ref_for_mpc = ref_traj.copy()
                ref_for_mpc[:, 2] = rl_ref_speed

            # Solve MPC
            try:
                mpc_action = mpc.solve(
                    ego_x=ego.position[0], ego_y=ego.position[1],
                    ego_heading=ego.heading, ego_speed=ego.speed,
                    reference=ref_for_mpc, ego_index=ego_idx,
                    is_collide=is_collide,
                    ca_mode="manual" if use_ca else "noCA")
            except Exception:
                mpc_action = np.array([0.0, 0.0])

            acc_norm = np.clip(mpc_action[0] / 5.0, -1.0, 1.0)
            steer_norm = np.clip(mpc_action[1] / (np.pi / 4), -1.0, 1.0)
            action = np.array([acc_norm, steer_norm], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1
            step_count += 1
            done = terminated or truncated

        crashed = bool(env.unwrapped.vehicle.crashed)
        arrived = bool(env.unwrapped.vehicle.position[0] > 370 and not crashed)

        all_rewards.append(ep_reward)
        all_lengths.append(ep_len)
        all_crashed.append(crashed)
        all_arrived.append(arrived)

        status = "CRASH" if crashed else ("ARRIVE" if arrived else "TIMEOUT")
        print(f"  Ep {ep+1:3d}/{n_episodes}: reward={ep_reward:8.2f}  "
              f"len={ep_len:3d}  x={env.unwrapped.vehicle.position[0]:.0f}  "
              f"{status}")

    env.close()

    n_crash = sum(all_crashed)
    n_arrive = sum(all_arrived)
    n_timeout = n_episodes - n_crash - n_arrive

    summary = {
        "model": model_name,
        "env": "merge-v0 (ego_on_ramp)",
        "checkpoint": checkpoint_path,
        "n_episodes": n_episodes,
        "collision_rate": n_crash / n_episodes,
        "arrival_rate": n_arrive / n_episodes,
        "timeout_rate": n_timeout / n_episodes,
        "n_crashed": n_crash,
        "n_arrived": n_arrive,
        "n_timeout": n_timeout,
        "reward_mean": float(np.mean(all_rewards)),
        "reward_std": float(np.std(all_rewards)),
        "length_mean": float(np.mean(all_lengths)),
        "per_episode": [
            {"episode": i+1, "reward": all_rewards[i], "length": all_lengths[i],
             "crashed": all_crashed[i], "arrived": all_arrived[i]}
            for i in range(n_episodes)
        ],
    }

    print(f"\n{'='*60}")
    print(f"  RESULTS: MPCRL {model_type} -> Merge ({n_episodes} episodes)")
    print(f"{'='*60}")
    print(f"  Collision rate : {summary['collision_rate']*100:5.1f}%  ({n_crash}/{n_episodes})")
    print(f"  Arrival rate   : {summary['arrival_rate']*100:5.1f}%  ({n_arrive}/{n_episodes})")
    print(f"  Timeout rate   : {summary['timeout_rate']*100:5.1f}%  ({n_timeout}/{n_episodes})")
    print(f"  Mean reward    : {summary['reward_mean']:8.2f}")
    print(f"  Mean length    : {summary['length_mean']:.1f} steps")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"eval_results_{model_name}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] Saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch eval MPCRL on merge-v0")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="manual",
                        choices=["noCA", "manual"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=str,
                        default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    run_evaluation(args.checkpoint, args.model, args.episodes,
                   args.output_dir, not args.no_video)


if __name__ == "__main__":
    main()
