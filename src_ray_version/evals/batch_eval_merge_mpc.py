"""
Zero-shot evaluate Pure MPC (intersection-designed) on merge-v0 with ego on ramp.

The intersection MPC uses a CasADi-based MPC that tracks a reference
trajectory [x, y, v, heading].  For zero-shot transfer to merging we:
  1. Put the ego on the merge ramp (ego_on_ramp=True)
  2. Generate a reference trajectory that follows the ramp geometry,
     curves through the SineLane, then lane-changes onto the highway
  3. Run the same MPC solver (kinematic bicycle, same costs/weights)

Difficulty presets (controls number of highway traffic vehicles):
  very_easy  : 1 vehicle
  easy       : 2 vehicles
  moderate   : 4 vehicles (default)

Usage:
    cd /users/saeani/src/mpcrl/MPC-RL_for_AVs/src_ray_version && \\
    PYTHONPATH="$PWD:$PWD/highway-env:$PYTHONPATH" \\
    python /users/saeani/src/mpcrl/DRL_MPC_for_AVs/src_ray_version/evals/batch_eval_merge_mpc.py \\
      --episodes 1000 --difficulty easy --no-video
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import gymnasium

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import highway_env  # noqa: F401

try:
    import casadi as ca
except ImportError:
    raise ImportError("casadi is required.  pip install casadi")


# ──────────────────────────────────────────────────────────────────────
#  Constants (must match intersection MPC env)
# ──────────────────────────────────────────────────────────────────────
VEHICLE_LENGTH = 5.0
VEHICLE_LENGTH_REAR = 2.5


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
    Reference trajectory following the merge ramp, SineLane curve,
    then lane-changing onto highway lane 1 (y=4).

    Returns np.ndarray shape (N, 4): [x, y, v, heading].
    """
    jk = road_network.get_lane(("j", "k", 0))
    kb = road_network.get_lane(("k", "b", 0))
    bc2 = road_network.get_lane(("b", "c", 2))

    # Find starting s on j->k
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

    # Phase 2: k->b SineLane (speed: ego_speed -> target_speed)
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
    merge_use = 50.0
    while s < merge_use:
        pos = bc2.position(s, 0)
        hdg = bc2.heading_at(s)
        waypoints.append([pos[0], pos[1], speed, hdg])
        s += speed * dt

    n_lane_sampled = len(waypoints)

    # Phase 4: Lane change y~8 -> y=4 over ~40 m (cubic S-curve)
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
        waypoints.append([x, y, speed, 0.0])

    # Phase 5: Continue straight y=4
    last_x = waypoints[-1][0]
    for i in range(1, 30):
        waypoints.append([last_x + dx * i, target_y, speed, 0.0])

    traj = np.array(waypoints)

    # Fix headings for phases 4-5
    for i in range(max(0, n_lane_sampled - 1), len(traj) - 1):
        ddx = traj[i + 1, 0] - traj[i, 0]
        ddy = traj[i + 1, 1] - traj[i, 1]
        if abs(ddx) > 1e-6 or abs(ddy) > 1e-6:
            traj[i, 3] = np.arctan2(ddy, ddx)
    if len(traj) > 1:
        traj[-1, 3] = traj[-2, 3]

    return traj


# ──────────────────────────────────────────────────────────────────────
#  Standalone MPC solver (mirrors intersection_mpc_env._solve_mpc)
# ──────────────────────────────────────────────────────────────────────
class StandaloneMPC:
    """CasADi MPC with the same model/costs/weights as the intersection env."""

    def __init__(self, horizon: int = 16, dt: float = 0.1):
        self.horizon = horizon
        self.dt = dt
        self.weights = {
            "weight_state": 10,
            "weight_speed": 1,
            "weight_control": 1,
            "weight_input_diff": 1,
            "weight_final_state": 1,
        }
        self.prev_solution = None

    def solve(self, ego_x, ego_y, ego_heading, ego_speed,
              reference, ego_index):
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

            state_cost += (
                4 * perp**2 + 2 * para**2 +
                self.weights["weight_speed"] * (x[3, k] - ref_v)**2 +
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
#  Difficulty presets (number of highway traffic vehicles)
# ──────────────────────────────────────────────────────────────────────
MERGE_DIFFICULTY = {
    "very_easy": {"other_vehicles_count": 1},
    "easy":      {"other_vehicles_count": 2},
    "moderate":  {"other_vehicles_count": 4},
}


# ──────────────────────────────────────────────────────────────────────
#  Evaluation
# ──────────────────────────────────────────────────────────────────────
def run_evaluation(n_episodes=100, output_dir=".", record_video=True,
                   difficulty="moderate"):
    diff_cfg = MERGE_DIFFICULTY[difficulty]
    model_name = "pure_mpc_merge"
    out_tag = f"{model_name}_{difficulty}_{n_episodes}ep"
    video_dir = os.path.join(output_dir, f"videos_{out_tag}")

    print(f"\n{'='*60}")
    print(f"  Model:      Pure MPC (zero-shot, ego merging)")
    print(f"  Difficulty: {difficulty} ({diff_cfg})")
    print(f"  Env:        merge-v0 (ego_on_ramp=True, ContinuousAction)")
    print(f"  Episodes:   {n_episodes}")
    print(f"  Output:     {output_dir}")
    print(f"{'='*60}\n")

    env_config = {
        "ego_on_ramp": True,
        "other_vehicles_count": diff_cfg["other_vehicles_count"],
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

        while not done:
            ego = env.unwrapped.vehicle
            ego_idx = int(np.argmin(
                np.linalg.norm(ref_xy - ego.position, axis=1)))

            try:
                mpc_action = mpc.solve(
                    ego_x=ego.position[0], ego_y=ego.position[1],
                    ego_heading=ego.heading, ego_speed=ego.speed,
                    reference=ref_traj, ego_index=ego_idx)
            except Exception:
                mpc_action = np.array([0.0, 0.0])

            acc_norm = np.clip(mpc_action[0] / 5.0, -1.0, 1.0)
            steer_norm = np.clip(mpc_action[1] / (np.pi / 4), -1.0, 1.0)
            action = np.array([acc_norm, steer_norm], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1
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
        "difficulty": difficulty,
        "difficulty_config": diff_cfg,
        "env": "merge-v0 (ego_on_ramp)",
        "n_episodes": n_episodes,
        "collision_rate": n_crash / n_episodes,
        "arrival_rate": n_arrive / n_episodes,
        "timeout_rate": n_timeout / n_episodes,
        "n_crashed": n_crash,
        "n_arrived": n_arrive,
        "n_timeout": n_timeout,
        "reward_mean": float(np.mean(all_rewards)),
        "reward_std": float(np.std(all_rewards)),
        "reward_min": float(np.min(all_rewards)),
        "reward_max": float(np.max(all_rewards)),
        "length_mean": float(np.mean(all_lengths)),
        "per_episode": [
            {"episode": i+1, "reward": all_rewards[i], "length": all_lengths[i],
             "crashed": all_crashed[i], "arrived": all_arrived[i]}
            for i in range(n_episodes)
        ],
    }

    print(f"\n{'='*60}")
    print(f"  RESULTS: Pure MPC -> Merge [{difficulty}] ({n_episodes} episodes)")
    print(f"{'='*60}")
    print(f"  Collision rate : {summary['collision_rate']*100:5.1f}%  ({n_crash}/{n_episodes})")
    print(f"  Arrival rate   : {summary['arrival_rate']*100:5.1f}%  ({n_arrive}/{n_episodes})")
    print(f"  Timeout rate   : {summary['timeout_rate']*100:5.1f}%  ({n_timeout}/{n_episodes})")
    print(f"  Mean reward    : {summary['reward_mean']:8.2f} +/- {summary['reward_std']:.2f}")
    print(f"  Mean length    : {summary['length_mean']:.1f} steps")
    if record_video:
        print(f"  Videos         : {os.path.abspath(video_dir)}")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"eval_results_{out_tag}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] Saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch eval Pure MPC on merge-v0")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--difficulty", type=str, default="moderate",
                        choices=["very_easy", "easy", "moderate"])
    parser.add_argument("--output-dir", type=str,
                        default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    run_evaluation(args.episodes, args.output_dir, not args.no_video,
                   args.difficulty)


if __name__ == "__main__":
    main()
