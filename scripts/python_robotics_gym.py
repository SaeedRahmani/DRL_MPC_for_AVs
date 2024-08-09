import gymnasium as gym
import highway_env
import numpy as np
import cvxpy as cp
import math
import pandas as pd

# Initialize the environment
env = gym.make("intersection-v0", render_mode="rgb_array")

# MPC parameters
NX = 4  # x = [x, y, v, yaw]
NU = 2  # u = [acceleration, steering]
T = 5  # Time horizon

R = np.diag([0.01, 0.01])  # Input cost matrix
Rd = np.diag([0.01, 1.0])  # Input difference cost matrix
Q = np.diag([1.0, 1.0, 0.5, 0.5])  # State cost matrix
Qf = Q  # State final cost matrix

GOAL_DIS = 1.5  # Goal distance
STOP_SPEED = 0.5 / 3.6  # Stop speed [m/s]
MAX_TIME = 500.0  # Max simulation time [s]

MAX_ITER = 3  # Max iterations
DU_TH = 0.1  # Iteration finish threshold

TARGET_SPEED = 10.0 / 3.6  # Target speed [m/s]
N_IND_SEARCH = 10  # Search index number

DT = 0.2  # Time tick [s]

# Vehicle parameters
WB = 2.5  # [m]

MAX_STEER = np.deg2rad(45.0)  # Max steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # Max steering speed [rad/s]
MAX_SPEED = 55.0 / 3.6  # Max speed [m/s]
MIN_SPEED = -20.0 / 3.6  # Min speed [m/s]
MAX_ACCEL = 1.0  # Max acceleration [m/s^2]

class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

def pi_2_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def get_linear_model_matrix(v, phi, delta):
    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = DT * math.cos(phi)
    A[0, 3] = -DT * v * math.sin(phi)
    A[1, 2] = DT * math.sin(phi)
    A[1, 3] = DT * v * math.cos(phi)
    A[3, 2] = DT * math.tan(delta) / WB

    B = np.zeros((NX, NU))
    B[2, 0] = DT
    B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)

    C = np.zeros(NX)
    C[0] = DT * v * math.sin(phi) * phi
    C[1] = -DT * v * math.cos(phi) * phi
    C[3] = -DT * v * delta / (WB * math.cos(delta) ** 2)

    return A, B, C

def update_state(state, a, delta):
    delta = np.clip(delta, -MAX_STEER, MAX_STEER)
    state.x = state.x + state.v * math.cos(state.yaw) * DT
    state.y = state.y + state.v * math.sin(state.yaw) * DT
    state.yaw = state.yaw + state.v / WB * math.tan(delta) * DT
    state.v = np.clip(state.v + a * DT, MIN_SPEED, MAX_SPEED)
    return state

def calc_nearest_index(state, cx, cy, cyaw, pind):
    dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]
    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)
    ind = d.index(mind) + pind
    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind

def predict_motion(x0, oa, od, xref):
    xbar = xref * 0.0
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
    for (ai, di, i) in zip(oa, od, range(1, T + 1)):
        state = update_state(state, ai, di)
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.v
        xbar[3, i] = state.yaw

    return xbar

def iterative_linear_mpc_control(xref, x0, dref, oa, od):
    if oa is None or od is None:
        oa = [0.0] * T
        od = [0.0] * T

    for _ in range(MAX_ITER):
        xbar = predict_motion(x0, oa, od, xref)
        poa, pod = oa[:], od[:]
        oa, od, ox, oy, oyaw, ov = linear_mpc_control(xref, xbar, x0, dref)
        du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
        if du <= DU_TH:
            break

    return oa, od, ox, oy, oyaw, ov

def linear_mpc_control(xref, xbar, x0, dref):
    x = cp.Variable((NX, T + 1))
    u = cp.Variable((NU, T))

    cost = 0.0
    constraints = []

    for t in range(T):
        cost += cp.quad_form(u[:, t], R)
        if t != 0:
            cost += cp.quad_form(xref[:, t] - x[:, t], Q)

        A, B, C = get_linear_model_matrix(xbar[2, t], xbar[3, t], dref[0, t])
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

        if t < (T - 1):
            cost += cp.quad_form(u[:, t + 1] - u[:, t], Rd)
            constraints += [cp.abs(u[1, t + 1] - u[1, t]) <= MAX_DSTEER * DT]

    cost += cp.quad_form(xref[:, T] - x[:, T], Qf)

    constraints += [x[:, 0] == x0]
    constraints += [x[2, :] <= MAX_SPEED]
    constraints += [x[2, :] >= MIN_SPEED]
    constraints += [cp.abs(u[0, :]) <= MAX_ACCEL]
    constraints += [cp.abs(u[1, :]) <= MAX_STEER]

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)

    if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
        ox = np.array(x.value[0, :]).flatten()
        oy = np.array(x.value[1, :]).flatten()
        ov = np.array(x.value[2, :]).flatten()
        oyaw = np.array(x.value[3, :]).flatten()
        oa = np.array(u.value[0, :]).flatten()
        odelta = np.array(u.value[1, :]).flatten()
    else:
        print("Error: Cannot solve mpc..")
        oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

    return oa, odelta, ox, oy, oyaw, ov

def calc_ref_trajectory(state, cx, cy, cyaw, pind):
    xref = np.zeros((NX, T + 1))
    dref = np.zeros((1, T + 1))
    ncourse = len(cx)

    ind, _ = calc_nearest_index(state, cx, cy, cyaw, pind)

    if pind >= ind:
        ind = pind

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = TARGET_SPEED
    xref[3, 0] = cyaw[ind]
    dref[0, 0] = 0.0  # steer operational point should be 0

    travel = 0.0

    for i in range(T + 1):
        travel += abs(state.v) * DT
        dind = int(round(travel / 1.0))

        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = TARGET_SPEED
            xref[3, i] = cyaw[ind + dind]
            dref[0, i] = 0.0
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = TARGET_SPEED
            xref[3, i] = cyaw[ncourse - 1]
            dref[0, i] = 0.0

    return xref, ind, dref

def generate_simple_reference_trajectory():
    x = np.linspace(0, 50, 100)  # 100 points from 0 to 50 meters
    y = np.zeros_like(x)  # Straight line along x-axis
    yaw = np.zeros_like(x)  # Constant heading (along x-axis)
    return x, y, yaw
    # filepath = "inter.txt"
    # data = pd.read_csv(filepath, header=None, names=['x', 'y', 'yaw'])
    # x = data['x'].values
    # y = data['y'].values
    # yaw = data['yaw'].values
    # return x, y, yaw

def print_vehicle_info(obs):
    for i, vehicle in enumerate(obs):
        if vehicle[0] == 1:  # If the vehicle is present
            print(f"Vehicle {i}: {'Ego' if i == 0 else 'Other'}")
            print(f"  Position: ({vehicle[1]:.2f}, {vehicle[2]:.2f})")
            print(f"  Velocity: ({vehicle[3]:.2f}, {vehicle[4]:.2f})")
            print(f"  Heading: {vehicle[5]:.2f}")
            print(f"  Presence: {vehicle[0]}")
    print("--------------------")
    
def convert_continuous_to_discrete_action(steering, acceleration):
    # This function needs to be implemented based on the specific action space of your environment
    # For example, if there are 5 discrete actions:
    # 0: do nothing, 1: accelerate, 2: decelerate, 3: turn left, 4: turn right
    if abs(steering) > abs(acceleration):
        return 3 if steering > 0 else 4  # turn left or right
    else:
        return 1 if acceleration > 0 else 2  # accelerate or decelerate
    
def main():
    print(__file__ + " start!!")

    cx, cy, cyaw = generate_simple_reference_trajectory()

    obs, info = env.reset()
    print("Observation shape:", obs.shape)
    print("Observation:", obs)
    
    # Print action space information
    print("Action space:", env.action_space)
    
    ego_info = obs[0]
    
    state = State(
        x=ego_info[1],
        y=ego_info[2],
        yaw=ego_info[5],
        v=np.sqrt(ego_info[3]**2 + ego_info[4]**2)
    )

    time = 0.0
    target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)

    odelta, oa = None, None

    while MAX_TIME >= time:
        xref, target_ind, dref = calc_ref_trajectory(state, cx, cy, cyaw, target_ind)

        x0 = [state.x, state.y, state.v, state.yaw]  # current state

        oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(xref, x0, dref, oa, odelta)

        if odelta is not None:
            di, ai = odelta[0], oa[0]

            # Convert MPC output to discrete action
            action = convert_continuous_to_discrete_action(di, ai)

            obs, reward, terminated, truncated, info = env.step(action)

            ego_info = obs[0]
            state.x = ego_info[1]
            state.y = ego_info[2]
            state.v = np.sqrt(ego_info[3]**2 + ego_info[4]**2)
            state.yaw = ego_info[5]

        time += env.config['simulation_frequency']

        if terminated or truncated:
            print("Episode finished")
            break

        env.render()

    env.close()

    
if __name__ == '__main__':
    main()