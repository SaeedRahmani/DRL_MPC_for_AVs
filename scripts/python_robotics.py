import gymnasium as gym
import highway_env
import numpy as np
import cvxpy as cp
import math
import matplotlib.pyplot as plt

import cubic_spline_planner
from angle import angle_mod

# Initialize the environment
env = gym.make("intersection-v1", render_mode="rgb_array")
obs, info = env.reset()

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
LENGTH = 4.5  # [m]
WIDTH = 2.0  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 2.5  # [m]

MAX_STEER = np.deg2rad(45.0)  # Max steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # Max steering speed [rad/s]
MAX_SPEED = 55.0 / 3.6  # Max speed [m/s]
MIN_SPEED = -20.0 / 3.6  # Min speed [m/s]
MAX_ACCEL = 1.0  # Max acceleration [m/s^2]

class State:
    """
    Vehicle state class
    """
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
    # Input check
    if delta >= MAX_STEER:
        delta = MAX_STEER
    elif delta <= -MAX_STEER:
        delta = -MAX_STEER

    state.x = state.x + state.v * math.cos(state.yaw) * DT
    state.y = state.y + state.v * math.sin(state.yaw) * DT
    state.yaw = state.yaw + state.v / WB * math.tan(delta) * DT
    state.v = state.v + a * DT

    if state.v > MAX_SPEED:
        state.v = MAX_SPEED
    elif state.v < MIN_SPEED:
        state.v = MIN_SPEED

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
    ox, oy, oyaw, ov = None, None, None, None

    if oa is None or od is None:
        oa = [0.0] * T
        od = [0.0] * T

    for i in range(MAX_ITER):
        xbar = predict_motion(x0, oa, od, xref)
        poa, pod = oa[:], od[:]
        oa, od, ox, oy, oyaw, ov = linear_mpc_control(xref, xbar, x0, dref)
        du = sum(abs(oa - poa)) + sum(abs(od - pod))  # Calc u change value
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

def generate_global_reference_trajectory():
    ax = [0.0, 30.0, 6.0, 20.0, 35.0]
    ay = [0.0, 0.0, 20.0, 35.0, 20.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=1.0)
    return cx, cy, cyaw

def plot_car(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover
    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")

def main():
    print(__file__ + " start!!")

    cx, cy, cyaw = generate_global_reference_trajectory()

    state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)

    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)

    odelta, oa = None, None

    plt.ion()
    fig, ax = plt.subplots()

    while MAX_TIME >= time:
        xref, target_ind, dref = calc_ref_trajectory(state, cx, cy, cyaw, target_ind)

        x0 = [state.x, state.y, state.v, state.yaw]  # current state

        oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(xref, x0, dref, oa, odelta)

        di, ai = 0.0, 0.0
        if odelta is not None:
            di, ai = odelta[0], oa[0]
            state = update_state(state, ai, di)

        time = time + DT

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)

        if abs(state.x - cx[-1]) < GOAL_DIS and abs(state.y - cy[-1]) < GOAL_DIS:
            print("Goal")
            break

        ax.cla()
        ax.plot(cx, cy, "-r", label="spline")
        ax.plot(x, y, "-g", label="tracking")
        plot_car(state.x, state.y, state.yaw, steer=di)
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-10, 50)
        ax.set_ylim(-10, 50)
        ax.grid(True)
        plt.pause(0.001)

    plt.ioff()
    plt.show()

    plt.figure()
    plt.plot(t, v, "-r", label="speed")
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [kmh]")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
