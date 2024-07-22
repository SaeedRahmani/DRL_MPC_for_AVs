#!/usr/bin/env python
# coding=utf-8

import casadi as ca
import numpy as np
import time
# from draw import Draw_MPC_tracking
import matplotlib.pyplot as plt
import gymnasium as gym
import math
import highway_env


def distance(x1, y1, x2, y2):
    dis = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dis


def ca_distance(x1, y1, x2, y2):
    dis = ca.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dis


def get_impulse(v1, v2_x, v2_y, theta1, theta2):
    v1_x = v1 * ca.cos(theta1)
    v1_y = v1 * ca.sin(theta1)
    imp_x = (mass_ego + mass_other) * (v2_x)
    imp_y = (mass_ego + mass_other) * (v2_y)
    # imp = np.hypot(imp_x, imp_y)
    # print(v1,"v2_x=",v2_x,"v2_y=",v2_y)
    # print(imp)
    imp = ca.sqrt(imp_x ** 2 + imp_y ** 2) * nor_imp

    return ca.fabs(imp)


def get_xcyc(x, y, theta, omega):  #
    R = (lr + lf) / (ca.tan(omega) + 0.00001)
    if omega < 0:
        R = -1 * R
    # if omega < 0 and x > 0:
    #    xc = ca.sin(theta*180/np.pi) * R + x
    #    yc = -ca.cos(theta*180/np.pi) * R + y
    # elif omega > 0 and x < 0:
    #    xc = -ca.sin(theta*180/np.pi) * R + x
    #    yc = ca.cos(theta*180/np.pi) * R + y
    # elif omega < 0 and x < 0:
    #    xc = -ca.sin(theta*180/np.pi) * R + x
    #    yc = -ca.cos(theta*180/np.pi) * R + y
    # elif omega > 0 and x > 0:
    #    xc = ca.sin(theta*180/np.pi) * R + x
    #    yc = ca.cos(theta*180/np.pi) * R + y
    dirction = np.array([ca.cos(theta), ca.sin(theta)])
    normal_vector = np.array([-ca.sin(theta), ca.cos(theta)])
    cross = np.cross(dirction, normal_vector)
    if (omega < 0 and cross > 0) or (omega > 0 and cross < 0):
        normal_vector *= -1
    # print(x)
    circle_point = np.array([x + R * normal_vector[0], y + R * normal_vector[1]])
    return circle_point  # [xc,yc]

# def get_xcyc(x, y, theta, omega):  #
#     R = (lr + lf) / (ca.tan(omega) + 0.00001)
#     if omega < 0:
#         R = -1 * R
#     dirction = np.array([ca.cos(theta), ca.sin(theta)])
#     normal_vector = np.array([-ca.sin(theta), ca.cos(theta)])
#     cross = np.cross(np.append(dirction, 0), np.append(normal_vector, 0))[:2]
#     if (omega < 0 and np.all(cross > 0)) or (omega > 0 and np.all(cross < 0)):
#         normal_vector *= -1
#     circle_point = np.array([x + R * normal_vector[0], y + R * normal_vector[1]])
#     return circle_point  # [xc,yc]



def arclen(x, y, xv, yv, xc, yc, R):  # xv = xcar yv=ycar
    theta1 = ca.atan2(yv - yc, xv - xc)
    theta2 = ca.atan2(y - yc, x - xc)
    arc_len = R * (theta2 - theta1)

    return arc_len


def driver_risk_field(x, y, xcar, ycar, v, theta, omega):  # DRF
    R = (lr + lf) / (ca.tan(omega) + 0.00001)
    if omega < 0:
        R = -1 * R
    circle_point = get_xcyc(xcar, ycar, theta, omega)
    xc = circle_point[0]
    yc = circle_point[1]
    arc_len = arclen(x, y, xcar, ycar, xc, yc, R)
    # arc_len = 1
    s = v * t_la
    if s < 1:
        s = 1
    dis = ca_distance(x, y, xc, yc)
    dis_to_circlepoint = dis - R
    # if dis_to_circlepoint < 0:
    width = (m + k_inner * ca.fabs(omega)) * arc_len + c
    # else:
    #      width = (m + k_outer * ca.fabs(omega)) * arc_len + c
    # 

    height1 = p * ((arc_len - s) ** 2)
    height = height1 * (ca.sign(s - arc_len) + 1) / 2 * (ca.sign(height1) + 1) / 2 * (ca.sign(arc_len) + 1) / 2
    risk = height * ca.exp(-(dis_to_circlepoint ** 2) / (2 * (width ** 2)))
    return risk


def pp_control(state, next_trajectories):
    Kp = 0.25
    e = (- state[3] + next_trajectories[1, 3])

    a = Kp * e + 0.01 * ei
    # 计算横向误差
    if ((state[0] - next_trajectories[1, 0]) * next_trajectories[1, 2] - (
            state[1] - next_trajectories[1, 1])).all() > 0:
        error = abs(math.sqrt((state[0] - next_trajectories[1, 0]) ** 2 + (state[1] - next_trajectories[1, 1]) ** 2))
    else:
        error = -abs(math.sqrt((state[0] - next_trajectories[1, 0]) ** 2 + (state[1] - next_trajectories[1, 1]) ** 2))
    delta = next_trajectories[1, 2] - state[2] + math.atan2(0.5 * error, state[3])

    ed = - state[1] + next_trajectories[1, 1]

    delta = np.arctan(2 * ed * (lr + lf) / (ld ** 2))

    if a > 3.0:
        delta = 3.0
    elif delta < - 5.0:
        delta = - 5.0

    #  限制车轮转角 [-45, 45]
    if delta > np.pi / 4.0:
        delta = np.pi / 4.0
    elif delta < - np.pi / 4.0:
        delta = - np.pi / 4.0
    # print(delta)
    return a, delta


def shift_movement(T, t0, u, x_n, ):
    #print(u[0])
    obs, reward, terminated, truncated, info = env.step([u[0, 0], u[0, 1]])
    done = terminated or truncated
    # ca.normalized_setup+
   # print("#################",obs,"########################")
    x, y, vx, vy, yaw = obs[0]
    env.render()
    st = np.array([x, y, yaw, info['speed'], math.hypot(vx, vy)])
    t = t0 + T
    u_end = np.concatenate((u[1:], u[-1:]))
    x_n = np.concatenate((x_n[1:], x_n[-1:]))
    # if distance(x,y,obs[1][0],obs[1][1]) > 30:
    #   obs[1][0] = 999999
    #   obs[1][1] = 999999

    print(info)
    ob_number = 1
    max_dis = 900000
    for i in range(1,len(obs)):
        dis = math.sqrt((obs[0][0] - obs[i][0])**2 + (obs[0][1] - obs[i][1])**2)
        if dis < max_dis and obs[0][0] != 0:
            max_dis = dis
            ob_number = i
            print("######################",ob_number)

    obstacle = np.array(
        [obs[ob_number][0], obs[ob_number][1], math.hypot(obs[ob_number][2], obs[ob_number][3]), obs[ob_number][4], obs[ob_number][2], obs[ob_number][3]])  # x,y,v,theta,vx,vy
    return t, st, u_end, x_n, obstacle, reward


def desired_command_and_trajectory(t, T, x0_: np.array, N_, v, index):
    # initial state / last state
    x_ = np.zeros((N_ + 1, 4))
    look_ahead_ = np.zeros((N_ + 1, 3))  # x y and angle
    x_[0] = x0_[:4]
    x_[0][0] = x_[0][0] + 1.3
    look_ahead_dis = 2 * v * T
    look_ahead_[0][0] = x_[0][0]
    look_ahead_[0][1] = x_[0][1]
    look_ahead_[0][2] = x_[0][2]
    tmp = index
    mindis = 2
    # index = index + 1
    #print(reference[0],"!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    for i in range( len(xnew) - N_ - 5, 5,-1):
        if mindis > math.sqrt((xnew[i] - x0_[0]) ** 2 + (ynew[i] - x0_[1]) ** 2):
            index = i
            # mindis = abs(xnew[i] - x0_[0])
            break
    if tmp > 1120:
        print("#####")
        index = tmp + 3
    #  if tmp >101:
    #      print("#####!!!!!!")
    #      index = tmp -10+2


    # states for the next N_ trajectories   (reference trajectories)
    print("index:",index)
    for i in range(N_):
        reference_index = index + i + preview
        v_ref_ = v_der  # preview
        # if v < 1:
        #      reference_index = index + i #+ preview
        #     v_ref_ = v_der
       # print(reference[1][reference_index], "!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        x_ref_ = xnew[reference_index] + 1
        y_ref_ = ynew[reference_index]
        theta_ref_ = thetanew[reference_index]
        x_[i + 1] = np.array([x_ref_, y_ref_, theta_ref_, v_ref_])
    # print(x_)
    # return pos
    for i in range(len(xnew) - N_ - 3, 2, -1):
        if mindis + look_ahead_dis > np.abs(xnew[i] - x0_[0]):
            index = i
            break
    for i in range(N_):
        reference_index = index + i + preview  #
        y_ref_ = ynew[reference_index]
        theta_ref_ = thetanew[reference_index]
        look_ahead_[i + 1] = np.array([x_ref_, y_ref_, theta_ref_])

    return x_, look_ahead_


def check_goal(x, y, goal):
    # check goal
    dx = x - goal[0]
    dy = y - goal[1]
    d = math.hypot(dx, dy)
    isgoal = (d <= 3)
    # isgoal = (abs(state.x - goal[0]) < 15)
    # if abs(tind - nind) >= 5:
    #    isgoal = False
    # isstop = (abs(state.v) <= STOP_SPEED)
    isstop = True
    if isgoal and isstop:
        return True
    return False


def prediction_state(x0, u, T, N):
    # define prediction horizon function
    states = np.zeros((N + 1, 3))
    states[0, :] = x0
    for i in range(N):
        states[i + 1, 0] = states[i, 0] + u[i, 0] * np.cos(states[i, 2]) * T  # states[i, 2]  jiaodu   u[i, 0] = v
        states[i + 1, 1] = states[i, 1] + u[i, 0] * np.sin(states[i, 2]) * T
        states[i + 1, 2] = states[i, 2]
    return states


r_x = []  # 创建空表存放x1数据
r_y = []
r_theta = []
with open('3.txt', 'r') as f1:  # 以只读形式打开txt文件
    for i, line in enumerate(f1):
        line = line.strip(',')  # 去掉换行符
        line = line.split(',')  # 分割掉两列数据之间的制表符
        r_x.append(line[0])
        r_y.append(line[1])
        r_theta.append(line[2])
xnew = np.array(r_x)
ynew = np.array(r_y)
thetanew = np.array(r_theta)
xnew = xnew.astype(float).tolist()
ynew = ynew.astype(float).tolist()
thetanew = thetanew.astype(float).tolist()
reference = np.array([r_x, r_y, r_theta])

safe_distance = 1  # m
preview = 0  # 2
v_der = 5.42  # 5.57 #6.82  #5

lr = 2.46
lf = 2.49
width = 1.98

mass_ego = 2200  # (kg)
mass_other = 2300  # (kg)

social_theta = 75 / 180 * np.pi  # 15
# param for risk model   #45

t_la = 1.8 # (s) lookahead time
m = 0.0001  # slope pf widening of DRF when driving straight
k_inner = 1.5  # k1k_inner
k_outer = 0  # k2
c = width*1.2  #/ 4
p = 0.0064  # steepness of the parabola

ei = 0
ld = 5.1  # (m)  lookhead distance in PP

use_constrain = False
use_mpcc = True
use_pp = False

nor_imp = 0.0001

if __name__ == '__main__':
    T = 0.25
    N = 10
    rob_diam = 0.3  # [m]
    v_max = 18
    omega_max = np.pi / 4.0

    slack = 0

    break_times = 0

    # env = gym.make('intersection-v0')
    # env = gym.make('roundabout-v0')
    env = gym.make("roundabout-v0", render_mode="rgb_array")
    env.configure({
        "action": {
            "type": "ContinuousAction",
            "action_config": {
                "type": "ContinuousAction",
            }
        }
    })
    # env.configure({
    #     "type": "MultiAgentAction",
    #     "action": {
    #         "type": "ContinuousAction"
    #     }
    # })
    # env.config["lanes_count"] = 2
    # env.configure({"controlled_vehicles": 2})
    env.reset()

    opti = ca.Opti()
    # control variables, a , string angle
    opt_controls = opti.variable(N, 2)
    a = opt_controls[:, 0]
    omega = opt_controls[:, 1]
    opt_states = opti.variable(N + 1, 5)
    x = opt_states[:, 0]
    y = opt_states[:, 1]
    theta = opt_states[:, 2]
    v = opt_states[:, 3]
    progress = opt_states[:, 4]

    # parameters, these parameters are the reference trajectories of the pose and inputs
    #  opt_u_ref = opti.parameter(N, 2)
    opt_x_ref = opti.parameter(N + 1, 4)
    opt_look_ahead_ref = opti.parameter(N + 1, 3)

    belta = np.arctan((lr / (lr + lf)) * np.tan(opt_controls[0, 1]))
    # create model
    f = lambda x_, u_: ca.vertcat(*[x_[3] * ca.cos(x_[2] + ca.arctan((lr / (lr + lf)) * ca.tan(u_[1]))),
                                    x_[3] * ca.sin(x_[2] + ca.arctan((lr / (lr + lf)) * ca.tan(u_[1]))),
                                    x_[3] * ca.cos(ca.arctan((lr / (lr + lf)) * ca.tan(u_[1]))) * ca.tan(
                                        u_[1] / (lr + lf)),
                                    u_[0],
                                    x_[3]])

    # add obstacle information
    obstacle = np.array([190, 190, 0, 0, 0, 0])
    last_obstacle = obstacle

    # obstacle_error_x = ca.fabs(opt_states[i,0] - 1)
    for i in range(N):
        obstacle_error_x = ca.fabs(opt_states[i, 0] - obstacle[0])
        obstacle_error_y = ca.fabs(opt_states[i, 1] - obstacle[1])
        obstacle_error = ca.sqrt(obstacle[0] ** 2 + obstacle[1] ** 2)
        risk = driver_risk_field(opt_states[i, 0], opt_states[i, 1], obstacle[0], obstacle[1], obstacle[2], obstacle[3],
                                 last_obstacle[3] - obstacle[3])
        # opti.subject_to(obstacle_error > 0.1)

    init_state = np.array([2.0, 30.0, 1.570, 1.0, 0.0])

    current_state = init_state.copy()

    next_trajectories, look_ahead = desired_command_and_trajectory(0, T, current_state, N, current_state[3], 0)
    opti.set_value(opt_x_ref, next_trajectories)
    opti.set_value(opt_look_ahead_ref, look_ahead)
    ## init_condition
    opti.subject_to(opt_states[0, :4] == opt_x_ref[0, :])
    for i in range(N):
        x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T * T
        opti.subject_to(opt_states[i + 1, :] == x_next)

    ## define the cost function
    ### some addition parameters
    Q = np.array([[0.006, 0.0, 0.0, 0.0], [0.0, .95, 0.0, 0.0], [0.0, 0.0, .7, 0.0], [0.0, 0.0, 0.0, .0]])
    R = np.array([[0.3, 0.0], [0.0, 0.5]])
    W = np.array([[0.5, 0.0], [0.0, .8]])
    L = np.array([[0.05, 0.0], [0.0, 0.05]])

    #### cost function
    obj = 0  #### cost
    if use_mpcc:
        for i in range(1, N):
            lateral_c_error = -(opt_states[i, 0] - opt_x_ref[i, 0]) * ca.sin(opt_x_ref[i, 2]) + (
                    opt_states[i, 1] - opt_x_ref[i, 1]) * ca.cos(opt_x_ref[i, 2])
            longitudinal_error = (opt_states[i, 0] - opt_x_ref[i, 0]) * ca.cos(opt_x_ref[i, 2]) + (
                    opt_states[i, 1] - opt_x_ref[i, 1]) * ca.sin(opt_x_ref[i, 2])
            orientation_error = 1 - ca.fabs(
                ca.cos(opt_x_ref[i, 2]) * ca.cos(opt_states[i, 2]) + ca.sin(opt_x_ref[i, 2]) * ca.sin(opt_states[i, 2]))
            last_control = opt_controls[i, :] - opt_controls[i - 1, :]
            control = opt_controls[i, :]
            v_error_ = opt_states[i, 3] - opt_x_ref[i + 1, 3]

            look_ahead_error = -(opt_states[i, 0] - opt_look_ahead_ref[i, 0]) * ca.sin(opt_look_ahead_ref[i, 2]) + (
                    opt_states[i, 1] - opt_look_ahead_ref[i, 1]) * ca.cos(opt_look_ahead_ref[i, 2])
            obj = obj + ca.mtimes([lateral_c_error, 0.8, lateral_c_error]) + ca.mtimes(
                [longitudinal_error, .0025, longitudinal_error]) + ca.mtimes(
                [last_control, W, last_control.T]) + ca.mtimes([v_error_, 0.09, v_error_]) + ca.mtimes(
                [orientation_error, 0.15, orientation_error]) + ca.mtimes(
                [look_ahead_error, 0.00, look_ahead_error.T]) + ca.mtimes([control, R, control.T])  # 0.03

    else:
        for i in range(1, N):
            look_ahead_error = opt_states[i, 1:3] - opt_look_ahead_ref[i + 1, 1:]
            state_error_ = opt_states[i, :4] - opt_x_ref[i + 1, :]
            # control_error_ = opt_controls[i, :] - opt_u_ref[i, :]  ca.mtimes([control_error_, R, control_error_.T])
            last_control = opt_controls[i, :] - opt_controls[i - 1, :]
            control = opt_controls[i, :]
            obj = obj + ca.mtimes([state_error_, Q, state_error_.T]) + ca.mtimes([control, R, control.T]) + ca.mtimes(
                [last_control, W, last_control.T]) + ca.mtimes([look_ahead_error, L, look_ahead_error.T])

    ### boundrary and control conditions

    opti.subject_to(opti.bounded(-60.0, y, 150.0))
    opti.subject_to(opti.bounded(-np.pi, theta, np.pi))
    opti.subject_to(opti.bounded(-25, v, 25))
    opti.subject_to(opti.bounded(-omega_max, omega, omega_max))
    opti.subject_to(opti.bounded(-1, a, 1.1))

    opts_setting = {'ipopt.max_iter': 3000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                    'ipopt.acceptable_obj_change_tol': 1e-6}

    opti.solver('ipopt', opts_setting)

    t0 = 0
    # init_state1 = np.array([2.0, 30.0, 1.570, 1.0])
    init_state1 = np.array([2.0, 30.0, 1.570, 1.0])
    current_state = init_state.copy()
    u0 = np.zeros((N, 2))
    next_trajectories = np.tile(init_state1, N + 1).reshape(N + 1,
                                                            -1)  # set the initial state as the first trajectories for
    # the robot
    next_controls = np.zeros((N, 2))
    next_states = np.zeros((N + 1, 5))
    x_c = []  # contains for the history of the state
    y_c = []
    x_state = []
    y_state = []
    u_c = []
    t_c = [t0]  # for the time
    xx = []

    a_state = []
    omega_state = []

    omega_state_a = []

    error_y = []

    goal = np.array([xnew[-1] - 20, ynew[-1]])
    sim_time = 60.0

    ## start MPC
    mpciter = 0
    start_time = time.time()
    index_t = []
    index_t_pp = []
    risk_show = []
    obj_obs = 0
    last_obstacle = np.array([0, 0, 0, 0])
    obstacle_error_ = 0
    int_obj = obj
    
    while (mpciter - sim_time / T < 0.0):
        #### boundrary and control conditions
        opti.subject_to(opti.bounded(-120.0, x, 120.0))

        ## set parameter, here only update initial state of x (x0)

        opti.minimize(obj)
        opti.set_value(opt_x_ref, next_trajectories)
        opti.set_value(opt_look_ahead_ref, look_ahead)

        ## provide the initial guess of the optimization targets
        opti.set_initial(opt_controls, u0.reshape(N, 2))  # (N, 2)
        opti.set_initial(opt_states, next_states)  # (N+1, 4)
        ## solve the problem

        t_ = time.time()
        sol = opti.solve()
        index_t.append(time.time() - t_)

        ## obtain the control input
        u_res = sol.value(opt_controls)
        x_m = sol.value(opt_states)
        # print(x_m[0])
        u_c.append(u_res[0, :])
        t_c.append(t0)
        x_c.append(x_m[0, 0])
        y_c.append(x_m[0, 1])

        x_state.append(current_state[0])
        y_state.append(current_state[1])

        omega_state_a.append(u_res[0, 1])
        #print("current_state[1]:",current_state[1])
        # print("next_trajectories[1,1]:",next_trajectories[0,1])
        error_y.append(
            np.abs(np.abs(current_state[1]) - np.abs(next_trajectories[1, 1])) / np.abs(next_trajectories[1, 1]))
        if use_pp:
            t_pp = time.time()
            u_res[0, 0], u_res[0, 1] = pp_control(current_state, next_trajectories)
            index_t_pp.append(time.time() - t_pp)

        a_state.append(u_res[0, 0])
        omega_state.append(u_res[0, 1])

        t0, current_state, u0, next_states, obstacle, reward = shift_movement(T, t0, u_res, x_m)
        xx.append(current_state)

        if u_res[0, 0] < -0.025:
            break_times = break_times + 1
        print("break distance:", math.hypot(abs(x_m[0, 0] - obstacle[0]), abs(x_m[0, 1] - obstacle[1])))
        break_distance = math.hypot(abs(x_m[0, 0] - obstacle[0]), abs(x_m[0, 1] - obstacle[1]))
       # print("obstacle[0]",obstacle[0],"obstacle[1]", obstacle[1])

        obj = int_obj
        #risk = 0
        if distance(current_state[0], current_state[1], obstacle[0], obstacle[1]) < 20 and current_state[0] > obstacle[0] and current_state[1] > obstacle[1]:
            print("!!!!!!!!111111111111")
            for i in range(N):
                obstacle_error_x = ca.fabs(opt_states[i, 0] - obstacle[0])
                obstacle_error_y = ca.fabs(opt_states[i, 1] - obstacle[1])
                obstacle_error = ca.sqrt(obstacle[0] ** 2 + obstacle[1] ** 2)

                # print(obstacle[2])
                imp = get_impulse(opt_states[i, 3], obstacle[4], obstacle[5], opt_states[i, 2], obstacle[3])
                risk += imp * driver_risk_field(opt_states[i, 0], opt_states[i, 1], obstacle[0], obstacle[1],
                                                obstacle[2], obstacle[3], last_obstacle[3] - obstacle[3])
               # if current_state[0]>0:
                  #  risk = 0
            # ((N-i)/N)  this method is not good

            if use_constrain:
                # imp = get_impulse(current_state[3],obstacle[4],obstacle[5],current_state[2],obstacle[3])
                opti.subject_to(risk < N)  # print(risk)
            else:
                # imp = get_impulse(current_state[3],obstacle[4],obstacle[5],current_state[2],obstacle[3])
                obj = ca.cos(social_theta) * obj + ca.sin(social_theta) * risk# - break_distance  * 1000000000
            # print(i)
            print(
                driver_risk_field(current_state[0], current_state[1], obstacle[0], obstacle[1], obstacle[2],
                                  obstacle[3], last_obstacle[3] - obstacle[3]))
        #(risk)
        # get_impulse(current_state[3],obstacle[4],obstacle[5],current_state[2],obstacle[3])
        last_obstacle = obstacle
        # risk_show.append(risk)
        # obj = obj - (obstacle_error_x +obstacle_error_y)*3.6
        # obj = obj + reward * 0.02

        ## estimate the new desired trajectories
        next_trajectories, look_ahead = desired_command_and_trajectory(t0, T, current_state, N, current_state[3],
                                                                       mpciter)
        mpciter = mpciter + 1

        #   if current_state[3] < 1:
        #      obs, reward, down, info = env.step(([4, 0], [1, 0]))
        # print(get_xcyc(current_state[0], current_state[1], current_state[2], u_res[0, 1]))

        # plt.imshow(env.render(mode="rgb_array"))
        plt.imshow(env.render())
        if check_goal(current_state[0], current_state[1], goal):
            print("goal!")
            break

    ## after MPC and drew sth
    print(mpciter)
    t_v = np.array(index_t)
    print(t_v.mean())
    print((time.time() - start_time) / (mpciter))

    t_v_pp = np.array(index_t_pp)
    print("pp time:", t_v_pp.mean() / (mpciter))

    print(break_times, "!!!!!!!!!!!!break times")

    # xnew = np.array(x_c)
    # ynew = np.array(y_c)

    fig = plt.figure(figsize=(10, 10))  # 创建绘图窗口，并设置窗口大小
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(xnew[:111], ynew[:111], 'b', label='Reference trajectory')
    ax1.plot(x_state, y_state, 'r', label='Vehicle track')

    ax1.invert_yaxis()
    ax1.legend(loc='upper left')  # 绘制图例，plot()中的label值
    ax1.set_xlabel('x-position (x)')  # 设置X轴名称
    ax1.set_ylabel('y-position (x)')  # 设置Y轴名称
    ax1.set_aspect(1)

    ax2 = fig.add_subplot(2, 2, 2)
    # ax2.plot(x_state, risk_show, 'b', label='risk_show')

    # ax2.invert_yaxis()
    ax2.legend(loc='upper left')  # 绘制图例，plot()中的label值
    ax2.set_xlabel('x-position (x)')  # 设置X轴名称
    ax2.set_ylabel('y-position (x)')  # 设置Y轴名称
    # ax2.set_aspect(1)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(x_state[3:], a_state[3:], 'b')

    # ax2.invert_yaxis()
    ax3.legend(loc='upper left')  # 绘制图例，plot()中的label值
    ax3.set_xlabel('x-position (x)')  # 设置X轴名称
    ax3.set_ylabel('Acceleration (m/s²)')  # 设置Y轴名称
    ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # ax3.set_aspect(4)
    # plt.show()  # 显示绘制的图

    ax4 = fig.add_subplot(2, 2, 4)
    # ax4.plot(x_state[3:], omega_state[3:],'b', label='omega_state')
    ax4.plot(x_state[3:], omega_state[3:], 'r', label='omega_state_a_MPC')

    print(x_state[3:], "!!!!!!!!!!!!!!!!")

    # ax2.invert_yaxis()
    ax4.legend(loc='upper left')  # 绘制图例，plot()中的label值
    ax4.set_xlabel('x-position (x)')  # 设置X轴名称
    ax4.set_ylabel('y-position (x)')  # 设置Y轴名称
    # ax4.set_aspect(1)

    print("pass time:", T * mpciter)

    print(a_state[3:])
    plt.show()
    ## draw function
#  draw_result = Draw_MPC_tracking(rob_diam=0.3, init_state=init_state, robot_states=xx )
