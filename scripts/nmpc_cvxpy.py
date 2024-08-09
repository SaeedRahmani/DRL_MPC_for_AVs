import gymnasium as gym
import highway_env
import numpy as np
import cvxpy as cp
import math
import matplotlib.pyplot as plt

# Initialize the environment
env = gym.make("intersection-v1", render_mode="rgb_array")
obs, info = env.reset()

# MPC parameters
horizon = 6
dt = 0.1

# Vehicle parameters
LENGTH = 5.0
WIDTH = 2.0
WHEELBASE = 2.5

MIN_SPEED = 0

def generate_global_reference_trajectory():
    trajectory = []
    x, y, v, heading = 2, 50, 10, - np.pi/2  # Starting with 10 m/s speed
    
    turn_start_y = 20
    radius = 5  # Radius of the curve
    turn_angle = np.pi / 2  # Total angle to turn (90 degrees for a left turn)

    # Go straight until reaching the turn start point
    for _ in range(38):
        y += v * dt * math.sin(heading)
        trajectory.append((x, y, v, heading))
    
    # Compute the turn
    angle_increment = turn_angle / 20  # Divide the turn into 20 steps
    for _ in range(20):
        heading += angle_increment  # Decrease heading to turn left
        x += v * dt * math.cos(heading)
        y += v * dt * math.sin(heading)
        trajectory.append((x, y, v, heading))
    
    # Continue straight after the turn
    for _ in range(25):  # Continue for a bit after the turn
        x += v * dt * math.cos(heading)
        trajectory.append((x, y, v, heading))
    
    return trajectory

def vehicle_model(state, action):
    x, y, v, psi = state
    a, delta = action
    beta = math.atan(0.5 * math.tan(delta))
    x_next = x + v * math.cos(psi + beta) * dt
    y_next = y + v * math.sin(psi + beta) * dt
    v_next = v + a * dt
    psi_next = psi + v * math.sin(beta) / WHEELBASE * dt
    return np.array([x_next, y_next, v_next, psi_next])

def find_closest_point(current_state, reference_trajectory):
    current_position = current_state[:2]
    distances = np.linalg.norm(current_position - np.array([point[:2] for point in reference_trajectory]), axis=1)
    closest_index = np.argmin(distances)
    return closest_index

def linearize_model(state, u, dt):
    x, y, v, psi = state
    a, delta = u
    beta = math.atan(0.5 * math.tan(delta))
    
    A = np.eye(4)
    A[0, 2] = dt * math.cos(psi + beta)
    A[0, 3] = -dt * v * math.sin(psi + beta)
    A[1, 2] = dt * math.sin(psi + beta)
    A[1, 3] = dt * v * math.cos(psi + beta)
    A[3, 2] = dt * math.sin(beta) / WHEELBASE
    
    B = np.zeros((4, 2))
    B[2, 0] = dt
    B[3, 1] = v * dt / (WHEELBASE * math.cos(delta)**2)
    
    C = np.zeros(4)
    C[0] = v * dt * math.sin(psi + beta) * delta / math.cos(delta)**2
    C[1] = -v * dt * math.cos(psi + beta) * delta / math.cos(delta)**2
    C[3] = v * dt * delta / (WHEELBASE * math.cos(delta)**2)
    
    return A, B, C

def mpc_control(current_state, reference_trajectory, obstacles, start_index):
    x = cp.Variable((4, horizon + 1))
    u = cp.Variable((2, horizon))
    
    cost = 0
    constraints = []
    
    constraints += [x[:, 0] == current_state]

    sin_ref_headings = np.sin([reference_trajectory[min(start_index + i, len(reference_trajectory) - 1)][3] for i in range(horizon)])
    cos_ref_headings = np.cos([reference_trajectory[min(start_index + i, len(reference_trajectory) - 1)][3] for i in range(horizon)])
    
    # Initial control input guess
    u_guess = np.zeros(2)
    
    for i in range(horizon):
        ref_index = min(start_index + i, len(reference_trajectory) - 1)
        ref_x, ref_y, ref_v, ref_heading = reference_trajectory[ref_index]
        
        dx = current_state[0] - ref_x
        dy = current_state[1] - ref_y
        perp_deviation = dx * sin_ref_headings[i] - dy * cos_ref_headings[i]
        para_deviation = dx * cos_ref_headings[i] + dy * sin_ref_headings[i]

        state_cost = 20 * cp.square(perp_deviation) + 1 * cp.square(para_deviation) + 1 * cp.square(x[2, i] - ref_v) + 0.5 * cp.square(x[3, i] - ref_heading)
        control_cost = 0.1 * cp.sum_squares(u[:, i])
        
        cost += state_cost + control_cost

        # Linearized vehicle model constraints
        A, B, C = linearize_model(current_state, u_guess, dt)
        constraints += [x[:, i+1] == A @ x[:, i] + B @ u[:, i] + C]
        
        # Update u_guess for the next iteration
        u_guess = np.array([u[0, i].value if u[0, i].value is not None else 0, u[1, i].value if u[1, i].value is not None else 0])
        
        # Predict the next state based on the guessed control input
        current_state = vehicle_model(current_state, u_guess)
        
        # Control input constraints
        constraints += [u[0, i] >= -10]
        constraints += [u[0, i] <= 3]
        constraints += [u[1, i] >= -np.pi/2]
        constraints += [u[1, i] <= np.pi/2]

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.ECOS, verbose=False)

    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return u[:, 0].value
    else:
        raise Exception("MPC failed to find a solution")

def process_observation(obs):
    ego_vehicle = obs[0]
    x, y = ego_vehicle[1], ego_vehicle[2]
    vx, vy = ego_vehicle[3], ego_vehicle[4]
    v = np.sqrt(vx**2 + vy**2)
    psi = np.arctan2(vy, vx)
    current_state = np.array([x, y, v, psi])
    obstacles = [vehicle[1:3] for vehicle in obs[1:] if vehicle[0] == 1]
    return current_state, obstacles

def calculate_distances(current_state, obstacles):
    current_position = current_state[:2]
    distances = [np.linalg.norm(current_position - np.array(obs)) for obs in obstacles]
    return distances

def plot_trajectory(real_path, ref_path):
    plt.clf()
    if ref_path:
        plt.plot(*zip(*ref_path), 'r--', label='Reference Trajectory')
    if real_path:
        plt.plot(*zip(*real_path), 'b-', label='Ego Vehicle Path')
        plt.scatter(real_path[-1][0], real_path[-1][1], color='blue', s=50, label='Current Position')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Vehicle Trajectory and Reference Path')
    plt.legend()
    plt.pause(0.1)

def get_imaginary_lines(x, y, psi, length=5.0):
    front_x = x + (length * np.cos(psi))
    front_y = y + (length * np.sin(psi))
    back_x = x - (length * np.cos(psi))
    back_y = y - (length * np.sin(psi))
    return (front_x, front_y), (back_x, back_y)

global_reference_trajectory = generate_global_reference_trajectory()
ref_path = [(x, y) for x, y, v, psi in global_reference_trajectory]
real_path = []
plt.ion()

max_steps = 500
for step in range(max_steps):
    env.render()
    current_state, obstacles = process_observation(obs)
    real_path.append((current_state[0], current_state[1]))
    plot_trajectory(real_path, ref_path)
    
    distances = calculate_distances(current_state, obstacles)
    closest_index = find_closest_point(current_state, global_reference_trajectory)
    current_reference = global_reference_trajectory[closest_index:closest_index+horizon]
    
    action = mpc_control(current_state, current_reference, obstacles, closest_index)
    x, y, v, psi = current_state
    a, delta = action
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated or closest_index >= len(global_reference_trajectory) - horizon:
        print(f"Finished after {step+1} steps")
        obs, info = env.reset()
        real_path = []
        continue
plt.ioff()
plt.show()
env.close()
