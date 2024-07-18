import gymnasium as gym
import highway_env
import numpy as np
import cvxpy as cp
import math

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
    x, y, v, heading = 0, 0, 10, np.pi/2  # Starting with 10 m/s speed
    
    # Go straight for 100 meters
    for _ in range(20):  # 1000 * 0.1 = 100 meters
        x += v * dt * math.cos(heading)
        y += v * dt * math.sin(heading)
        trajectory.append((x, y, v, heading))
    
    return trajectory

def linearized_vehicle_model(state, v_ref, psi_ref):
    x, y, v, psi = state
    
    A = np.array([
        [1, 0, dt*np.cos(psi_ref), -dt*v_ref*np.sin(psi_ref)],
        [0, 1, dt*np.sin(psi_ref),  dt*v_ref*np.cos(psi_ref)],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    B = np.array([
        [0, 0],
        [0, 0],
        [dt, 0],
        [0, dt*v_ref/WHEELBASE]
    ])
    
    return A, B

def distance_to_nearest_obstacle(current_state, obstacles):
    current_position = current_state[:2]
    distances = [np.linalg.norm(current_position - np.array(obs)) for obs in obstacles]
    return min(distances) if distances else float('inf')

def mpc_control(current_state, reference_trajectory, obstacles, start_index):
    x = cp.Variable((horizon + 1, 4))
    u = cp.Variable((horizon, 2))

    constraints = [x[0] == current_state]
    objective = 0

    nearest_obstacle_distance = distance_to_nearest_obstacle(current_state, obstacles)
    
    for t in range(horizon):
        ref_index = min(start_index + t, len(reference_trajectory) - 1)
        ref_x, ref_y, ref_v, ref_heading = reference_trajectory[ref_index]
        
        A, B = linearized_vehicle_model(current_state, ref_v, ref_heading)
        
        constraints += [x[t+1] == A @ x[t] + B @ u[t]]
        constraints += [u[t, 0] >= -10, u[t, 0] <= 3]  # acceleration bounds
        constraints += [u[t, 1] >= -np.pi/4, u[t, 1] <= np.pi/4]  # steering bounds
        constraints += [x[t+1, 2] >= MIN_SPEED]  # minimum speed constraint
        
        # Add speed reduction term based on obstacle proximity
        speed_reduction_factor = max(0, min(1, (nearest_obstacle_distance - 10) / 20))
        target_speed = ref_v * speed_reduction_factor
        
        objective += cp.sum_squares(x[t, :2] - [ref_x, ref_y]) + \
                     0.5 * cp.square(x[t, 2] - target_speed) + \
                     0.5 * cp.square(x[t, 3] - ref_heading) + \
                     0.1 * cp.sum_squares(u[t])
        
        # Add obstacle avoidance constraint using quadratic penalty
        for obs in obstacles:
            objective += 1000 * cp.sum_squares(x[t, :2] - obs)

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.ECOS, verbose=False)

    if problem.status != cp.OPTIMAL:
        print(f"Warning: problem status is {problem.status}")
        return np.zeros(2)  # Return zero action if optimization fails

    return u[0].value

def process_observation(obs):
    ego_vehicle = obs[0]
    
    x, y = ego_vehicle[1], ego_vehicle[2]
    vx, vy = ego_vehicle[3], ego_vehicle[4]
    v = np.sqrt(vx**2 + vy**2)
    psi = ego_vehicle[5]  # Using provided heading
    
    current_state = np.array([x, y, v, psi])
    
    obstacles = [vehicle[1:3] for vehicle in obs[1:] if vehicle[0] == 1]
    
    return current_state, obstacles

def calculate_distances(current_state, obstacles):
    current_position = current_state[:2]
    distances = [np.linalg.norm(current_position - np.array(obs)) for obs in obstacles]
    return distances

def find_closest_point(current_state, reference_trajectory):
    current_position = current_state[:2]
    distances = [np.linalg.norm(current_position - np.array(point[:2])) for point in reference_trajectory]
    closest_index = np.argmin(distances)
    return closest_index

# Generate the global reference trajectory
global_reference_trajectory = generate_global_reference_trajectory()

# Main simulation loop
max_steps = 500

for step in range(max_steps):
    env.render()
    
    current_state, obstacles = process_observation(obs)
    
    # Calculate distances to all detected obstacles
    distances = calculate_distances(current_state, obstacles)
    
    # Print the detected obstacles and their distances
    # print(f"Step {step}: Detected Obstacles: {obstacles}, Distances: {distances}")
    
    # Find the closest point on the reference trajectory
    closest_index = find_closest_point(current_state, global_reference_trajectory)
    
    # Use only the next 'horizon' points of the reference trajectory
    current_reference = global_reference_trajectory[closest_index:closest_index+horizon]
    
    action = mpc_control(current_state, global_reference_trajectory, obstacles, closest_index)
    
    # Print the speed, location, and acceleration of the vehicle
    x, y, v, psi = current_state
    a, delta = action
    print(f"Step {step}: Location (x, y) = ({x:.2f}, {y:.2f}), Speed = {v:.2f} m/s, Acceleration = {a:.2f} m/s²")
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated or closest_index >= len(global_reference_trajectory) - horizon:
        print(f"Finished after {step+1} steps")
        break

env.close()