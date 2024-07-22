import gymnasium as gym
import highway_env
import numpy as np
from scipy.optimize import minimize
import math

# Initialize the environment
env = gym.make("intersection-v1", render_mode="rgb_array")
obs, info = env.reset()

# MPC parameters
horizon = 12
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
    distances = [np.linalg.norm(current_position - np.array(point[:2])) for point in reference_trajectory]
    closest_index = np.argmin(distances)
    return closest_index

def cost_function(u, current_state, reference_trajectory, obstacles, start_index):
    state = current_state
    total_cost = 0
    
    for i in range(horizon):
        action = u[2*i:2*i+2]
        state = vehicle_model(state, action)
        
        ref_index = min(start_index + i, len(reference_trajectory) - 1)
        ref_x, ref_y, ref_v, ref_heading = reference_trajectory[ref_index]
        
        state_cost = (state[0] - ref_x)**2 + (state[1] - ref_y)**2 + 0.5*(state[2] - ref_v)**2 + 0.5*(state[3] - ref_heading)**2
        control_cost = 0.1 * np.sum(action**2)
        
        obstacle_cost = 0
        for obs in obstacles:
            dist = np.linalg.norm(state[:2] - obs[:2])
            if dist < 2.0:  # Only penalize if within a certain distance
                obstacle_cost += 1000 / (dist + 1e-6)**2
        
        total_cost += state_cost + control_cost + obstacle_cost
    
    return total_cost

def mpc_control(current_state, reference_trajectory, obstacles, start_index):
    u0 = np.zeros(2 * horizon)
    
    bounds = [(-10, 3), (-np.pi/4, np.pi/4)] * horizon
    
    result = minimize(cost_function, u0, args=(current_state, reference_trajectory, obstacles, start_index), 
                      method='SLSQP', bounds=bounds, options={'maxiter': 100})
    
    return result.x[:2]  # Return only the first action

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

def shift_movement(T, t0, u, x_n):
    obs, reward, terminated, truncated, info = env.step(u)
    print(f"obs structure: {obs}")  # Add this line to inspect the structure of obs
    done = terminated or truncated
    
    # Adjust unpacking based on actual structure of obs
    x, y, vx, vy, yaw = obs[0][:5]  # Assuming obs[0] is an array with at least 5 elements
    env.render()
    st = np.array([x, y, yaw, info['speed'], math.hypot(vx, vy)])
    t = t0 + T
    u_end = np.concatenate((u[1:], u[-1:]))
    x_n = np.concatenate((x_n[1:], x_n[-1:]))
    
    obstacle_data = [vehicle for vehicle in obs[1:] if vehicle[0] == 1]
    obstacles = np.array([[veh[1], veh[2]] for veh in obstacle_data])
    
    return t, st, u_end, x_n, obstacles, reward, done

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
    print(f"Step {step}: Detected Obstacles: {obstacles}, Distances: {distances}")
    
    print(f"Step {step}: Vehicle position: x = {current_state[0]}, y = {current_state[1]}")
    
    # Find the closest point on the reference trajectory
    closest_index = find_closest_point(current_state, global_reference_trajectory)
    
    # Use only the next 'horizon' points of the reference trajectory
    current_reference = global_reference_trajectory[closest_index:closest_index+horizon]
    
    action = mpc_control(current_state, global_reference_trajectory, obstacles, closest_index)
    
    # Print the speed, location, and acceleration of the vehicle
    x, y, v, psi = current_state
    a, delta = action
    print(f"Step {step}: Location (x, y) = ({x:.2f}, {y:.2f}), Speed = {v:.2f} m/s, Acceleration = {a:.2f} m/s²")
    
    # Update the state using shift_movement
    t0 = step * dt
    u = [a, delta]
    t0, current_state, u, x_n, obstacles, reward, done = shift_movement(dt, t0, u, current_state)
    
    if done or closest_index >= len(global_reference_trajectory) - horizon:
        print(f"Finished after {step+1} steps")
        break

env.close()
