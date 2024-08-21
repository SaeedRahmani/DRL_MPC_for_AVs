import gymnasium as gym
import highway_env
import numpy as np
from scipy.optimize import minimize
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
    x, y, v, heading = 2, 60, 10, - np.pi/2  # Starting with 10 m/s speed
    
    turn_start_y = 20
    radius = 10  # Radius of the curve
    turn_angle = np.pi / 2  # Total angle to turn (90 degrees for a left turn)

    # Go straight until reaching the turn start point
    for _ in range(50):
        x += 0
        y += v * dt * math.sin(heading)
        trajectory.append((x, y, v, heading))
    
    # Compute the turn
    angle_increment = turn_angle / 30  # Divide the turn into 20 steps
    for _ in range(30):
        heading += angle_increment  # Decrease heading to turn left
        x += v * dt * math.cos(heading)
        y += v * dt * math.sin(heading)
        trajectory.append((x, y, v, heading))
    
    # Continue straight after the turn
    for _ in range(50):  # Continue for a bit after the turn
        x += v * dt * math.cos(heading)
        y += 0
        trajectory.append((x, y, v, heading))
    
    return trajectory

def vehicle_model(state, action):
    x, y, v, psi = state
    # print('heading of the vehicle mode:', psi)
    a, delta = action
    # print('acceleration:', a, 'steering:', delta)
    
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
    print('CURRENT STATE: x:', state[0], 'y:', state[1], 'v:', state[2], 'heading:', state[3])
    print('REFERENCE TRAJECTORY:', reference_trajectory)
    total_cost = 0
    (front_x, front_y), (back_x, back_y) = get_imaginary_lines(state[0], state[1], state[3])
    
    for i in range(horizon):
        # print('u:', u)
        action = u[i:i+2] # check this 
        # print('ACTION:', action)
        state = vehicle_model(state, action)
        # print('HORIZON STATES: x:', state[0], 'y:', state[1], 'v:', state[2], 'heading:', state[3])

        ref_index = min(start_index + i, len(reference_trajectory) - 1)
        ref_x, ref_y, ref_v, ref_heading = reference_trajectory[ref_index]
        state_cost = 20*(state[0] - ref_x)**2 + 1*(state[1] - ref_y)**2 + 1*(state[2] - ref_v)**2 + 0*(state[3] - ref_heading)**2
        control_cost = 0.1 * np.sum(action**2) # Change this 
        
        obstacle_cost = 0
        for obs in obstacles:
            obs_x, obs_y = obs
            # Distance from the front and back line
            dist_to_front_line = np.abs((front_y - obs_y) - np.tan(state[3]) * (obs_x - front_x)) / np.sqrt(1 + np.tan(state[3])**2)
            dist_to_back_line = np.abs((back_y - obs_y) - np.tan(state[3]) * (obs_x - back_x)) / np.sqrt(1 + np.tan(state[3])**2)
            
            # Check if the obstacle is close to the lines
            if dist_to_front_line < 2.0 or dist_to_back_line < 2.0:
                # Significantly higher cost if the obstacle is on the imaginary line
                obstacle_cost += 1000 / (dist_to_front_line + 1e-6)**2
            else:
                # Normal cost if not directly on the line
                obstacle_cost += 100 / (np.linalg.norm(state[:2] - np.array([obs_x, obs_y])) + 1e-6)**2
        
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
    # psi = ego_vehicle[5]  # Using provided heading
    psi = np.arctan2(vy, vx)  # Calculate heading from velocity
    
    current_state = np.array([x, y, v, psi])
    
    obstacles = [vehicle[1:3] for vehicle in obs[1:] if vehicle[0] == 1]
    
    return current_state, obstacles

def calculate_distances(current_state, obstacles):
    current_position = current_state[:2]
    distances = [np.linalg.norm(current_position - np.array(obs)) for obs in obstacles]
    return distances

def plot_trajectory(real_path, ref_path):
    plt.clf()  # Clear the current figure to update the plot dynamically
    if ref_path:  # Check if reference path is not empty
        plt.plot(*zip(*ref_path), 'r--', label='Reference Trajectory')  # Plot reference trajectory
    if real_path:  # Check if real path is not empty
        plt.plot(*zip(*real_path), 'b-', label='Ego Vehicle Path')  # Plot actual path of the vehicle
        plt.scatter(real_path[-1][0], real_path[-1][1], color='blue', s=50, label='Current Position')  # Mark the current position
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Vehicle Trajectory and Reference Path')
    plt.legend()
    plt.pause(0.1)  # Pause to update the plot

def get_imaginary_lines(x, y, psi, length=5.0):
    """ Calculate points on the imaginary lines extending from the vehicle's front and back. """
    # Front line endpoint
    front_x = x + (length * np.cos(psi))
    front_y = y + (length * np.sin(psi))
    
    # Back line endpoint (opposite direction)
    back_x = x - (length * np.cos(psi))
    back_y = y - (length * np.sin(psi))
    
    return (front_x, front_y), (back_x, back_y)



# Generate the global reference trajectory
global_reference_trajectory = generate_global_reference_trajectory()
ref_path = [(x, y) for x, y, v, psi in global_reference_trajectory]

# Variables to store real path
real_path = []
plt.ion()
# Main simulation loop
max_steps = 500

for step in range(max_steps):
    env.render()
    
    current_state, obstacles = process_observation(obs)
    print('current state:', current_state)
    real_path.append((current_state[0], current_state[1]))  # Append current position to the path

    # Update the plot
    plot_trajectory(real_path, ref_path)
    
    # Calculate distances to all detected obstacles
    distances = calculate_distances(current_state, obstacles)
    
    # Print the detected obstacles and their distances
    # print(f"Step {step}: Detected Obstacles: {obstacles}, Distances: {distances}")
    
    print(f"Step {step}: Vehicle position: x = {current_state[0]}, y = {current_state[1]}")

    
    # Find the closest point on the reference trajectory
    closest_index = find_closest_point(current_state, global_reference_trajectory)
    # Use only the next 'horizon' points of the reference trajectory
    current_reference = global_reference_trajectory[closest_index:closest_index+horizon]
    
    action = mpc_control(current_state, current_reference, obstacles, closest_index)
    
    # Print the speed, location, and acceleration of the vehicle
    x, y, v, psi = current_state
    a, delta = action
    print(f"Step {step}: Location (x, y) = ({x:.2f}, {y:.2f}), Speed = {v:.2f} m/s, Acceleration = {a:.2f} m/s²\n")
    # print(f"Global Reference trajectory: {global_reference_trajectory}\n")
    print(f"Step {step}: Closest index: {closest_index}, Current Reference trajectory: {current_reference}\n")
    obs, reward, terminated, truncated, info = env.step(action)
    print('Observation:', obs)
    print('Reward:', reward)
    
    
    if terminated or truncated or closest_index >= len(global_reference_trajectory) - horizon:
        print(f"Finished after {step+1} steps")
        obs, info = env.reset()
        real_path = []
        continue
plt.ioff()  # Turn off interactive mode
plt.show()
env.close()