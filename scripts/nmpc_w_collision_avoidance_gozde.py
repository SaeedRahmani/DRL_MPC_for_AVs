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
dt = 0.1  # Increased time step

# Vehicle parameters
LENGTH = 5.0
WIDTH = 2.0
WHEELBASE = 2.5
buffer = 2

MIN_SPEED = 0

def generate_global_reference_trajectory():
    trajectory = []
    x, y, v, v_f, heading = 2, 50, 10, 0, - np.pi/2  # Starting with 10 m/s speed
    
    turn_start_y = 20
    radius = 5  # Radius of the curve
    turn_angle = np.pi / 2  # Total angle to turn (90 degrees for a left turn)

    # Go straight until reaching the turn start point
    for _ in range(40):
        x += 0
        y += v * dt * math.sin(heading)
        trajectory.append((x, y, v, heading))
    
    # Compute the turn
    angle_increment = turn_angle / 20  # Divide the turn into 20 steps
    for _ in range(20):
        heading += angle_increment  # Decrease heading to turn left
        x -= v * dt * math.cos(heading)
        y += v * dt * math.sin(heading)
     
        trajectory.append((x, y, v, heading))
    
    # Continue straight after the turn
    for _ in range(25):  # Continue for a bit after the turn
        x -= v * dt * math.cos(heading)
        y += 0
       
        trajectory.append((x, y, v, heading))
    
    return trajectory

def vehicle_model(state, action):
    x, y, v, psi = state
    a, delta = action
    
    beta = math.atan(0.5 * math.tan(delta))
    x_next = x + v * math.cos(psi + beta) * dt
    y_next = y + v * math.sin(psi + beta) * dt
    v_next = max(0, v + a * dt)  # Ensure that speed does not go negative
    psi_next = psi + v * math.sin(beta) / WHEELBASE * dt
    
    return np.array([x_next, y_next, v_next, psi_next])

def find_closest_point(current_state, reference_trajectory):
    current_position = current_state[:2]
    distances = [np.linalg.norm(current_position - np.array(point[:2])) for point in reference_trajectory]
    closest_index = np.argmin(distances)
    return closest_index

def predict_vehicle_positions(state, action, steps, dt=0.1):
    """ Predict future positions of a vehicle based on current state and action """
    state = np.array(state)
    predictions = []

    for _ in range(steps):
        state = vehicle_model(state, action)
        predictions.append(state[:2])  # Append only x, y positions
    
    return predictions

def predict_others_future_positions(obstacles, steps, dt):
    """ Predict future positions of all obstacles """
    future_positions = []
    
    for obstacle in obstacles:
        if len(obstacle) == 2:
            x, y = obstacle
            vx, vy = 0, 0  # Default to zero velocity if not provided
            psi = 0
        else:
            x, y, vx, vy = obstacle
            psi = np.arctan2(vy, vx) if vx or vy else 0

        v = np.sqrt(vx**2 + vy**2)
        state = [x, y, v, psi]
        
        action = [0, 0]  # Assuming constant velocity model for simplicity
        future_positions.append(predict_vehicle_positions(state, action, steps, dt))
    
    return future_positions

def cost_function(u, current_state, reference_trajectory, obstacles, start_index, dt, collision_detected):
    state = current_state
    
    total_cost = 0
    extended_horizon = 10  # Extend the prediction horizon for other vehicles
    predicted_obstacles = predict_others_future_positions(obstacles, extended_horizon, dt)
    
    
    
    for i in range(horizon):
        action = u[i*2:(i+1)*2]
        state = vehicle_model(state, action)
        
        ref_index = min(start_index + i, len(reference_trajectory) - 1)
        ref_x, ref_y, ref_v, ref_heading = reference_trajectory[ref_index]
        
        # Calculate perpendicular and parallel deviations
        dx = state[0] - ref_x
        dy = state[1] - ref_y
        perp_deviation = dx * np.sin(ref_heading) - dy * np.cos(ref_heading)
        para_deviation = dx * np.cos(ref_heading) + dy * np.sin(ref_heading)

        state_cost = 20 * perp_deviation**2 + 1 * para_deviation**2 + 1 * (state[2] - ref_v)**2 + 0.1 * (state[3] - ref_heading)**2
        control_cost = 0.1 * action[0]**2 + 0.1 * action[1]**2
        
        obstacle_cost = 0
        for obs_future_positions in predicted_obstacles:
            obs_x, obs_y = obs_future_positions[min(i, len(obs_future_positions) - 1)]
            distance_to_obstacle = np.sqrt((state[0] - obs_x)**2 + (state[1] - obs_y)**2)
            if distance_to_obstacle < 1.0:
                obstacle_cost += 1000 / (distance_to_obstacle + 1e-6)**2
                
            else:
                obstacle_cost += 100 / (distance_to_obstacle + 1e-6)**2
        
        total_cost += state_cost + control_cost + obstacle_cost
        
        if collision_detected:
           # state[2] = 0  # Set speed to zero if a collision is detected
            total_cost += 5000 * state[2]**2  # Penalize speed if a collision is detected
            

        if i > 0:
            prev_action = u[(i-1)*2:i*2]
            input_diff_cost = 0.5 * ((action[0] - prev_action[0])**2 + (action[1] - prev_action[1])**2)
            total_cost += input_diff_cost

        desired_final_state = reference_trajectory[-1]
        final_state_cost = 100 * ((state[0] - desired_final_state[0])**2 + (state[1] - desired_final_state[1])**2 + 
                                  (state[2] - desired_final_state[2])**2 + (state[3] - desired_final_state[3])**2)
        
        total_cost += state_cost + control_cost + obstacle_cost + final_state_cost 
    
    return total_cost

def mpc_control(current_state, reference_trajectory, obstacles, start_index,collision_detected):
    u0 = np.zeros(2 * horizon)
    
    bounds = [(-10, 3), (-np.pi/2, np.pi/2)] * horizon
    
    result = minimize(cost_function, u0, args=(current_state, reference_trajectory, obstacles, start_index, dt,collision_detected), 
                      method='SLSQP', bounds=bounds, options={'maxiter': 100})
    
    return result.x[:2]  # Return only the first action

def process_observation(obs):
    ego_vehicle = obs[0]
    
    x, y = ego_vehicle[1], ego_vehicle[2]
    vx, vy = ego_vehicle[3], ego_vehicle[4]
    v = np.sqrt(vx**2 + vy**2)
    
    psi = np.arctan2(vy, vx)  # Calculate heading from velocity
    
    current_state = np.array([x, y, v, psi])
    
    obstacles = []
    for vehicle in obs[1:]:
        if vehicle[0] == 1:
            ox, oy = vehicle[1], vehicle[2]
            ovx = vehicle[3] if len(vehicle) > 3 else 0  # Set default velocity components if not available
            ovy = vehicle[4] if len(vehicle) > 4 else 0
            obstacles.append([ox, oy, ovx, ovy])
    
    return current_state, obstacles

def calculate_distances(current_state, obstacles):
    current_position = current_state[:2]
    distances = [np.linalg.norm(current_position - np.array(obs[:2])) for obs in obstacles]
    return distances

def plot_trajectory(real_path, ref_path, predicted_obstacles, collision_points):
    plt.clf()  # Clear the current figure to update the plot dynamically
    buffer = 2.6
    if ref_path:  # Check if reference path is not empty
        plt.plot(*zip(*ref_path), 'r--', label='Reference Trajectory')  # Plot reference trajectory
    if real_path:  # Check if real path is not empty
        plt.plot(*zip(*real_path), 'b-', label='Ego Vehicle Path')  # Plot actual path of the vehicle
        plt.scatter(real_path[-1][0], real_path[-1][1], color='blue', s=50, label='Current Position')  # Mark the current position

    # Plot predicted positions of other vehicles
    for i, obs_future_positions in enumerate(predicted_obstacles):
        if obs_future_positions:  # Ensure there are predicted positions
            plt.plot(*zip(*obs_future_positions), 'g--', label=f'Predicted Obstacle Path {i}')

    # Highlight collision points
    for collision_point in collision_points:
        plt.scatter(*collision_point, color='red', s=100, label='Predicted Collision')
    
    # Plot the buffer around the ego vehicle path
    half_width = buffer / 2
    for px, py in real_path:
        rectangle = plt.Rectangle((px - half_width, py - half_width), buffer, buffer, linewidth=1, edgecolor='purple', facecolor='none', linestyle='--')
        plt.gca().add_patch(rectangle)
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Vehicle Trajectory and Reference Path')
    plt.legend()
    plt.pause(0.1)

def get_imaginary_lines(x, y, psi, length=5.0):
    """ Calculate points on the imaginary lines extending from the vehicle's front and back. """
    front_x = x + (length * np.cos(psi))
    front_y = y + (length * np.sin(psi))
    
    back_x = x - (length * np.cos(psi))
    back_y = y - (length * np.sin(psi))
    
    return (front_x, front_y), (back_x, back_y)

def check_collisions(predicted_ego_path, predicted_obstacles,  start_index=0):
    collision_points = []
    collision_detected = False
    buffer = 2.6
    half_width = buffer / 2
    time_steps_window = int(1.0 / dt)  # Number of steps to check within the time window
    
    for step, (px, py) in enumerate(predicted_ego_path[start_index:]):
        ego_box = [(px - half_width, py - half_width), (px + half_width, py + half_width)]
        for obs_future_positions in predicted_obstacles:
            for obs_step in range(max(0, step - time_steps_window), min(len(obs_future_positions), step + time_steps_window + 1)):
                ox, oy = obs_future_positions[obs_step]
                distance_to_obstacle = np.sqrt((px - ox)**2 + (py - oy)**2)
                if distance_to_obstacle < 1.0:
                    if (ego_box[0][0] <= ox <= ego_box[1][0]) and (ego_box[0][1] <= oy <= ego_box[1][1]):
                        collision_points.append((px, py))
                        collision_detected = True
                          # No need to check further if a collision is detected in this step
    
    print(collision_points, "collision----")
    return collision_points, collision_detected


# Generate the global reference trajectory
global_reference_trajectory = generate_global_reference_trajectory()
ref_path = [(x, y) for x, y, v, psi in global_reference_trajectory]

# Variables to store real path
real_path = []
plt.ion()
# Main simulation loop
max_steps = 500
crashed = 0
sim = 0
for step in range(max_steps):
    env.render()
    
    current_state, obstacles = process_observation(obs)
    print('current state:', current_state)
    real_path.append((current_state[0], current_state[1]))  # Append current position to the path

    # Predict future positions of obstacles
    extended_horizon = 14  # Extended prediction horizon for other vehicles
    predicted_obstacles = predict_others_future_positions(obstacles, extended_horizon, dt)
    
    # Find the closest point on the reference trajectory
    closest_index = find_closest_point(current_state, global_reference_trajectory)
    # Use only the next 'horizon' points of the reference trajectory
    current_reference = global_reference_trajectory[closest_index:closest_index+horizon]

    # Check for collisions
    collision_points, collision_detected = check_collisions(ref_path, predicted_obstacles, start_index=closest_index)
    
    action = mpc_control(current_state, current_reference, obstacles, closest_index,collision_detected)
    
    x, y, v, psi = current_state
    a, delta = action
    print(f"Step {step}: Location (x, y) = ({x:.2f}, {y:.2f}), Speed = {v:.2f} m/s, Acceleration = {a:.2f} m/s²\n")
    print(f"Step {step}: Closest index: {closest_index}, Current Reference trajectory: {current_reference}\n")
    obs, reward, done, truncated, info = env.step(action)
    
    # Update the current state after taking the action
    current_state, _ = process_observation(obs)
    
    
    # Update the plot
    plot_trajectory(real_path, ref_path, predicted_obstacles, collision_points)
    
    if info["crashed"]:
        crashed += 1
    
    if done or truncated or closest_index >= len(global_reference_trajectory) - horizon:
        print(f"Finished after {step+1} steps")
        obs, info = env.reset()
        real_path = []
        sim += 1
        print("crash count:", crashed, "sim count:", sim)
        continue

plt.ioff()  # Turn off interactive mode
plt.show()
env.close()
