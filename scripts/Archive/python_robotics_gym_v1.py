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

def read_reference_trajectory():
    # filename = 'inter1.txt'
    # r_x, r_y, r_theta = [], [], []
    # with open(filename, 'r') as f:
    #     for line in f:
    #         line = line.strip(',').split(',')
    #         r_x.append(float(line[0]))
    #         r_y.append(float(line[1]))
    #         r_theta.append(float(line[2]))
    # return np.array(r_x), np.array(r_y), np.array(r_theta)
    length = 50
    dt = 0.1
    speed = 10
    
    r_x, r_y, r_theta = [], [], []
    x, y, theta = 0, 0, 0  # Starting position (x=0, y=0) and heading (theta=0, pointing along the x-axis)
    
    steps = int(length / (speed * dt))  # Calculate the number of steps needed to cover the length at the given speed

    for _ in range(steps):
        x += speed * dt * np.cos(theta)
        y += speed * dt * np.sin(theta)
        r_x.append(x)
        r_y.append(y)
        r_theta.append(theta)  # Heading remains constant

    return np.array(r_x), np.array(r_y), np.array(r_theta)

def predict_interference(obstacle, obstacle_velocity, reference_trajectory, time_horizon):
    safety_distance = LENGTH + WIDTH + 2  # Add extra safety margin
    future_positions = [
        (obstacle[0] + obstacle_velocity[0] * t, obstacle[1] + obstacle_velocity[1] * t)
        for t in np.linspace(0, time_horizon * dt, 20)  # More granular prediction
    ]
    
    for ref_point in reference_trajectory[:time_horizon]:
        for future_pos in future_positions:
            if np.linalg.norm(np.array(ref_point[:2]) - np.array(future_pos)) < safety_distance:
                return True
    return False

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

    interfering_obstacles = [
        obs['position'] for obs in obstacles 
        if predict_interference(obs['position'], obs['velocity'], reference_trajectory[start_index:], horizon)
    ]
    
    collision_risk = len(interfering_obstacles) > 0
    
    for t in range(horizon):
        ref_index = min(start_index + t, len(reference_trajectory) - 1)
        ref_x, ref_y, ref_v, ref_heading = reference_trajectory[ref_index]
        
        A, B = linearized_vehicle_model(current_state, ref_v, ref_heading)
        
        constraints += [x[t+1] == A @ x[t] + B @ u[t]]
        constraints += [u[t, 0] >= -10, u[t, 0] <= 3]  # acceleration bounds
        constraints += [u[t, 1] >= -np.pi/4, u[t, 1] <= np.pi/4]  # steering bounds
        constraints += [x[t+1, 2] >= 0]  # allow full stop
        
        if collision_risk:
            # Encourage deceleration when there's a collision risk
            target_speed = max(0, ref_v - 5)  # Reduce target speed
            deceleration_weight = 10  # Increase weight on deceleration
        else:
            target_speed = ref_v
            deceleration_weight = 0.5
        
        objective += cp.sum_squares(x[t, :2] - [ref_x, ref_y]) + \
                     deceleration_weight * cp.square(x[t, 2] - target_speed) + \
                     0.5 * cp.square(x[t, 3] - ref_heading) + \
                     0.1 * cp.sum_squares(u[t])
        
        # Add obstacle avoidance constraint only for interfering obstacles
        for obs in interfering_obstacles:
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
    
    obstacles = []
    for vehicle in obs[1:]:
        if vehicle[0] == 1:  # If the vehicle is present
            obstacles.append({
                'position': vehicle[1:3],
                'velocity': vehicle[3:5]
            })
    
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

# Read the global reference trajectory from file
xnew, ynew, thetanew = read_reference_trajectory()
global_reference_trajectory = list(zip(xnew, ynew, [10]*len(xnew), thetanew))

# Main simulation loop
max_steps = 500

for step in range(max_steps):
    env.render()
    
    current_state, obstacles = process_observation(obs)
    
    # Calculate distances to all detected obstacles
    distances = [np.linalg.norm(current_state[:2] - np.array(obs['position'])) for obs in obstacles]
    
    # Find the closest point on the reference trajectory
    closest_index = find_closest_point(current_state, global_reference_trajectory)
    
    # Check for interfering obstacles
    interfering_obstacles = [
        obs for obs in obstacles 
        if predict_interference(obs['position'], obs['velocity'], global_reference_trajectory[closest_index:], horizon)
    ]
    
    print(f"Step {step}: Detected Obstacles: {len(obstacles)}, Interfering Obstacles: {len(interfering_obstacles)}")
    
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
