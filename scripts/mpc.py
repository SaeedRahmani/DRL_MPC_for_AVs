import gymnasium as gym
import highway_env
import numpy as np
from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt
import mpc_controller as mpc

"""pure mpc with not normalized intersectionv1 for testing"""

env = gym.make("intersectionmpc-v1", render_mode="rgb_array")


obs, info = env.reset()


# MPC parameters
horizon = 6
dt = 0.1  # Increased time step

# Vehicle parameters
LENGTH = 5.0
WIDTH = 2.0
WHEELBASE = 2.5

MIN_SPEED = 0

original_reference_trajectory = mpc.generate_global_reference_trajectory()
global_reference_trajectory = original_reference_trajectory.copy()
ref_path = [(x, y) for x, y, v, psi in global_reference_trajectory]

# Variables to store real path
real_path = []
plt.ion()
# Main simulation loop
max_steps = 500
crashed = 0
sim = 0
resume_original_trajectory = False
collision_wait_time = 0.3  # 0.3 seconds
collision_timer = 0

for step in range(max_steps):
    env.render()
    
    current_state, obstacles, directions = mpc.process_observation(obs)
    print('current state:', current_state)
    real_path.append((current_state[0], current_state[1]))  # Append current position to the path

    # Predict future positions of obstacles
    extended_horizon = 12  #prediction horizon for other vehicles
    predicted_obstacles = mpc.predict_others_future_positions(obstacles, current_state[2], extended_horizon, dt)
    
    # Find the closest point on the reference trajectory
    closest_index = mpc.find_closest_point(current_state, global_reference_trajectory)
    # Use only the next 'horizon' points of the reference trajectory
    current_reference = global_reference_trajectory[closest_index:closest_index+horizon]

    # Check for collisions
    collision_points, collision_detected = mpc.check_collisions(ref_path, predicted_obstacles, start_index=closest_index)
    
    if collision_detected:
        global_reference_trajectory = mpc.generate_global_reference_trajectory(collision_points, speed_override = None)
        ref_path = [(x, y) for x, y, v, psi in global_reference_trajectory]
        resume_original_trajectory = False
        collision_timer = 0
    elif not resume_original_trajectory and not collision_detected:
        collision_timer += dt
        if collision_timer >= collision_wait_time:
            global_reference_trajectory = original_reference_trajectory
            ref_path = [(x, y) for x, y, v, psi in global_reference_trajectory]
            resume_original_trajectory = True

    action = mpc.mpc_control(current_state, current_reference, obstacles, closest_index, collision_detected)
    
    x, y, v, psi = current_state
    a, delta = action
    print(f"Step {step}: Location (x, y) = ({x:.2f}, {y:.2f}), Speed = {v:.2f} m/s, Acceleration = {a:.2f} m/s²\n")
    print(f"Step {step}: Closest index: {closest_index}, Current Reference trajectory: {current_reference}\n")
    obs, reward, done, truncated, info = env.step(action)
    
    
    # Update the current state after taking the action
    current_state, obstacles,directions = mpc.process_observation(obs)
    
    # Update the plot
    mpc.plot_trajectory(real_path, ref_path, predicted_obstacles, collision_points, directions)
    
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