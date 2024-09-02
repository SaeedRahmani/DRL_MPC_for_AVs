import gymnasium as gym
import highway_env
import numpy as np
from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt
import highway_env.envs.mpc_controller as mpc
from stable_baselines3 import DDPG,SAC
import torch
import pprint


env = gym.make("intersectiondrl-v5", render_mode="rgb_array")


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

"""model = SAC('MlpPolicy', env,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=0.0005,  # Adjusted learning rate
            buffer_size=15000,  # Increased buffer size
            learning_starts=200,  # Increased learning starts
            batch_size=64,  # Increased batch size
            tau=0.02,  # Adjusted tau
            gamma=0.8,  # Adjusted gamma
            train_freq=1,
            gradient_steps=1,  # Do as many gradient steps as steps done in the environment during the rollout
            optimize_memory_usage=False,
            verbose=1,
            device='auto',
            tensorboard_log="intersection_sac/")


# uncomment the lines below if you want to train a new model

model.learn(total_timesteps=int(1000),progress_bar=True)  


model.save("intersection_sac11/model")"""

model = SAC.load("intersection_sac_custom/model")
for step in range(max_steps):
    action, _states = model.predict(obs, deterministic=True)
    env.render()
    
    current_state, obstacles, directions = mpc.process_observation(obs)
      
    
    x, y, v, psi = current_state
    a, delta,speed = action
    print(f"Step {step}: Location (x, y) = ({x:.2f}, {y:.2f}), Speed = {v:.2f} m/s, Acceleration = {a:.2f} m/s²\n")
    
    #print(f"Step {step}: Closest index: {closest_index}, Current Reference trajectory: {current_reference}\n")
    obs, reward, done, truncated, info = env.step(action)
    speed_override  = env.simulate(action)
    print(f"speed override:{speed_override:.2f}")
    
    
    #print("speed_override:",speed_override)
    
    # Update the current state after taking the action
    current_state, obstacles,directions = mpc.process_observation(obs)
    
    # Update the plot
    #mpc.plot_trajectory(real_path, ref_path, predicted_obstacles, collision_points, directions)
    
    if info["crashed"]: 
        crashed += 1
    
    if done or truncated:
        print(f"Finished after {step+1} steps")
        obs, info = env.reset()
        real_path = []
        sim += 1
        print("crash count:", crashed, "sim count:", sim)
        continue
    

plt.ioff()  # Turn off interactive mode
plt.show()
env.close()

