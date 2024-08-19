import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

# Create the environment
env = make_vec_env("intersectiondrl-v5", n_envs=1)

# Set training mode (adjust based on your environment structure)
try:
    env.env_method("set_training_mode", True)
except AttributeError:
    print("Warning: Could not set training mode. Make sure your environment supports this method.")

# Initialize the SAC model
model = SAC('MlpPolicy', env,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=0.0005,
            buffer_size=15000,
            learning_starts=200,
            batch_size=64,
            tau=0.02,
            gamma=0.8,
            train_freq=1,
            gradient_steps=1,
            optimize_memory_usage=False,
            verbose=1,
            device='auto',
            tensorboard_log="intersection_sac/")

# Training parameters
num_episodes = 1000
episode_rewards = []

# Set up the plot for live updating
plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
line, = ax.plot([], [])
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.set_title('Episode Rewards During Training')

# Custom training loop
for episode in range(num_episodes):
    obs = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        step_result = env.step(action)
        
        # Handle both Gym (4 returns) and Gymnasium (5 returns) interfaces
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_obs, reward, done, info = step_result
        
        # Process the info object
        processed_info = {"TimeLimit.truncated": False}
        if isinstance(info, dict):
            processed_info["TimeLimit.truncated"] = info.get("TimeLimit.truncated", False)
        
        model.replay_buffer.add(
            obs, 
            next_obs, 
            action, 
            reward, 
            np.array([done]), 
            [processed_info]
        )
        
        obs = next_obs
        episode_reward += reward.item()
        
        if model.num_timesteps > model.learning_starts:
            model.train()
    
    episode_rewards.append(episode_reward)
    
    print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}")
    
    # Update the plot
    line.set_xdata(range(len(episode_rewards)))
    line.set_ydata(episode_rewards)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

# Save the model
model.save("intersection_sac_custom/model")

# Final plot update
plt.ioff()
plt.show()

# Set the environment back to evaluation mode
try:
    env.env_method("set_training_mode", False)
except AttributeError:
    print("Warning: Could not set evaluation mode. Make sure your environment supports this method.")

# Your evaluation code here...