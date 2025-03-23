import numpy as np
import gymnasium
import highway_env
from pprint import pprint

np.set_printoptions(suppress=True)

if __name__ == "__main__":
    # Create the MPC environment
    env = gymnasium.make("intersection-mpc-v0", render_mode="rgb_array")
    
    # Print environment information for debugging
    # print(f"Environment: {env}")
    # print(f"Observation space: {env.observation_space}")
    # print(f"Action space: {env.action_space}")
    
    # Reset the environment
    obs, info = env.reset()
    
    # Run simulation
    for i in range(200):
        # For MPC environments, we pass an empty a dummy action. but
        # The MPC controller inside the environment will generate the actual control (inside _simulate)
        action = env.action_space.sample()  # Sample a valid action from the action space to start the simulation
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        # Checking the reward
        # print(f"Step {i}, Reward: {reward}")
        
        if terminated or truncated:
            # print("Episode finished")
            break

    env.close()