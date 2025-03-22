import gymnasium as gym
import highway_env
import hydra
import numpy as np

from agents.pure_mpc_no_collision import PureMPC_Agent
from config.config import build_env_config, build_pure_mpc_agent_config

np.set_printoptions(suppress=True)

@hydra.main(config_name="cfg", config_path="./config", version_base="1.3")
def test_pure_mpc_agent(cfg):
    # config
    gym_env_config = build_env_config(cfg)
    pure_mpc_agent_config = build_pure_mpc_agent_config(cfg, use_collision_avoidance=False)
    
    # env
    env = gym.make("intersection-v1", render_mode="rgb_array", config=gym_env_config)
    print(env.unwrapped.default_config())

    # agent
    mpc_agent = PureMPC_Agent(env, pure_mpc_agent_config)

    observation, _ = env.reset()
    
    # Test different reference speeds
    test_speeds = [5.0, 10.0, 15.0]  # Test three different speeds
    current_speed_idx = 0
    steps_per_speed = 33  # Change speed every 33 steps
    step_count = 0
    
    for i in range(200):
        # Update reference speed periodically
        if i % steps_per_speed == 0 and current_speed_idx < len(test_speeds):
            ref_speed = test_speeds[current_speed_idx]
            current_speed_idx = (current_speed_idx + 1) % len(test_speeds)
            print(f"\nSetting new reference speed: {ref_speed} m/s")

        # getting action from agent
        action = mpc_agent.predict(
            observation, 
            return_numpy=False,
            ref_speed=np.array([[ref_speed]])  # Pass current reference speed
        )
        
        # Print current state
        # print(f"Step {i}: Speed={observation[0,3]:.2f}, Ref={ref_speed:.2f}, "
        #       f"Acceleration={action.acceleration:.2f}")
        # print(action)
        
        # Take step in environment
        observation, reward, done, truncated, info = env.step(
            [action.acceleration/5, action.steer/(np.pi/3)]
        )
        
        # Render for visualization
        env.render()
        
        if done or truncated:
            break
            observation, _ = env.reset()

    env.close()

if __name__ == "__main__":
    test_pure_mpc_agent()