import gymnasium as gym
import highway_env
import hydra
import numpy as np
from pprint import pprint
from agents.pure_mpc import PureMPC_Agent
from config.config import build_env_config, build_pure_mpc_agent_config

np.set_printoptions(suppress=True)

@hydra.main(config_name="cfg", config_path="./config", version_base="1.3")
def test_pure_mpc_agent(cfg):
    # config
    gym_env_config = build_env_config(cfg)
    pure_mpc_agent_config = build_pure_mpc_agent_config(cfg)

    # env
    env = gym.make("intersection-v1", render_mode="rgb_array", config=gym_env_config)
    pprint(env.unwrapped.default_config())
    
    # agent
    mpc_agent = PureMPC_Agent(env, pure_mpc_agent_config)
    pprint(mpc_agent.default_weights)
    
    observation, _ = env.reset()
    
    # print(env.unwrapped.road.network.graph['o0']['ir0'][0].start)

    # for _, roads in env.unwrapped.road.network.graph.items():
    #     for _, road in roads.items():
    #         print(road[0].start, road[0].end)

    for i in range(150):
        # getting action from agent
        action = mpc_agent.predict(observation, False)
        # mpc_agent.plot()
        # mpc_agent.visualize_predictions()
        # print(np.array([action.acceleration, action.steer]))
        observation, reward, done, truncated, info = env.step([action.acceleration/5, action.steer/(np.pi/3)])
        # observation, reward, done, truncated, info = env.step([action.acceleration, action.steer])
        # print('speed', observation[0,3])
        # print('obs', observation[0][0:8])
        # rendering animation
        env.render()
        
        # checking end conditions
        if done or truncated:
            break
            state = env.reset()

    # destroy all handles
    env.close()


if __name__ == "__main__":
    test_pure_mpc_agent()