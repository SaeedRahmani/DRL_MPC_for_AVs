import highway_env
from highway_env.envs import IntersectionMpcEnv


if __name__ == "__main__":
    env = IntersectionMpcEnv()    
    print(env, env.observation_space, env.action_space)
    obs, info = env.reset()
    