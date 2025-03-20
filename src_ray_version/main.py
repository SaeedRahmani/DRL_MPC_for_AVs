import numpy as np
import gymnasium
import highway_env

from pprint import pprint

from highway_env.envs import IntersectionMpcEnv, IntersectionEnv


if __name__ == "__main__":
    env = gymnasium.make("intersection-mpc-v0", render_mode="rgb_array")
    print(env, env.observation_space.shape, env.action_space.n)
    pprint(env.unwrapped.default_config())

    obs: np.ndarray
    obs, _ = env.reset()

    for i in range(100):
        action = np.random.random(env.action_space.n)  # agent predicts
        obs, reward, terminated, truncated, info = env.step(action=action)
        env.render()
        if terminated:
            break
