import numpy as np
import gymnasium
import highway_env

from pprint import pprint

np.set_printoptions(suppress=True)

if __name__ == "__main__":

    # env = gymnasium.make("intersection-mpcrl-v0", render_mode="rgb_array")
    # env = gymnasium.make("intersection-mpcrl-v1", render_mode="rgb_array")

    env = gymnasium.make("intersection-mpc-v0", render_mode="rgb_array")
    # print(env, env.observation_space.shape, env.action_space.shape)
    pprint(env.unwrapped.default_config())

    obs: np.ndarray
    obs, _ = env.reset()

    for i in range(200):
        # action = np.random.random(env.action_space.n)  # agent predicts
        obs, reward, terminated, truncated, info = env.step(action=None)
        env.render()
        if terminated:
            break

    env.close()
