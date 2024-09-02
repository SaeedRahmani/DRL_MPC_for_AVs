import cv2
import gymnasium as gym
import numpy as np

import highway_env
from stable_baselines3 import DDPG,SAC
import torch
import pprint




env = gym.make("intersection-v1", render_mode ="rgb_array")




env.reset()

pprint.pprint(env.unwrapped.config)

model = SAC('MlpPolicy', env,
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


model.save("intersection_sac/model")


########## Load and test saved model##############

model = SAC.load("intersection_sac/model")
#while True:
for f in range(40):
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)#env.step(action.item(0))

    #print(reward)
    #print(info)
    #input("Press Enter to continue...") 
    env.render()
  


#cur_frame = env.render(mode="rgb_array")
#out.write(cur_frame)


print('DONE')


#print(env_reward())

#NOTE
#rewards is the gives rewards along different categories,
#reward combines the values from rewards into 1 value
#reward does this calculation using config and rewards

