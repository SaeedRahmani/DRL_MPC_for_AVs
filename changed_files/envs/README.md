# This is a instruction file for the uploaded files. Please write descriptions of each file you upload and why is it uploaded.

## intersectiondrl_env_temp.py
I added this file to be able to run the RL model. Changes comapred to original intersectiondrl_env.py include:
- the collision checking is changed a bit
- a condition is added to do not plot the simulation during training

but it should be checked with the other environment code and merge them.

## __init__ file
for registering the new env

## Env files
different environment files for difference purposes. @GozdeKorpe: please add a brief explanation for each env and what it is doing. Thanks!