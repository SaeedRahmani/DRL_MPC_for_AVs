# RLlib-optimized MPC_RL_for_AVs

## TODO

- [x] Define new obs/action space?
  - [ ] V1 - Ref speed
  - [ ] V2 - Dynamic weights  
- [x] Use `gym.make("env_name-verison", render_mode)`
- [x] Add a solver inside `simulate` method?
- [x] Use `rllib` to distribute the environments?
- [ ] Understand the parameters in `rllib`.
- [ ] Add argparse/config for train_rl_agent.
- [ ] The config is not loaded in the old version

## Install locally

1. pip install -r requirements.txt
2. go to folder highway-env --> pip install -e .
