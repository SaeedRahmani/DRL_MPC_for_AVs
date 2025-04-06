# RLlib-optimized MPC_RL_for_AVs

## Introduction

### Environments

- [x] Pure MPC
  - [x] Without collision avoidance (`"intersection-mpc-v0"`)
  - [x] Manual collision avoidance (`"intersection-mpc-v1"`)
  - [ ] Collision avoidance in cost function (?)
- [x] MPC-RL: Dynamic weights
  - [x] Without collision avoidance
  - [ ] Manual collision avoidance
- [x] MPC-RL: Reference speeds
  - [x] Without collision avoidance
  - [ ] Manual collision avoidance

### Train MPC-RL agents (using 2nd-gen `RLlib` API stack)

1. Check the configuration file to specify the `env_version`; 
2. Run `python ray_mpc.py`;

### Run MPC agents

1. Check the configuration file to specify the `env_version` to use (`"intersection-mpc` `-v0` or `-v1`); 
2. Run `python run_mpc.py`;
3. Observations:
    - `v0` is more aggressive;
    - `v1` is more conservation because of the avoidance collision checking.

## Local installation

1. `cd` to directory `./src_ray_verison`;
2. Install dependencies: `pip install -r requirements.txt`;
3. `cd` to directory `./highway-env` and run `pip install -e .` to install our customized `highway-env` package.
