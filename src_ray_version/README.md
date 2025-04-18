<!-- markdownlint-disable MD033 -->

# RLlib-optimized MPC_RL_for_AVs

## TODO

- [ ] Configure envs using cfg.
- [x] Manual MPC is working
- [x] Check the if-statement w/ Saeed.
  - [x] What else constraint components do we consider or not?
  - [x] Add collision constraints in MPC (TODO)
- [x] V1 MPCRL Agent
  - [x] Current use action_dim = 7, including all the weight components, 
  - [ ] **Issue is that it may conflict with noCA, but can omit the exceeding weights.**
- [x] V0 MPCRL Agent
  - [x] Only use the action_dim = 1.
  - [ ] Boundary of the action? I find it succeed with random action of ref speed.
  - [ ] action_dim = 6 to interpolate
  - [ ] action_dim = 16 to cover the whole future reference states.


## Environments

<table>
  <tr>
    <th>Algorithm</th>
    <th>Gym environment</th>
    <th>Features</th>
  </tr>
  <tr>
    <td rowspan="4">Pure MPC</td>
    <td> <code> "intersection-mpc-noCA" </code> </td>
    <td> No collison avoidance </td>
  </tr>
  <tr>
    <td> <code> "intersection-mpc-manual" </code> </td>
    <td> Manual collision avoidance outside MPC </td>
  </tr>
  <tr>
    <td> <code> "intersection-mpc-cost" </code> </td>
    <td> Include collision cost in MPC </td>
  </tr>
  <tr>
    <td> <code> "intersection-mpc-constraint" </code> </td>
    <td> Add collision constraint in MPC </td>
  </tr>
  <tr>
    <td rowspan="3"> MPC-RL<br>(Dynamic<br>weights)</td>
    <td> <code> "intersection-mpcrl-dynamicweights-noCA" </code> </td>
    <td> No collison avoidance </td>
  </tr>
  <tr>
    <td> <code> "intersection-mpcrl-dynamicweights-manual" </code> </td>
    <td> Manual collision avoidance outside MPC </td>
  </tr>
  <tr>
    <td> <code> "intersection-mpcrl-dynamicweights-cost" </code> </td>
    <td> Include collision cost in MPC </td>
  </tr>
  <tr>
    <td rowspan="4"> MPC-RL<br>(Reference<br>speed)</td>
    <td> <code> "intersection-mpcrl-refspeed-noCA" </code> </td>
    <td> No collison avoidance </td>
  </tr>
  <tr>
    <td> <code> "intersection-mpcrl-refspeed-manual" </code> </td>
    <td> Manual collision avoidance outside MPC </td>
  </tr>
  <tr>
    <td> <code> "intersection-mpcrl-refspeed-cost" </code> </td>
    <td> Include collision cost in MPC </td>
  </tr>
  <tr>
    <td> <code> "intersection-mpcrl-refspeed-constraint" </code> </td>
    <td> Add collision constraint in MPC </td>
  </tr>
</table>

## Run MPC agents

1. Check the configuration file to specify the `CA_mode` (options include `noCA`, `manual`, `cost` or `constraint`);
2. Run `python run_mpc.py`;
<!-- 3. Observations of behaviors:
    - `v0` is more aggressive;
    - `v1` is more conservation because of the avoidance collision checking. -->

## Train MPC-RL agents

1. Check the configuration file to specify 
  - `env_version`: `"v0"` or `"v1"`; 
  - `sub_env_version`: options include `noCA`, `manual`, `cost` or `constraint`;
2. Run `python train_mpcrl.py`;

## Test MPC-RL agents 

1. Run `python test_mpcrl.py`;

## Local installation

1. `cd` to directory `./src_ray_verison`;
2. Install dependencies: `pip install -r requirements.txt`;
3. `cd` to directory `./highway-env` and run `pip install -e .` to install our customized `highway-env` package.
