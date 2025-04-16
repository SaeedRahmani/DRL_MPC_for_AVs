<!-- markdownlint-disable MD033 -->

# RLlib-optimized MPC_RL_for_AVs

## TODO

- [ ] Manual MPC is not working
- [ ] Check the if-statement w/ Saeed.
  - [ ] What else constraint components do we consider or not?
  - [ ] Add collision constraints in MPC (TODO)

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

## Train MPC-RL agents (using 2nd-gen `RLlib` API stack)

1. Check the configuration file to specify the `env_version`;
2. Run `python ray_mpc.py`;

## Run MPC agents

1. Check the configuration file to specify the `env_version` to use (`"intersection-mpc` `-v0` or `-v1`);
2. Run `python run_mpc.py`;
3. Observations:
    - `v0` is more aggressive;
    - `v1` is more conservation because of the avoidance collision checking.

## Local installation

1. `cd` to directory `./src_ray_verison`;
2. Install dependencies: `pip install -r requirements.txt`;
3. `cd` to directory `./highway-env` and run `pip install -e .` to install our customized `highway-env` package.
