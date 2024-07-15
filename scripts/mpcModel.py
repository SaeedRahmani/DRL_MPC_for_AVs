import gymnasium as gym
import highway_env
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import math

# Define the dynamics of the vehicle
def vehicle_dynamics(x, u):
    x_pos = x[0]
    y_pos = x[1]
    theta = x[2]
    v = x[3]
    delta = u[1]
    
    lr = 1.5  # distance from rear to center of mass
    lf = 1.5  # distance from front to center of mass
    
    beta = ca.arctan(lr / (lr + lf) * ca.tan(delta))
    
    dx_pos = v * ca.cos(theta + beta)
    dy_pos = v * ca.sin(theta + beta)
    dtheta = v * ca.sin(beta) / lr
    dv = u[0]
    
    return ca.vertcat(dx_pos, dy_pos, dtheta, dv)

# Define the MPC controller
def mpc_controller(env, N=10, T=0.1):
    opti = ca.Opti()
    
    # Decision variables
    X = opti.variable(4, N+1)
    U = opti.variable(2, N)
    
    # Parameters (initial state, reference state, and obstacles)
    X0 = opti.parameter(4, 1)
    X_ref = opti.parameter(4, N+1)
    obs_pos = opti.parameter(2, 1)  # Position of the nearest obstacle
    obs_speed = opti.parameter(1, 1)  # Speed of the nearest obstacle
    
    # Initial condition
    opti.subject_to(X[:, 0] == X0)
    
    # Dynamics constraints
    for k in range(N):
        x_next = X[:, k] + T * vehicle_dynamics(X[:, k], U[:, k])
        opti.subject_to(X[:, k+1] == x_next)
    
    # Cost function
    Q = np.diag([10, 10, 1, 1])
    R = np.diag([1, 1])
    cost = 0
    for k in range(N):
        state_error = X[:, k] - X_ref[:, k]
        control_error = U[:, k]
        cost += ca.mtimes([state_error.T, Q, state_error]) + ca.mtimes([control_error.T, R, control_error])
        
        # Add obstacle avoidance cost
        obs_dist = ca.sqrt((X[0, k] - obs_pos[0])**2 + (X[1, k] - obs_pos[1])**2 + 1e-6)  # Avoid division by zero
        cost += 100 / obs_dist  # Penalize being too close to the obstacle
        
        # Penalize higher speed when near obstacles
        speed_penalty = 500 * ca.fmax(0, 1 - obs_dist / 10) * X[3, k]
        cost += speed_penalty
    
    opti.minimize(cost)
    
    # Control constraints
    opti.subject_to(opti.bounded(-1, U[0, :], 1))  # acceleration
    opti.subject_to(opti.bounded(-ca.pi/4, U[1, :], ca.pi/4))  # steering angle
    opti.subject_to(opti.bounded(0, X[3, :], 10))  # speed constraints
    
    # Solver options
    opts = {'ipopt.max_iter': 3000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-6}
    opti.solver('ipopt', opts)
    
    return opti, X0, X_ref, X, U, obs_pos, obs_speed

# Create a more dynamic reference trajectory with sinusoidal curves
def create_reference_trajectory(N, current_state, T):
    ref_trajectory = np.zeros((4, N+1))
    for i in range(N+1):
        ref_trajectory[0, i] = current_state[0, 0] + current_state[3, 0] * T * i  # x position
        ref_trajectory[1, i] = current_state[1, 0] + 2 * np.sin(0.1 * i)  # y position with sinusoidal changes
        ref_trajectory[2, i] = current_state[2, 0] + 0.1 * np.sin(0.1 * i)  # orientation to follow the curve
        ref_trajectory[3, i] = current_state[3, 0]  # maintain the same velocity
    return ref_trajectory

def find_nearest_obstacle(state, others):
    ego_position = state[1:3]  # Position X, Y
    min_distance = float('inf')
    nearest_obstacle = None
    nearest_speed = 0
    for other in others:
        other_position = other[1:3]  # Position X, Y
        other_speed = np.linalg.norm(other[3:5])  # Speed calculated from vx, vy
        distance = np.linalg.norm(ego_position - other_position)
        if distance < min_distance:
            min_distance = distance
            nearest_obstacle = other_position
            nearest_speed = other_speed
    return np.array(nearest_obstacle).reshape(2, 1) if nearest_obstacle is not None else np.array([0, 0]).reshape(2, 1), nearest_speed

def debug_obstacle_info(state, nearest_obstacle, nearest_obstacle_speed):
    print("Ego State:")
    print(state)
    print("Ego Speed:")
    print(np.linalg.norm(state[3:5]))  # Calculate speed from vx, vy
    print("Nearest Obstacle Position:")
    print(nearest_obstacle)
    print("Nearest Obstacle Speed:")
    print(nearest_obstacle_speed)

def simulate_mpc(env, opti, X0, X_ref, X, U, obs_pos, obs_speed, N=10, T=0.1):
    state, info = env.reset()
    
    while True:
        if isinstance(state, tuple):
            state = state[0]

        vehicle_state = state[0]  # Ego vehicle state
        other_vehicles = state[1:-1]  # Other vehicles, ignoring the last placeholder row

        # Extract current state for the ego vehicle
        current_state = np.array([
            vehicle_state[1],  # Position X
            vehicle_state[2],  # Position Y
            np.arctan2(vehicle_state[4], vehicle_state[3]),  # Orientation from vx, vy
            np.linalg.norm(vehicle_state[3:5])  # Speed from vx, vy
        ]).reshape(4, 1)

        ref_state = create_reference_trajectory(N, current_state, T)
        nearest_obstacle, nearest_obstacle_speed = find_nearest_obstacle(vehicle_state, other_vehicles)

        # Debug: Print the reference trajectory and nearest obstacle information
        print("Reference trajectory:")
        print(ref_state)
        debug_obstacle_info(vehicle_state, nearest_obstacle, nearest_obstacle_speed)

        opti.set_value(X0, current_state)
        opti.set_value(X_ref, ref_state)
        opti.set_value(obs_pos, nearest_obstacle)
        opti.set_value(obs_speed, np.array([nearest_obstacle_speed]).reshape(1, 1))

        try:
            sol = opti.solve()
        except RuntimeError as e:
            print("Solver failed:", e)
            print("X0:", opti.debug.value(X0))
            print("X_ref:", opti.debug.value(X_ref))
            print("obs_pos:", opti.debug.value(obs_pos))
            print("obs_speed:", opti.debug.value(obs_speed))
            print("X:", opti.debug.value(X))
            print("U:", opti.debug.value(U))
            break

        control_input = sol.value(U[:, 0])

        state, _, done, truncated, info = env.step(control_input)
        done = done or truncated
        env.render()

        if info['crashed']:
            print("Collision detected. Restarting the simulation...")
            state, info = env.reset()
            continue

        if done and not info['crashed']:
            print("Simulation successful. Restarting the simulation...")
            state, info = env.reset()
            continue

# Main function
if __name__ == '__main__':
    env = gym.make('intersection-v1', render_mode="rgb_array")
    env.unwrapped.configure({"simulation_frequency": 15})  # Corrected way to configure the environment
    
    opti, X0, X_ref, X, U, obs_pos, obs_speed = mpc_controller(env)
    simulate_mpc(env, opti, X0, X_ref, X, U, obs_pos, obs_speed)
