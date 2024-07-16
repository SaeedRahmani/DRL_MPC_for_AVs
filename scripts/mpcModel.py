import gymnasium as gym
import highway_env
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# Define the dynamics of the vehicle
def vehicle_dynamics(x, u):
    x_pos, y_pos, vx, vy = x[0], x[1], x[2], x[3]
    acceleration, steering = u[0], u[1]
    
    v = ca.sqrt(vx**2 + vy**2)
    theta = ca.arctan2(vy, vx)
    
    dx_pos = vx
    dy_pos = vy
    dvx = acceleration * ca.cos(theta) - v * steering * ca.sin(theta)
    dvy = acceleration * ca.sin(theta) + v * steering * ca.cos(theta)
    
    return ca.vertcat(dx_pos, dy_pos, dvx, dvy)

# Define the MPC controller
def mpc_controller(env, N=10, T=0.1):
    opti = ca.Opti()
    
    # Decision variables
    X = opti.variable(4, N+1)
    U = opti.variable(2, N)
    
    # Parameters (initial state, reference state, and obstacles)
    X0 = opti.parameter(4, 1)
    X_ref = opti.parameter(4, N+1)
    obs_pos = opti.parameter(2, 5)  # Position of up to 5 nearest obstacles
    obs_speed = opti.parameter(1, 5)  # Speed of up to 5 nearest obstacles
    
    # Initial condition
    opti.subject_to(X[:, 0] == X0)
    
    # Dynamics constraints
    for k in range(N):
        x_next = X[:, k] + T * vehicle_dynamics(X[:, k], U[:, k])
        opti.subject_to(X[:, k+1] == x_next)
    
    # Cost function
    Q = np.diag([100, 10, 1, 1])
    R = np.diag([0.1, 0.1])
    cost = 0
    for k in range(N):
        state_error = X[:, k] - X_ref[:, k]
        control_error = U[:, k]
        cost += ca.mtimes([state_error.T, Q, state_error]) + ca.mtimes([control_error.T, R, control_error])
        
        # Add obstacle avoidance cost
        for i in range(5):  # For each of the 5 potential obstacles
            obs_dist = ca.sqrt((X[0, k] - obs_pos[0, i])**2 + (X[1, k] - obs_pos[1, i])**2 + 1e-6)
            cost += 50 / obs_dist  # Reduced weight
        
        # Penalize higher speed when near obstacles
        speed = ca.sqrt(X[2, k]**2 + X[3, k]**2)
        speed_penalty = 100 * ca.fmax(0, 1 - obs_dist / 20) * speed
        cost += speed_penalty
    
    opti.minimize(cost)
    
    # Control constraints
    opti.subject_to(opti.bounded(-5, U[0, :], 5))  # acceleration
    opti.subject_to(opti.bounded(-0.5, U[1, :], 0.5))  # steering
    opti.subject_to(opti.bounded(0, ca.sqrt(X[2, :]**2 + X[3, :]**2), 20))  # speed constraints
    
    # Solver options
    opts = {
        'ipopt.max_iter': 5000,
        'ipopt.print_level': 0,
        'print_time': 0,
        'ipopt.acceptable_tol': 1e-8,
        'ipopt.acceptable_obj_change_tol': 1e-6,
        'ipopt.warm_start_init_point': 'yes',
        'ipopt.mu_strategy': 'adaptive',
        'ipopt.hessian_approximation': 'limited-memory'
    }
    opti.solver('ipopt', opts)
    
    return opti, X0, X_ref, X, U, obs_pos, obs_speed

def create_reference_trajectory(N, current_state, T, target_speed=10):
    ref_trajectory = np.zeros((4, N+1))
    x_pos, y_pos, vx, vy = current_state.flatten()
    
    v = np.sqrt(vx**2 + vy**2)
    if v < 1e-6:
        theta = 0  # Default direction if speed is near zero
    else:
        theta = np.arctan2(vy, vx)
    
    for i in range(N+1):
        ref_trajectory[0, i] = x_pos + np.sin(theta) * target_speed * T * i
        ref_trajectory[1, i] = y_pos + np.cos(theta) * target_speed * T * i
        ref_trajectory[2, i] = target_speed * np.cos(theta)
        ref_trajectory[3, i] = target_speed * np.sin(theta)
    
    return ref_trajectory

def plot_trajectory_on_image(image, trajectory, scale=10):
    fig, ax = plt.subplots()
    ax.imshow(image)
    x = trajectory[0, :] * scale + image.shape[1] / 2
    y = -trajectory[1, :] * scale + image.shape[0] / 2
    ax.plot(x, y, 'r-', label='Reference Trajectory')
    ax.scatter(x, y, c='blue', label='Waypoints')
    ax.set_xlim([0, image.shape[1]])
    ax.set_ylim([image.shape[0], 0])
    ax.legend()
    plt.show()

def find_nearest_obstacles(ego_vehicle, other_vehicles, max_obstacles=5):
    ego_position = ego_vehicle[1:3]  # Position X, Y
    obstacles = []
    for other in other_vehicles:
        if other[0] == 1:  # Check if vehicle is present
            other_position = other[1:3]  # Position X, Y
            other_speed = np.linalg.norm(other[3:5])  # Speed calculated from vx, vy
            distance = np.linalg.norm(ego_position - other_position)
            obstacles.append((distance, other_position, other_speed))
    
    obstacles.sort(key=lambda x: x[0])  # Sort by distance
    nearest_obstacles = obstacles[:max_obstacles]
    
    # Pad with dummy obstacles if less than max_obstacles
    while len(nearest_obstacles) < max_obstacles:
        nearest_obstacles.append((float('inf'), np.array([0, 0]), 0))
    
    return np.array([obs[1] for obs in nearest_obstacles]).T, np.array([obs[2] for obs in nearest_obstacles])

def debug_obstacle_info(ego_vehicle, nearest_obstacles, nearest_obstacle_speeds):
    print("Ego State:")
    print(ego_vehicle)
    print("Ego Speed:")
    print(np.linalg.norm(ego_vehicle[3:5]))  # Calculate speed from vx, vy
    print("Nearest Obstacle Positions:")
    print(nearest_obstacles)
    print("Nearest Obstacle Speeds:")
    print(nearest_obstacle_speeds)

def simulate_mpc(env, opti, X0, X_ref, X, U, obs_pos, obs_speed, N=10, T=0.1):
    state, info = env.reset()
    
    while True:
        if isinstance(state, tuple):
            state = state[0]

        ego_vehicle = state[0]  # Ego vehicle state
        other_vehicles = state[1:]  # Other vehicles

        # Extract current state for the ego vehicle
        current_state = np.array([
            ego_vehicle[1],  # Position X
            ego_vehicle[2],  # Position Y
            ego_vehicle[3],  # Velocity X
            ego_vehicle[4]   # Velocity Y
        ]).reshape(4, 1)

        ref_state = create_reference_trajectory(N, current_state, T)
        nearest_obstacles, nearest_obstacle_speeds = find_nearest_obstacles(ego_vehicle, other_vehicles)

        # Debug: Print the reference trajectory and nearest obstacle information
        print("Reference trajectory:")
        print(ref_state)
        # debug_obstacle_info(ego_vehicle, nearest_obstacles, nearest_obstacle_speeds)

        opti.set_value(X0, current_state)
        opti.set_value(X_ref, ref_state)
        opti.set_value(obs_pos, nearest_obstacles)
        opti.set_value(obs_speed, nearest_obstacle_speeds.reshape(1, -1))

        try:
            sol = opti.solve()
            control_input = sol.value(U[:, 0])
        except RuntimeError as e:
            print("Solver failed:", e)
            print("Using fallback control strategy")
            # Simple fallback strategy: maintain current velocity
            control_input = np.array([0, 0])

        # Convert MPC output to environment action
        # Assuming the environment expects [steering, acceleration]
        action = np.array([control_input[1], control_input[0]])
        
        state, reward, done, truncated, info = env.step(action)
        done = done or truncated
        env.render()
        image = env.render()
        plot_trajectory_on_image(image, ref_state)

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