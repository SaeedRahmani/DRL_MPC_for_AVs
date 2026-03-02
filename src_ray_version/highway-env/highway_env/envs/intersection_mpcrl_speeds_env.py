import numpy as np
from gymnasium import spaces

from .intersection_mpc_env import (
    IntersectionMpcEnv_noCA,
    IntersectionMpcEnv_manual,
    IntersectionMpcEnv_cost,
    IntersectionMpcEnv_constraint,
)
from highway_env.envs.common.action import Action
from highway_env.envs.common.abstract import Observation


class IntersectionMpcrlSpeedsEnv_noCA(IntersectionMpcEnv_noCA):
    """
    MPCRL v0: RL outputs a reference speed that replaces manual collision avoidance.

    The RL agent learns WHEN to decelerate/accelerate based on traffic context,
    replacing the conservative heuristic CA module in pure MPC.

    Observation: flattened & normalized kinematics (80) + MPC context features (6) = 86D vector.
    Action: single scalar in [-1, 1] mapped to [0, MAX_REF_SPEED] m/s.
    """

    # Number of extra MPC-context features appended to the flattened kinematic obs
    N_MPC_FEATURES = 6

    # Normalization ranges for kinematic features (col_idx → (low, high))
    _FEATURE_RANGES = {
        0: (0.0, 1.0),          # presence
        1: (-100.0, 100.0),     # x
        2: (-100.0, 100.0),     # y
        3: (-20.0, 20.0),       # vx
        4: (-20.0, 20.0),       # vy
        5: (-np.pi, np.pi),     # heading
        6: (-1.0, 1.0),         # sin_h
        7: (-1.0, 1.0),         # cos_h
    }

    def __init__(self, config: dict = None, render_mode: str | None = None, action_dim: int = 1):
        super().__init__(config=config, render_mode=render_mode)
        self.action_dim = action_dim
        self.config["action"]["num_reference_points"] = self.action_dim
        self.define_spaces()

        self.CA_mode = "noCA"
        assert self.CA_mode == "noCA", "Expect CA mode to be `noCA`."
        self.agent_mode = "MPC-RL<Reference speed>"
        assert self.agent_mode == "MPC-RL<Reference speed>", "Expect agent mode to be `MPC-RL<Reference speed>`."

    # ------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------

    def define_spaces(self) -> None:
        """Override to provide a flat 1-D observation space with MPC features appended."""
        super().define_spaces()
        vehicles_count = self.config["observation"].get("vehicles_count", 10)
        n_features = len(self.config["observation"].get(
            "features", ["presence", "x", "y", "vx", "vy", "heading", "sin_h", "cos_h"]))
        flat_size = vehicles_count * n_features + self.N_MPC_FEATURES
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(flat_size,), dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Observation augmentation
    # ------------------------------------------------------------------

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Flatten & normalise kinematics, then append MPC-context features.

        The raw ``obs`` (shape (V, 8)) is kept in ``self.current_observation``
        for MPC use; this method produces a *separate* 1-D vector for the RL
        policy only.
        """
        # --- normalise kinematic features to roughly [-1, 1] ---
        norm = obs.copy().astype(np.float32)
        for col, (lo, hi) in self._FEATURE_RANGES.items():
            if hi > lo:
                norm[:, col] = 2.0 * (norm[:, col] - lo) / (hi - lo) - 1.0
        flat = norm.flatten()

        # --- MPC-context features (all in [0, ~1]) ---
        # 1. Ego speed normalised by MAX_REF_SPEED
        vx, vy = obs[0, 3], obs[0, 4]
        ego_speed = np.sqrt(vx ** 2 + vy ** 2)
        ego_speed_norm = np.clip(ego_speed / self.MAX_REF_SPEED, 0.0, 2.0)

        # 2. Progress along the reference trajectory [0, 1]
        ref_traj = self.reference_trajectory
        ego_pos = obs[0, 1:3]
        if hasattr(self, "ego_index"):
            ego_progress = self.ego_index / max(len(ref_traj) - 1, 1)
        else:
            # Fallback for first call (before _prepare_obs)
            dists = np.linalg.norm(ref_traj - ego_pos, axis=1)
            ego_progress = float(np.argmin(dists)) / max(len(ref_traj) - 1, 1)

        # 3. Min Euclidean distance to nearest other vehicle [0, 1]
        #    Smooth continuous signal — much better than binary is_collide.
        #    0.0 = touching/collision, 1.0 = far away (>= PROXIMITY_RANGE m).
        PROXIMITY_RANGE = 30.0  # metres
        min_dist_norm = 1.0
        for i in range(1, obs.shape[0]):
            if obs[i, 0] > 0.5:  # presence flag
                d = np.sqrt((obs[0, 1] - obs[i, 1])**2 + (obs[0, 2] - obs[i, 2])**2)
                min_dist_norm = min(min_dist_norm, d / PROXIMITY_RANGE)
        min_dist_norm = float(np.clip(min_dist_norm, 0.0, 1.0))
        self._last_min_dist = min_dist_norm  # cache for reward

        # 4. Normalised distance to nearest conflict point along trajectory
        #    1.0 → no conflict;  0.0 → conflict is at ego position
        min_conflict_dist = 1.0
        if getattr(self, "is_collide", False) and hasattr(self, "conflict_index"):
            valid = [idx for idx in self.conflict_index if idx is not None]
            if valid and hasattr(self, "ego_index"):
                dists_along = [abs(idx - self.ego_index) for idx in valid]
                min_conflict_dist = float(min(dists_along)) / max(len(ref_traj), 1)
                min_conflict_dist = np.clip(min_conflict_dist, 0.0, 1.0)

        # 5. Number of visible vehicles (normalised)
        vehicles_count = self.config["observation"].get("vehicles_count", 10)
        n_visible = int(np.sum(obs[1:, 0]))  # presence flags (skip ego)
        n_vehicles_norm = n_visible / max(vehicles_count - 1, 1)

        # 6. Time remaining in the episode [0, 1]
        duration = self.config.get("duration", 13)
        time_remaining = np.clip(1.0 - getattr(self, "time", 0) / duration, 0.0, 1.0)

        mpc_features = np.array([
            ego_speed_norm,
            ego_progress,
            min_dist_norm,
            min_conflict_dist,
            n_vehicles_norm,
            time_remaining,
        ], dtype=np.float32)

        return np.concatenate([flat, mpc_features])

    # ------------------------------------------------------------------
    # Reset / Step  (augment obs for RL, keep raw obs for MPC)
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev_mpc_acc = 0.0
        self._last_min_dist = 1.0   # no proximity penalty at start

        # Clear stale collision state from the previous episode so the
        # first observation of the new episode is clean.
        self.is_collide = False
        self.collision_memory = 0
        self.memorized_conflict_points = None
        self.memorized_conflict_indices = None
        self.last_valid_stop_point = None
        if hasattr(self, "conflict_index"):
            self.conflict_index = []
            self.conflict_points = []

        # Guard: reference_trajectory is set in the parent __init__ AFTER
        # super().__init__() returns, but super().__init__() already calls
        # self.reset() once.  On that first init-time call the attribute
        # doesn't exist yet, so we must skip augmentation.
        if not hasattr(self, "reference_trajectory"):
            return obs, info

        # Compute a fresh ego_index for the starting position
        ego_pos = obs[0, 1:3]
        dists = np.linalg.norm(self.reference_trajectory - ego_pos, axis=1)
        self.ego_index = int(np.argmin(dists))

        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return self._augment_obs(obs), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Reward  (v0-specific: speed efficiency + safety)
    # ------------------------------------------------------------------

    def _reward(self, action) -> float:
        """
        v0-specific reward that balances speed efficiency with safety.

        The RL agent should learn to keep speed high when safe and slow down
        just enough when dangerous — replacing the conservative manual CA.

        Components
        ----------
        - **Alive bonus** (+0.2 / step): survive longer = more reward.
        - **Speed efficiency** (up to +1.5 / step): proportional to ego speed
          vs MAX_REF_SPEED.
        - **Proximity warning** (up to -2.0 / step): continuous penalty that
          increases as ego gets closer to any other vehicle.  Provides a
          smooth gradient signal BEFORE a crash happens.
        - **Smoothness** (penalty): penalises jerky MPC accelerations.
        - **Terminal**: crash −300, off-road −50, arrived +100.
        """
        vehicle = self.controlled_vehicles[0]

        # ---- terminal events ----
        # Keep terminals within ~10x of per-step reward to avoid
        # advantage spikes that cause NaN in the policy network.
        if vehicle.crashed:
            return -50.0
        if not vehicle.on_road:
            return -10.0
        if self.has_arrived(vehicle):
            return 20.0

        reward = 0.0

        # 1. Alive bonus — incentivise survival
        reward += 0.2

        # 2. Speed efficiency
        speed_ratio = np.clip(vehicle.speed / self.MAX_REF_SPEED, 0.0, 1.5)
        reward += 1.5 * speed_ratio

        # 3. Proximity warning — penalise being close to other vehicles.
        #    _last_min_dist is in [0, 1]:  0 = touching,  1 = far away.
        #    Penalty activates when distance < 0.5 (i.e., < 15 m).
        min_dist = getattr(self, "_last_min_dist", 1.0)
        if min_dist < 0.5:
            # Quadratic penalty: grows fast as distance shrinks
            reward -= 2.0 * (1.0 - 2.0 * min_dist) ** 2

        # 4. Smoothness — penalise abrupt MPC acceleration changes
        if hasattr(self, "_prev_mpc_acc"):
            acc_change = abs(self.last_acc - self._prev_mpc_acc)
            reward -= 0.3 * (acc_change / 5.0)  # normalise by max_acc
        self._prev_mpc_acc = getattr(self, "last_acc", 0.0)

        # 5. Lane centering (minor, MPC responsibility)
        if vehicle.on_road:
            lateral = vehicle.lane.local_coordinates(vehicle.position)[1]
            centering = 1.0 - min(1.0, abs(lateral) / (vehicle.lane.width / 2))
            reward += 0.3 * centering

        return reward

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "ReferenceSpeedAction",
                    "num_reference_points": 1,
                },
            }
        )
        return config


class IntersectionMpcrlSpeedsEnv_manual(IntersectionMpcEnv_manual):
    """
    MPCRL v0 + manual CA: RL outputs a reference speed to make the vehicle
    more aggressive (faster), while the built-in manual CA module still
    provides a safety net by overriding to stop when conflicts are detected.

    The RL agent learns to push speed higher when safe, knowing that the
    CA will bail it out if things get dangerous.
    """

    N_MPC_FEATURES = 6

    _FEATURE_RANGES = {
        0: (0.0, 1.0),
        1: (-100.0, 100.0),
        2: (-100.0, 100.0),
        3: (-20.0, 20.0),
        4: (-20.0, 20.0),
        5: (-np.pi, np.pi),
        6: (-1.0, 1.0),
        7: (-1.0, 1.0),
    }

    def __init__(self, config: dict = None, render_mode: str | None = None, action_dim: int = 1):
        super().__init__(config=config, render_mode=render_mode)
        self.action_dim = action_dim
        self.config["action"]["num_reference_points"] = self.action_dim
        self.define_spaces()

        self.CA_mode = "manual"
        assert self.CA_mode == "manual", "Expect CA mode to be `manual`."
        self.agent_mode = "MPC-RL<Reference speed>"
        assert self.agent_mode == "MPC-RL<Reference speed>"

    # ------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------

    def define_spaces(self) -> None:
        """Override to provide a flat 1-D observation space with MPC features."""
        super().define_spaces()
        vehicles_count = self.config["observation"].get("vehicles_count", 10)
        n_features = len(self.config["observation"].get(
            "features", ["presence", "x", "y", "vx", "vy", "heading", "sin_h", "cos_h"]))
        flat_size = vehicles_count * n_features + self.N_MPC_FEATURES
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(flat_size,), dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Observation augmentation  (same as noCA)
    # ------------------------------------------------------------------

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        norm = obs.copy().astype(np.float32)
        for col, (lo, hi) in self._FEATURE_RANGES.items():
            if hi > lo:
                norm[:, col] = 2.0 * (norm[:, col] - lo) / (hi - lo) - 1.0
        flat = norm.flatten()

        vx, vy = obs[0, 3], obs[0, 4]
        ego_speed = np.sqrt(vx ** 2 + vy ** 2)
        ego_speed_norm = np.clip(ego_speed / self.MAX_REF_SPEED, 0.0, 2.0)

        ref_traj = self.reference_trajectory
        if hasattr(self, "ego_index"):
            ego_progress = self.ego_index / max(len(ref_traj) - 1, 1)
        else:
            dists = np.linalg.norm(ref_traj - obs[0, 1:3], axis=1)
            ego_progress = float(np.argmin(dists)) / max(len(ref_traj) - 1, 1)

        PROXIMITY_RANGE = 30.0
        min_dist_norm = 1.0
        for i in range(1, obs.shape[0]):
            if obs[i, 0] > 0.5:
                d = np.sqrt((obs[0, 1] - obs[i, 1])**2 + (obs[0, 2] - obs[i, 2])**2)
                min_dist_norm = min(min_dist_norm, d / PROXIMITY_RANGE)
        min_dist_norm = float(np.clip(min_dist_norm, 0.0, 1.0))
        self._last_min_dist = min_dist_norm

        min_conflict_dist = 1.0
        if getattr(self, "is_collide", False) and hasattr(self, "conflict_index"):
            valid = [idx for idx in self.conflict_index if idx is not None]
            if valid and hasattr(self, "ego_index"):
                dists_along = [abs(idx - self.ego_index) for idx in valid]
                min_conflict_dist = float(min(dists_along)) / max(len(ref_traj), 1)
                min_conflict_dist = np.clip(min_conflict_dist, 0.0, 1.0)

        vehicles_count = self.config["observation"].get("vehicles_count", 10)
        n_visible = int(np.sum(obs[1:, 0]))
        n_vehicles_norm = n_visible / max(vehicles_count - 1, 1)

        duration = self.config.get("duration", 13)
        time_remaining = np.clip(1.0 - getattr(self, "time", 0) / duration, 0.0, 1.0)

        mpc_features = np.array([
            ego_speed_norm, ego_progress, min_dist_norm,
            min_conflict_dist, n_vehicles_norm, time_remaining,
        ], dtype=np.float32)

        return np.concatenate([flat, mpc_features])

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev_mpc_acc = 0.0
        self._last_min_dist = 1.0

        self.is_collide = False
        self.collision_memory = 0
        self.memorized_conflict_points = None
        self.memorized_conflict_indices = None
        self.last_valid_stop_point = None
        if hasattr(self, "conflict_index"):
            self.conflict_index = []
            self.conflict_points = []

        if not hasattr(self, "reference_trajectory"):
            return obs, info

        ego_pos = obs[0, 1:3]
        dists = np.linalg.norm(self.reference_trajectory - ego_pos, axis=1)
        self.ego_index = int(np.argmin(dists))

        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return self._augment_obs(obs), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Reward  (manual-CA variant: reward aggressiveness, CA is safety net)
    # ------------------------------------------------------------------

    def _reward(self, action) -> float:
        """
        Reward for the manual-CA variant.

        Since the CA module provides a safety net, the agent can be more
        speed-aggressive. Crash penalty is moderate (CA should prevent most).
        The main signal is: drive as fast as possible while still being smooth.
        """
        vehicle = self.controlled_vehicles[0]

        # Keep terminals within ~10x of per-step reward to avoid
        # advantage spikes that cause NaN in the policy network.
        if vehicle.crashed:
            return -30.0
        if not vehicle.on_road:
            return -10.0
        if self.has_arrived(vehicle):
            return 20.0

        reward = 0.0

        # 1. Alive bonus
        reward += 0.2

        # 2. Speed efficiency — higher weight than noCA since CA protects us
        speed_ratio = np.clip(vehicle.speed / self.MAX_REF_SPEED, 0.0, 1.5)
        reward += 2.0 * speed_ratio

        # 3. Mild proximity warning (CA handles the hard stops)
        min_dist = getattr(self, "_last_min_dist", 1.0)
        if min_dist < 0.3:  # tighter threshold since CA is active
            reward -= 1.0 * (1.0 - min_dist / 0.3) ** 2

        # 4. Smoothness
        if hasattr(self, "_prev_mpc_acc"):
            acc_change = abs(self.last_acc - self._prev_mpc_acc)
            reward -= 0.3 * (acc_change / 5.0)
        self._prev_mpc_acc = getattr(self, "last_acc", 0.0)

        # 5. Lane centering
        if vehicle.on_road:
            lateral = vehicle.lane.local_coordinates(vehicle.position)[1]
            centering = 1.0 - min(1.0, abs(lateral) / (vehicle.lane.width / 2))
            reward += 0.3 * centering

        return reward

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "ReferenceSpeedAction",  # use 6 to predict 30, optimal: 16
                    "num_reference_points": 1,
                },
            }
        )
        return config


class IntersectionMpcrlSpeedsEnv_cost(IntersectionMpcEnv_cost):
    """ MPCRL: Reference speed with collision cost in MPC. """

    def __init__(self, config: dict = None, render_mode: str | None = None, action_dim: int = 1):
        super().__init__(config=config, render_mode=render_mode)
        self.action_dim = action_dim
        self.config["action"]["num_reference_points"] = self.action_dim
        self.define_spaces()
        
        self.CA_mode = "cost"
        assert self.CA_mode == "cost", "Expect CA mode to be `cost`."
        self.agent_mode = "MPC-RL<Reference speed>"

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "ReferenceSpeedAction",
                    "num_reference_points": 1,
                },
            }
        )
        return config


class IntersectionMpcrlSpeedsEnv_constraint(IntersectionMpcEnv_constraint):
    """ MPCRL: Reference speed with collision avoidance constraint in MPC. """

    def __init__(self, config: dict = None, render_mode: str | None = None):
        super().__init__(config=config, render_mode=render_mode)
        self.CA_mode = "constraint"
        assert self.CA_mode == "constraint", "Expect CA mode to be `constraint`."
        self.agent_mode = "MPC-RL<Reference speed>"
        assert self.agent_mode == "MPC-RL<Reference speed>", "Expect agent mode to be `MPC-RL<Reference speed>`."

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "ReferenceSpeedAction",
                    "num_reference_points": 1,
                },
            }
        )
        return config
