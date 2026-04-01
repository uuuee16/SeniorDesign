import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from typing import List, Tuple, Dict, Any
from config import MapConfig


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))

class AUVState:
    def __init__(self, initial_pos: Tuple[float, float, float] = (0, 0, 0)):
        x, y, z = initial_pos
        self.state = np.array([x, y, z, 0.0, 0.0], dtype=np.float32)
    
    @property
    def position(self) -> np.ndarray:
        return self.state[:3]
    
    @property
    def attitude(self) -> Tuple[float, float]:
        return self.state[3], self.state[4]
    
    @property
    def yaw(self) -> float:
        return self.state[3]
    
    @property
    def pitch(self) -> float:
        return self.state[4]
    
    def update_position(self, new_pos: np.ndarray):
        assert new_pos.shape == (3,), "位置必须是3维数组"
        self.state[:3] = new_pos
    
    def update_attitude(self, new_yaw: float, new_pitch: float):
        self.state[3] = normalize_angle(new_yaw)
        self.state[4] = np.clip(new_pitch, MapConfig.pitch_limit[0], MapConfig.pitch_limit[1])
    
    def get_full_state(self) -> np.ndarray:
        return self.state.copy()

class AUVEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(
        self,
        k_obst: int = MapConfig.k_nearest_obstacles,
        obstacles: List[Tuple[float, float, float, float]] = MapConfig.obstacles_maze,
        env_bound: List[float] = None,
        target_pos: Tuple[float, float, float] = MapConfig.target_pos_maze,
        dt: float = MapConfig.dt
    ):
        super().__init__()
        self.k_obst = k_obst
        self.dt = dt
        self.env_bound = env_bound if env_bound is not None else MapConfig.env_bound[:]
        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = self.env_bound
        self.scene_scale = max(
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            self.z_max - self.z_min
        )
        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min
        self.z_range = self.z_max - self.z_min

        self.obstacles = obstacles if obstacles is not None else self._generate_default_obstacles()
        self.r_min, self.r_max = MapConfig.obstacle_radius_range
        self.r_range = self.r_max - self.r_min

        self.target_fixed = target_pos is not None
        self.target_pos = np.array(
            target_pos if target_pos is not None else MapConfig.default_target_pos,
            dtype=np.float32
        )
        self.action_dim = 3
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.action_dim,), 
            dtype=np.float32
        )
        self.obs_dim = 3 + 2 + 4 * self.k_obst
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.obs_dim,), 
            dtype=np.float32
        )
        self.auv_state = AUVState()
    
    def _generate_default_obstacles(self) -> List[Tuple[float, float, float, float]]:
        obstacles = []

        max_radius = max(MapConfig.obstacle_radius_range)
        
        # 2. 计算安全边界（考虑最大半径）
        x_low = self.x_min + max_radius
        x_high = self.x_max - max_radius
        y_low = self.y_min + max_radius
        y_high = self.y_max - max_radius
        z_low = self.z_min + max_radius
        z_high = self.z_max - max_radius
        
        # 3. 验证范围有效性
        if x_low >= x_high or y_low >= y_high or z_low >= z_high:
            raise ValueError(
                f"环境空间太小，无法生成障碍物。"
                f"x范围: {x_low}-{x_high}, y范围: {y_low}-{y_high}, z范围: {z_low}-{z_high}"
            )
        
        # 4. 生成障碍物
        for _ in range(self.k_obst):
            ox = self.np_random.uniform(x_low, x_high)
            oy = self.np_random.uniform(y_low, y_high)
            oz = self.np_random.uniform(z_low, z_high)
            r = self.np_random.uniform(*MapConfig.obstacle_radius_range)
            obstacles.append((ox, oy, oz, r))
        
        return obstacles
    
    def _find_nearest_k(self, auv_pos: np.ndarray) -> List[Dict[str, Any]]:
        assert auv_pos.shape == (3,), "AUV位置必须是3维数组"
        
        if not self.obstacles:  # 无障碍物时直接返回空列表
            return []
        
        obs_positions = np.array([(ox, oy, oz) for ox, oy, oz, r in self.obstacles], dtype=np.float32)
        obs_radii = np.array([r for ox, oy, oz, r in self.obstacles], dtype=np.float32)
        distances = np.linalg.norm(obs_positions - auv_pos[None, :], axis=1)
        
        obstacle_info = [
            {'distance': dist, 'position': pos, 'radius': r}
            for dist, pos, r in zip(distances, obs_positions, obs_radii)
        ]
        obstacle_info.sort(key=lambda x: x['distance'])
        return obstacle_info[:self.k_obst]
    
    def _get_nearest_surface_distance(self, auv_pos: np.ndarray) -> float:
        nearest_obstacles = self._find_nearest_k(auv_pos)
        if not nearest_obstacles:
            return float('inf')  # 无障碍物，距离为无穷大（逻辑正确）
        min_surface_dist = float('inf')
        for obs in nearest_obstacles:
            surface_dist = obs['distance'] - (obs['radius'] + MapConfig.collision_threshold)
            min_surface_dist = min(min_surface_dist, surface_dist)
        return min_surface_dist  # 移除多余的inf判断（nearest_obstacles非空时，min_surface_dist不可能是inf）
    
    
    def _check_collision(self, auv_pos: np.ndarray) -> bool:
        return self._get_nearest_surface_distance(auv_pos) < 0.0

    def _compute_velocity_jacobian(self, yaw: float, pitch: float) -> np.ndarray:
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        return np.array([
            [cy * cp, -sy, -cy * sp],
            [sy * cp,  cy, -sy * sp],
            [sp,      0,   cp]
        ], dtype=np.float32)
    
    def _update_auv_dynamics(self, action: np.ndarray):
        body_vel = (action[0] + 1.0) / 2.0 * MapConfig.max_body_vel
        yaw_rate = action[1] * MapConfig.max_yaw_vel
        pitch_rate = action[2] * MapConfig.max_pitch_vel

        current_pos = self.auv_state.position
        current_yaw, current_pitch = self.auv_state.attitude

        body_vel_vec = np.array([body_vel, 0.0, 0.0], dtype=np.float32)
        J = self._compute_velocity_jacobian(current_yaw, current_pitch)
        
        ned_vel = J @ body_vel_vec
        new_pos = current_pos + ned_vel * self.dt
        new_yaw = current_yaw + yaw_rate * self.dt
        new_pitch = current_pitch + pitch_rate * self.dt
        self.auv_state.update_position(new_pos)
        self.auv_state.update_attitude(new_yaw, new_pitch)

    def _check_out_of_bound(self) -> bool:
        x, y, z = self.auv_state.position
        return (x < self.x_min or x > self.x_max or
                y < self.y_min or y > self.y_max or
                z < self.z_min or z > self.z_max)
    
    def _check_termination(self) -> bool:
        dist_to_target = np.linalg.norm(self.auv_state.position - self.target_pos)
        if dist_to_target < MapConfig.goal_threshold:
            return True
        if self._check_out_of_bound():
            return True
        if self._check_collision(self.auv_state.position):
            return True
        return False

    def _normalize_observation(self) -> np.ndarray:
        obs_vec = []
        
        rel_target = (self.target_pos - self.auv_state.position) / self.scene_scale
        rel_target = np.clip(rel_target, -1.0, 1.0)
        obs_vec.extend(rel_target.tolist())

        norm_yaw = self.auv_state.yaw / math.pi
        norm_pitch = self.auv_state.pitch / math.pi
        obs_vec.extend([norm_yaw, norm_pitch])

        nearest_obstacles = self._find_nearest_k(self.auv_state.position)
        for i in range(self.k_obst):
            if i < len(nearest_obstacles):
                obs = nearest_obstacles[i]
                rel_obs_pos = (obs['position'] - self.auv_state.position) / self.scene_scale
                rel_obs_pos = np.clip(rel_obs_pos, -1.0, 1.0)
                norm_radius = 2 * (obs['radius'] - self.r_min) / self.r_range - 1
                norm_radius = np.clip(norm_radius, -1.0, 1.0)
                obs_vec.extend(list(rel_obs_pos) + [norm_radius])
            else:
                # 障碍物不足k个时
                obs_vec.extend([1.0, 1.0, 1.0, 0.0])
        return np.array(obs_vec, dtype=np.float32)
    
    def _calculate_reward(self, pos_before: np.ndarray) -> Tuple[float, Dict[str, float]]:
        auv_pos = self.auv_state.position
        dist_to_target = np.linalg.norm(auv_pos - self.target_pos)
        collision = self._check_collision(auv_pos)
        out_of_bound = self._check_out_of_bound()
        nearest_obst_dist = self._get_nearest_surface_distance(auv_pos)
        reward_detail = {}
        if dist_to_target < MapConfig.goal_threshold:
            task_reward = MapConfig.reward_task_goal
        # 先不检查边界碰撞
        elif collision:
        # elif collision or out_of_bound:
            task_reward = MapConfig.reward_task_collision
        elif nearest_obst_dist < MapConfig.dangerous_distance:
            task_reward = MapConfig.reward_task_danger
        else:
            task_reward = 0.0
        reward_detail['task_reward'] = task_reward
        dist_before = np.linalg.norm(pos_before - self.target_pos)
        dist_after = dist_to_target
        movement = np.linalg.norm(auv_pos - pos_before)
        progress_bonus = MapConfig.reward_dist_eps1 * (dist_before - dist_after)
        inefficiency_penalty = MapConfig.reward_dist_eps2 * movement
        distance_reward = progress_bonus - inefficiency_penalty
        reward_detail['distance_reward'] = distance_reward
        total_reward = task_reward + MapConfig.reward_dist_lambda * distance_reward
        return total_reward, reward_detail

    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.np_random.seed(seed)
        while True:
            # 固定AUV起始状态
            x = 10
            y = 10
            z = 10
            yaw = 0
            pitch = 0

            # yaw = math.pi / 4
            # pitch = math.pi / 4
            # 随机AUV起始状态
            # x = self.np_random.uniform(self.x_min + 5, self.x_min + self.x_range // 4)
            # y = self.np_random.uniform(self.y_min + 5, self.y_min + self.y_range // 4)
            # z = self.np_random.uniform(self.z_min + 5, self.z_min + self.z_range // 4)
            # yaw = self.np_random.uniform(-5*math.pi/6, 5*math.pi/6)
            # pitch = self.np_random.uniform(*MapConfig.pitch_limit)

            self.auv_state = AUVState((x, y, z))
            self.auv_state.update_attitude(yaw, pitch)
            # 检查AUV初始化位置与target和obstacle是否有重叠
            safe_start = True
            for (ox, oy, oz, r) in self.obstacles:
                dist = np.linalg.norm(self.auv_state.position - np.array([ox, oy, oz]))
                if dist < (r + 2.0):
                    safe_start = False
                    break
            if not safe_start:
                continue
            if not self.target_fixed:
                self.target_pos = np.array([
                    self.np_random.uniform(self.x_min + 5, self.x_max - 5),
                    self.np_random.uniform(self.y_min + 5, self.y_max - 5),
                    self.np_random.uniform(self.z_min + 5, self.z_max - 5)
                ], dtype=np.float32)
            dist2target = np.linalg.norm(self.auv_state.position - self.target_pos)
            if dist2target > 10.0:
                break
        obs = self._normalize_observation()
        info = {
            'auv_full_state': self.auv_state.get_full_state(),
            'target_pos': self.target_pos.copy(),
            'initial_dist2target': dist2target
        }
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # 真实值，而非归一化后的值
        pos_before = self.auv_state.position.copy()
        self._update_auv_dynamics(action)
        # 只对obs进行归一化，reward和info里包含的状态信息保持原始值
        obs = self._normalize_observation()
        reward, reward_detail = self._calculate_reward(pos_before)
        terminated = self._check_termination()
        truncated = False
        info = {
            'auv_full_state': self.auv_state.get_full_state(),
            'dist2target': np.linalg.norm(self.auv_state.position - self.target_pos),
            'nearest_obstacle_dist': self._get_nearest_surface_distance(self.auv_state.position),
            'reward_detail': reward_detail
        }
        return obs, reward, terminated, truncated, info
