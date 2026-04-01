import math
from datetime import timedelta

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import MapConfig as cfg


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def wrap_angle(x: float) -> float:
    return math.atan2(math.sin(x), math.cos(x))


# ──────────────────────────────────────────────────────────────────────────────
# 合成海流：定常流 + Lamb 涡流叠加
# ──────────────────────────────────────────────────────────────────────────────

class OceanCurrent:
    enable_temporal = False

    def __init__(self, const=None, vortices=None, bounds=None):
        self.cfg       = cfg
        self.bounds    = np.array(bounds or cfg.bounds, dtype=np.float32)
        self.const     = np.array(cfg.current_const if const is None else const, dtype=np.float32)
        self.vortices  = list(cfg.vortices if vortices is None else vortices)
        self.max_speed = float(cfg.current_clip)

    def get_current_at_position(self, pos) -> np.ndarray:
        return self.get_current_at_positions(np.asarray(pos, dtype=np.float32).reshape(1, 3))[0]

    def get_current_at_positions(self, pos) -> np.ndarray:
        pos = np.asarray(pos, dtype=np.float32).reshape(-1, 3)
        vel = np.tile(self.const, (pos.shape[0], 1))

        for vtx in self.vortices:
            c  = np.asarray(vtx["center"], dtype=np.float32)
            dx = pos[:, 0] - c[0]
            dy = pos[:, 1] - c[1]
            dz = pos[:, 2] - c[2]

            r              = np.sqrt(dx * dx + dy * dy)
            safe_r         = np.maximum(r, 1e-4)
            radius         = float(vtx["radius"])
            gamma          = float(vtx["gamma"])
            depth_decay    = float(vtx.get("depth_decay", 50.0))
            vertical_scale = float(vtx.get("vertical_scale", 0.3))

            decay_z = np.exp(-(dz * dz) / (2.0 * depth_decay ** 2))
            v_theta = (
                gamma / (2.0 * math.pi * safe_r)
                * (1.0 - np.exp(-(safe_r ** 2) / (radius ** 2)))
            )

            vx = -v_theta * dy / safe_r * decay_z
            vy =  v_theta * dx / safe_r * decay_z
            vz = (
                vertical_scale * gamma / (2.0 * math.pi * radius)
                * np.exp(-(safe_r ** 2) / (2.0 * radius ** 2))
                * np.sin(dz / depth_decay * math.pi)
                * decay_z
            )

            mask = r < 0.1
            vx[mask] = vy[mask] = vz[mask] = 0.0

            vel += np.stack([vx, vy, vz], axis=1)

        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        over  = speed[:, 0] > self.max_speed
        vel[over] = vel[over] / speed[over] * self.max_speed
        return vel.astype(np.float32)

# ──────────────────────────────────────────────────────────────────────────────
# AUV 3-D 路径规划环境
# ──────────────────────────────────────────────────────────────────────────────

class AUVEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, obstacle_name=None, target=None, current=None):
        super().__init__()
        self.cfg    = cfg
        self.dt     = float(cfg.dt)
        self.bounds = np.array(cfg.bounds, dtype=np.float32)
        self.x0, self.x1, self.y0, self.y1, self.z0, self.z1 = self.bounds
        self.scale  = float(max(self.x1 - self.x0, self.y1 - self.y0, self.z1 - self.z0))

        self.start_fixed  = cfg.start_fixed
        self.target_fixed = cfg.target_fixed
        self.start        = cfg.start.astype(np.float32).copy()
        self.target       = (
            cfg.target if target is None else np.asarray(target, dtype=np.float32)
        ).astype(np.float32)
        self.obstacles    = list(cfg.get_obstacles(obstacle_name))
        self.k_obs        = int(cfg.k_obs)
        self.r_min, self.r_max = cfg.obs_radius
        self.r_span       = float(self.r_max - self.r_min)

        self.current_mode = cfg.current_mode
        self.current_on   = self.current_mode != "none"
        self.current      = self._init_current(current)

        if self.current_mode == "synthetic":
            self.vortices  = list(cfg.vortices)
        self.obs_dim = int(cfg.obs_dim)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(self.obs_dim,), dtype=np.float32)

        self.pos         = self.start.copy()
        self.yaw         = 0.0
        self.pitch       = 0.0
        self.body_vec    = np.zeros(3, dtype=np.float32)
        self.current_vec = np.zeros(3, dtype=np.float32)
        self.applied_current_vec = np.zeros(3, dtype=np.float32)
        self.sim_time    = None

    def _init_current(self, current):
        if not self.current_on:
            return None
        if current is not None:
            return current
        if self.current_mode == "synthetic":
            return OceanCurrent()
        raise ValueError(
            "current_mode='real' requires a pre-built current object. "
            "Use build_env() in main.py which calls ocean_current.build_current()."
        )

    @property
    def env_bound(self):
        return tuple(self.bounds.tolist())

    @property
    def target_pos(self):
        return self.target

    def _nearest_obstacles(self, pos):
        data = [
            (
                np.array([ox, oy, oz], dtype=np.float32),
                float(r),
                float(np.linalg.norm(pos - np.array([ox, oy, oz], dtype=np.float32)) - r),
            )
            for ox, oy, oz, r in self.obstacles
        ]
        data.sort(key=lambda x: x[2])
        return data[: self.k_obs]

    def _nearest_dist(self, pos) -> float:
        nearest = self._nearest_obstacles(pos)
        return float(nearest[0][2]) if nearest else float("inf")

    def _out(self, pos) -> bool:
        x, y, z = pos
        return (
            x < self.x0 or x > self.x1
            or y < self.y0 or y > self.y1
            or z < self.z0 or z > self.z1
        )

    def _sync_current_time(self):
        if (
            self.current_on
            and self.current is not None
            and getattr(self.current, "enable_temporal", False)
            and self.sim_time is not None
            and hasattr(self.current, "set_simulation_time")
        ):
            self.current.set_simulation_time(self.sim_time)

    def _advance_current_time(self):
        if not (
            self.current_on
            and self.current is not None
            and getattr(self.current, "enable_temporal", False)
            and self.sim_time is not None
        ):
            return

        if hasattr(self.current, "advance_time"):
            self.sim_time = self.current.advance_time(self.dt)
        else:
            self.sim_time = self.sim_time + timedelta(seconds=self.dt)
            self._sync_current_time()

    def _update_current(self):
        if self.current_on:
            self._sync_current_time()
            self.current_vec = self.current.get_current_at_position(self.pos)
        else:
            self.current_vec[:] = 0.0

    def _compute_velocity_jacobian(self, yaw: float, pitch: float) -> np.ndarray:
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        return np.array([
            [cy * cp, -sy, -cy * sp],
            [sy * cp,  cy, -sy * sp],
            [sp,      0.0,  cp],
        ], dtype=np.float32)

    def _update_auv_dynamics(self, action: np.ndarray):
        body_speed = (float(action[0]) + 1.0) * 0.5 * self.cfg.max_body_speed
        yaw_rate   = float(action[1]) * self.cfg.max_yaw_rate
        pitch_rate = float(action[2]) * self.cfg.max_pitch_rate

        body_vel_vec = np.array([body_speed, 0.0, 0.0], dtype=np.float32)
        J = self._compute_velocity_jacobian(self.yaw, self.pitch)
        ned_vel = J @ body_vel_vec
        self.body_vec = ned_vel.copy()

        self.yaw   = wrap_angle(self.yaw + yaw_rate * self.dt)
        self.pitch = float(np.clip(self.pitch + pitch_rate * self.dt, *self.cfg.pitch_range))

        self._update_current()
        self.applied_current_vec = self.current_vec.copy()
        self.pos = self.pos + (self.body_vec + self.applied_current_vec) * self.dt

    def _get_obs(self) -> np.ndarray:
        obs = []

        rel = np.clip((self.target - self.pos) / self.scale, -1.0, 1.0)
        obs.extend(rel.tolist())
        obs.extend([self.yaw / math.pi, self.pitch / math.pi])

        nearest = self._nearest_obstacles(self.pos)
        for i in range(self.k_obs):
            if i < len(nearest):
                center, radius, _ = nearest[i]
                rel_obs = np.clip((center - self.pos) / self.scale, -1.0, 1.0)
                r_norm = float(np.clip(2.0 * (radius - self.r_min) / self.r_span - 1.0, -1.0, 1.0))
                obs.extend(rel_obs.tolist())
                obs.append(r_norm)
            else:
                obs.extend([1.0, 1.0, 1.0, 0.0])

        if self.current_on:
            obs.extend(np.clip(self.current_vec / self.cfg.current_clip, -1.0, 1.0).tolist())

        return np.asarray(obs, dtype=np.float32)

    def _get_flags(self) -> dict:
        dist     = float(np.linalg.norm(self.pos - self.target))
        obs_dist = self._nearest_dist(self.pos)
        return {
            "goal":     dist <= self.cfg.goal_dist,
            "hit":      obs_dist <= self.cfg.hit_dist,
            "warn":     obs_dist <= self.cfg.warn_dist,
            "out":      self._out(self.pos),
            "dist":     dist,
            "obs_dist": obs_dist,
        }

    def _get_reward(self, pos0) -> tuple[float, dict]:
        flags = self._get_flags()

        if flags["goal"]:
            r_task = self.cfg.r_goal
        elif flags["hit"]:
            r_task = self.cfg.r_hit
        elif flags["warn"]:
            r_task = self.cfg.r_warn
        else:
            r_task = 0.0

        effective_move = float(np.linalg.norm(pos0 - self.target)) - float(np.linalg.norm(self.pos - self.target))
        actual_move    = float(np.linalg.norm(self.pos - pos0))
        r_dis = self.cfg.r_epsilon1 * effective_move - self.cfg.r_epsilon2 * actual_move

        r_cur = 0.0
        if self.current_on:
            v_t   = self.body_vec
            v_cur = self.applied_current_vec
            norm_t   = float(np.linalg.norm(v_t))
            norm_cur = float(np.linalg.norm(v_cur))
            if norm_t > 1e-6 and norm_cur > 1e-6:
                cos_alpha = float(np.dot(v_t, v_cur)) / (norm_t * norm_cur)
                r_cur = 2.0 * cos_alpha ** 2 - 1.0

        reward = r_task + self.cfg.r_lamda1 * r_dis + self.cfg.r_lamda2 * r_cur
        part = {
            "r_task":     r_task,
            "r_distance": self.cfg.r_lamda1 * r_dis,
            "r_current":  self.cfg.r_lamda2 * r_cur,
        }
        return float(reward), part

    def _get_info(self, part=None) -> dict:
        flags = self._get_flags()
        return {
            "state":         np.array([*self.pos, self.yaw, self.pitch], dtype=np.float32),
            "target":        self.target.copy(),
            "dist":          flags["dist"],
            "obs_dist":      flags["obs_dist"],
            "goal":          flags["goal"],
            "hit":           flags["hit"],
            "warn":          flags["warn"],
            "out":           flags["out"],
            "current":       self.current_vec.copy() if self.current_on else None,
            "current_speed": float(np.linalg.norm(self.current_vec)) if self.current_on else 0.0,
            "reward_part":   part or {},
            "sim_time":      None if self.sim_time is None else str(self.sim_time),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        # 固定AUV起始位置
        self.yaw      = 0.0
        self.pitch    = 0.0
        self.body_vec = np.zeros(3, dtype=np.float32)
        self.current_vec = np.zeros(3, dtype=np.float32)
        self.applied_current_vec = np.zeros(3, dtype=np.float32)
        # 是否使用时变海流
        if self.current_on and getattr(self.current, "enable_temporal", False):
            base_time = getattr(self.current, "time0", None)
            if base_time is not None:
                self.sim_time = base_time   # 每个 episode 均从数据起始时刻开始
                self._sync_current_time()
            else:
                self.sim_time = None
        else:
            self.sim_time = None

        rng = np.random.default_rng(seed)
        # 检查随机化的起点，终点，障碍物的初始位置是否合理
        while True:
            if self.start_fixed:
                self.pos = self.start.copy()
            else:
                margin = 5.0
                x = rng.uniform(self.x0 + margin, self.x0 + (self.x1 - self.x0) * 0.25)
                y = rng.uniform(self.y0 + margin, self.y0 + (self.y1 - self.y0) * 0.25)
                z = rng.uniform(self.z0 + margin, self.z0 + (self.z1 - self.z0) * 0.25)
                self.pos = np.array([x, y, z], dtype=np.float32)
                self.yaw   = rng.uniform(-5 * math.pi / 6, 5 * math.pi / 6)
                self.pitch = rng.uniform(*self.cfg.pitch_range)

            safe_start = True
            for (ox, oy, oz, r) in self.obstacles:
                dist = np.linalg.norm(self.pos - np.array([ox, oy, oz], dtype=np.float32))
                if dist < (r + 2.0):
                    safe_start = False
                    break
            if not safe_start:
                continue
            if not self.target_fixed:
                margin = 5.0
                self.target = np.array([
                    rng.uniform(self.x1 - (self.x1 - self.x0) * 0.25, self.x1 - margin),
                    rng.uniform(self.y1 - (self.y1 - self.y0) * 0.25, self.y1 - margin),
                    rng.uniform(self.z1 - (self.z1 - self.z0) * 0.25, self.z1 - margin),
                ], dtype=np.float32)

            dist2target = float(np.linalg.norm(self.pos - self.target))
            if dist2target > 10.0:
                break
        self._update_current()
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(3), -1.0, 1.0)
        pos0   = self.pos.copy()

        self._update_auv_dynamics(action)
        applied_current = self.applied_current_vec.copy()
        reward, part = self._get_reward(pos0)

        flags = self._get_flags()
        done = flags["goal"] or flags["hit"] or flags["out"]
        if flags["goal"]:
            termination_reason = "goal"
        elif flags["hit"]:
            termination_reason = "hit_obstacle"
        elif flags["out"]:
            termination_reason = "hit_boundary"
        else:
            termination_reason = "running"

        self._advance_current_time()
        self._update_current()

        obs = self._get_obs()
        info = self._get_info(part)
        info.update({
            "current_applied": applied_current,
            "current_speed_applied": float(np.linalg.norm(applied_current)) if self.current_on else 0.0,
            "step_terminated": bool(done),
            "termination_reason": termination_reason,
        })
        return obs, reward, done, False, info
