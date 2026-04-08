import glob
import os
import re

import numpy as np
import torch as T
import torch.nn.functional as F

from config import FileAddress, NetworkConfig
from networks import ActorNetwork, CriticNetwork


# ══════════════════════════════════════════════════════════════════════════════
# SumTree
# ══════════════════════════════════════════════════════════════════════════════

class SumTree:

    def __init__(self, capacity: int):
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data      = np.zeros(capacity, dtype=np.int32)
        self.n_entries = 0
        self.write     = 0   

    # ------------------------------------------------------------------
    # 内部方法（迭代版，替代原递归实现）
    # ------------------------------------------------------------------

    def _propagate(self, idx: int, change: float) -> None:
        """自叶节点向根节点传播优先级变化量（迭代）。"""
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def _retrieve(self, s: float) -> int:
        """按概率 s 查找叶节点索引（迭代）。"""
        idx = 0
        while True:
            left  = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx
            if s <= self.tree[left]:
                idx = left
            else:
                s  -= self.tree[left]
                idx = right

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def total(self) -> float:
        return self.tree[0]

    def add(self, p: float, data) -> None:
        """存入一条转移，并赋予优先级 p。"""
        leaf_idx              = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(leaf_idx, p)

        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, p: float) -> None:
        """更新叶节点 idx 的优先级，并向根传播差值。"""
        change         = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float):
        """
        按概率值 s 取样，返回 (tree_idx, priority, data)。

        修复：将 s 夹紧至 [0, total - ε]，避免浮点误差导致
        _retrieve 走入零优先级的空叶节点。
        """
        s        = np.clip(s, 0.0, self.total() - 1e-8)
        idx      = self._retrieve(s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def __len__(self) -> int:
        return self.n_entries


# ══════════════════════════════════════════════════════════════════════════════
# Standard Uniform Replay Buffer（无 PER，保持原样）
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, max_size: int, input_shape: tuple, n_actions: int):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory     = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory    = np.zeros((self.mem_size, n_actions),    dtype=np.float32)
        self.reward_memory    = np.zeros(self.mem_size,                 dtype=np.float32)
        self.terminal_memory  = np.zeros(self.mem_size,                 dtype=bool)

    def store_transition(self, state, action, reward, next_state, done) -> None:
        index = self.mem_cntr % self.mem_size
        self.state_memory[index]     = state
        self.new_state_memory[index] = next_state
        self.action_memory[index]    = action
        self.reward_memory[index]    = reward
        self.terminal_memory[index]  = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size: int):
        max_mem     = min(self.mem_cntr, self.mem_size)
        batch       = np.random.choice(max_mem, batch_size, replace=False)
        states      = self.state_memory[batch]
        next_states = self.new_state_memory[batch]
        actions     = self.action_memory[batch]
        rewards     = self.reward_memory[batch]
        dones       = self.terminal_memory[batch]
        return states, actions, rewards, next_states, dones

    def ready(self, batch_size: int) -> bool:
        return self.mem_cntr >= batch_size

    def __len__(self) -> int:
        return min(self.mem_cntr, self.mem_size)


# ══════════════════════════════════════════════════════════════════════════════
# Prioritized Experience Replay Buffer（PER）
# ══════════════════════════════════════════════════════════════════════════════

class PERReplayBuffer:
    def __init__(self, max_size: int, input_shape: tuple, n_actions: int):
        self.epsilon        : float = NetworkConfig.epsilon
        self.alpha          : float = NetworkConfig.alpha
        self.beta           : float = NetworkConfig.beta
        self.beta_increment : float = NetworkConfig.beta_increment

        self.capacity    = max_size
        self.n_actions   = n_actions
        self.input_shape = input_shape

        self.tree = SumTree(max_size)
        self._max_raw_error: float = 1.0

        self.state_memory     = np.zeros((self.capacity, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.capacity, *input_shape), dtype=np.float32)
        self.action_memory    = np.zeros((self.capacity, n_actions),    dtype=np.float32)
        self.reward_memory    = np.zeros(self.capacity,                 dtype=np.float32)
        self.terminal_memory  = np.zeros(self.capacity,                 dtype=bool)

    def _priority(self, error: float) -> float:
        """将原始 TD 误差转换为存储优先级。"""
        return (abs(error) + self.epsilon) ** self.alpha

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    @property
    def mem_cntr(self) -> int:
        """已存储的转移数量，与 ReplayBuffer.mem_cntr 语义一致。"""
        return self.tree.n_entries

    def store_transition(self, state, action, reward, next_state, done) -> None:
        # 🟢 核心修改 2：把数据存入连续的 Numpy 数组中
        index = self.tree.write
        self.state_memory[index]     = state
        self.new_state_memory[index] = next_state
        self.action_memory[index]    = action
        self.reward_memory[index]    = reward
        self.terminal_memory[index]  = done
        
        # 🟢 核心修改 3：SumTree 里只存这个整型 index，不存数据！
        p = self._priority(self._max_raw_error)
        self.tree.add(p, index) 
        

    def sample_buffer(self, batch_size: int):
        batch_indices, tree_idxs, priorities = [], [], []
        segment = self.tree.total() / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, p, data_index = self.tree.get(s) # data_index 现在只是一个整数
            priorities.append(p)
            batch_indices.append(data_index)
            tree_idxs.append(idx)

        batch_indices = np.array(batch_indices, dtype=np.int32)

        states      = self.state_memory[batch_indices]
        next_states = self.new_state_memory[batch_indices]
        actions     = self.action_memory[batch_indices]
        rewards     = self.reward_memory[batch_indices]
        dones       = self.terminal_memory[batch_indices]

        # 计算 IS 权重
        priorities_arr = np.array(priorities, dtype=np.float64)
        sampling_probs = priorities_arr / self.tree.total()
        is_weights     = np.power(self.tree.n_entries * sampling_probs, -self.beta)
        is_weights     = (is_weights / is_weights.max()).astype(np.float32)

        return states, actions, rewards, next_states, dones, tree_idxs, is_weights

    def update_priorities(self, tree_idxs, td_errors: np.ndarray) -> None:
        """
        用本批次的 TD 误差回写 SumTree 中各转移的优先级。

        Parameters
        ----------
        tree_idxs : list[int]   来自 sample_buffer 的叶节点索引
        td_errors : np.ndarray  每条转移对应的 TD 误差（shape: [batch]）
        """
        # 【新增】设置 TD Error 的最大上限。
        # 即使撞墙产生了 200 的误差，在这里也会被强行压平到 10.0。

        max_error_clip = 10.0
        for idx, err in zip(tree_idxs, td_errors):
            raw_err = float(abs(err))
            clipped_err = min(raw_err, max_error_clip)
            self._max_raw_error = max(self._max_raw_error, clipped_err)
            self.tree.update(idx, self._priority(clipped_err))

    def ready(self, batch_size: int) -> bool:
        return self.tree.n_entries >= batch_size

    def __len__(self) -> int:
        return self.tree.n_entries

# ══════════════════════════════════════════════════════════════════════════════
# TD3 Agent
# ══════════════════════════════════════════════════════════════════════════════

class TD3:
    def __init__(self, state_dims=None, n_actions=None, use_PER=None):
        self.state_dims = state_dims if state_dims is not None else NetworkConfig.state_dim
        self.n_actions  = n_actions  if n_actions  is not None else NetworkConfig.action_dim
        self.use_PER    = use_PER    if use_PER    is not None else NetworkConfig.use_PER

        self.update_cnt         = 0
        self.gamma              = NetworkConfig.gamma
        self.tau                = NetworkConfig.tau
        self.actor_delay        = NetworkConfig.actor_delay
        self.target_noise_clip  = NetworkConfig.target_noise_clip
        self.target_noise_range = NetworkConfig.target_noise
        self.action_noise_range = NetworkConfig.action_noise
        self.chkpt_dir          = FileAddress.net

        # ── 根据标志位选择 Replay Buffer ──────────────────────────────────
        input_shape = (self.state_dims,)
        if self.use_PER:
            self.memory = PERReplayBuffer(NetworkConfig.buffer, input_shape, self.n_actions)
            print("[TD3] Prioritized Experience Replay (PER) is ENABLED.")
        else:
            self.memory = ReplayBuffer(NetworkConfig.buffer, input_shape, self.n_actions)
            print("[TD3] Using standard uniform Replay Buffer.")

        self.actor = ActorNetwork(
            NetworkConfig.actor_lr, self.state_dims, self.n_actions,
            fc1_dims=NetworkConfig.hidden[0], fc2_dims=NetworkConfig.hidden[1],
        )
        self.target_actor = ActorNetwork(
            NetworkConfig.actor_lr, self.state_dims, self.n_actions,
            fc1_dims=NetworkConfig.hidden[0], fc2_dims=NetworkConfig.hidden[1],
        )
        self.critic_1 = CriticNetwork(
            NetworkConfig.critic_lr, self.state_dims, self.n_actions,
            fc1_dims=NetworkConfig.hidden[0], fc2_dims=NetworkConfig.hidden[1],
        )
        self.critic_2 = CriticNetwork(
            NetworkConfig.critic_lr, self.state_dims, self.n_actions,
            fc1_dims=NetworkConfig.hidden[0], fc2_dims=NetworkConfig.hidden[1],
        )
        self.target_critic_1 = CriticNetwork(
            NetworkConfig.critic_lr, self.state_dims, self.n_actions,
            fc1_dims=NetworkConfig.hidden[0], fc2_dims=NetworkConfig.hidden[1],
        )
        self.target_critic_2 = CriticNetwork(
            NetworkConfig.critic_lr, self.state_dims, self.n_actions,
            fc1_dims=NetworkConfig.hidden[0], fc2_dims=NetworkConfig.hidden[1],
        )

        self.soft_update_network_parameters(tau=1.0)
        self.device = self.actor.device

    # ──────────────────────────────────────────────────────────────────────────
    # 公共接口（与原始 TD3 完全一致）
    # ──────────────────────────────────────────────────────────────────────────

    def soft_update_network_parameters(self, tau=None) -> None:
        tau = self.tau if tau is None else tau
        for tp, p in zip(self.target_actor.parameters(),    self.actor.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
        for tp, p in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
        for tp, p in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

    def select_action(self, observation, evaluate: bool = False) -> np.ndarray:
        if not isinstance(observation, T.Tensor):
            state = T.tensor(observation, dtype=T.float32, device=self.device)
        else:
            state = observation.to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with T.no_grad():
            actions = self.actor(state).squeeze(0).cpu().numpy()

        if not evaluate:
            noise   = np.random.normal(0, self.action_noise_range, size=self.n_actions)
            actions = np.clip(actions + noise, -1.0, 1.0)
        return actions

    def smooth_target_action(self, next_state: T.Tensor) -> T.Tensor:
        """计算带平滑噪声的目标动作（Target Policy Smoothing）。"""
        if not isinstance(next_state, T.Tensor):
            next_state = T.tensor(next_state, dtype=T.float32, device=self.device)
        else:
            next_state = next_state.to(self.device)
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)

        with T.no_grad():
            actions = self.target_actor(next_state)
            noise   = T.normal(0, self.target_noise_range, size=actions.shape, device=self.device)
            noise   = T.clamp(noise, -self.target_noise_clip, self.target_noise_clip)
            return  T.clamp(actions + noise, -1.0, 1.0)

    def update(self, batch_size: int = NetworkConfig.batch) -> dict | None:
        if not self.memory.ready(batch_size):
            return None

        self.update_cnt += 1

        # ── 采样 ──────────────────────────────────────────────────────────────
        if self.use_PER:
            states, actions, rewards, next_states, dones, tree_idxs, is_weights = \
                self.memory.sample_buffer(batch_size)
            is_weights_t = T.tensor(is_weights, dtype=T.float32, device=self.device).unsqueeze(1)
        else:
            states, actions, rewards, next_states, dones = \
                self.memory.sample_buffer(batch_size)

        states      = T.tensor(states,      dtype=T.float32, device=self.device)
        actions     = T.tensor(actions,     dtype=T.float32, device=self.device)
        rewards     = T.tensor(rewards,     dtype=T.float32, device=self.device).unsqueeze(1)
        next_states = T.tensor(next_states, dtype=T.float32, device=self.device)
        dones       = T.tensor(dones,       dtype=T.float32, device=self.device).unsqueeze(1)

        # ── 目标 Q 值（两种模式共用）─────────────────────────────────────────
        with T.no_grad():
            smoothed_next_actions = self.smooth_target_action(next_states)
            target_q_min   = T.min(
                self.target_critic_1(next_states, smoothed_next_actions),
                self.target_critic_2(next_states, smoothed_next_actions),
            )
            target_q_value = rewards + (1 - dones) * self.gamma * target_q_min

        # ── Critic 1 & 2 更新（合并 zero_grad，分别 backward）──────────────
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        predicted_q1 = self.critic_1(states, actions)
        predicted_q2 = self.critic_2(states, actions)

        if self.use_PER:
            # 提前保存两个 critic 的 TD 误差，用于稍后回写优先级
            td_errors_1 = (predicted_q1 - target_q_value).detach()
            td_errors_2 = (predicted_q2 - target_q_value).detach()

            # IS 权重加权 MSE（逐样本）以消除采样偏差
            loss_fn  = lambda pred: (is_weights_t * F.mse_loss(pred, target_q_value, reduction="none")).mean()
            q_loss_1 = loss_fn(predicted_q1)
            q_loss_2 = loss_fn(predicted_q2)
        else:
            q_loss_1 = F.mse_loss(predicted_q1, target_q_value)
            q_loss_2 = F.mse_loss(predicted_q2, target_q_value)

        q_loss_1.backward()
        q_loss_2.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # ── PER 优先级回写（使用两个 Critic TD 误差的均值，更稳定）──────────
        if self.use_PER:
            td_errors_np = ((td_errors_1.abs() + td_errors_2.abs()) / 2).squeeze(1).cpu().numpy()
            self.memory.update_priorities(tree_idxs, td_errors_np)

        # ── Delayed Actor 更新 ────────────────────────────────────────────────
        actor_loss_value = np.nan
        actor_updated    = False
        if self.update_cnt % self.actor_delay == 0:
            self.actor.optimizer.zero_grad()
            actor_loss = -self.critic_1(states, self.actor(states)).mean()
            actor_loss.backward()
            self.actor.optimizer.step()
            self.soft_update_network_parameters()
            actor_loss_value = float(actor_loss.item())
            actor_updated    = True

        info = {
            "update_index":  int(self.update_cnt),
            "critic1_loss":  float(q_loss_1.item()),
            "critic2_loss":  float(q_loss_2.item()),
            "actor_loss":    actor_loss_value,
            "actor_updated": bool(actor_updated),
            "target_q_mean": float(target_q_value.mean().item()),
            "q1_mean":       float(predicted_q1.mean().item()),
            "q2_mean":       float(predicted_q2.mean().item()),
        }
        # PER 专属监控指标（供 TensorBoard 等工具记录 beta 退火过程）
        if self.use_PER:
            info["per_beta"]         = float(self.memory.beta)
            info["per_max_priority"] = float(self.memory._max_raw_error)

        return info

    # ──────────────────────────────────────────────────────────────────────────
    # 模型存取（与原始 TD3 完全一致）
    # ──────────────────────────────────────────────────────────────────────────

    def _build_weight_paths(self, episode=None) -> dict:
        os.makedirs(self.chkpt_dir, exist_ok=True)
        suffix = f"_episode{episode}.pth" if episode is not None else ".pth"
        return {
            "actor":          os.path.join(self.chkpt_dir, f"td3_actor_net{suffix}"),
            "critic1":        os.path.join(self.chkpt_dir, f"td3_critic_1_net{suffix}"),
            "critic2":        os.path.join(self.chkpt_dir, f"td3_critic_2_net{suffix}"),
            "target_actor":   os.path.join(self.chkpt_dir, f"td3_target_actor_net{suffix}"),
            "target_critic1": os.path.join(self.chkpt_dir, f"td3_target_critic_1_net{suffix}"),
            "target_critic2": os.path.join(self.chkpt_dir, f"td3_target_critic_2_net{suffix}"),
        }

    def best_saved_episode(self) -> int | None:
        import pandas as pd

        candidates = glob.glob(os.path.join(self.chkpt_dir, "td3_actor_net_episode*.pth"))
        if not candidates:
            return None

        saved_episodes = [
            int(m.group(1))
            for path in candidates
            if (m := re.search(r"episode(\d+)\.pth$", os.path.basename(path)))
        ]
        if not saved_episodes:
            return None

        summary_file = FileAddress.summary_csv("train")
        if os.path.exists(summary_file):
            try:
                df       = pd.read_csv(summary_file)
                df_saved = df[df["episode"].isin(saved_episodes)]
                if not df_saved.empty:
                    best_idx    = df_saved["episode_reward"].idxmax()
                    best_ep     = int(df_saved.loc[best_idx, "episode"])
                    best_reward = df_saved.loc[best_idx, "episode_reward"]
                    print(f"[TD3] Auto-selected BEST episode: {best_ep} (Reward: {best_reward:.2f})")
                    return best_ep
            except Exception as e:
                print(f"[TD3] Warning: Failed to parse summary file for best episode ({e}).")

        latest_ep = max(saved_episodes)
        print(f"[TD3] Auto-selected LATEST episode: {latest_ep}")
        return latest_ep

    def save_weights(self, episode=None, include_target: bool = True) -> None:
        paths = self._build_weight_paths(episode)
        T.save(self.actor.state_dict(),    paths["actor"])
        T.save(self.critic_1.state_dict(), paths["critic1"])
        T.save(self.critic_2.state_dict(), paths["critic2"])
        if include_target:
            T.save(self.target_actor.state_dict(),    paths["target_actor"])
            T.save(self.target_critic_1.state_dict(), paths["target_critic1"])
            T.save(self.target_critic_2.state_dict(), paths["target_critic2"])
        print(f"[TD3] model saved successfully to: {self.chkpt_dir} (episode={episode})")

    def load_weights(self, episode=None, strict: bool = True, include_target: bool = True) -> None:
        paths = self._build_weight_paths(episode)
        actor_p, c1_p, c2_p = paths["actor"], paths["critic1"], paths["critic2"]

        if not all(os.path.exists(p) for p in (actor_p, c1_p, c2_p)):
            raise FileNotFoundError(
                f"[TD3] Cannot find weight files: {actor_p}, {c1_p}, {c2_p}"
            )

        def _load(net, path):
            net.load_state_dict(T.load(path, map_location=net.device, weights_only=True), strict=strict)

        _load(self.actor,    actor_p)
        _load(self.critic_1, c1_p)
        _load(self.critic_2, c2_p)

        if include_target:
            _load(self.target_actor,    paths["target_actor"]   if os.path.exists(paths["target_actor"])   else actor_p)
            _load(self.target_critic_1, paths["target_critic1"] if os.path.exists(paths["target_critic1"]) else c1_p)
            _load(self.target_critic_2, paths["target_critic2"] if os.path.exists(paths["target_critic2"]) else c2_p)

        print(f"[TD3] model loaded successfully: episode={episode}")


class DDPG:
    def __init__(self, state_dims=None, n_actions=None, use_PER=None):
        self.state_dims = state_dims if state_dims is not None else NetworkConfig.state_dim
        self.n_actions  = n_actions  if n_actions  is not None else NetworkConfig.action_dim
        self.use_PER    = use_PER    if use_PER    is not None else NetworkConfig.use_PER

        self.update_cnt         = 0
        self.gamma              = NetworkConfig.gamma
        self.tau                = NetworkConfig.tau
        self.action_noise_range = NetworkConfig.action_noise
        self.chkpt_dir          = FileAddress.net

        # ── 根据标志位选择 Replay Buffer ──────────────────────────────────
        input_shape = (self.state_dims,)
        if self.use_PER:
            self.memory = PERReplayBuffer(NetworkConfig.buffer, input_shape, self.n_actions)
            print("[DDPG] Prioritized Experience Replay (PER) is ENABLED.")
        else:
            self.memory = ReplayBuffer(NetworkConfig.buffer, input_shape, self.n_actions)
            print("[DDPG] Using standard uniform Replay Buffer.")

        self.actor = ActorNetwork(
            NetworkConfig.actor_lr, self.state_dims, self.n_actions,
            fc1_dims=NetworkConfig.hidden[0], fc2_dims=NetworkConfig.hidden[1],
        )
        self.target_actor = ActorNetwork(
            NetworkConfig.actor_lr, self.state_dims, self.n_actions,
            fc1_dims=NetworkConfig.hidden[0], fc2_dims=NetworkConfig.hidden[1],
        )
        # DDPG 只需要一个 Critic
        self.critic = CriticNetwork(
            NetworkConfig.critic_lr, self.state_dims, self.n_actions,
            fc1_dims=NetworkConfig.hidden[0], fc2_dims=NetworkConfig.hidden[1],
        )
        self.target_critic = CriticNetwork(
            NetworkConfig.critic_lr, self.state_dims, self.n_actions,
            fc1_dims=NetworkConfig.hidden[0], fc2_dims=NetworkConfig.hidden[1],
        )

        self.soft_update_network_parameters(tau=1.0)
        self.device = self.actor.device

    # ──────────────────────────────────────────────────────────────────────────
    # 公共接口
    # ──────────────────────────────────────────────────────────────────────────

    def soft_update_network_parameters(self, tau=None) -> None:
        tau = self.tau if tau is None else tau
        for tp, p in zip(self.target_actor.parameters(), self.actor.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

    def select_action(self, observation, evaluate: bool = False) -> np.ndarray:
        if not isinstance(observation, T.Tensor):
            state = T.tensor(observation, dtype=T.float32, device=self.device)
        else:
            state = observation.to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with T.no_grad():
            actions = self.actor(state).squeeze(0).cpu().numpy()

        if not evaluate:
            # 标准 DDPG 论文常用 OU Noise，但为了统一风格，这里继续保留高斯噪声
            noise   = np.random.normal(0, self.action_noise_range, size=self.n_actions)
            actions = np.clip(actions + noise, -1.0, 1.0)
        return actions

    def update(self, batch_size: int = NetworkConfig.batch) -> dict | None:
        if not self.memory.ready(batch_size):
            return None

        self.update_cnt += 1

        # ── 采样 ──────────────────────────────────────────────────────────────
        if self.use_PER:
            states, actions, rewards, next_states, dones, tree_idxs, is_weights = \
                self.memory.sample_buffer(batch_size)
            is_weights_t = T.tensor(is_weights, dtype=T.float32, device=self.device).unsqueeze(1)
        else:
            states, actions, rewards, next_states, dones = \
                self.memory.sample_buffer(batch_size)

        states      = T.tensor(states,      dtype=T.float32, device=self.device)
        actions     = T.tensor(actions,     dtype=T.float32, device=self.device)
        rewards     = T.tensor(rewards,     dtype=T.float32, device=self.device).unsqueeze(1)
        next_states = T.tensor(next_states, dtype=T.float32, device=self.device)
        dones       = T.tensor(dones,       dtype=T.float32, device=self.device).unsqueeze(1)

        # ── 目标 Q 值（DDPG 无目标策略平滑）───────────────────────────────────
        with T.no_grad():
            target_actions = self.target_actor(next_states)
            target_q_value = rewards + (1 - dones) * self.gamma * self.target_critic(next_states, target_actions)

        # ── Critic 更新 ───────────────────────────────────────────────────────
        self.critic.optimizer.zero_grad()
        predicted_q = self.critic(states, actions)

        if self.use_PER:
            td_errors = (predicted_q - target_q_value).detach()
            loss_fn = lambda pred: (is_weights_t * F.mse_loss(pred, target_q_value, reduction="none")).mean()
            q_loss  = loss_fn(predicted_q)
        else:
            q_loss = F.mse_loss(predicted_q, target_q_value)

        q_loss.backward()
        self.critic.optimizer.step()

        # ── PER 优先级回写（直接使用唯一的 TD error）──────────────────────────
        if self.use_PER:
            td_errors_np = td_errors.abs().squeeze(1).cpu().numpy()
            self.memory.update_priorities(tree_idxs, td_errors_np)

        # ── Actor 更新（无延迟，每步更新）─────────────────────────────────────
        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic(states, self.actor(states)).mean()
        actor_loss.backward()
        self.actor.optimizer.step()
        
        # 每步均进行目标网络软更新
        self.soft_update_network_parameters()

        info = {
            "update_index":  int(self.update_cnt),
            "critic_loss":   float(q_loss.item()),
            "actor_loss":    float(actor_loss.item()),
            "actor_updated": True,
            "target_q_mean": float(target_q_value.mean().item()),
            "q_mean":        float(predicted_q.mean().item()),
        }
        
        # PER 专属监控指标
        if self.use_PER:
            info["per_beta"]         = float(self.memory.beta)
            info["per_max_priority"] = float(self.memory._max_raw_error)

        return info

    # ──────────────────────────────────────────────────────────────────────────
    # 模型存取 (文件名前缀变更为 ddpg_)
    # ──────────────────────────────────────────────────────────────────────────

    def _build_weight_paths(self, episode=None) -> dict:
        os.makedirs(self.chkpt_dir, exist_ok=True)
        suffix = f"_episode{episode}.pth" if episode is not None else ".pth"
        return {
            "actor":         os.path.join(self.chkpt_dir, f"ddpg_actor_net{suffix}"),
            "critic":        os.path.join(self.chkpt_dir, f"ddpg_critic_net{suffix}"),
            "target_actor":  os.path.join(self.chkpt_dir, f"ddpg_target_actor_net{suffix}"),
            "target_critic": os.path.join(self.chkpt_dir, f"ddpg_target_critic_net{suffix}"),
        }

    def best_saved_episode(self) -> int | None:
        import pandas as pd

        candidates = glob.glob(os.path.join(self.chkpt_dir, "ddpg_actor_net_episode*.pth"))
        if not candidates:
            return None

        saved_episodes = [
            int(m.group(1))
            for path in candidates
            if (m := re.search(r"episode(\d+)\.pth$", os.path.basename(path)))
        ]
        if not saved_episodes:
            return None

        summary_file = FileAddress.summary_csv("train")
        if os.path.exists(summary_file):
            try:
                df       = pd.read_csv(summary_file)
                df_saved = df[df["episode"].isin(saved_episodes)]
                if not df_saved.empty:
                    best_idx    = df_saved["episode_reward"].idxmax()
                    best_ep     = int(df_saved.loc[best_idx, "episode"])
                    best_reward = df_saved.loc[best_idx, "episode_reward"]
                    print(f"[DDPG] Auto-selected BEST episode: {best_ep} (Reward: {best_reward:.2f})")
                    return best_ep
            except Exception as e:
                print(f"[DDPG] Warning: Failed to parse summary file for best episode ({e}).")

        latest_ep = max(saved_episodes)
        print(f"[DDPG] Auto-selected LATEST episode: {latest_ep}")
        return latest_ep

    def save_weights(self, episode=None, include_target: bool = True) -> None:
        paths = self._build_weight_paths(episode)
        T.save(self.actor.state_dict(),  paths["actor"])
        T.save(self.critic.state_dict(), paths["critic"])
        if include_target:
            T.save(self.target_actor.state_dict(),  paths["target_actor"])
            T.save(self.target_critic.state_dict(), paths["target_critic"])
        print(f"[DDPG] model saved successfully to: {self.chkpt_dir} (episode={episode})")

    def load_weights(self, episode=None, strict: bool = True, include_target: bool = True) -> None:
        paths = self._build_weight_paths(episode)
        actor_p, c_p = paths["actor"], paths["critic"]

        if not all(os.path.exists(p) for p in (actor_p, c_p)):
            raise FileNotFoundError(
                f"[DDPG] Cannot find weight files: {actor_p}, {c_p}"
            )

        def _load(net, path):
            net.load_state_dict(T.load(path, map_location=net.device, weights_only=True), strict=strict)

        _load(self.actor,  actor_p)
        _load(self.critic, c_p)

        if include_target:
            _load(self.target_actor,  paths["target_actor"]  if os.path.exists(paths["target_actor"])  else actor_p)
            _load(self.target_critic, paths["target_critic"] if os.path.exists(paths["target_critic"]) else c_p)

        print(f"[DDPG] model loaded successfully: episode={episode}")