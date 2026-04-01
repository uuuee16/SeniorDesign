import glob
import os
import re

import numpy as np
import torch as T
import torch.nn.functional as F

from config import FileAddress, NetworkConfig
from networks import ActorNetwork, CriticNetwork


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        next_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, next_states, dones

    def ready(self, batch_size):
        return self.mem_cntr >= batch_size


class TD3:
    def __init__(self, state_dims=None, n_actions=None):
        self.state_dims = state_dims if state_dims is not None else NetworkConfig.state_dim
        self.n_actions = n_actions if n_actions is not None else NetworkConfig.action_dim

        self.gamma = NetworkConfig.gamma
        self.tau = NetworkConfig.tau
        self.actor_update_interval = NetworkConfig.actor_update_interval
        self.smooth_noise_clip = NetworkConfig.target_noise_clip
        self.smooth_noise_range = NetworkConfig.target_noise
        self.explore_noise_range = NetworkConfig.explore_noise
        self.update_cnt = 0
        self.chkpt_dir = FileAddress.net

        self.actor = ActorNetwork(
            NetworkConfig.actor_lr,
            self.state_dims, self.n_actions,
            fc1_dims=NetworkConfig.hidden[0],
            fc2_dims=NetworkConfig.hidden[1],
        )
        self.target_actor = ActorNetwork(
            NetworkConfig.actor_lr,
            self.state_dims, self.n_actions,
            fc1_dims=NetworkConfig.hidden[0],
            fc2_dims=NetworkConfig.hidden[1],
        )
        self.critic_1 = CriticNetwork(
            NetworkConfig.critic_lr,
            self.state_dims, self.n_actions,
            fc1_dims=NetworkConfig.hidden[0],
            fc2_dims=NetworkConfig.hidden[1],
        )
        self.critic_2 = CriticNetwork(
            NetworkConfig.critic_lr,
            self.state_dims, self.n_actions,
            fc1_dims=NetworkConfig.hidden[0],
            fc2_dims=NetworkConfig.hidden[1],
        )
        self.target_critic_1 = CriticNetwork(
            NetworkConfig.critic_lr,
            self.state_dims, self.n_actions,
            fc1_dims=NetworkConfig.hidden[0],
            fc2_dims=NetworkConfig.hidden[1],
        )
        self.target_critic_2 = CriticNetwork(
            NetworkConfig.critic_lr,
            self.state_dims, self.n_actions,
            fc1_dims=NetworkConfig.hidden[0],
            fc2_dims=NetworkConfig.hidden[1],
        )

        self.soft_update_network_parameters(tau=1.0)
        self.memory = ReplayBuffer(NetworkConfig.buffer, input_shape=(self.state_dims,), n_actions=self.n_actions)
        self.device = self.actor.device

    def soft_update_network_parameters(self, tau=None):
        tau = self.tau if tau is None else tau
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def select_action(self, observation, evaluate=False):
        if not isinstance(observation, T.Tensor):
            state = T.tensor(observation, dtype=T.float32, device=self.device)
        else:
            state = observation.to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        with T.no_grad():
            actions = self.actor.forward(state)
        actions = actions.squeeze(0).cpu().numpy()
        if not evaluate:
            noise = np.random.normal(0, self.explore_noise_range, size=self.n_actions)
            actions = np.clip(actions + noise, -1.0, 1.0)
        return actions

    def smooth_target_action(self, next_state):
        if not isinstance(next_state, T.Tensor):
            state = T.tensor(next_state, dtype=T.float32, device=self.device)
        else:
            state = next_state.to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        with T.no_grad():
            actions = self.target_actor.forward(state)
            noise = T.normal(0, self.smooth_noise_range, size=actions.shape, device=self.device)
            noise = T.clamp(noise, -self.smooth_noise_clip, self.smooth_noise_clip)
            smoothed_actions = T.clamp(actions + noise, -1.0, 1.0)
        return smoothed_actions

    def update(self, batch_size=NetworkConfig.batch):
        if not self.memory.ready(batch_size):
            return None

        self.update_cnt += 1

        states, actions, rewards, next_states, dones = self.memory.sample_buffer(batch_size)
        states = T.tensor(states, dtype=T.float32, device=self.device)
        actions = T.tensor(actions, dtype=T.float32, device=self.device)
        rewards = T.tensor(rewards, dtype=T.float32, device=self.device).unsqueeze(1)
        next_states = T.tensor(next_states, dtype=T.float32, device=self.device)
        dones = T.tensor(dones, dtype=T.float32, device=self.device).unsqueeze(1)

        smoothed_next_actions = self.smooth_target_action(next_states)
        with T.no_grad():
            target_q_min = T.min(
                self.target_critic_1(next_states, smoothed_next_actions),
                self.target_critic_2(next_states, smoothed_next_actions),
            )
            target_q_value = rewards + (1 - dones) * self.gamma * target_q_min

        self.critic_1.optimizer.zero_grad()
        predicted_q_value_1 = self.critic_1(states, actions)
        q_value_loss_1 = F.mse_loss(predicted_q_value_1, target_q_value)
        q_value_loss_1.backward()
        self.critic_1.optimizer.step()

        self.critic_2.optimizer.zero_grad()
        predicted_q_value_2 = self.critic_2(states, actions)
        q_value_loss_2 = F.mse_loss(predicted_q_value_2, target_q_value)
        q_value_loss_2.backward()
        self.critic_2.optimizer.step()

        actor_loss_value = np.nan
        actor_updated = False
        if self.update_cnt % self.actor_update_interval == 0:
            self.actor.optimizer.zero_grad()
            actor_loss = -self.critic_1(states, self.actor(states)).mean()
            actor_loss.backward()
            self.actor.optimizer.step()
            self.soft_update_network_parameters()
            actor_loss_value = float(actor_loss.item())
            actor_updated = True

        return {
            "update_index": int(self.update_cnt),
            "critic1_loss": float(q_value_loss_1.item()),
            "critic2_loss": float(q_value_loss_2.item()),
            "actor_loss": actor_loss_value,
            "actor_updated": bool(actor_updated),
            "target_q_mean": float(target_q_value.mean().item()),
            "q1_mean": float(predicted_q_value_1.mean().item()),
            "q2_mean": float(predicted_q_value_2.mean().item()),
        }

    def _build_weight_paths(self, episode=None):
        os.makedirs(self.chkpt_dir, exist_ok=True)
        suffix = f"_episode{episode}.pth" if episode is not None else ".pth"
        return {
            "actor": os.path.join(self.chkpt_dir, f"td3_actor_net{suffix}"),
            "critic1": os.path.join(self.chkpt_dir, f"td3_critic_1_net{suffix}"),
            "critic2": os.path.join(self.chkpt_dir, f"td3_critic_2_net{suffix}"),
            "target_actor": os.path.join(self.chkpt_dir, f"td3_target_actor_net{suffix}"),
            "target_critic1": os.path.join(self.chkpt_dir, f"td3_target_critic_1_net{suffix}"),
            "target_critic2": os.path.join(self.chkpt_dir, f"td3_target_critic_2_net{suffix}"),
        }

    def best_saved_episode(self):
        import pandas as pd
        from config import FileAddress
        
        # 1. 先找出所有实际保存在硬盘上的 episode 权重编号
        pattern = os.path.join(self.chkpt_dir, "td3_actor_net_episode*.pth")
        candidates = glob.glob(pattern)
        if not candidates:
            return None
            
        saved_episodes = []
        for path in candidates:
            m = re.search(r"episode(\d+)\.pth$", os.path.basename(path))
            if m:
                saved_episodes.append(int(m.group(1)))
                
        if not saved_episodes:
            return None
            
        # 2. 尝试从训练总结日志中寻找 Reward 最高的 episode
        summary_file = FileAddress.summary_csv("train")
        if os.path.exists(summary_file):
            try:
                df = pd.read_csv(summary_file)
                # 仅筛选出那些已经保存了网络权重的 episode 记录
                df_saved = df[df["episode"].isin(saved_episodes)]
                
                if not df_saved.empty:
                    # 找到 episode_reward 最高的索引
                    best_idx = df_saved["episode_reward"].idxmax()
                    best_ep = int(df_saved.loc[best_idx]["episode"])
                    best_reward = df_saved.loc[best_idx]["episode_reward"]
                    print(f"[TD3] Auto-selected BEST episode: {best_ep} (Reward: {best_reward:.2f})")
                    return best_ep
            except Exception as e:
                print(f"[TD3] Warning: Failed to parse summary file for best episode ({e}).")
                
        # 3. 如果没有日志文件或解析失败，退化为返回最新的 episode
        latest_ep = max(saved_episodes)
        print(f"[TD3] Auto-selected LATEST episode: {latest_ep}")
        return latest_ep

    def save_weights(self, episode=None, include_target=True):
        paths = self._build_weight_paths(episode)
        T.save(self.actor.state_dict(), paths["actor"])
        T.save(self.critic_1.state_dict(), paths["critic1"])
        T.save(self.critic_2.state_dict(), paths["critic2"])

        if include_target:
            T.save(self.target_actor.state_dict(), paths["target_actor"])
            T.save(self.target_critic_1.state_dict(), paths["target_critic1"])
            T.save(self.target_critic_2.state_dict(), paths["target_critic2"])

        print(f"[TD3] model saved successfully to: {self.chkpt_dir} (episode={episode})")

    def load_weights(self, episode=None, strict=True, include_target=True):
        paths = self._build_weight_paths(episode)
        actor_path, critic1_path, critic2_path = paths["actor"], paths["critic1"], paths["critic2"]

        if not (os.path.exists(actor_path) and os.path.exists(critic1_path) and os.path.exists(critic2_path)):
            raise FileNotFoundError(f"can not find weight files: {actor_path}, {critic1_path}, {critic2_path}")

        self.actor.load_state_dict(T.load(actor_path, map_location=self.actor.device), strict=strict)
        self.critic_1.load_state_dict(T.load(critic1_path, map_location=self.critic_1.device), strict=strict)
        self.critic_2.load_state_dict(T.load(critic2_path, map_location=self.critic_2.device), strict=strict)

        if include_target:
            if os.path.exists(paths["target_actor"]):
                self.target_actor.load_state_dict(T.load(paths["target_actor"], map_location=self.target_actor.device), strict=strict)
            else:
                self.target_actor.load_state_dict(self.actor.state_dict(), strict=strict)

            if os.path.exists(paths["target_critic1"]):
                self.target_critic_1.load_state_dict(T.load(paths["target_critic1"], map_location=self.target_critic_1.device), strict=strict)
            else:
                self.target_critic_1.load_state_dict(self.critic_1.state_dict(), strict=strict)

            if os.path.exists(paths["target_critic2"]):
                self.target_critic_2.load_state_dict(T.load(paths["target_critic2"], map_location=self.target_critic_2.device), strict=strict)
            else:
                self.target_critic_2.load_state_dict(self.critic_2.state_dict(), strict=strict)

        print(f"[TD3] model loaded successfully: episode={episode}")
