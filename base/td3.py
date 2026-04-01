import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from networks import ActorNetwork, CriticNetwork
from config import NetworkConfig, FileAddress, MapConfig  # 导入配置文件


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
        next_state = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, next_state, dones
    
    def ready(self, batch_size):
        if self.mem_cntr >= batch_size:
            return True
        return False

class TD3:
    def __init__(self, state_dims=None, n_actions=None):
        self.state_dims = state_dims if state_dims is not None else NetworkConfig.nn_input_dim
        self.n_actions = n_actions if n_actions is not None else NetworkConfig.nn_output_dim
        
        self.gamma = NetworkConfig.gamma
        self.tau = NetworkConfig.soft_update_tau
        self.n_actions = NetworkConfig.nn_output_dim
        self.actor_update_interval = NetworkConfig.actor_update_interval  # Actor 更新间隔，Critic 更新频率为1
        self.smooth_noise_clip = NetworkConfig.smooth_noise_clip
        self.smooth_noise_range = NetworkConfig.smooth_noise_range
        self.explore_noise_range = NetworkConfig.explore_noise_range
        self.update_cnt = 0
        self.chkpt_dir = FileAddress.td3_network_folder

        self.actor = ActorNetwork(
            NetworkConfig.learning_rate_actor,
            state_dims, n_actions,
            fc1_dims=NetworkConfig.nn_hidden_dim[0],
            fc2_dims=NetworkConfig.nn_hidden_dim[1])
        self.target_actor = ActorNetwork(
            NetworkConfig.learning_rate_actor,
            state_dims, n_actions,
            fc1_dims=NetworkConfig.nn_hidden_dim[0],
            fc2_dims=NetworkConfig.nn_hidden_dim[1])
        self.critic_1 = CriticNetwork(
            NetworkConfig.learning_rate_critic,
            state_dims, n_actions,
            fc1_dims=NetworkConfig.nn_hidden_dim[0],
            fc2_dims=NetworkConfig.nn_hidden_dim[1])
        self.critic_2 = CriticNetwork(
            NetworkConfig.learning_rate_critic,
            state_dims, n_actions,
            fc1_dims=NetworkConfig.nn_hidden_dim[0],
            fc2_dims=NetworkConfig.nn_hidden_dim[1])
        self.target_critic_1 = CriticNetwork(
            NetworkConfig.learning_rate_critic,
            state_dims, n_actions,
            fc1_dims=NetworkConfig.nn_hidden_dim[0],
            fc2_dims=NetworkConfig.nn_hidden_dim[1])
        self.target_critic_2 = CriticNetwork(
            NetworkConfig.learning_rate_critic,
            state_dims, n_actions,
            fc1_dims=NetworkConfig.nn_hidden_dim[0],
            fc2_dims=NetworkConfig.nn_hidden_dim[1])

        # init target networks parameters = original networks parameters
        self.soft_update_network_parameters(tau=1)
        # init replay buffer input_dims = state维度 
        self.memory = ReplayBuffer(NetworkConfig.buffer_size, input_shape=(state_dims,), n_actions=n_actions)
        self.device = self.actor.device
        
        
    # soft update target networks
    def soft_update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    # 根据当前的state，由当前的actor选择action，再添加噪声用于探索
    def select_action(self, observation, evaluate=False):
        """
        [交互阶段] 选择动作与环境交互
        输出范围：[-1, 1] (由 Environment 负责映射到物理世界)
        """
        # 数据预处理
        if not isinstance(observation, T.Tensor):
            state = T.tensor(observation, dtype=T.float32, device=self.device)
        else:
            state = observation.to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0) # [Batch, Dim]

        with T.no_grad():
            actions = self.actor.forward(state)
        actions = actions.squeeze(0).cpu().numpy()
        if not evaluate:
            noise = np.random.normal(0, self.explore_noise_range, size=self.n_actions)
            actions = np.clip(actions + noise, -1.0, 1.0)
        
        return actions
    
    def smooth_target_action(self, next_state):
        """
        [训练阶段] 生成带平滑噪声的目标动作 (Target Policy Smoothing)
        用于计算 Target Q 值：y = r + γ * min(Q1(s', ã), Q2(s', ã))
        输出类型：Tensor (保留在计算图中)
        输出范围：[-1, 1]
        """
        if not isinstance(next_state, T.Tensor):
            state = T.tensor(next_state, dtype=T.float32, device=self.device)
        else:
            state = next_state.to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        with T.no_grad():
            actions = self.target_actor.forward(state)
            # 生成截断高斯噪声
            noise = T.normal(0, self.smooth_noise_range, size=actions.shape, device=self.device)
            noise = T.clamp(noise, -self.smooth_noise_clip, self.smooth_noise_clip)
            smoothed_actions = actions + noise
            smoothed_actions = T.clamp(smoothed_actions, -1.0, 1.0)
            
        return smoothed_actions
    
    def update(self, batch_size=NetworkConfig.batch_size):
        if not self.memory.ready(batch_size):
            return
        # 每调用一次update方法，update_cnt加1，用于控制Actor的更新频率
        self.update_cnt += 1

        # 从经验回放中采样一个batch的数据，并将其转换为tensor
        states, actions, rewards, next_states, dones = self.memory.sample_buffer(batch_size)
        states = T.tensor(states, dtype=T.float32).to(self.device)
        actions = T.tensor(actions, dtype=T.float32).to(self.device)    
        rewards = T.tensor(rewards, dtype=T.float32).to(self.device).unsqueeze(1)
        next_states = T.tensor(next_states, dtype=T.float32).to(self.device)
        dones = T.tensor(dones, dtype=T.float32).to(self.device).unsqueeze(1)
        # 计算带平滑噪声的target actions
        smoothed_next_actions = self.smooth_target_action(next_states)
        # 计算 target Q value
        with T.no_grad():
            target_q_min = T.min(self.target_critic_1(next_states, smoothed_next_actions), self.target_critic_2(next_states, smoothed_next_actions))
            target_q_value = rewards + (1 - dones) * self.gamma * target_q_min

        self.critic_1.optimizer.zero_grad()
        predicted_q_value_1 = self.critic_1(states, actions)
        q_value_loss_1 = F.mse_loss(predicted_q_value_1, target_q_value)
        q_value_loss_1.backward()
        self.critic_1.optimizer.step()
        # self.critic_1.scheduler.step()

        self.critic_2.optimizer.zero_grad()
        predicted_q_value_2 = self.critic_2(states, actions)
        q_value_loss_2 = F.mse_loss(predicted_q_value_2, target_q_value)
        q_value_loss_2.backward()
        self.critic_2.optimizer.step()
        # self.critic_2.scheduler.step()

        # delayed update of actor and target networks
        if self.update_cnt % self.actor_update_interval == 0:
            self.actor.optimizer.zero_grad()
            actor_loss = -self.critic_1(states, self.actor(states)).mean() # Actor loss: maximize Q value
            actor_loss.backward()
            self.actor.optimizer.step()
            # self.actor.scheduler.step()
            # soft update target networks
            self.soft_update_network_parameters()


    # def _build_weight_paths(self, episode=None):
    #     os.makedirs(self.chkpt_dir, exist_ok=True)
    #     if episode is not None:
    #         suffix = f"_episode{episode}.pth"
    #     else:
    #         suffix = ".pth"
    #     return {
    #         'critic1': os.path.join(self.chkpt_dir, f"td3_critic_1_net{suffix}"),
    #         'critic2': os.path.join(self.chkpt_dir, f"td3_critic_2_net{suffix}"),
    #         'actor': os.path.join(self.chkpt_dir, f"td3_actor_net{suffix}"),
    #         'target_actor': os.path.join(self.chkpt_dir, f"td3_target_actor_net{suffix}"),
    #         'target_critic1': os.path.join(self.chkpt_dir, f"td3_target_critic_1_net{suffix}"),
    #         'target_critic2': os.path.join(self.chkpt_dir, f"td3_target_critic_2_net{suffix}"),
    #     }

    # def save_weights(self, episode=None, include_target=True):
    #     paths = self._build_weight_paths(episode)
    #     T.save(self.actor.state_dict(), paths['policy'])
    #     T.save(self.critic_1.state_dict(), paths['critic1'])
    #     T.save(self.critic_2.state_dict(), paths['critic2'])

    #     if include_target:
    #         T.save(self.target_actor.state_dict(), paths['target_actor'])
    #         T.save(self.target_critic_1.state_dict(), paths['target_critic1'])
    #         T.save(self.target_critic_2.state_dict(), paths['target_critic2'])

    #     print(f"[TD3] 模型已保存到：{self.chkpt_dir} (episode={episode})")

    # def load_weights(self, episode=None, strict=True, include_target=True):
    #     paths = self._build_weight_paths(episode)
    #     policy_path, critic1_path, critic2_path = paths['policy'], paths['critic1'], paths['critic2']

    #     if not (os.path.exists(policy_path) and os.path.exists(critic1_path) and os.path.exists(critic2_path)):
    #         raise FileNotFoundError(f"无法找到权重文件：{policy_path}, {critic1_path}, {critic2_path}")

    #     self.actor.load_state_dict(T.load(policy_path, map_location=self.actor.device), strict=strict)
    #     self.critic_1.load_state_dict(T.load(critic1_path, map_location=self.critic_1.device), strict=strict)
    #     self.critic_2.load_state_dict(T.load(critic2_path, map_location=self.critic_2.device), strict=strict)

    #     if include_target:
    #         if os.path.exists(paths['target_actor']):
    #             self.target_actor.load_state_dict(T.load(paths['target_actor'], map_location=self.target_actor.device), strict=strict)
    #         else:
    #             self.target_actor.load_state_dict(self.actor.state_dict(), strict=strict)

    #         if os.path.exists(paths['target_critic1']):
    #             self.target_critic_1.load_state_dict(T.load(paths['target_critic1'], map_location=self.target_critic_1.device), strict=strict)
    #         else:
    #             self.target_critic_1.load_state_dict(self.critic_1.state_dict(), strict=strict)

    #         if os.path.exists(paths['target_critic2']):
    #             self.target_critic_2.load_state_dict(T.load(paths['target_critic2'], map_location=self.target_critic_2.device), strict=strict)
    #         else:
    #             self.target_critic_2.load_state_dict(self.critic_2.state_dict(), strict=strict)

    #     print(f"[TD3] 模型加载成功：episode={episode}")

    # # 兼容旧接口
    # def save_checkpoint(self, episode=None):
    #     self.save_weights(episode=episode, include_target=True)

    # def load_checkpoint(self, episode=None):
    #     self.load_weights(episode=episode, include_target=True)

    # # 依据指定路径加载已保存的模型参数（兼容 main_ai.py 自定义 .pth 文件）
    # def load_weights_from_files(self, policy_path, critic1_path, critic2_path):
    #     if not os.path.exists(policy_path) or not os.path.exists(critic1_path) or not os.path.exists(critic2_path):
    #         raise FileNotFoundError('部分模型文件不存在，请确认路径正确')

    #     self.actor.load_state_dict(T.load(policy_path, map_location=self.actor.device))
    #     self.critic_1.load_state_dict(T.load(critic1_path, map_location=self.critic_1.device))
    #     self.critic_2.load_state_dict(T.load(critic2_path, map_location=self.critic_2.device))

    #     self.target_actor.load_state_dict(self.actor.state_dict())
    #     self.target_critic_1.load_state_dict(self.critic_1.state_dict())
    #     self.target_critic_2.load_state_dict(self.critic_2.state_dict())
    #     print(f'加载权重成功：{policy_path}, {critic1_path}, {critic2_path}')


    def _build_weight_paths(self, episode=None):
        os.makedirs(self.chkpt_dir, exist_ok=True)
        if episode is not None:
            suffix = f"_episode{episode}.pth"
        else:
            suffix = ".pth"
        return {
            'actor': os.path.join(self.chkpt_dir, f"td3_actor_net{suffix}"),
            'critic1': os.path.join(self.chkpt_dir, f"td3_critic_1_net{suffix}"),
            'critic2': os.path.join(self.chkpt_dir, f"td3_critic_2_net{suffix}"),
            'target_actor': os.path.join(self.chkpt_dir, f"td3_target_actor_net{suffix}"),
            'target_critic1': os.path.join(self.chkpt_dir, f"td3_target_critic_1_net{suffix}"),
            'target_critic2': os.path.join(self.chkpt_dir, f"td3_target_critic_2_net{suffix}"),
        }
    def save_weights(self, episode=None, include_target=True):
        paths = self._build_weight_paths(episode)
        # 保存主网络
        T.save(self.actor.state_dict(), paths['actor'])
        T.save(self.critic_1.state_dict(), paths['critic1'])
        T.save(self.critic_2.state_dict(), paths['critic2'])

        # 可选保存目标网络
        if include_target:
            T.save(self.target_actor.state_dict(), paths['target_actor'])
            T.save(self.target_critic_1.state_dict(), paths['target_critic1'])
            T.save(self.target_critic_2.state_dict(), paths['target_critic2'])

        print(f"[TD3] model saved successfully to: {self.chkpt_dir} (episode={episode})")

    def load_weights(self, episode=None, strict=True, include_target=True):
        paths = self._build_weight_paths(episode)
        actor_path, critic1_path, critic2_path = paths['actor'], paths['critic1'], paths['critic2']

        # 校验主网络文件是否存在
        if not (os.path.exists(actor_path) and os.path.exists(critic1_path) and os.path.exists(critic2_path)):
            raise FileNotFoundError(f"can not find weight files: {actor_path}, {critic1_path}, {critic2_path}")

        # 加载主网络
        self.actor.load_state_dict(T.load(actor_path, map_location=self.actor.device), strict=strict)
        self.critic_1.load_state_dict(T.load(critic1_path, map_location=self.critic_1.device), strict=strict)
        self.critic_2.load_state_dict(T.load(critic2_path, map_location=self.critic_2.device), strict=strict)

        # 加载目标网络（可选）
        if include_target:
            # 目标网络文件不存在时，从主网络同步
            if os.path.exists(paths['target_actor']):
                self.target_actor.load_state_dict(T.load(paths['target_actor'], map_location=self.target_actor.device), strict=strict)
            else:
                self.target_actor.load_state_dict(self.actor.state_dict(), strict=strict)

            if os.path.exists(paths['target_critic1']):
                self.target_critic_1.load_state_dict(T.load(paths['target_critic1'], map_location=self.target_critic_1.device), strict=strict)
            else:
                self.target_critic_1.load_state_dict(self.critic_1.state_dict(), strict=strict)

            if os.path.exists(paths['target_critic2']):
                self.target_critic_2.load_state_dict(T.load(paths['target_critic2'], map_location=self.target_critic_2.device), strict=strict)
            else:
                self.target_critic_2.load_state_dict(self.critic_2.state_dict(), strict=strict)

        print(f"[TD3] model loaded successfully: episode={episode}")
