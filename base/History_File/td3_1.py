import os
import torch as T
import torch.nn.functional as F
import numpy as np
from networks import ActorNetwork, CriticNetwork
from config import NetworkConfig, FileAddress

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
    def __init__(self, state_dims, n_actions, scenario='simple', max_size=10000, alpha=0.0001, beta=0.0002, fc1=128, fc2=128, gamma=0.99, tau=0.002,
                actor_update_interval=2, smooth_noise_clip=0.5, smooth_noise_range=0.2, explore_noise_range=0.1, chkpt_dir='tmp/td3/'):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.actor_update_interval = actor_update_interval # Actor 更新间隔，Critic 更新频率为1
        self.smooth_noise_clip = smooth_noise_clip
        self.smooth_noise_range = smooth_noise_range
        self.explore_noise_range = explore_noise_range
        self.update_cnt = 0
        self.chkpt_dir = os.path.join(chkpt_dir, scenario)

        self.actor = ActorNetwork(alpha, state_dims, n_actions, fc1, fc2, self.chkpt_dir, 'actor')
        self.target_actor = ActorNetwork(alpha, state_dims, n_actions, fc1, fc2, self.chkpt_dir, 'target_actor')

        self.critic_1 = CriticNetwork(beta, state_dims, n_actions, fc1, fc2, self.chkpt_dir, 'critic_1')
        self.critic_2 = CriticNetwork(beta, state_dims, n_actions, fc1, fc2, self.chkpt_dir, 'critic_2')
        self.target_critic_1 = CriticNetwork(beta, state_dims, n_actions, fc1, fc2, self.chkpt_dir, 'target_critic_1')
        self.target_critic_2 = CriticNetwork(beta, state_dims, n_actions, fc1, fc2, self.chkpt_dir, 'target_critic_2')

        # init target networks parameters = original networks parameters
        self.soft_update_network_parameters(tau=1)
        # init replay buffer input_dims = state维度 
        self.memory = ReplayBuffer(max_size, input_shape=(state_dims,), n_actions=n_actions)
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

        # # 更新target_actor网络参数
        # target_actor_params = self.target_actor.named_parameters()
        # actor_params = self.actor.named_parameters()
        # target_actor_dict = dict(target_actor_params)
        # actor_dict = dict(actor_params)
        # for name in target_actor_dict:
        #     target_actor_dict[name] = (1-tau)*target_actor_dict[name].clone() + tau*actor_dict[name].clone()
        # self.target_actor.load_state_dict(target_actor_dict)

        # # 更新target_critic_1网络参数
        # target_critic_1_params = self.target_critic_1.named_parameters()
        # critic_1_params = self.critic_1.named_parameters()
        # target_critic_1_dict = dict(target_critic_1_params)
        # critic_1_dict = dict(critic_1_params)
        # for name in target_critic_1_dict:
        #     target_critic_1_dict[name] = (1-tau)*target_critic_1_dict[name].clone() + tau*critic_1_dict[name].clone()
        # self.target_critic_1.load_state_dict(target_critic_1_dict)

        # # 更新target_critic_2网络参数
        # target_critic_2_params = self.target_critic_2.named_parameters()
        # critic_2_params = self.critic_2.named_parameters()
        # target_critic_2_dict = dict(target_critic_2_params)
        # critic_2_dict = dict(critic_2_params)
        # for name in target_critic_2_dict:
        #     target_critic_2_dict[name] = (1-tau)*target_critic_2_dict[name].clone() + tau*critic_2_dict[name].clone()
        # self.target_critic_2.load_state_dict(target_critic_2_dict)

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
    
    def update(self, batch_size):
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
        self.critic_1.scheduler.step()

        self.critic_2.optimizer.zero_grad()
        predicted_q_value_2 = self.critic_2(states, actions)
        q_value_loss_2 = F.mse_loss(predicted_q_value_2, target_q_value)
        q_value_loss_2.backward()
        self.critic_2.optimizer.step()
        self.critic_2.scheduler.step()

        # delayed update of actor and target networks
        if self.update_cnt % self.actor_update_interval == 0:
            self.actor.optimizer.zero_grad()
            actor_loss = -self.critic_1(states, self.actor(states)).mean() # Actor loss: maximize Q value
            actor_loss.backward()
            self.actor.optimizer.step()
            self.actor.scheduler.step()
            # soft update target networks
            self.soft_update_network_parameters()

    def save_models(self):
        print('... saving checkpoint ...')
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        print('... loading checkpoint ...')
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()