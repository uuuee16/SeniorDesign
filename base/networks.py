import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims, fc2_dims):
        super(CriticNetwork, self).__init__()
        self.full_input_dims = input_dims + n_actions

        self.fc1 = nn.Linear(self.full_input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.33)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q_value = self.q(x)  # 直接输出，无激活
        return q_value


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, n_actions, fc1_dims, fc2_dims):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.8)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        前向传播：输入状态，输出归一化动作
        :param state: 形状 [batch_size, state_dims]
        :return: 动作，形状 [batch_size, n_actions]，范围[-1,1]
        """
        x = F.leaky_relu(self.fc1(state))  # 直接激活，无归一化
        x = F.leaky_relu(self.fc2(x))
        # TD3原论文指定tanh，替代Softsign（核心修正保留）
        actions = T.tanh(self.pi(x))  # 输出严格限制在[-1,1]
        # actions = F.softsign(self.pi(x))  # 输出严格限制在[-1,1]
        return actions



