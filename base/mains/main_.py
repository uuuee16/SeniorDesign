import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os

from config import FileAddress, MapConfig, NetworkConfig
from td3 import TD3, ReplayBuffer
from env import AUVEnv


if __name__ == '__main__':
    # 参数定义
    total_episodes = MapConfig.train_episodes
    evaluate = False

    env = AUVEnv()
    # buffer 已经在TD3中定义了
    agent = TD3()

    if evaluate:
            agent.load_checkpoint()
            print('----  evaluating  ----')
    else:
        print('----training start----')

    for i in range(total_episodes):
        obs, info = env.reset() # 初始化时进行了归一化
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action) # step中进行了obs的归一化
            done = terminated or truncated
            agent.memory.store_transition(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_reward += reward
            
            if not evaluate:
                agent.update()

        # 可视化，记录轨迹，奖励，速度，yaw，pitch，是否成功，记录为2000个csv文件
    
        print(f"Episode {i+1}/{total_episodes}, Reward: {episode_reward:.2f}")
    # 总的奖励函数曲线， 1图 1文件
