"""
main_ai.py
----------
TD3 AUV 训练/评估入口。

用法示例
--------
# 从头训练
python main_ai.py

# 加载 checkpoint 继续训练
python main_ai.py --load --load_episode 200

# 评估已训练模型
python main_ai.py --evaluate --load --load_episode 500

# 评估最新模型，自定义 episode 数
python main_ai.py --evaluate --load --eval_episodes 100
"""

import os
import argparse
import torch

from config import FileAddress, MapConfig, NetworkConfig
from td3 import TD3
from env import AUVEnv
from runner import train_agent, evaluate_agent


# ─────────────────────────────────────────────
# 初始化
# ─────────────────────────────────────────────

def _load_weights_legacy(agent, load_episode):
    """兼容旧版路径的模型加载（新版 load_weights 失败时回退）。"""
    folder = FileAddress.td3_network_folder

    def _path(name, ep):
        return (os.path.join(folder, f"{name}_episode{ep}.pth")
                if ep is not None else getattr(FileAddress, f"td3_{name}_path"))

    paths = {
        'actor':    _path('policy_net',    load_episode),
        'critic_1': _path('critic_1_net',  load_episode),
        'critic_2': _path('critic_2_net',  load_episode),
    }

    if all(os.path.exists(p) for p in paths.values()):
        agent.actor.load_state_dict(   torch.load(paths['actor'],    map_location=agent.device))
        agent.critic_1.load_state_dict(torch.load(paths['critic_1'], map_location=agent.device))
        agent.critic_2.load_state_dict(torch.load(paths['critic_2'], map_location=agent.device))
        # 同步 target 网络
        agent.target_actor.load_state_dict(agent.actor.state_dict())
        agent.target_critic_1.load_state_dict(agent.critic_1.state_dict())
        agent.target_critic_2.load_state_dict(agent.critic_2.state_dict())
        print(f"[加载模型] 兼容旧路径成功：{list(paths.values())}")
    else:
        print(f"[加载模型] 未找到有效模型文件：{list(paths.values())}")


def init_agent_and_env(load_model: bool, load_episode):
    """初始化环境、智能体，并按需加载权重。"""
    env   = AUVEnv()
    agent = TD3(
        state_dims=NetworkConfig.nn_input_dim,
        n_actions=NetworkConfig.nn_output_dim,
    )

    if load_model:
        try:
            agent.load_weights(episode=load_episode, include_target=True)
            print(f"[加载模型] Checkpoint 加载成功 (episode={load_episode})")
        except FileNotFoundError:
            _load_weights_legacy(agent, load_episode)
        except Exception as e:
            print(f"[加载模型] 错误: {e}，训练将从头开始。")

    # 确保输出目录存在
    for folder in [
        FileAddress.auv_trajectory_folder_path,
        FileAddress.auv_plots_folder_path,
        FileAddress.auv_rewards_steps_results,
        FileAddress.td3_network_folder,
    ]:
        os.makedirs(folder, exist_ok=True)

    return env, agent


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='TD3 AUV — Training / Evaluation')

    parser.add_argument('--evaluate',      action='store_true',
                        help='评估模式（不更新网络），默认为训练模式')
    parser.add_argument('--load',          action='store_true',
                        help='运行前加载模型权重')
    parser.add_argument('--load_episode',  type=int, default=None,
                        help='加载指定 episode 的 checkpoint（默认加载最新）')
    parser.add_argument('--eval_episodes', type=int, default=50,
                        help='评估 episode 数（默认 50）')
    parser.add_argument('--train_episodes', type=int, default=MapConfig.train_episodes,
                        help='训练 episode 数（默认读取 MapConfig）')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # 评估模式必须加载预训练模型
    if args.evaluate and not args.load:
        print("⚠️  评估模式需要加载预训练模型，请添加 --load 参数。")
        raise SystemExit(1)

    env, agent = init_agent_and_env(args.load, args.load_episode)

    if args.evaluate:
        evaluate_agent(env, agent, args.eval_episodes, MapConfig.max_episode_steps)
    else:
        train_agent(env, agent, args.train_episodes, MapConfig.max_episode_steps)
