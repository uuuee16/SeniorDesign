"""
main.py  ── 统一训练 / 评估入口
=============================================
通过修改 ALGORITHMS_TO_RUN 可以控制：
    - 单算法运行，例如 ["TD3-PER"]
    - 多算法顺序运行，例如 ["TD3-PER", "TD3", "DDPG"]

无论单算法还是多算法，在全部算法完成后都会自动生成：
    - 1 张总 reward 折线图
    - 1 张总 success rate 折线图
    - 1 张总 energy boxplot
    - 每个 episode 的联合轨迹图（3D / XOY / XOZ 同图）
    - 每个 episode 的单独 3D 联合轨迹图
"""
from __future__ import annotations

import random

import numpy as np
import torch

from config import FileAddress, MapConfig, NetworkConfig
from env import AUVEnv
from ocean_current import build_current
from plot_comparison import generate_experiment_reports
from runner import evaluate_agent, train_agent
from algos import TD3, DDPG
from visualization import visualize_current_environment


# ALGORITHMS_TO_RUN = ["TD3-PER", "TD3", "DDPG"]
ALGORITHMS_TO_RUN = ["TD3-PER", "TD3"]


def set_global_seed(seed: int):
    """锁死底层随机种子，保证实验可复现与公平性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[*] Global Seed fixed to: {seed}")


def build_agent(env, algo_name: str):
    if algo_name == "TD3-PER":
        return TD3(state_dims=env.obs_dim, n_actions=NetworkConfig.action_dim, use_PER=True)
    elif algo_name == "TD3":
        return TD3(state_dims=env.obs_dim, n_actions=NetworkConfig.action_dim, use_PER=False)
    elif algo_name == "DDPG":
        return DDPG(state_dims=env.obs_dim, n_actions=NetworkConfig.action_dim, use_PER=False)


def _resolve_eval_episode(agent):
    if MapConfig.eval_model_episode is not None:
        return MapConfig.eval_model_episode
    return agent.best_saved_episode()


def _build_env(scene=None):
    current = None
    if MapConfig.current_mode == "real":
        current = build_current(getattr(MapConfig, "center_name", None))
    return AUVEnv(obstacle_name=scene, current=current)

# ── 可视化当前环境并保存快照 ───────────────────────────────────────────────
def save_current_snapshot(algorithms: list[str], scene=None):
    env = _build_env(scene=scene)
    if len(algorithms) == 1:
        FileAddress.update_algo(algorithms[0])
        FileAddress.make_dirs()
        save_path = FileAddress.current_png(f"{MapConfig.current_mode}_{algorithms[0]}")
    else:
        save_path = FileAddress.comparison_png(MapConfig.mode, algorithms, f"current_environment_{MapConfig.current_mode}")

    visualize_current_environment(env, save_path=str(save_path))
    print(f"Current figure saved: {save_path}")
    return save_path


def run_single_algorithm(algo_name: str, mode: str, scene=None):
    set_global_seed(MapConfig.seed)

    FileAddress.update_algo(algo_name)
    FileAddress.make_dirs()

    env = _build_env(scene=scene)
    agent = build_agent(env, algo_name)

    if mode == "train":
        train_agent(
            env,
            agent,
            total_episodes=MapConfig.train_eps,
            max_episode_steps=MapConfig.max_steps,
        )
        return

    load_episode = _resolve_eval_episode(agent)
    agent.load_weights(episode=load_episode, include_target=False)
    evaluate_agent(
        env,
        agent,
        eval_episodes=MapConfig.eval_eps,
        max_episode_steps=MapConfig.max_steps,
    )


def main():
    algorithms = list(ALGORITHMS_TO_RUN)
    mode = str(MapConfig.mode)
    scene = getattr(MapConfig, "scene", None)

    if MapConfig.show_current:
        save_current_snapshot(algorithms, scene=scene)
        return

    for algo in algorithms:
        print(f"\n{'=' * 72}")
        print(f"🚀 STARTING [{mode.upper()}] FOR ALGORITHM: {algo}")
        print(f"{'=' * 72}\n")

        run_single_algorithm(algo, mode=mode, scene=scene)

        print(f"\n{'=' * 72}")
        print(f"✅ FINISHED [{mode.upper()}] FOR ALGORITHM: {algo}")
        print(f"{'=' * 72}\n")

    outputs = generate_experiment_reports(algorithms, mode=mode, scene=scene)
    if outputs:
        print(f"\n[{mode.upper()}] Generated experiment reports:")
        for output in outputs:
            print(f"  - {output}")


if __name__ == "__main__":
    main()
