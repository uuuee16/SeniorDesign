"""
main.py  ── 训练 / 评估入口
=============================================
三种海流模式由 MapConfig.current_mode 控制：
    "none"       无海流
    "synthetic"  合成海流（OceanCurrent 自动构建）
    "real"       真实 CMEMS 海流（build_current() 构建后注入）
"""
from __future__ import annotations

from pathlib import Path

from config import FileAddress, MapConfig, NetworkConfig
from env import AUVEnv
from ocean_current import build_current
from runner import evaluate_agent, train_agent
from td3 import TD3
from visualization import visualize_current_environment


def build_env(scene=None) -> AUVEnv:
    current = None
    if MapConfig.current_mode == "real":
        current = build_current()
    return AUVEnv(obstacle_name=scene, current=current)


def build_agent(env):
    return TD3(state_dims=env.obs_dim, n_actions=NetworkConfig.action_dim)


def _resolve_eval_episode(agent: TD3):
    # 1. 如果在 config.py 中明确指定了加载哪个 episode，则以指定的为准
    if MapConfig.eval_model_episode is not None:
        return MapConfig.eval_model_episode

    # 2. 核心修改：优先寻找表现最好的已保存 episode
    best_ep = agent.best_saved_episode()
    if best_ep is not None:
        return best_ep

    # 3. 如果找不到带编号的最好权重，退而求其次寻找训练结束时保存的最终权重(.pth)
    default_paths = agent._build_weight_paths(None)
    if all(Path(default_paths[k]).exists() for k in ["actor", "critic1", "critic2"]):
        print("[TD3] No numbered checkpoints found. Using final saved weights.")
        return None

    raise FileNotFoundError(
        "No trained weights were found. Please run training first or set MapConfig.eval_model_episode explicitly."
    )


def main(mode=MapConfig.mode, scene=None):
    FileAddress.make_dirs()
    env = build_env(scene=scene)

    if MapConfig.show_current:
        if not env.current_on or env.current is None:
            raise ValueError("show_current=True requires current_mode != 'none'.")
        tag = MapConfig.current_mode
        save_path = FileAddress.current_png(tag)
        visualize_current_environment(env, save_path=str(save_path))
        print(f"Current figure saved: {save_path}")
        return

    agent = build_agent(env)
    if mode == "train":
        train_agent(
            env,
            agent,
            total_episodes=MapConfig.train_eps,
            max_episode_steps=MapConfig.max_steps,
        )
    elif mode == "eval":
        load_episode = _resolve_eval_episode(agent)
        agent.load_weights(episode=load_episode, include_target=False)
        evaluate_agent(
            env,
            agent,
            eval_episodes=MapConfig.eval_eps,
            max_episode_steps=MapConfig.max_steps,
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose 'train' or 'eval'.")


if __name__ == "__main__":
    main()
