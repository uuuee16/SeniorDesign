import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time

from config import FileAddress, MapConfig, NetworkConfig

# ─────────────────────────────────────────────
# 1. 单 Episode 仿真
# ─────────────────────────────────────────────

def run_episode(env, agent, mode: str, max_steps: int):
    """
    执行一个完整 episode，返回所有采集数据。
    """
    is_train = (mode == 'train')

    obs, info = env.reset()
    start_pos = env.auv_state.position.copy()
    prev_pos  = start_pos.copy()

    done            = False
    step_count      = 0
    ep_reward       = 0.0
    ep_path_length  = 0.0
    trajectory_data = []
    step_data       = []
    boundary_collision_points = []

    while not done and step_count < max_steps:
        # 动作选择：训练模式加噪声探索，评估模式无探索
        action = agent.select_action(obs, evaluate=(not is_train))
        next_obs, reward, terminated, truncated, info = env.step(action)

        if step_count + 1 >= max_steps:
            truncated = True
            info['truncated_reason'] = 'max_episode_steps_reached'

        done = terminated or truncated

        # AUV 当前状态
        current_state = env.auv_state
        x, y, z       = current_state.position
        yaw, pitch    = current_state.attitude

        # 边界碰撞检测
        hit_boundary = bool(
            x <= 0 or x >= MapConfig.env_limit or
            y <= 0 or y >= MapConfig.env_limit or
            z <= 0 or z >= MapConfig.env_limit
        )
        if hit_boundary:
            boundary_collision_points.append([x, y, z])

        # 动作解析（[-1,1] → 物理量）
        actual_speed      = (action[0] + 1.0) / 2.0 * MapConfig.max_body_vel
        actual_yaw_rate   = action[1] * MapConfig.max_yaw_vel
        actual_pitch_rate = action[2] * MapConfig.max_pitch_vel

        # 路径长度累积
        current_pos     = current_state.position
        ep_path_length += np.linalg.norm(current_pos - prev_pos)
        prev_pos        = current_pos.copy()

        # 单步记录
        step_data.append({
            'step': step_count,
            'x': x, 'y': y, 'z': z,
            'yaw': yaw, 'pitch': pitch,
            'actual_speed':      actual_speed,
            'actual_yaw_rate':   actual_yaw_rate,
            'actual_pitch_rate': actual_pitch_rate,
            'action_raw_0': action[0],
            'action_raw_1': action[1],
            'action_raw_2': action[2],
            'reward':               reward,
            'dist2target':          info['dist2target'],
            'nearest_obstacle_dist': info['nearest_obstacle_dist'],
            'step_terminated': done,
            'hit_boundary':    hit_boundary,
        })
        trajectory_data.append([x, y, z])

        # 仅训练模式存入经验池并更新网络
        if is_train:
            agent.memory.store_transition(obs, action, reward, next_obs, done)
            agent.update()

        obs        = next_obs
        ep_reward += reward
        step_count += 1

    # Episode 收尾
    ep_time      = step_count * MapConfig.dt
    success_flag = int(
        info.get('reach_target', False) or
        (info['dist2target'] < MapConfig.goal_threshold)
    )
    for record in step_data:
        record['episode_success'] = success_flag

    return {
        'trajectory_data':          trajectory_data,
        'step_data':                step_data,
        'boundary_collision_points': boundary_collision_points,
        'start_pos':        start_pos,
        'ep_reward':        ep_reward,
        'step_count':       step_count,
        'ep_time':          ep_time,
        'ep_path_length':   ep_path_length,
        'success_flag':     success_flag,
        'info':             info,
    }


# ─────────────────────────────────────────────
# 2. 3D 轨迹可视化
# ─────────────────────────────────────────────

# 模式专属配置
_PLOT_STYLE = {
    'train': dict(traj_color='r',    scatter_color='crimson', bc_color='darkviolet', label_suffix=''),
    'eval':  dict(traj_color='blue', scatter_color='blue',    bc_color='red',        label_suffix=' (Eval)'),
}

def plot_trajectory(env, ep_result: dict, ep_idx: int, mode: str, save_path: str) -> None:
    style        = _PLOT_STYLE[mode]
    traj_array   = np.array(ep_result['trajectory_data'])
    start_pos    = ep_result['start_pos']
    ep_reward    = ep_result['ep_reward']
    ep_time      = ep_result['ep_time']
    ep_path_length = ep_result['ep_path_length']
    success_flag = ep_result['success_flag']
    bc_points    = ep_result['boundary_collision_points']

    fig = plt.figure(figsize=(14, 11), facecolor='white')
    ax  = fig.add_subplot(111, projection='3d')

    # 边界框
    for i in [0, MapConfig.env_limit]:
        for j in [0, MapConfig.env_limit]:
            kw = dict(linewidth=0.8, alpha=0.3)
            ax.plot([i, i], [j, j], [0, MapConfig.env_limit], 'k-', **kw)
            ax.plot([i, i], [0, MapConfig.env_limit], [j, j], 'k-', **kw)
            ax.plot([0, MapConfig.env_limit], [i, i], [j, j], 'k-', **kw)

    # 轨迹
    if len(traj_array) > 1:
        ax.plot(traj_array[:, 0], traj_array[:, 1], traj_array[:, 2],
                color=style['traj_color'], linewidth=2.5,
                label=f'AUV Trajectory{style["label_suffix"]}', alpha=0.8, zorder=3)
        step_sample = max(1, len(traj_array) // 50)
        ax.scatter(traj_array[::step_sample, 0],
                   traj_array[::step_sample, 1],
                   traj_array[::step_sample, 2],
                   c=style['scatter_color'], s=15, alpha=0.6, zorder=3)

    # 边界碰撞点
    if bc_points:
        bc_array = np.array(bc_points)
        ax.scatter(bc_array[:, 0], bc_array[:, 1], bc_array[:, 2],
                   c=style['bc_color'], s=250, marker='X',
                   label='Boundary Collision', zorder=10,
                   edgecolors='white', linewidths=2.5)

    # 起点 / 目标点
    ax.scatter(*start_pos, c='black', s=180, marker='o',
               label='Start Point', zorder=5, edgecolors='gold', linewidths=2)
    ax.scatter(*env.target_pos, c='limegreen', s=220, marker='*',
               label='Target Point', zorder=5, edgecolors='yellow', linewidths=2)

    # 障碍物球体
    obstacle_colors = ['dodgerblue', 'deepskyblue', 'steelblue', 'cornflowerblue', 'royalblue']
    for idx, (ox, oy, oz, r) in enumerate(env.obstacles):
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        ax.plot_surface(
            ox + r * np.cos(u) * np.sin(v),
            oy + r * np.sin(u) * np.sin(v),
            oz + r * np.cos(v),
            color=obstacle_colors[idx % len(obstacle_colors)],
            alpha=0.35, edgecolor='navy', linewidth=0.3, antialiased=True,
        )

    # 坐标轴样式
    ax.set_xlabel('North (m)', fontsize=13, fontweight='bold', color='darkblue',  labelpad=10)
    ax.set_ylabel('East (m)',  fontsize=13, fontweight='bold', color='darkgreen', labelpad=10)
    ax.set_zlabel('Down (m)',  fontsize=13, fontweight='bold', color='darkred',   labelpad=10)
    for setter, ticks in zip([ax.set_xticks, ax.set_yticks, ax.set_zticks],
                              [[0,MapConfig.env_limit*1.0/4.0,MapConfig.env_limit/2.0,MapConfig.env_limit*3.0/4.0,MapConfig.env_limit]]*3):
        setter(ticks)
    ax.tick_params(labelsize=10)
    ax.set_xlim(0, MapConfig.env_limit); ax.set_ylim(0, MapConfig.env_limit); ax.set_zlim(0, MapConfig.env_limit)
    ax.set_box_aspect([1, 1, 1])
    ax.invert_zaxis()

    # 标题
    tag = mode.upper()
    ax.set_title(
        f"[{tag}] Episode {ep_idx} Trajectory Visualization\n"
        f"Reward: {ep_reward:.2f}  |  Time: {ep_time:.2f}s  |  "
        f"Path: {ep_path_length:.2f}m  |  Success: {'✓' if success_flag else '✗'}",
        fontsize=13, fontweight='bold', pad=25, color='darkslategray',
    )

    ax.view_init(elev=28, azim=-50, roll=0)
    legend = ax.legend(loc='upper left', fontsize=11,
                       framealpha=0.9, edgecolor='gray',
                       facecolor='white', fancybox=True)
    legend.get_frame().set_linewidth(1.2)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4, color='gray')
    ax.set_facecolor('aliceblue')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)


# ─────────────────────────────────────────────
# 3. 单 Episode 数据持久化
# ─────────────────────────────────────────────

def save_episode_data(ep_result: dict, ep_idx: int, mode: str) -> None:
    """保存单 episode 的步骤数据 CSV 和轨迹 CSV。"""
    prefix = mode  # 'train' | 'eval'

    pd.DataFrame(ep_result['step_data']).to_csv(
        os.path.join(FileAddress.auv_rewards_steps_results,
                     f"{prefix}_step_data_episode{ep_idx}.csv"),
        index=False,
    )
    pd.DataFrame(ep_result['trajectory_data'], columns=['x', 'y', 'z']).to_csv(
        os.path.join(FileAddress.auv_trajectory_folder_path,
                     f"{prefix}_trajectory_episode{ep_idx}.csv"),
        index=False,
    )


# ─────────────────────────────────────────────
# 4. 全局统计汇总
# ─────────────────────────────────────────────

def save_summary(records: dict, mode: str, total_episodes: int,
                 elapsed: float = 0.0, device: str = 'unknown') -> None:
    """
    保存全局汇总 CSV、统计 CSV、奖励曲线，并打印终端统计。

    Parameters
    ----------
    records        : 各列表 {'rewards', 'lengths', 'times', 'path_lengths', 'success_flags'}
    mode           : 'train' | 'eval'
    total_episodes : episode 总数
    elapsed        : 总耗时
    device         : 使用的设备
    """
    tag    = mode  # 用于文件名前缀
    ep_ids = np.arange(1, total_episodes + 1)

    # ── 汇总 CSV ──
    summary_df = pd.DataFrame({
        'episode':       ep_ids,
        'total_reward':  records['rewards'],
        'steps':         records['lengths'],
        'time_cost_s':   records['times'],
        'path_length_m': records['path_lengths'],
        'success':       records['success_flags'],
    })
    summary_df.to_csv(
        os.path.join(FileAddress.auv_reward_plot_and_infos_path,
                     f"{tag}_summary.csv"),
        index=False,
    )

    # ── 统计 CSV ──
    success_eps = summary_df[summary_df['success'] == 1]
    success_rate = len(success_eps) / total_episodes * 100
    if len(success_eps) > 0:
        shortest_path = success_eps['path_length_m'].min()
        shortest_time = success_eps['time_cost_s'].min()
        average_path  = success_eps['path_length_m'].mean()
        average_time  = success_eps['time_cost_s'].mean()
    else:
        shortest_path = shortest_time = average_path = average_time = np.nan

    stats = {
        f'{"eval" if mode=="eval" else "total"}_episodes': total_episodes,
        'success_count':       len(success_eps),
        'success_rate_percent': success_rate,
        'shortest_path_m':     shortest_path,
        'shortest_time_s':     shortest_time,
        'average_path_m':      average_path,
        'average_time_s':      average_time,
        'max_reward':          np.max(records['rewards']),
        'mean_reward':         np.mean(records['rewards']),
    }
    pd.DataFrame([stats]).to_csv(
        os.path.join(FileAddress.auv_reward_plot_and_infos_path,
                     f"{tag}_statistics.csv"),
        index=False,
    )

    # ── 奖励曲线 ──
    raw_color    = {'train': 'lightgray', 'eval': 'lightblue'}[mode]
    smooth_color = {'train': 'red',       'eval': 'blue'}[mode]
    smooth_window = (min(50, total_episodes // 10)
                     if mode == 'train' and total_episodes >= 10
                     else min(10, total_episodes // 5) if total_episodes >= 5
                     else 1)

    plt.figure(figsize=(12, 6))
    plt.plot(ep_ids, records['rewards'],
             color=raw_color, alpha=0.5, label='Raw Reward')
    smoothed = pd.Series(records['rewards']).rolling(smooth_window, min_periods=1).mean()
    plt.plot(ep_ids, smoothed, color=smooth_color, linewidth=2,
             label=f'Smoothed Reward (window={smooth_window})')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title(f'{"Training" if mode=="train" else "Evaluation"} Reward Curve', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(
        os.path.join(FileAddress.auv_reward_plot_and_infos_path,
                     f"{tag}_reward_curve.png"),
        dpi=300, bbox_inches='tight',
    )
    plt.close()

    # ── 终端打印 ──
    label = 'Training' if mode == 'train' else 'Evaluation'

    device_str = str(device) if isinstance(device, torch.device) else device

    # 设备信息（修复后）
    if device_str.startswith('cuda'):
        try:
            # 兼容 "cuda" / "cuda:0" / torch.device("cuda:0") 多种格式
            idx = int(device_str.split(':')[1]) if ':' in device_str else 0
            gpu_name = torch.cuda.get_device_name(idx)
            device_info = f"GPU ({gpu_name})"
        except Exception:
            device_info = f"GPU ({device_str})"
    else:
        device_info = "CPU"

    # 耗时格式化：自动选择 h/m/s
    hours,   rem  = divmod(int(elapsed), 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        elapsed_str = f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        elapsed_str = f"{minutes}m {secs}s"
    else:
        elapsed_str = f"{elapsed:.1f}s"

    print("\n" + "=" * 60)
    print(f"{label} Finished! Final Statistics:")
    print(f"Total Episodes:   {total_episodes}")
    print(f"Success Rate:     {success_rate:.2f}%")
    print(f"Shortest Path (Success): {shortest_path:.2f}m")
    print(f"Shortest Time (Success): {shortest_time:.2f}s")
    print(f"Average Path  (Success): {average_path:.2f}m")
    print(f"Average Time  (Success): {average_time:.2f}s")
    print(f"Data saved to: {FileAddress.auv_trajectory_folder_path}")
    print("-" * 60)
    print(f"Compute Device:   {device_info}")
    print(f"Wall-clock Time:  {elapsed_str}")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────
# 5. 训练主循环
# ─────────────────────────────────────────────

def train_agent(env, agent, total_episodes: int, max_episode_steps: int) -> None:
    """训练模式主循环。"""
    records = dict(rewards=[], lengths=[], times=[], path_lengths=[], success_flags=[])
    wall_start = time.time()
    device = agent.device if hasattr(agent, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')

    for ep in range(total_episodes):
        result = run_episode(env, agent, mode='train', max_steps=max_episode_steps)

        # 数据持久化
        save_episode_data(result, ep_idx=ep + 1, mode='train')
        plot_trajectory(
            env, result, ep_idx=ep + 1, mode='train',
            save_path=os.path.join(FileAddress.auv_plots_folder_path,
                                   f"train_trajectory_episode{ep+1}.png"),
        )

        # 全局记录更新
        records['rewards'].append(result['ep_reward'])
        records['lengths'].append(result['step_count'])
        records['times'].append(result['ep_time'])
        records['path_lengths'].append(result['ep_path_length'])
        records['success_flags'].append(result['success_flag'])

        # 定期保存模型 checkpoint
        if (ep + 1) % NetworkConfig.save_interval == 0:
            agent.save_weights(episode=ep + 1, include_target=False)
            print(f"[Checkpoint] Model saved at episode {ep + 1}")

        print(
            f"[TRAIN] Episode {ep+1}/{total_episodes} | "
            f"Reward: {result['ep_reward']:.2f} | "
            f"Steps: {result['step_count']} | "
            f"Time: {result['ep_time']:.2f}s | "
            f"Path: {result['ep_path_length']:.2f}m | "
            f"Success: {result['success_flag']}"
        )

    elapsed = time.time() - wall_start
    # 保存最终模型 + 统计汇总
    agent.save_weights(episode=None, include_target=False)
    print(f"\n[Final Model] Saved to {FileAddress.td3_network_folder}")
    save_summary(records, mode='train', total_episodes=total_episodes, elapsed=elapsed, device=device)


# ─────────────────────────────────────────────
# 6. 评估主循环
# ─────────────────────────────────────────────

def evaluate_agent(env, agent, eval_episodes: int, max_episode_steps: int) -> None:
    """评估模式主循环（不更新网络）。"""
    records = dict(rewards=[], lengths=[], times=[], path_lengths=[], success_flags=[])
    wall_start = time.time()
    device = agent.device if hasattr(agent, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nStarting evaluation for {eval_episodes} episodes...")
    for ep in range(eval_episodes):
        result = run_episode(env, agent, mode='eval', max_steps=max_episode_steps)

        # 数据持久化
        save_episode_data(result, ep_idx=ep + 1, mode='eval')
        plot_trajectory(
            env, result, ep_idx=ep + 1, mode='eval',
            save_path=os.path.join(FileAddress.auv_plots_folder_path,
                                   f"eval_trajectory_episode{ep+1}.png"),
        )

        # 全局记录更新
        records['rewards'].append(result['ep_reward'])
        records['lengths'].append(result['step_count'])
        records['times'].append(result['ep_time'])
        records['path_lengths'].append(result['ep_path_length'])
        records['success_flags'].append(result['success_flag'])

        print(
            f"[EVAL] Episode {ep+1}/{eval_episodes} | "
            f"Reward: {result['ep_reward']:.2f} | "
            f"Steps: {result['step_count']} | "
            f"Time: {result['ep_time']:.2f}s | "
            f"Path: {result['ep_path_length']:.2f}m | "
            f"Success: {result['success_flag']}"
        )
    
    elapsed = time.time() - wall_start
    save_summary(records, mode='eval', total_episodes=eval_episodes, elapsed=elapsed, device=device)
