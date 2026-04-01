import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import torch
import os
import argparse

from config import FileAddress, MapConfig, NetworkConfig
from td3 import TD3
from env import AUVEnv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TD3 AUV Training/Inference')
    parser.add_argument('--evaluate', action='store_true', help='evaluate mode (no training updates)')
    parser.add_argument('--load', action='store_true', help='load model weights before running')
    parser.add_argument('--load_episode', type=int, default=None, help='load checkpoint for specific episode')
    args = parser.parse_args()

    # 基础配置预加载
    total_episodes = MapConfig.train_episodes
    max_episode_steps = MapConfig.max_episode_steps
    evaluate = args.evaluate

    # 仿真核心参数预提取
    dt = MapConfig.dt
    max_body_vel = MapConfig.max_body_vel
    max_yaw_rate = MapConfig.yaw_limit[1]
    max_pitch_rate = MapConfig.pitch_limit[1]
    target_reach_radius = MapConfig.goal_threshold
    save_interval = NetworkConfig.save_interval

    # 环境与智能体初始化
    env = AUVEnv()
    agent = TD3(state_dims=NetworkConfig.nn_input_dim, n_actions=NetworkConfig.nn_output_dim)

    # 加载现有模型（如果用户要求）
    if args.load:
        try:
            # 有 episode 参数时负载对应周期模型，否者加载最新默认路径
            agent.load_weights(episode=args.load_episode, include_target=True)
            print(f"[加载模型] 已从 checkpoint 目录加载模型 (episode={args.load_episode})")
        except FileNotFoundError as e:
            # 兼容旧版路径名
            policy_load_path = FileAddress.td3_policy_net_path if args.load_episode is None else os.path.join(FileAddress.td3_network_folder, f"td3_policy_net_episode{args.load_episode}.pth")
            critic1_load_path = FileAddress.td3_critic_1_net_path if args.load_episode is None else os.path.join(FileAddress.td3_network_folder, f"td3_critic_1_net_episode{args.load_episode}.pth")
            critic2_load_path = FileAddress.td3_critic_2_net_path if args.load_episode is None else os.path.join(FileAddress.td3_network_folder, f"td3_critic_2_net_episode{args.load_episode}.pth")
            if os.path.exists(policy_load_path) and os.path.exists(critic1_load_path) and os.path.exists(critic2_load_path):
                agent.actor.load_state_dict(torch.load(policy_load_path, map_location=agent.device))
                agent.critic_1.load_state_dict(torch.load(critic1_load_path, map_location=agent.device))
                agent.critic_2.load_state_dict(torch.load(critic2_load_path, map_location=agent.device))
                agent.target_actor.load_state_dict(agent.actor.state_dict())
                agent.target_critic_1.load_state_dict(agent.critic_1.state_dict())
                agent.target_critic_2.load_state_dict(agent.critic_2.state_dict())
                print(f"[加载模型] 兼容旧路径完成加载：{policy_load_path}, {critic1_load_path}, {critic2_load_path}")
            else:
                print(f"[加载模型] 无有效模型文件：{policy_load_path}, {critic1_load_path}, {critic2_load_path}。训练将从头开始。")
        except Exception as e:
            print(f"[加载模型] 其他错误: {e}，训练将从头开始。")

    # 存储文件夹创建
    os.makedirs(FileAddress.auv_trajectory_folder_path, exist_ok=True)
    os.makedirs(FileAddress.auv_plots_folder_path, exist_ok=True)
    os.makedirs(FileAddress.auv_rewards_steps_results, exist_ok=True)
    os.makedirs(FileAddress.td3_network_folder, exist_ok=True)

    # 全局训练数据记录
    episode_rewards = []
    episode_lengths = []
    episode_times = []
    episode_path_lengths = []
    episode_success_flags = []

    for ep in range(total_episodes):
        # 环境重置
        obs, info = env.reset()
        start_pos = env.auv_state.position.copy()
        done = False
        step_count = 0
        ep_reward = 0
        ep_path_length = 0.0
        prev_pos = start_pos.copy()

        # 单episode数据存储容器
        trajectory_data = []
        step_data = []
        boundary_collision_points = [] 
        # ========== 新增：存储撞边界的位置坐标 ==========
        boundary_collision_points = [] 

        # 单步仿真循环
        while not done and step_count < max_episode_steps:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # 手动触发步数截断
            if step_count + 1 >= max_episode_steps:
                truncated = True
                info['truncated_reason'] = 'max_episode_steps_reached'
            
            done = terminated or truncated
            
            # 提取AUV当前状态
            current_state = env.auv_state
            x, y, z = current_state.position
            yaw, pitch = current_state.attitude
            
            # ========== 新增：边界碰撞检测 ==========
            # 假设边界为 [0, 100]，根据你的 MapConfig 调整此处逻辑
            # 也可以直接使用 info 中可能已有的 'collision' 信息
            hit_boundary = False
            if x <= 0 or x >= 100 or y <= 0 or y >= 100 or z <= 0 or z >= 100:
                hit_boundary = True
                boundary_collision_points.append([x, y, z]) # 记录碰撞点
            
            # 动作解析
            actual_speed = (action[0] + 1.0) / 2.0 * max_body_vel
            actual_yaw_rate = action[1] * max_yaw_rate
            actual_pitch_rate = action[2] * max_pitch_rate
            
            # 路径长度计算
            current_pos = current_state.position
            step_displacement = np.linalg.norm(current_pos - prev_pos)
            ep_path_length += step_displacement
            prev_pos = current_pos.copy()

            # 单步记录
            step_record = {
                'step': step_count,
                'x': x, 'y': y, 'z': z,
                'yaw': yaw, 'pitch': pitch,
                'actual_speed': actual_speed,
                'actual_yaw_rate': actual_yaw_rate,
                'actual_pitch_rate': actual_pitch_rate,
                'action_raw_0': action[0],
                'action_raw_1': action[1],
                'action_raw_2': action[2],
                'reward': reward,
                'dist2target': info['dist2target'],
                'nearest_obstacle_dist': info['nearest_obstacle_dist'],
                'step_terminated': done,
                'hit_boundary': hit_boundary  # 记录该步是否撞墙
            }
            step_data.append(step_record)
            trajectory_data.append([x, y, z])

            # 经验池存储与网络更新
            agent.memory.store_transition(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward
            step_count += 1

            if not evaluate:
                agent.update()

        # Episode结束后收尾处理
        ep_time = step_count * dt
        success_flag = 1 if (info.get('reach_target', False) or (info['dist2target'] < target_reach_radius)) else 0
        for record in step_data:
            record['episode_success'] = success_flag

        # 保存CSV
        step_df = pd.DataFrame(step_data)
        step_csv_path = os.path.join(FileAddress.auv_rewards_steps_results, f"step_data_episode{ep+1}.csv")
        step_df.to_csv(step_csv_path, index=False)
        
        traj_df = pd.DataFrame(trajectory_data, columns=['x','y','z'])
        traj_csv_path = os.path.join(FileAddress.auv_trajectory_folder_path, f"trajectory_episode{ep+1}.csv")
        traj_df.to_csv(traj_csv_path, index=False)

        # ========== 可视化绘图（修改部分） ==========
        fig = plt.figure(figsize=(14, 11), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        traj_array = np.array(trajectory_data)
        
        # 1. 绘制环境边界框
        for i in [0, 100]:
            for j in [0, 100]:
                ax.plot([i, i], [j, j], [0, 100], 'k-', linewidth=0.8, alpha=0.3)
                ax.plot([i, i], [0, 100], [j, j], 'k-', linewidth=0.8, alpha=0.3)
                ax.plot([0, 100], [i, i], [j, j], 'k-', linewidth=0.8, alpha=0.3)
        
        # 2. 核心元素绘制
        if len(traj_array) > 1:
            ax.plot(traj_array[:,0], traj_array[:,1], traj_array[:,2], 
                   'r-', linewidth=2.5, label='AUV Trajectory', alpha=0.8, zorder=3)
            step_sample = max(1, len(traj_array) // 50)
            ax.scatter(traj_array[::step_sample, 0], traj_array[::step_sample, 1], 
                      traj_array[::step_sample, 2], c='crimson', s=15, 
                      alpha=0.6, zorder=3)
        
        # ========== 新增：绘制边界碰撞点 ==========
        if len(boundary_collision_points) > 0:
            bc_array = np.array(boundary_collision_points)
            # 使用大号的 'X' 标记，白色边缘，深紫色填充，确保醒目
            ax.scatter(bc_array[:, 0], bc_array[:, 1], bc_array[:, 2], 
                      c='darkviolet',          # 颜色
                      s=250,                   # 大小
                      marker='X',              # 形状
                      label='Boundary Collision', 
                      zorder=10,               # 图层置顶
                      edgecolors='white',      # 描边颜色
                      linewidths=2.5)          # 描边宽度
        
        # 起点
        ax.scatter(start_pos[0], start_pos[1], start_pos[2], 
                  c='black', s=180, marker='o', label='Start Point', 
                  zorder=5, edgecolors='gold', linewidths=2)
        
        # 目标点
        ax.scatter(env.target_pos[0], env.target_pos[1], env.target_pos[2], 
                  c='limegreen', s=220, marker='*', label='Target Point', 
                  zorder=5, edgecolors='yellow', linewidths=2)
        
        # 3. 障碍物绘制
        obstacle_colors = ['dodgerblue', 'deepskyblue', 'steelblue', 'cornflowerblue', 'royalblue']
        for idx, (ox, oy, oz, r) in enumerate(env.obstacles):
            u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
            xs = ox + r * np.cos(u) * np.sin(v)
            ys = oy + r * np.sin(u) * np.sin(v)
            zs = oz + r * np.cos(v)
            ax.plot_surface(xs, ys, zs, 
                           color=obstacle_colors[idx % len(obstacle_colors)], 
                           alpha=0.35, 
                           edgecolor='navy', 
                           linewidth=0.3,
                           antialiased=True)
        
        # 4. 坐标系与样式设置
        ax.set_xlabel('North (m)', fontsize=13, fontweight='bold', color='darkblue', labelpad=10)
        ax.set_ylabel('East (m)', fontsize=13, fontweight='bold', color='darkgreen', labelpad=10)
        ax.set_zlabel('Down (m)', fontsize=13, fontweight='bold', color='darkred', labelpad=10)
        
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_zticks([0, 25, 50, 75, 100])
        ax.tick_params(labelsize=10)
        
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_zlim(0, 100)
        ax.set_box_aspect([1, 1, 1])
        ax.invert_zaxis()
        
        # 5. 标题
        title_text = (
            f'Episode {ep+1} Trajectory Visualization\n'
            f'Reward: {ep_reward:.2f}  |  Time: {ep_time:.2f}s  |  '
            f'Path: {ep_path_length:.2f}m  |  Success: {"✓" if success_flag else "✗"}'
        )
        ax.set_title(title_text, fontsize=13, fontweight='bold', pad=25, color='darkslategray')
        
        # 6. 视角与图例
        ax.view_init(elev=28, azim=-50, roll=0)
        legend = ax.legend(loc='upper left', fontsize=11, framealpha=0.9, 
                          edgecolor='gray', facecolor='white', fancybox=True)
        legend.get_frame().set_linewidth(1.2)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4, color='gray')
        ax.set_facecolor('aliceblue')
        
        # 7. 保存
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_save_path = os.path.join(FileAddress.auv_plots_folder_path, 
                                    f"trajectory_episode{ep+1}.png")
        plt.savefig(fig_save_path, dpi=300, bbox_inches='tight', 
                   facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)

        # 全局数据更新
        episode_rewards.append(ep_reward)
        episode_lengths.append(step_count)
        episode_times.append(ep_time)
        episode_path_lengths.append(ep_path_length)
        episode_success_flags.append(success_flag)

        # 定期保存模型
        if (ep + 1) % save_interval == 0:
            agent.save_weights(episode=ep+1, include_target=False)
            print(f"[Checkpoint] Model saved at episode {ep+1}")

        # 终端打印
        print(
            f"Episode {ep+1}/{total_episodes} | "
            f"Reward: {ep_reward:.2f} | "
            f"Steps: {step_count} | "
            f"Time: {ep_time:.2f}s | "
            f"Path Length: {ep_path_length:.2f}m | "
            f"Success: {success_flag}"
        )

    # 训练结束，全局统计
    summary_df = pd.DataFrame({
        'episode': np.arange(1, total_episodes+1),
        'total_reward': episode_rewards,
        'steps': episode_lengths,
        'time_cost_s': episode_times,
        'path_length_m': episode_path_lengths,
        'success': episode_success_flags
    })
    summary_csv_path = os.path.join(FileAddress.auv_reward_plot_and_infos_path, "training_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    
    valid_success_episodes = summary_df[summary_df['success'] == 1]
    total_success_rate = len(valid_success_episodes) / total_episodes * 100
    if len(valid_success_episodes) > 0:
        shortest_path = valid_success_episodes['path_length_m'].min()
        shortest_time = valid_success_episodes['time_cost_s'].min()
        average_path = valid_success_episodes['path_length_m'].mean()
        average_time = valid_success_episodes['time_cost_s'].mean()
    else:
        shortest_path = shortest_time = average_path = average_time = np.nan
    
    stats_dict = {
        'total_episodes': total_episodes,
        'success_count': len(valid_success_episodes),
        'success_rate_percent': total_success_rate,
        'shortest_path_m': shortest_path,
        'shortest_time_s': shortest_time,
        'average_path_m': average_path,
        'average_time_s': average_time,
        'max_reward': np.max(episode_rewards),
        'mean_reward': np.mean(episode_rewards)
    }
    stats_df = pd.DataFrame([stats_dict])
    stats_csv_path = os.path.join(FileAddress.auv_reward_plot_and_infos_path, "training_statistics.csv")
    stats_df.to_csv(stats_csv_path, index=False)

    # 奖励曲线
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, total_episodes+1), episode_rewards, color='lightgray', alpha=0.5, label='Raw Reward')
    smooth_window = min(50, total_episodes//10) if total_episodes >=10 else 1
    smoothed_rewards = pd.Series(episode_rewards).rolling(window=smooth_window, min_periods=1).mean()
    plt.plot(range(1, total_episodes+1), smoothed_rewards, color='red', linewidth=2, label=f'Smoothed Reward (window={smooth_window})')
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('Training Reward Curve', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    reward_curve_path = os.path.join(FileAddress.auv_reward_plot_and_infos_path, "total_reward_curve.png")
    plt.savefig(reward_curve_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 保存最终模型
    agent.save_weights(episode=None, include_target=False)
    print(f"\n[Final Model] Saved to {FileAddress.td3_network_folder}")

    # 最终打印
    print("\n" + "="*60)
    print("Training Finished! Final Statistics:")
    print(f"Total Episodes: {total_episodes}")
    print(f"Success Rate: {total_success_rate:.2f}%")
    print(f"Shortest Path (Success Episodes): {shortest_path:.2f}m")
    print(f"Shortest Time (Success Episodes): {shortest_time:.2f}s")
    print(f"Average Path (Success Episodes): {average_path:.2f}m")
    print(f"Average Time (Success Episodes): {average_time:.2f}s")
    print(f"All data saved to: {FileAddress.auv_trajectory_folder_path}")
    print("="*60 + "\n")