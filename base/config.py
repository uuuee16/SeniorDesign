import numpy as np
import math

# 文件存储路径配置
class FileAddress:
    # 圆形和线形障碍物参数储存
    obstacle_path = 'results/obstacle_parameter/obstacle_parameter.csv'
    # 数据保存格式:shape,x,y,z,length x,length y,length z,length l,row,pitch,yaw
    grid_obstacle_path = 'results/obstacle_parameter/grid_obstacle_parameter.csv'

    # 神经网络路径
    td3_policy_net_path = 'results/net_storage/td3_policy_net.pth'
    td3_critic_1_net_path = 'results/net_storage/td3_critic_1_net.pth'
    td3_critic_2_net_path = 'results/net_storage/td3_critic_2_net.pth'
    td3_network_folder = 'results/net_storage'

    # 训练过程中的信息存储
    # auv轨迹储存（注：使用时需要.format(episodes)）
    auv_trajectory_path = 'results/training_data/path_data/trajectory_episode{}.csv'
    auv_trajectory_folder_path = 'results/training_data/path_data'
    # 输出图片储存路径（注：使用时需要.format(episodes)）
    auv_plots_path = 'results/training_data/path_picture/picture_episode{}.jpg'
    auv_plots_folder_path = 'results/training_data/path_picture'
    # 训练过程中信息存储
    auv_rewards_steps_results_path = 'results/training_data/step_infos/auv_rewards_steps.csv'
    auv_rewards_steps_results = 'results/training_data/step_infos'
    # 输出图片路径
    auv_reward_plot_and_infos_path = 'results/training_data'




# 地图/环境核心配置
class MapConfig:

    train_episodes = 2000
    max_episode_steps = 4000
    dt = 0.5
    # 环境边界 [x_min, x_max, y_min, y_max, z_min, z_max]
    env_bound = [0, 500, 0, 500, 0, 500]
    env_limit = env_bound[1]
    # AUV动作范围
    max_pitch_vel = math.pi / 6
    max_yaw_vel = math.pi / 4
    max_body_vel = 5.0
    pitch_limit = [-math.pi/4, math.pi/4]
    yaw_limit = [-math.pi, math.pi]
    # 障碍物参数
    obstacles0 = [
        (20.0, 20.0, 20.0, 10.0),
        (10.0, 50.0, 50.0, 12.0),
        (70.0, 20.0, 30.0, 15.0),
        (30.0, 70.0, 40.0, 13.0),
        (70.0, 60.0, 70.0, 11.0)
        ]
    obstacles1 = [
        (125.0, 125.0, 250.0, 30.0),
        (375.0, 125.0, 250.0, 35.0),
        (250.0, 375.0, 250.0, 50.0),
        (250.0, 125.0, 125.0, 40.0),
        (250.0, 375.0, 375.0, 45.0)
        ]
    # 方案1（核心路径封锁+多层立体障碍）- 推荐难度
    obstacles_challenging1 = [
        (250.0, 250.0, 150.0, 50.0),  # 障碍1：封锁上层直线路径，半径50增加封锁范围
        (100.0, 100.0, 150.0, 40.0),  # 障碍2：上层狭窄瓶颈，半径40缩小容错空间
        (200.0, 350.0, 300.0, 40.0),  # 障碍3：中层侧方交叉障碍，半径40
        (350.0, 200.0, 350.0, 45.0),  # 障碍4：中层另一侧交叉障碍，半径45
        (400.0, 400.0, 450.0, 50.0),  # 障碍5：下层终点前障碍，半径50增加最后阶段压力
    ]
    obstacles_challenging2 = [
        (250.0, 250.0, 150.0, 50.0),  # 障碍1：封锁上层直线路径，半径50增加封锁范围
        (100.0, 100.0, 100.0, 40.0),  # 障碍2：上层狭窄瓶颈，半径40缩小容错空间
        (200.0, 350.0, 300.0, 40.0),  # 障碍3：中层侧方交叉障碍，半径40
        (350.0, 200.0, 350.0, 45.0),  # 障碍4：中层另一侧交叉障碍，半径45
        (400.0, 400.0, 400.0, 50.0),  # 障碍5：下层终点前障碍，半径50增加最后阶段压力
    ]

    # 方案2（峡谷型狭窄通道）- 高难度
    obstacles_canyon = [
        (150.0, 300.0, 100.0, 50.0),
        (350.0, 100.0, 150.0, 50.0),
        (250.0, 150.0, 300.0, 40.0),
        (350.0, 350.0, 350.0, 45.0),
        (450.0, 250.0, 450.0, 30.0)
    ]

    # 方案3（迷宫式密集障碍）- 极高难度
    obstacles_maze = [
        (200.0, 200.0, 200.0, 40.0),
        (250.0, 250.0, 250.0, 45.0),
        (300.0, 300.0, 300.0, 50.0),
        (200.0, 300.0, 350.0, 40.0),
        (300.0, 200.0, 400.0, 45.0)
    ]

    obstacles_hard = [
    (250.0, 250.0, 150.0, 50.0),  
    (100.0, 100.0, 100.0, 40.0), 
    (150.0, 350.0, 180.0, 40.0), 
    (350.0, 150.0, 180.0, 40.0),  
    (200.0, 350.0, 300.0, 40.0),  
    (350.0, 200.0, 300.0, 45.0),  
    (300.0, 300.0, 250.0, 45.0),  
    (420.0, 420.0, 400.0, 50.0), 
    ]



    obstacle_radius_range = (30.0, 50.0)
    k_nearest_obstacles = 5
    # 目标参数
    default_target_pos = (480.0, 480.0, 480.0)
    target_pos_maze = (400.0, 400.0, 325.0)
    # 判断成功和碰撞的阈值
    goal_threshold = 15.0
    collision_threshold = 5.0   # 碰撞阈值
    dangerous_distance = 10.0    # 危险距离
    
    # 奖励函数参数
    reward_task_goal = 100.0
    reward_task_collision = -200.0
    reward_task_danger = -2.0
    # reward_task_danger = -1.0 防止过度保守
    
    reward_dist_eps1 = 2.0
    reward_dist_eps2 = 1.0
    reward_dist_lambda = 1.25

class NetworkConfig:
    k_obstacle = MapConfig.k_nearest_obstacles
    nn_input_dim = 3 + 2 + 4 * k_obstacle 
    nn_output_dim = 3  # (艏向速度, yaw_vel, pitch_vel)
    nn_hidden_dim = np.array([128, 128], dtype=int)  
    
    gamma = 0.99
    learning_rate_actor = 0.0003  # TD3 Actor学习率
    learning_rate_critic = 0.0003  # TD3 Critic学习率
    soft_update_tau = 0.005  # 软更新系数
    batch_size = 64  # 回放缓冲区采样批次大小
    buffer_size = int(1e6)  # 回放缓冲区最大容量
    actor_update_interval = 2  # TD3 Actor延迟更新间隔
    smooth_noise_clip = 0.5  # TD3目标策略平滑噪声裁剪值
    smooth_noise_range = 0.2  # TD3目标策略平滑噪声范围
    explore_noise_range = 0.2  # TD3探索噪声范围

    save_interval = 50  # 网络模型保存间隔