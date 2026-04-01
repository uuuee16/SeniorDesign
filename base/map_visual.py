import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 1. 定义缺失的配置类（原代码中用到的MapConfig）
class MapConfig:
    env_limit = 500  # 空间范围设置为500，匹配需求

# 2. 定义环境对象（原代码中用到的env），包含指定的障碍物列表
class Env:
    def __init__(self):
            
        # 你的目标障碍物列表
        # self.obstacles = [
        #     (250.0, 250.0, 150.0, 50.0),  # 障碍1：封锁上层直线路径
        #     (100.0, 100.0, 100.0, 40.0),  # 障碍2：上层狭窄瓶颈
        #     (200.0, 350.0, 300.0, 40.0),  # 障碍3：中层侧方交叉障碍
        #     (350.0, 200.0, 350.0, 45.0),  # 障碍4：中层另一侧交叉障碍
        #     (400.0, 400.0, 400.0, 50.0),  # 障碍5：下层终点前障碍
        # ]
            # self.obstacles_canyon = [
            # (150.0, 300.0, 100.0, 50.0),
            # (350.0, 100.0, 150.0, 50.0),
            # (250.0, 150.0, 300.0, 40.0),
            # (350.0, 350.0, 350.0, 45.0),
            # (450.0, 250.0, 450.0, 30.0),
        # ]

        # 方案3（迷宫式密集障碍）- 极高难度
        self.obstacles_maze = [
            (200.0, 200.0, 200.0, 40.0),
            (250.0, 250.0, 250.0, 45.0),
            (300.0, 300.0, 300.0, 50.0),
            (200.0, 300.0, 350.0, 40.0),
            (300.0, 200.0, 400.0, 45.0),
        ]
        self.obstacles_hard = [
        # 上层区域（z < 200）：封锁起点到中层的直线路径
        (250.0, 250.0, 150.0, 50.0),  # 障碍1：上层核心障碍，阻断中心直线路径
        (100.0, 100.0, 100.0, 40.0),  # 障碍2：起点附近瓶颈，限制初始方向
        (150.0, 350.0, 180.0, 40.0),  # 障碍3：上层右侧障碍，与障碍2形成窄通道
        (350.0, 150.0, 180.0, 40.0),  # 障碍4：上层左侧障碍，与障碍3对称压缩通道
        
        # 中层区域（200 ≤ z ≤ 350）：交叉障碍，进一步压缩路径
        (200.0, 350.0, 300.0, 40.0),  # 障碍5：中层右侧障碍
        (350.0, 200.0, 300.0, 45.0),  # 障碍6：中层左侧障碍
        (300.0, 300.0, 250.0, 45.0),  # 障碍7：中层中心障碍，形成迷宫式交叉
        
        # 下层区域（z > 350）：终点前最后障碍
        (420.0, 420.0, 400.0, 50.0),  # 障碍8：终点前障碍，阻断最后直线路径
        ]

# 3. 初始化环境和绘图
env = Env()
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 4. 绘制障碍物球体（保留你原有的样式，仅适配数据）
obstacle_colors = ['dodgerblue', 'deepskyblue', 'steelblue', 'cornflowerblue', 'royalblue']
for idx, (ox, oy, oz, r) in enumerate(env.obstacles_maze):
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]  # 原代码的网格密度
    ax.plot_surface(
        ox + r * np.cos(u) * np.sin(v),
        oy + r * np.sin(u) * np.sin(v),
        oz + r * np.cos(v),
        color=obstacle_colors[idx % len(obstacle_colors)],
        alpha=0.35, edgecolor='navy', linewidth=0.3, antialiased=True,
    )
    # 新增：在球心标注障碍编号，方便识别
    ax.text(ox, oy, oz, f'Obstacle {idx+1}', fontsize=11, fontweight='bold', color='black')

# 5. 坐标轴样式（完全保留你原有的逻辑，适配500范围）
ax.set_xlabel('North (m)', fontsize=13, fontweight='bold', color='darkblue', labelpad=10)
ax.set_ylabel('East (m)',  fontsize=13, fontweight='bold', color='darkgreen', labelpad=10)
ax.set_zlabel('Down (m)',  fontsize=13, fontweight='bold', color='darkred',   labelpad=10)

# 设置刻度：0, 125, 250, 375, 500（对应500的1/4、1/2、3/4）
for setter, ticks in zip([ax.set_xticks, ax.set_yticks, ax.set_zticks],
                          [[0, MapConfig.env_limit*1.0/4.0, MapConfig.env_limit/2.0, MapConfig.env_limit*3.0/4.0, MapConfig.env_limit]]*3):
    setter(ticks)

ax.tick_params(labelsize=10)
ax.set_xlim(0, MapConfig.env_limit)
ax.set_ylim(0, MapConfig.env_limit)
ax.set_zlim(0, MapConfig.env_limit)
ax.set_box_aspect([1, 1, 1])  # 保持三轴等比例
ax.invert_zaxis()  # 反转z轴（Down方向，z值越大表示越向下）

# 6. 添加标题
ax.set_title('3D Obstacle Distribution (500×500×500 m)', fontsize=15, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()