from __future__ import annotations
import math
import datetime
from pathlib import Path
import numpy as np

ROOT_PATH = Path(__file__).resolve().parent

# ══════════════════════════════════════════════════════════════════════════════
# 地图 / 环境配置
# ══════════════════════════════════════════════════════════════════════════════

class MapConfig:

    # ── Flag_Initialization ──────────────────────────────────────────────────
    start_fixed = True
    target_fixed = True

    seed = 36
    algo_name = "TD3-PER"  # 可选: "TD3-PER", "TD3", "DDPG"
    # ── 运行模式 ──────────────────────────────────────────────────────────────
    mode      = "train"   # "train" | "eval"
    show_current = False
    train_eps = 2000
    eval_eps  = 100
    max_steps = 4096
    dt        = 0.5       # 仿真步长，秒

    # ── eval 权重加载 ───────────────────────────────────────────────────────
    eval_model_episode = None   # None -> 优先加载最终权重(.pth)，若不存在则加载最近一次 episode 快照

    # ── 海流模式 ──────────────────────────────────────────────────────────────
    current_mode = "synthetic"    # 可选: "none" | "synthetic" | "real"

    # ── (real mode)真实海流参数 ──────────────────────────────────────────────────────────
    use_downloader           = False
    use_temporal_current     = True
    use_small_scale_gradient = True
    time_compress_hours      = 1.0   # ── 将下载数据的完整时间跨度压缩到该小时数内

    operation_center = {
            "South_China_Sea_Deep_Center": (115.0, 15.0, 0.1, 250),
            "Philippine_Sea_Deep_Center":  (123.0, 20.0, 0.1, 250),
            "Western_Pacific_Open_Center": (125.0, 18.0, 0.1, 250),
            "Custom_Operation_Center":     (120.0, 20.0, 0.1, 250),
        }
    center_name = "South_China_Sea_Deep_Center"

    t = datetime.datetime.now()

    horiz = f"currents_horizontal_{center_name}_{t:%Y%m%d}_{(t + datetime.timedelta(days=2)):%Y%m%d}.nc"
    vert = f"currents_vertical_{center_name}_{t:%Y%m%d}_{(t + datetime.timedelta(days=2)):%Y%m%d}.nc"
    cmems_horiz_file = str(ROOT_PATH / "cmems_data" / horiz)
    cmems_vert_file  = str(ROOT_PATH / "cmems_data" / vert)

    # ── 场景空间 ──────────────────────────────────────────────────────────────
    bounds = (0.0, 500.0, 0.0, 500.0, 0.0, 500.0)
    start  = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    target = np.array([480.0, 480.0, 480.0], dtype=np.float32)
    target_maze = np.array([400.0, 400.0, 325.0], dtype=np.float32)

    # ── AUV 运动参数 ──────────────────────────────────────────────────────────
    max_body_speed = 5.0
    max_yaw_rate   = math.pi / 4
    max_pitch_rate = math.pi / 6
    pitch_range    = (-math.pi / 4, math.pi / 4)
    yaw_range      = (-math.pi, math.pi)

    # ── 障碍物场景 (x, y, z, radius)，米 ─────────────────────────────────────
    obstacles0 = [
        (20.0, 20.0, 20.0, 10.0), (10.0, 50.0, 50.0, 12.0), (70.0, 20.0, 30.0, 15.0),
        (30.0, 70.0, 40.0, 13.0), (70.0, 60.0, 70.0, 11.0),
    ]
    obstacles1 = [
        (125.0, 125.0, 250.0, 30.0), (375.0, 125.0, 250.0, 35.0), (250.0, 375.0, 250.0, 50.0),
        (250.0, 125.0, 125.0, 40.0), (250.0, 375.0, 375.0, 45.0),
    ]
    challenging1 = [
        (250.0, 250.0, 150.0, 50.0), (100.0, 100.0, 150.0, 40.0), (200.0, 350.0, 300.0, 40.0),
        (350.0, 200.0, 350.0, 45.0), (400.0, 400.0, 450.0, 50.0),
    ]
    challenging2 = [
        (250.0, 250.0, 150.0, 50.0), (100.0, 100.0, 100.0, 40.0), (200.0, 350.0, 300.0, 40.0),
        (350.0, 200.0, 350.0, 45.0), (400.0, 400.0, 400.0, 50.0),
    ]
    canyon = [
        (150.0, 300.0, 100.0, 50.0), (350.0, 100.0, 150.0, 50.0), (250.0, 150.0, 300.0, 40.0),
        (350.0, 350.0, 350.0, 45.0), (450.0, 250.0, 450.0, 30.0),
    ]
    maze = [
        (200.0, 200.0, 200.0, 40.0), (250.0, 250.0, 250.0, 45.0), (300.0, 300.0, 300.0, 50.0),
        (200.0, 300.0, 350.0, 40.0), (300.0, 200.0, 400.0, 45.0),
    ]

    hard_easy = [
        (250.0, 250.0, 150.0, 50.0), (100.0, 100.0, 100.0, 45.0), (150.0, 350.0, 180.0, 30.0), (350.0, 150.0, 180.0, 35.0),
        (200.0, 350.0, 300.0, 40.0), (350.0, 200.0, 300.0, 45.0), (300.0, 300.0, 250.0, 45.0), (420.0, 420.0, 400.0, 33.0),
    ]

    hard = [
        (250.0, 250.0, 150.0, 50.0), (100.0, 100.0, 100.0, 30.0), (150.0, 350.0, 180.0, 30.0), (350.0, 150.0, 180.0, 35.0),
        (200.0, 350.0, 300.0, 40.0), (350.0, 200.0, 300.0, 45.0), (300.0, 300.0, 250.0, 45.0), (420.0, 420.0, 400.0, 50.0),
    ]
    obstacle_map = {
        "obstacles0":   obstacles0,
        "obstacles1":   obstacles1,
        "challenging1": challenging1,
        "challenging2": challenging2,
        "canyon":       canyon,
        "maze":         maze,
        "hard":         hard,
        "hard_easy":    hard_easy,
    }

    scene      = "challenging2"
    k_obs      = 5
    obs_radius = (30.0, 50.0)

    @classmethod
    def get_obstacles(cls, name=None):
        return cls.obstacle_map.get(name or cls.scene, cls.obstacle_map[cls.scene])

    # ── 终止 / 危险判定距离 ───────────────────────────────────────────────────
    goal_dist = 15.0
    hit_dist  = 5.0
    warn_dist = 10.0

    # ── 奖励函数权重 ──────────────────────────────────────────────────────────
    r_goal    = 100.0
    r_hit     = -200.0
    r_warn    = -5.0
    r_lambda_dist  = 1.25
    r_lambda_cur  = 0.80
    r_lambda_eng = 0.10  
    r_epsilon1 = 2.0
    r_epsilon2 = 1.0

    # ── (Synthetic Mode)合成海流参数 ──────────────────────────────────────────────────────────
    # current_clip  = 2.0
    current_clip = 1.0
    current_const = (0.25, 0.15, 0.05)
    meters_per_deg_lat = 111320.0
    vortices = [
        {"center": (150.0, 150.0, 200.0), "gamma": 50.0,  "radius": 40.0, "depth_decay": 50.0, "vertical_scale": 0.30},
        {"center": (350.0, 350.0, 300.0), "gamma": -60.0, "radius": 50.0, "depth_decay": 60.0, "vertical_scale": 0.30},
        {"center": (250.0, 100.0, 150.0), "gamma": 40.0,  "radius": 35.0, "depth_decay": 45.0, "vertical_scale": 0.25},
    ]
    vortices_vib = [
        {"center": (250.0, 250.0, 250.0), "gamma": 1.0, "radius": 200.0, "depth_decay": 250.0, "vertical_scale": 0.30}
    ]
    #── 真实海流随机化参数 (仅在 mode = "real" 时生效) ─────────────────────────────────────

    cmems_usrname = "1686546803@qq.com"
    cmems_pswd    = "@Lwk51805852004"

    # ── 观测维度（自动推导） ─────────────────────────────────────────────────
    obs_dim = 3 + 2 + 4 * k_obs + (3 if current_mode != "none" else 0)


# ══════════════════════════════════════════════════════════════════════════════
# 神经网络配置
# ══════════════════════════════════════════════════════════════════════════════

class NetworkConfig:
    state_dim  = MapConfig.obs_dim
    action_dim = 3
    hidden     = np.array([128, 128], dtype=int)

    # PER 超参数
    use_PER         = True
    epsilon         = 0.5
    alpha           = 0.5
    beta            = 0.4
    # beta_increment  = 0.001
    beta_increment  = (1.0 - beta) / MapConfig.train_eps  # 自动计算：(1-0.4)/2000=0.0003
    # TD3 超参数
    gamma                 = 0.99
    actor_lr              = 3e-4
    critic_lr             = 3e-4
    tau                   = 0.005
    batch                 = 64
    buffer                = int(1e6)
    actor_delay           = 2
    target_noise_clip     = 0.5
    target_noise          = 0.2
    # action noise 从0.1增大到了0.2
    action_noise         = 0.2
    save_gap              = 50


# ══════════════════════════════════════════════════════════════════════════════
# 文件路径
# ══════════════════════════════════════════════════════════════════════════════

class FileAddress:
    root  = ROOT_PATH
    cmems = root / "cmems_data"

    # 运行时由 update_algo() 绑定到当前算法目录
    results = None
    net     = None
    train   = None
    traj    = None
    fig     = None
    step    = None
    update  = None
    summary = None
    current = None

    @classmethod
    def results_root(cls) -> Path:
        p = cls.root / "results"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def algorithm_results_dir(cls, algo_name: str) -> Path:
        return cls.results_root() / algo_name

    @classmethod
    def comparison_root(cls) -> Path:
        p = cls.results_root() / "comparison"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def comparison_tag(cls, algorithms) -> str:
        if isinstance(algorithms, str):
            algorithms = [algorithms]
        return "__".join(str(a) for a in algorithms)

    @classmethod
    def comparison_dir(cls, mode: str, algorithms) -> Path:
        p = cls.comparison_root() / str(mode) / cls.comparison_tag(algorithms)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def comparison_png(cls, mode: str, algorithms, stem: str) -> Path:
        return cls.comparison_dir(mode, algorithms) / f"{stem}.png"

    @classmethod
    def comparison_csv(cls, mode: str, algorithms, stem: str) -> Path:
        return cls.comparison_dir(mode, algorithms) / f"{stem}.csv"

    @classmethod
    def comparison_episode_root(cls, mode: str, algorithms) -> Path:
        p = cls.comparison_dir(mode, algorithms) / "episode_trajectory"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ──────────────────────────────────────────────────────────────────────────
    # 【修改区域开始】适配新版 plot_comparison.py 的轨迹对比路径生成
    # ──────────────────────────────────────────────────────────────────────────

    # 【删除】删除了旧版的 comparison_episode_dir(cls, mode, algorithms, view)
    # 【删除】删除了旧版的 comparison_episode_png(cls, mode, algorithms, ep, view)

    @classmethod
    def comparison_episode_combo_dir(cls, mode: str, algorithms) -> Path:
        """【新增】为 combo（组合视图）单独生成并确保目录存在"""
        p = cls.comparison_episode_root(mode, algorithms) / "combo"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def comparison_episode_3d_dir(cls, mode: str, algorithms) -> Path:
        """【新增】为 single_3d（纯 3D 视图）单独生成并确保目录存在"""
        p = cls.comparison_episode_root(mode, algorithms) / "single_3d"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def comparison_episode_combo_png(cls, mode: str, algorithms, episode: int) -> Path:
        """【修改】去掉了末尾错误的放置，移到这里，并使用正确的 combo 目录"""
        return cls.comparison_episode_combo_dir(mode, algorithms) / f"episode_{int(episode):04d}_combo.png"

    @classmethod
    def comparison_episode_3d_png(cls, mode: str, algorithms, episode: int) -> Path:
        """【新增】专门用于获取 3D 轨迹对比图的保存路径"""
        return cls.comparison_episode_3d_dir(mode, algorithms) / f"episode_{int(episode):04d}_3d.png"

    # ──────────────────────────────────────────────────────────────────────────
    # 【修改区域结束】
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def summary_csv_for(cls, algo_name: str, mode: str) -> Path:
        return cls.algorithm_results_dir(algo_name) / "training_data" / "summary" / f"{mode}_summary.csv"

    @classmethod
    def stat_csv_for(cls, algo_name: str, mode: str) -> Path:
        return cls.algorithm_results_dir(algo_name) / "training_data" / "summary" / f"{mode}_stat.csv"

    @classmethod
    def traj_csv_for(cls, algo_name: str, mode: str, ep: int) -> Path:
        return cls.algorithm_results_dir(algo_name) / "training_data" / "path_data" / mode / f"trajectory_{ep:04d}.csv"

    @classmethod
    def step_csv_for(cls, algo_name: str, mode: str, ep: int) -> Path:
        return cls.algorithm_results_dir(algo_name) / "training_data" / "step_data" / mode / f"step_{ep:04d}.csv"

    @classmethod
    def update_csv_for(cls, algo_name: str, mode: str, ep: int) -> Path:
        return cls.algorithm_results_dir(algo_name) / "training_data" / "update_data" / mode / f"update_{ep:04d}.csv"

    @classmethod
    def figure_png_for(cls, algo_name: str, mode: str, ep: int) -> Path:
        return cls.algorithm_results_dir(algo_name) / "training_data" / "path_picture" / mode / f"episode_{ep:04d}.png"

    @classmethod
    def update_algo(cls, algo_name: str) -> None:
        """根据算法名称动态切换结果保存目录。"""
        MapConfig.algo_name = algo_name
        cls.results = cls.algorithm_results_dir(algo_name)
        cls.net     = cls.results / "net_storage"
        cls.train   = cls.results / "training_data"
        cls.traj    = cls.train   / "path_data"
        cls.fig     = cls.train   / "path_picture"
        cls.step    = cls.train   / "step_data"
        cls.update  = cls.train   / "update_data"
        cls.summary = cls.train   / "summary"
        cls.current = cls.train   / "current_field"

    @classmethod
    def make_dirs(cls) -> None:
        if cls.results is None:
            cls.update_algo(MapConfig.algo_name)

        for p in [
            cls.results_root(), cls.results, cls.net, cls.train,
            cls.traj, cls.fig, cls.step, cls.update,
            cls.summary, cls.current, cls.cmems,
        ]:
            p.mkdir(parents=True, exist_ok=True)

    @classmethod
    def ep_dir(cls, mode: str) -> Path:
        p = cls.fig / mode
        p.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def traj_csv(cls, mode: str, ep: int) -> Path:
        p = cls.traj / mode
        p.mkdir(parents=True, exist_ok=True)
        return p / f"trajectory_{ep:04d}.csv"

    @classmethod
    def step_csv(cls, mode: str, ep: int) -> Path:
        p = cls.step / mode
        p.mkdir(parents=True, exist_ok=True)
        return p / f"step_{ep:04d}.csv"

    @classmethod
    def update_csv(cls, mode: str, ep: int) -> Path:
        p = cls.update / mode
        p.mkdir(parents=True, exist_ok=True)
        return p / f"update_{ep:04d}.csv"

    @classmethod
    def ep_png(cls, mode: str, ep: int, step: int | None = None) -> Path:
        p = cls.ep_dir(mode)
        return p / (f"episode_{ep:04d}.png" if step is None else f"step_{step:04d}.png")

    @classmethod
    def current_png(cls, tag: str = "default") -> Path:
        cls.current.mkdir(parents=True, exist_ok=True)
        return cls.current / f"current_{tag}.png"

    @classmethod
    def current_ep_png(cls, mode: str, ep: int) -> Path:
        p = cls.current / mode
        p.mkdir(parents=True, exist_ok=True)
        return p / f"current_episode_{ep:04d}.png"

    @classmethod
    def summary_csv(cls, mode: str) -> Path:
        cls.summary.mkdir(parents=True, exist_ok=True)
        return cls.summary / f"{mode}_summary.csv"

    @classmethod
    def stat_csv(cls, mode: str) -> Path:
        cls.summary.mkdir(parents=True, exist_ok=True)
        return cls.summary / f"{mode}_stat.csv"

    @classmethod
    def curve_png(cls, mode: str) -> Path:
        cls.summary.mkdir(parents=True, exist_ok=True)
        return cls.summary / f"{mode}_metrics.png"

# 初始化全局算法目录
FileAddress.update_algo(MapConfig.algo_name)


# ══════════════════════════════════════════════════════════════════════════════
# 可视化配置
# ══════════════════════════════════════════════════════════════════════════════

class VisualizationConfig:
    dpi          = 300
    episode_size = (24, 8)
    comparison_size = (24, 8)
    current_size = (18, 10)
    reward_size  = (10, 6)     # 单张图表尺寸优化
    metrics_size = (10, 6)
    boxplot_size = (8, 6)
    single_3d_size = (10, 8)

    metric_raw_alpha = 0.15
    metric_line_alpha = 0.95
    metric_linewidth = 2.2
    metric_line_alpha_raw = metric_raw_alpha
    metric_line_alpha_smooth = metric_line_alpha
    metric_line_width = metric_linewidth

    box_face_alpha = 0.90
    box_edge_color = "#333333"
    box_median_color = "#1C1C1C"
    box_flier_color = "#404040"
    box_flier_marker = "d"
    box_flier_size = 5
    box_grid_alpha = 0.25
    metric_box_size = boxplot_size

    tick_num   = 5
    grid_3d    = 30 
    obstacle_u = 36
    obstacle_v = 18
    current_stride_3d = 2 

    arrow_scale_3d = 16.0
    arrow_scale_2d = 18.0
    arrow_alpha_3d = 0.75
    arrow_alpha_2d = 0.55
    arrow_width_3d = 0.35
    arrow_width_2d = 0.0022
    arrow_ratio_3d = 0.18
    global_v_max = MapConfig.current_clip
    cmap = "viridis"

    traj_width_3d = 2.6
    traj_width_2d = 2.1
    start_size    = 120
    target_size   = 180
    pos_size      = 70
    hit_size      = 90

    # ── 统一颜色管理 ────────────────────────────────────────────────────────
    c_obs_face  = "#C8D3DE"
    c_obs_edge  = "#4A5568"
    c_col_bound = "#8B5CF6"
    c_col_obs   = "#EF4444"
    c_start_fc  = "#F59E0B"
    c_start_ec  = "#1C1C1C"
    c_target_fc = "#FACC15"
    c_target_ec = "#1C1C1C"
    c_vortex_fc = "#E74C3C"
    c_vortex_ec = "#922B21"
    c_traj_train= "#C0392B"
    c_traj_eval = "#1D6FA4"

    # 新版 2D 投影箭头背景设置
    speed_background_alpha = 0.96
    speed_background_levels = 28
    arrow_color_2d_background = "#111111"
    arrow_alpha_2d_background = 0.88
    arrow_width_2d_background = max(arrow_width_2d * 0.9, 0.0016)

    algorithm_colors = {
        "TD3-PER": "#D62728",
        "TD3": "#1F77B4",
        "DDPG": "#2CA02C",
    }
    fallback_colors = ["#8E44AD", "#16A085", "#E67E22", "#7F7F7F", "#BCBD22", "#17BECF"]

    equal_axis = True
    invert_z   = True
    colorbar   = True
    show_speed_background = False  # 是否启用 2D 热力底色

    reward_window_train = 50
    reward_window_eval  = 10
    success_window_train = 100
    success_window_eval  = 10

    traj_dot_gap  = 20
    save_ep_fig   = True
    save_current_fig = False

    @classmethod
    def get_algorithm_color(cls, algo_name: str, index: int = 0) -> str:
        if algo_name in cls.algorithm_colors:
            return cls.algorithm_colors[algo_name]
        fallback = list(getattr(cls, "fallback_colors", ["#8E44AD", "#16A085", "#E67E22", "#7F7F7F"]))
        return fallback[index % len(fallback)]

