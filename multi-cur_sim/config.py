import math
import datetime
from pathlib import Path
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# 文件路径
# ══════════════════════════════════════════════════════════════════════════════

class FileAddress:
    root    = Path(__file__).resolve().parent
    results = root / "results"

    net     = results / "net_storage"
    train   = results / "training_data"
    traj    = train   / "path_data"
    fig     = train   / "path_picture"
    step    = train   / "step_data"
    update  = train   / "update_data"
    summary = train   / "summary"
    current = train   / "current_field"
    cmems   = root    / "cmems_data"

    @classmethod
    def make_dirs(cls) -> None:
        for p in [
            cls.results, cls.net, cls.train,
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
        return cls.summary / f"{mode}_reward_success.png"


# ══════════════════════════════════════════════════════════════════════════════
# 地图 / 环境配置
# ══════════════════════════════════════════════════════════════════════════════

class MapConfig:

    # ── Flag_Initialization ──────────────────────────────────────────────────
    start_fixed = True
    target_fixed = True

    # ── 运行模式 ──────────────────────────────────────────────────────────────
    mode      = "train"   # "train" | "eval"
    show_current = True
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
    cmems_horiz_file = str(FileAddress.cmems / horiz)
    cmems_vert_file  = str(FileAddress.cmems / vert)

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
    # r_step    = -0.02
    # r_bound   = -150.0
    r_lamda1  = 1.25
    r_lamda2  = 0.80
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

    gamma                 = 0.99
    actor_lr              = 3e-4
    critic_lr             = 3e-4
    tau                   = 0.005
    batch                 = 64
    buffer                = int(1e6)
    actor_update_interval = 2
    target_noise_clip     = 0.5
    target_noise          = 0.2
    explore_noise         = 0.2
    save_gap              = 50


# ══════════════════════════════════════════════════════════════════════════════
# 可视化配置
# ══════════════════════════════════════════════════════════════════════════════

class VisualizationConfig:
    dpi          = 300
    episode_size = (24, 8)
    # episode_size = (18, 10)
    current_size = (18, 10)
    reward_size  = (12, 8)

    tick_num   = 5
    grid_3d    = 30 # 3D图网格分辨率
    obstacle_u = 36
    obstacle_v = 18
    current_stride_3d = 2

    arrow_scale_3d = 16.0
    arrow_scale_2d = 18.0
    arrow_alpha_3d = 0.75
    # arrow_alpha_3d = 0.28
    arrow_alpha_2d = 0.55
    arrow_width_3d = 0.35
    arrow_width_2d = 0.0022
    arrow_ratio_3d = 0.18
    global_v_max = MapConfig.current_clip
    cmap           = "viridis"

    traj_width_3d = 2.6
    traj_width_2d = 2.1
    start_size    = 120
    target_size   = 180
    pos_size      = 70
    hit_size      = 90

    equal_axis = True
    invert_z   = True
    colorbar   = True

    reward_window_train = 50
    reward_window_eval  = 10

    step_gap      = 50
    save_ep_fig   = True
    save_current_fig = False # 每个episode保存当前海流环境照片
