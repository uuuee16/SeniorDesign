"""
visualization.py  ──  学术品质可视化模块
==========================================
坐标系约定（全文统一，NED 坐标系）：
    pos[0] → East  (m)   X 轴
    pos[1] → North (m)   Y 轴
    pos[2] → Down  (m)   Z 轴（翻转：0 在顶，增大方向向下）

图布局（save_episode_combo_figure）：
    +────────────────────+──────────────+
    │                    │  XY 俯视图   │
    │    3D 轨迹图        ├──────────────┤
    │                    │  XZ 侧视图   │
    +────────────────────+──────────────+
    ╰─────── 海流色标 (底部水平 colorbar) ──────╯
"""
from __future__ import annotations

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from config import VisualizationConfig as Vc

# ══════════════════════════════════════════════════════════════════════════════
# 颜色 / 样式常量
# ══════════════════════════════════════════════════════════════════════════════

OBS_FACE  = "#C8D3DE"   # 障碍物填充
OBS_EDGE  = "#4A5568"   # 障碍物边框
COL_BOUND = "#8B5CF6"   # 越界碰撞 (紫)
COL_OBS   = "#EF4444"   # 障碍物碰撞 (红)

# 起点：金色填充 + 深色边框
_START_FC  = "#F59E0B"
_START_EC  = "#1C1C1C"
# 目标：荧光黄 + 深色边框
_TARGET_FC = "#FACC15"
_TARGET_EC = "#1C1C1C"

# 终止点：橙色菱形
# _TERM_FC   = "#FB923C"
# 涡旋中心标记：红色五角星（显眼且易区分）
_VORTEX_FC = "#E74C3C"    # 填充色（鲜红）
_VORTEX_EC = "#922B21"    # 边框色（深红）

# 轨迹颜色
_TRAJ_TRAIN = "#C0392B"
_TRAJ_EVAL  = "#1D6FA4"

# 轨迹打点间隔（每 N 步一个点）
_DOT_GAP = getattr(Vc, "traj_dot_gap", 20)


def _traj_color(mode: str) -> str:
    return _TRAJ_TRAIN if mode == "train" else _TRAJ_EVAL


# ══════════════════════════════════════════════════════════════════════════════
# 海流场采样
# ══════════════════════════════════════════════════════════════════════════════

def _current_batch(current, pos):
    if hasattr(current, "get_current_at_positions"):
        return np.asarray(current.get_current_at_positions(pos), dtype=np.float32)
    return np.asarray([current.get_current_at_position(p) for p in pos], dtype=np.float32)


def sample_current_field(current, bounds, n=None):
    n = n or Vc.grid_3d
    x0, x1, y0, y1, z0, z1 = bounds
    x = np.linspace(x0, x1, n, dtype=np.float32)
    y = np.linspace(y0, y1, n, dtype=np.float32)
    z = np.linspace(z0, z1, n, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    vel = _current_batch(current, pos)
    return {
        "x": x, "y": y, "z": z,
        "X": X, "Y": Y, "Z": Z,
        "u": vel[:, 0].reshape(X.shape),
        "v": vel[:, 1].reshape(X.shape),
        "w": vel[:, 2].reshape(X.shape),
        "speed": np.linalg.norm(vel, axis=1).reshape(X.shape),
    }


def _nearest_idx(values, ref=None):
    return len(values) // 2 if ref is None else int(np.argmin(np.abs(values - ref)))


def extract_xy_slice(field, z_ref=None):
    """提取指定深度 z_ref 处的 XY 截面海流。"""
    k = _nearest_idx(field["z"], z_ref)
    X, Y = np.meshgrid(field["x"], field["y"], indexing="ij")
    u, v = field["u"][:, :, k], field["v"][:, :, k]
    return {"X": X, "Y": Y, "u": u, "v": v,
            "speed": np.sqrt(u*u + v*v), "z_ref": float(field["z"][k])}


def extract_xz_slice(field, y_ref=None):
    """提取指定 y_ref 处的 XZ 截面海流。"""
    j = _nearest_idx(field["y"], y_ref)
    X, Z = np.meshgrid(field["x"], field["z"], indexing="ij")
    u, w = field["u"][:, j, :], field["w"][:, j, :]
    return {"X": X, "Z": Z, "u": u, "w": w,
            "speed": np.sqrt(u*u + w*w), "y_ref": float(field["y"][j])}


# ══════════════════════════════════════════════════════════════════════════════
# 坐标轴设置（统一 NED 标签）
# ══════════════════════════════════════════════════════════════════════════════

def set_equal_axis(ax, bounds):
    """3D 轴设置：坐标标签 East / North / Down，Z 轴可选翻转。"""
    x0, x1, y0, y1, z0, z1 = bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z1)
    ax.set_xlabel("East (m)",  labelpad=6, fontsize=10, color="#1A6496", fontweight="bold")
    ax.set_ylabel("North (m)", labelpad=6, fontsize=10, color="#196F3D", fontweight="bold")
    ax.set_zlabel("Down (m)",  labelpad=6, fontsize=10, color="#922B21", fontweight="bold")
    ticks = Vc.tick_num
    ax.set_xticks(np.linspace(x0, x1, ticks))
    ax.set_yticks(np.linspace(y0, y1, ticks))
    ax.set_zticks(np.linspace(z0, z1, ticks))
    if Vc.equal_axis:
        ax.set_box_aspect((x1-x0, y1-y0, z1-z0))
    if Vc.invert_z:
        ax.invert_zaxis()
    ax.grid(True, linestyle="--", alpha=0.22)
    ax.tick_params(labelsize=8)


def _setup_xy_axis(ax, env, z_ref=None):
    ax.set_xlim(env.x0, env.x1)
    ax.set_ylim(env.y0, env.y1)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.30)
    ax.set_xlabel("East (m)",  fontsize=10)
    ax.set_ylabel("North (m)", fontsize=10)
    ax.tick_params(labelsize=8)
    subtitle = "XOY Plain"
    if z_ref is not None:
        subtitle += f"  ·  current at z = {z_ref:.0f} m"
    ax.set_title(subtitle, fontsize=11)


def _setup_xz_axis(ax, env, y_ref=None):
    ax.set_xlim(env.x0, env.x1)
    ax.set_ylim(env.z0, env.z1)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.30)
    ax.set_xlabel("East (m)",  fontsize=10)
    ax.set_ylabel("Down (m)",  fontsize=10)
    ax.tick_params(labelsize=8)
    if Vc.invert_z:
        ax.invert_yaxis()
    subtitle = "XOZ Plain"
    if y_ref is not None:
        subtitle += f"  ·  current at y = {y_ref:.0f} m"
    ax.set_title(subtitle, fontsize=11)


# ══════════════════════════════════════════════════════════════════════════════
# 障碍物绘制
# ══════════════════════════════════════════════════════════════════════════════

def draw_obstacles_3d(ax, obstacles):
    u_grid, v_grid = np.mgrid[
        0 : 2*np.pi : complex(0, Vc.obstacle_u),
        0 : np.pi   : complex(0, Vc.obstacle_v),
    ]
    for ox, oy, oz, r in obstacles:
        ax.plot_surface(
            ox + r * np.cos(u_grid) * np.sin(v_grid),
            oy + r * np.sin(u_grid) * np.sin(v_grid),
            oz + r * np.cos(v_grid),
            color=OBS_FACE, alpha=0.28, edgecolor=OBS_EDGE, linewidth=0.20,
        )

def draw_obstacles_xy(ax, obstacles):
    for ox, oy, _, r in obstacles:
        ax.add_patch(Circle((ox, oy), r,
                            edgecolor=OBS_EDGE, facecolor=OBS_FACE, alpha=0.38, linewidth=1.1))


def draw_obstacles_xz(ax, obstacles):
    for ox, _, oz, r in obstacles:
        ax.add_patch(Circle((ox, oz), r,
                            edgecolor=OBS_EDGE, facecolor=OBS_FACE, alpha=0.38, linewidth=1.1))


# ══════════════════════════════════════════════════════════════════════════════
# 海流绘制（统一接受外部 norm，保证三图颜色比例尺一致）
# ══════════════════════════════════════════════════════════════════════════════

def draw_current_3d(ax, field, bounds, norm=None):
    stride = max(1, int(Vc.current_stride_3d))
    X = field["X"][::stride, ::stride, ::stride]
    Y = field["Y"][::stride, ::stride, ::stride]
    Z = field["Z"][::stride, ::stride, ::stride]
    U = field["u"][::stride, ::stride, ::stride]
    V = field["v"][::stride, ::stride, ::stride]
    W = field["w"][::stride, ::stride, ::stride]
    S = field["speed"][::stride, ::stride, ::stride]

    if norm is None:
        norm = Normalize(vmin=0.0, vmax=max(float(np.max(S)), 1e-6))

    cmap_fn = cm.get_cmap(Vc.cmap)
    colors  = cmap_fn(norm(S.ravel()))

    scene = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
    scale = scene / max(X.shape) * 0.45 * Vc.arrow_scale_3d / 10.0

    ax.quiver(
        X.ravel(), Y.ravel(), Z.ravel(),
        U.ravel(), V.ravel(), W.ravel(),
        colors=colors,
        length=scale, normalize=True,
        alpha=Vc.arrow_alpha_3d,
        linewidth=Vc.arrow_width_3d,
        arrow_length_ratio=Vc.arrow_ratio_3d,
    )
    sm = cm.ScalarMappable(cmap=Vc.cmap, norm=norm)
    sm.set_array([])
    return sm


def draw_current_xy(ax, field_xy, bounds, norm=None):
    if norm is None:
        norm = Normalize(vmin=0.0, vmax=max(float(np.max(field_xy["speed"])), 1e-6))
    scene = max(bounds[1]-bounds[0], bounds[3]-bounds[2])
    base  = scene / field_xy["X"].shape[0] / 1.8
    u = field_xy["u"] / np.maximum(field_xy["speed"], 1e-9) * base * Vc.arrow_scale_2d / 10.0
    v = field_xy["v"] / np.maximum(field_xy["speed"], 1e-9) * base * Vc.arrow_scale_2d / 10.0
    return ax.quiver(
        field_xy["X"], field_xy["Y"], u, v, field_xy["speed"],
        cmap=Vc.cmap, norm=norm,
        angles="xy", scale_units="xy", scale=1.0,
        alpha=Vc.arrow_alpha_2d, width=Vc.arrow_width_2d,
    )


def draw_current_xz(ax, field_xz, bounds, norm=None):
    if norm is None:
        norm = Normalize(vmin=0.0, vmax=max(float(np.max(field_xz["speed"])), 1e-6))
    scene = max(bounds[1]-bounds[0], bounds[5]-bounds[4])
    base  = scene / field_xz["X"].shape[0] / 1.8
    u = field_xz["u"] / np.maximum(field_xz["speed"], 1e-9) * base * Vc.arrow_scale_2d / 10.0
    w = field_xz["w"] / np.maximum(field_xz["speed"], 1e-9) * base * Vc.arrow_scale_2d / 10.0
    return ax.quiver(
        field_xz["X"], field_xz["Z"], u, w, field_xz["speed"],
        cmap=Vc.cmap, norm=norm,
        angles="xy", scale_units="xy", scale=1.0,
        alpha=Vc.arrow_alpha_2d, width=Vc.arrow_width_2d,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 轨迹绘制（线 + 等间距打点 + 终止标记）
# ══════════════════════════════════════════════════════════════════════════════

def _traj_dot_indices(n: int) -> list[int]:
    """每 _DOT_GAP 步取一个打点索引（不含首尾）。"""
    return list(range(_DOT_GAP, n - 1, _DOT_GAP))


def _plot_traj_3d(ax, traj: np.ndarray, mode: str):
    if len(traj) < 2:
        return
    color = _traj_color(mode)
    # 轨迹线
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
            color=color, linewidth=Vc.traj_width_3d, alpha=0.90,
            label="AUV Trajectory", zorder=4)
    # 时间打点
    idx = _traj_dot_indices(len(traj))
    if idx:
        ax.scatter(traj[idx, 0], traj[idx, 1], traj[idx, 2],
                   c=color, s=Vc.pos_size * 0.45, alpha=0.65,
                   edgecolors="none", zorder=5)
    # # 终止点（菱形，橙色）
    # ax.scatter(*traj[-1], c=_TERM_FC, s=Vc.pos_size * 1.2, marker="D",
    #            edgecolors="#1C1C1C", linewidths=1.3, zorder=7, label="Terminal")


def _plot_traj_xy(ax, traj: np.ndarray, mode: str):
    if len(traj) < 2:
        return
    color = _traj_color(mode)
    ax.plot(traj[:, 0], traj[:, 1],
            color=color, linewidth=Vc.traj_width_2d, alpha=0.90,
            label="AUV Trajectory", zorder=4)
    idx = _traj_dot_indices(len(traj))
    if idx:
        ax.scatter(traj[idx, 0], traj[idx, 1],
                   c=color, s=Vc.pos_size * 0.35, alpha=0.65,
                   edgecolors="none", zorder=5)
    # ax.scatter(traj[-1, 0], traj[-1, 1],
    #            c=_TERM_FC, s=Vc.pos_size * 1.1, marker="D",
    #            edgecolors="#1C1C1C", linewidths=1.1, zorder=7, label="Terminal")


def _plot_traj_xz(ax, traj: np.ndarray, mode: str):
    if len(traj) < 2:
        return
    color = _traj_color(mode)
    ax.plot(traj[:, 0], traj[:, 2],
            color=color, linewidth=Vc.traj_width_2d, alpha=0.90,
            label="AUV Trajectory", zorder=4)
    idx = _traj_dot_indices(len(traj))
    if idx:
        ax.scatter(traj[idx, 0], traj[idx, 2],
                   c=color, s=Vc.pos_size * 0.35, alpha=0.65,
                   edgecolors="none", zorder=5)
    # ax.scatter(traj[-1, 0], traj[-1, 2],
    #            c=_TERM_FC, s=Vc.pos_size * 1.1, marker="D",
    #            edgecolors="#1C1C1C", linewidths=1.1, zorder=7, label="Terminal")


# ══════════════════════════════════════════════════════════════════════════════
# 起点 / 目标标记
# ══════════════════════════════════════════════════════════════════════════════

def _draw_start_target_3d(ax, env):
    ax.scatter(*env.start,
               c=_START_FC, s=Vc.start_size, marker="o",
               edgecolors=_START_EC, linewidths=2.2, zorder=8, label="Start")
    ax.scatter(*env.target,
               c=_TARGET_FC, s=Vc.target_size, marker="*",
               edgecolors=_TARGET_EC, linewidths=1.5, zorder=8, label="Target")


def _draw_start_target_xy(ax, env):
    ax.scatter(env.start[0], env.start[1],
               c=_START_FC, s=Vc.start_size * 0.85, marker="o",
               edgecolors=_START_EC, linewidths=2.0, zorder=8, label="Start")
    ax.scatter(env.target[0], env.target[1],
               c=_TARGET_FC, s=Vc.target_size * 0.85, marker="*",
               edgecolors=_TARGET_EC, linewidths=1.4, zorder=8, label="Target")


def _draw_start_target_xz(ax, env):
    ax.scatter(env.start[0], env.start[2],
               c=_START_FC, s=Vc.start_size * 0.85, marker="o",
               edgecolors=_START_EC, linewidths=2.0, zorder=8, label="Start")
    ax.scatter(env.target[0], env.target[2],
               c=_TARGET_FC, s=Vc.target_size * 0.85, marker="*",
               edgecolors=_TARGET_EC, linewidths=1.4, zorder=8, label="Target")


# ══════════════════════════════════════════════════════════════════════════════
# 碰撞点
# ══════════════════════════════════════════════════════════════════════════════

def _draw_collision_points(ax3d, ax_xy, ax_xz, collision_points):
    bound_pts = np.asarray(collision_points.get("boundary", []), dtype=np.float32)
    obs_pts   = np.asarray(collision_points.get("obstacle",  []), dtype=np.float32)

    for pts, color, label in [
        (bound_pts, COL_BOUND, "Boundary Hit"),
        (obs_pts,   COL_OBS,   "Obstacle Hit"),
    ]:
        if pts.size == 0:
            continue
        if ax3d is not None:
            ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                         c=color, s=Vc.hit_size, marker="X",
                         edgecolors="white", linewidths=0.8, zorder=9, label=label)
        if ax_xy is not None:
            ax_xy.scatter(pts[:, 0], pts[:, 1],
                          c=color, s=Vc.hit_size, marker="X",
                          edgecolors="white", linewidths=0.8, zorder=9, label=label)
        if ax_xz is not None:
            ax_xz.scatter(pts[:, 0], pts[:, 2],
                          c=color, s=Vc.hit_size, marker="X",
                          edgecolors="white", linewidths=0.8, zorder=9, label=label)
            

# ══════════════════════════════════════════════════════════════════════════════
# 涡旋中心绘制（新增函数）
# ══════════════════════════════════════════════════════════════════════════════


def _get_vortex_center(v):
    """辅助函数：兼容字典格式和元组格式的涡旋坐标提取"""
    return v["center"] if isinstance(v, dict) else v

def _draw_vortices_3d(ax, vortices):
    """
    在3D轴上绘制涡旋中心点。
    """
    if not vortices:
        return
    # 提取中心坐标，通过 _get_vortex_center 兼容你的字典定义
    xs, ys, zs = zip(*[_get_vortex_center(v) for v in vortices])
    ax.scatter(
        xs, ys, zs,
        c=_VORTEX_FC, s=Vc.start_size, marker="D",
        edgecolors=_VORTEX_EC, linewidths=2.0, zorder=10,  # zorder确保在最上层
        label="Vortex Center"
    )

def _draw_vortices_xy(ax, vortices):
    """在XY俯视图上绘制涡旋中心投影"""
    if not vortices:
        return
    centers = [_get_vortex_center(v) for v in vortices]
    xs, ys = zip(*[(c[0], c[1]) for c in centers])
    ax.scatter(
        xs, ys,
        c=_VORTEX_FC, s=Vc.start_size * 0.8, marker="D",
        edgecolors=_VORTEX_EC, linewidths=1.8, zorder=10,
        label="Vortex Center"
    )

def _draw_vortices_xz(ax, vortices):
    """在XZ侧视图上绘制涡旋中心投影"""
    if not vortices:
        return
    centers = [_get_vortex_center(v) for v in vortices]
    xs, zs = zip(*[(c[0], c[2]) for c in centers])
    ax.scatter(
        xs, zs,
        c=_VORTEX_FC, s=Vc.start_size * 0.8, marker="D",
        edgecolors=_VORTEX_EC, linewidths=1.8, zorder=10,
        label="Vortex Center"
    )

# ══════════════════════════════════════════════════════════════════════════════
# 图例辅助
# ══════════════════════════════════════════════════════════════════════════════

def _add_legend(ax, is_3d=False, fontsize=9, loc="upper left"):
    """
    统一风格图例，只在有 label 时添加。
    可通过 loc 参数指定图例位置，支持:
    'upper left' (左上), 'upper right' (右上),
    'lower left' (左下), 'lower right' (右下), 
    或 'best' (自动寻找避开数据点的最佳位置)
    """
    try:
        handles, labels = ax.get_legend_handles_labels()
    except Exception:
        return
    if not handles:
        return
    # 去重（matplotlib 有时会重复）
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
            
    ax.legend(
        list(seen.values()), list(seen.keys()),
        loc=loc,  # <--- 在这里使用传入的参数
        fontsize=fontsize,
        framealpha=0.82,
        edgecolor="#CCCCCC",
        borderpad=0.5,
        handlelength=1.4,
        handletextpad=0.5,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 三面板布局
# ══════════════════════════════════════════════════════════════════════════════

def _build_three_panel_figure():
    """
    左侧大 3D 轴（占全高），右侧上下各一个 2D 轴。
    为底部 colorbar 预留 8% 空间。
    """
    fig = plt.figure(figsize=Vc.episode_size, facecolor="white")
    gs = gridspec.GridSpec(
        1, 3,          # 单行三列
        left=0.04, right=0.97,
        bottom=0.12, top=0.88,
        wspace=0.28,
    )
    ax3d  = fig.add_subplot(gs[0, 0], projection="3d")
    ax_xy = fig.add_subplot(gs[0, 1])
    ax_xz = fig.add_subplot(gs[0, 2])
    return fig, ax3d, ax_xy, ax_xz


# ══════════════════════════════════════════════════════════════════════════════
# 主回合图（runner.py 调用）
# ══════════════════════════════════════════════════════════════════════════════

def save_episode_combo_figure(
    env,
    traj,
    collision_points,
    ep_reward,
    time_s,
    success,
    path_len,
    mode,
    save_path,
    field=None,
    ep=None,
):
    """
    保存三面板回合轨迹图（3D + XY 俯视 + XZ 侧视）。

    Parameters
    ----------
    env              : AUVEnv 实例
    traj             : ndarray (N, 3)  轨迹点序列
    collision_points : dict {"boundary": [...], "obstacle": [...]}
    ep_reward        : 本回合总奖励
    time_s           : 仿真时长（秒）
    success          : 是否成功
    path_len         : 路径长度（米）
    mode             : "train" | "eval"
    save_path        : 保存路径
    field            : sample_current_field() 返回值（可为 None）
    ep               : 回合编号（可为 None，用于标题）
    """
    traj = np.asarray(traj, dtype=np.float32)
    if len(traj) == 0:
        return

    # 以轨迹均值决定二维切片参考位置
    z_ref = float(np.mean(traj[:, 2]))
    y_ref = float(np.mean(traj[:, 1]))

    # ── 全局色标（三图共用） ──────────────────────────────────────────────────
    norm = Normalize(vmin=0.0, vmax=Vc.global_v_max)

    # ── 图布局 ────────────────────────────────────────────────────────────────
    fig, ax3d, ax_xy, ax_xz = _build_three_panel_figure()

    # ── 海流 ──────────────────────────────────────────────────────────────────
    sm = None
    if field is not None:
        sm = draw_current_3d(ax3d, field, env.env_bound, norm=norm)
        field_xy = extract_xy_slice(field, z_ref)
        field_xz = extract_xz_slice(field, y_ref)
        draw_current_xy(ax_xy, field_xy, env.env_bound, norm=norm)
        draw_current_xz(ax_xz, field_xz, env.env_bound, norm=norm)
    else:
        field_xy = field_xz = None

    # ── 障碍物 ────────────────────────────────────────────────────────────────
    draw_obstacles_3d(ax3d, env.obstacles)
    draw_obstacles_xy(ax_xy, env.obstacles)
    draw_obstacles_xz(ax_xz, env.obstacles)

    # ── 轨迹（线 + 等间距打点 + 终止标记） ────────────────────────────────────
    _plot_traj_3d(ax3d, traj, mode)
    _plot_traj_xy(ax_xy, traj, mode)
    _plot_traj_xz(ax_xz, traj, mode)

    # ── 碰撞点 ────────────────────────────────────────────────────────────────
    _draw_collision_points(ax3d, ax_xy, ax_xz, collision_points)

    # ── 起点 / 目标（最后绘制，保证 zorder 在最顶层） ─────────────────────────
    _draw_start_target_3d(ax3d, env)
    _draw_start_target_xy(ax_xy, env)
    _draw_start_target_xz(ax_xz, env)

    # 安全获取 vortices 数据，这样无论是 real、synthetic 还是 none 模式都不会报错
    vortices_data = getattr(env.current, "vortices", getattr(env, "vortices", []))
    _draw_vortices_3d(ax3d, vortices_data)
    _draw_vortices_xy(ax_xy, vortices_data)
    _draw_vortices_xz(ax_xz, vortices_data)

    # ── 3D 轴设置 ─────────────────────────────────────────────────────────────
    set_equal_axis(ax3d, env.env_bound)
    ax3d.view_init(elev=28, azim=-50)
    ax3d.set_title("3D Trajectory", fontsize=12, pad=8)

    # ── 2D 轴设置 ─────────────────────────────────────────────────────────────
    z_label = None if field_xy is None else field_xy.get("z_ref")
    y_label = None if field_xz is None else field_xz.get("y_ref")
    _setup_xy_axis(ax_xy, env, z_label)
    _setup_xz_axis(ax_xz, env, y_label)

    # ── Colorbar（全局共享，底部水平） ─────────────────────────────────────────
    if Vc.colorbar and sm is not None:
        cax = fig.add_axes([0.15, 0.04, 0.70, 0.025])
        cb  = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label("Ocean Current Speed (m/s)", fontsize=10)
        cb.ax.tick_params(labelsize=9)

    # ── 图例 ──────────────────────────────────────────────────────────────────
    _add_legend(ax3d, is_3d=True, fontsize=9)
    _add_legend(ax_xy, fontsize=8)
    _add_legend(ax_xz, fontsize=8, loc="best")

    # ── 主标题（单行，包含所有回合信息） ─────────────────────────────────────
    ep_str    = f"Episode {ep:04d}   |   " if ep is not None else ""
    status    = "✓" if success else "✗"
    suptitle  = (
        f"[{mode.upper()}]   {ep_str}"
        f"Reward: {ep_reward:+.2f}   |   "
        f"Time: {time_s:.1f} s   |   "
        f"Path: {path_len:.1f} m   |   "
        f"Success: {status}"
    )
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=0.97,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F4F8",
                           edgecolor="#B0BEC5", alpha=0.85))

    fig.savefig(save_path, dpi=Vc.dpi, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 海流场可视化（独立，runner / main 调用）
# ══════════════════════════════════════════════════════════════════════════════


def visualize_current_environment(
    env,
    save_path=None,
    title="Ocean Current Field",
    field=None,
    z_ref=None,
    y_ref=None,
):
    """渲染当前海流场（三面板 + 单独的3D视图），无轨迹。"""
    if not env.current_on or env.current is None:
        raise ValueError("visualize_current_environment requires current_on=True.")

    field    = field or sample_current_field(env.current, env.env_bound, Vc.grid_3d)
    field_xy = extract_xy_slice(field, z_ref)
    field_xz = extract_xz_slice(field, y_ref)

    norm = Normalize(vmin=0.0, vmax=Vc.global_v_max)
    
    # ════════════════════════════════════════════════════════════════════════
    # 第一部分：绘制原有的三面板图 (3D + XY + XZ)
    # ════════════════════════════════════════════════════════════════════════
    fig, ax3d, ax_xy, ax_xz = _build_three_panel_figure()

    sm = draw_current_3d(ax3d, field, env.env_bound, norm=norm)
    draw_obstacles_3d(ax3d, env.obstacles)
    _draw_start_target_3d(ax3d, env)
    set_equal_axis(ax3d, env.env_bound)
    ax3d.view_init(elev=28, azim=-50)
    ax3d.set_title("3D Current Field", fontsize=12, pad=8)

    draw_current_xy(ax_xy, field_xy, env.env_bound, norm=norm)
    draw_obstacles_xy(ax_xy, env.obstacles)
    _draw_start_target_xy(ax_xy, env)
    _setup_xy_axis(ax_xy, env, field_xy["z_ref"])

    draw_current_xz(ax_xz, field_xz, env.env_bound, norm=norm)
    draw_obstacles_xz(ax_xz, env.obstacles)
    _draw_start_target_xz(ax_xz, env)
    _setup_xz_axis(ax_xz, env, field_xz["y_ref"])

    # 增加vortex绘制 (兼容获取方式)
    # 替换为这行（调换了 env.current 和 env 的顺序）：
    vortices_data = getattr(env.current, "vortices", getattr(env, "vortices", []))
    
    _draw_vortices_3d(ax3d, vortices_data)
    _draw_vortices_xy(ax_xy, vortices_data)
    _draw_vortices_xz(ax_xz, vortices_data)

    if Vc.colorbar:
        cax = fig.add_axes([0.15, 0.04, 0.70, 0.025])
        cb  = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label("Ocean Current Speed (m/s)", fontsize=10)
        cb.ax.tick_params(labelsize=9)

    _add_legend(ax3d, is_3d=True, fontsize=9)
    _add_legend(ax_xy, fontsize=8)
    _add_legend(ax_xz, fontsize=8, loc="best")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.97)

    # ════════════════════════════════════════════════════════════════════════
    # 第二部分：绘制一张独立的放大版 3D 视图
    # ════════════════════════════════════════════════════════════════════════
    fig_3d = plt.figure(figsize=(10, 8), facecolor="white")
    ax_single_3d = fig_3d.add_subplot(111, projection="3d")
    
    # 复用绘制函数
    sm_single = draw_current_3d(ax_single_3d, field, env.env_bound, norm=norm)
    draw_obstacles_3d(ax_single_3d, env.obstacles)
    _draw_start_target_3d(ax_single_3d, env)
    _draw_vortices_3d(ax_single_3d, vortices_data)
    
    set_equal_axis(ax_single_3d, env.env_bound)
    ax_single_3d.view_init(elev=28, azim=-50)
    ax_single_3d.set_title("3D Current Field", fontsize=12, pad=8)
    
    if Vc.colorbar:
        # 单张图的 colorbar 可以直接绑在下方
        cb_single = fig_3d.colorbar(sm_single, ax=ax_single_3d, shrink=0.7, pad=0.08, orientation="horizontal")
        cb_single.set_label("Ocean Current Speed (m/s)", fontsize=10)
        
    _add_legend(ax_single_3d, is_3d=True, fontsize=9)
    fig_3d.suptitle(title, fontsize=13, fontweight="bold", y=0.95)

    # ════════════════════════════════════════════════════════════════════════
    # 第三部分：双重保存逻辑
    # ════════════════════════════════════════════════════════════════════════
    if save_path is not None:
        from pathlib import Path
        p = Path(save_path)
        # 为单 3D 图自动生成新路径，例如： "current_episode_0001.png" -> "current_episode_0001_3D.png"
        save_path_3d = str(p.with_name(f"{p.stem}_3D{p.suffix}"))
        
        # 保存两张图片
        fig.savefig(save_path, dpi=Vc.dpi, bbox_inches="tight")
        fig_3d.savefig(save_path_3d, dpi=Vc.dpi, bbox_inches="tight")
        
        # 关闭内存，防止泄漏
        plt.close(fig)
        plt.close(fig_3d)
        return None
        
    return fig, fig_3d
