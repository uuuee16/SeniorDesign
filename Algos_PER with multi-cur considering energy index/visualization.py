"""Unified plotting helpers for experiment reports and trajectory visualization.

This module centralizes:
- current-field plotting
- single / multi-algorithm trajectory plotting
- aggregate metric plotting (reward / success / energy)

It is designed so that `plot_comparison.py` only does data loading and orchestration,
while all Matplotlib drawing stays here.
"""
from __future__ import annotations

from pathlib import Path
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Circle

from config import VisualizationConfig as Vc


# ============================================================================
# Generic style helpers
# ============================================================================


def get_algorithm_color(algo_name: str, index: int = 0) -> str:
    return Vc.get_algorithm_color(algo_name, index)



def _traj_color(mode: str) -> str:
    return Vc.c_traj_train if str(mode).lower() == "train" else Vc.c_traj_eval



def _current_batch(current, pos):
    if hasattr(current, "get_current_at_positions"):
        return np.asarray(current.get_current_at_positions(pos), dtype=np.float32)
    return np.asarray([current.get_current_at_position(p) for p in pos], dtype=np.float32)



def _nearest_idx(values, ref=None):
    return len(values) // 2 if ref is None else int(np.argmin(np.abs(values - ref)))



def _build_speed_norm(vmax=None):
    vmax = float(getattr(Vc, "global_v_max", 1.0) if vmax is None else vmax)
    return Normalize(vmin=0.0, vmax=max(vmax, 1e-6))



def _build_speed_mappable(norm):
    sm = cm.ScalarMappable(cmap=Vc.cmap, norm=norm)
    sm.set_array([])
    return sm



def _traj_dot_indices(n: int) -> list[int]:
    gap = max(1, int(getattr(Vc, "traj_dot_gap", 20)))
    return list(range(gap, n - 1, gap))



def _safe_title(title: str | None, default: str) -> str:
    return default if title is None or str(title).strip() == "" else str(title)


# ============================================================================
# Current field sampling
# ============================================================================


def sample_current_field(current, bounds, n=None):
    n = int(n or Vc.grid_3d)
    x0, x1, y0, y1, z0, z1 = bounds
    x = np.linspace(x0, x1, n, dtype=np.float32)
    y = np.linspace(y0, y1, n, dtype=np.float32)
    z = np.linspace(z0, z1, n, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    vel = _current_batch(current, pos)
    return {
        "x": x,
        "y": y,
        "z": z,
        "X": X,
        "Y": Y,
        "Z": Z,
        "u": vel[:, 0].reshape(X.shape),
        "v": vel[:, 1].reshape(X.shape),
        "w": vel[:, 2].reshape(X.shape),
        "speed": np.linalg.norm(vel, axis=1).reshape(X.shape),
    }



def extract_xy_slice(field, z_ref=None):
    k = _nearest_idx(field["z"], z_ref)
    X, Y = np.meshgrid(field["x"], field["y"], indexing="ij")
    u = field["u"][:, :, k]
    v = field["v"][:, :, k]
    return {
        "X": X,
        "Y": Y,
        "u": u,
        "v": v,
        "speed": np.sqrt(u * u + v * v),
        "z_ref": float(field["z"][k]),
    }



def extract_xz_slice(field, y_ref=None):
    j = _nearest_idx(field["y"], y_ref)
    X, Z = np.meshgrid(field["x"], field["z"], indexing="ij")
    u = field["u"][:, j, :]
    w = field["w"][:, j, :]
    return {
        "X": X,
        "Z": Z,
        "u": u,
        "w": w,
        "speed": np.sqrt(u * u + w * w),
        "y_ref": float(field["y"][j]),
    }


# ============================================================================
# Axes / environment helpers
# ============================================================================


def set_equal_axis(ax, bounds):
    x0, x1, y0, y1, z0, z1 = bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z1)
    ax.set_xlabel("East (m)", labelpad=6, fontsize=10, color="#1A6496", fontweight="bold")
    ax.set_ylabel("North (m)", labelpad=6, fontsize=10, color="#196F3D", fontweight="bold")
    ax.set_zlabel("Down (m)", labelpad=6, fontsize=10, color="#922B21", fontweight="bold")
    ticks = int(Vc.tick_num)
    ax.set_xticks(np.linspace(x0, x1, ticks))
    ax.set_yticks(np.linspace(y0, y1, ticks))
    ax.set_zticks(np.linspace(z0, z1, ticks))
    if getattr(Vc, "equal_axis", True):
        ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    if getattr(Vc, "invert_z", True):
        ax.invert_zaxis()
    ax.grid(True, linestyle="--", alpha=0.22)
    ax.tick_params(labelsize=8)



def _setup_xy_axis(ax, env, z_ref=None):
    ax.set_xlim(env.x0, env.x1)
    ax.set_ylim(env.y0, env.y1)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.30)
    ax.set_xlabel("East (m)", fontsize=10)
    ax.set_ylabel("North (m)", fontsize=10)
    ax.tick_params(labelsize=8)
    title = "XOY Projection"
    if z_ref is not None:
        title += f"  |  speed at z = {z_ref:.0f} m"
    ax.set_title(title, fontsize=11)



def _setup_xz_axis(ax, env, y_ref=None):
    ax.set_xlim(env.x0, env.x1)
    ax.set_ylim(env.z0, env.z1)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.30)
    ax.set_xlabel("East (m)", fontsize=10)
    ax.set_ylabel("Down (m)", fontsize=10)
    ax.tick_params(labelsize=8)
    if getattr(Vc, "invert_z", True):
        ax.invert_yaxis()
    title = "XOZ Projection"
    if y_ref is not None:
        title += f"  |  speed at y = {y_ref:.0f} m"
    ax.set_title(title, fontsize=11)



def _draw_obstacles_3d(ax, obstacles):
    u_grid, v_grid = np.mgrid[
        0 : 2 * np.pi : complex(0, int(Vc.obstacle_u)),
        0 : np.pi : complex(0, int(Vc.obstacle_v)),
    ]
    for ox, oy, oz, r in obstacles:
        ax.plot_surface(
            ox + r * np.cos(u_grid) * np.sin(v_grid),
            oy + r * np.sin(u_grid) * np.sin(v_grid),
            oz + r * np.cos(v_grid),
            color=Vc.c_obs_face,
            alpha=0.28,
            edgecolor=Vc.c_obs_edge,
            linewidth=0.20,
        )



def _draw_obstacles_xy(ax, obstacles):
    for ox, oy, _, r in obstacles:
        ax.add_patch(
            Circle(
                (ox, oy),
                r,
                edgecolor=Vc.c_obs_edge,
                facecolor=Vc.c_obs_face,
                alpha=0.38,
                linewidth=1.1,
                zorder=1,
            )
        )



def _draw_obstacles_xz(ax, obstacles):
    for ox, _, oz, r in obstacles:
        ax.add_patch(
            Circle(
                (ox, oz),
                r,
                edgecolor=Vc.c_obs_edge,
                facecolor=Vc.c_obs_face,
                alpha=0.38,
                linewidth=1.1,
                zorder=1,
            )
        )


# ============================================================================
# Current rendering
# ============================================================================


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
        norm = _build_speed_norm(float(np.max(S)))

    cmap_fn = cm.get_cmap(Vc.cmap)
    colors = cmap_fn(norm(S.ravel()))
    scene = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    scale = scene / max(X.shape) * 0.45 * Vc.arrow_scale_3d / 10.0

    ax.quiver(
        X.ravel(),
        Y.ravel(),
        Z.ravel(),
        U.ravel(),
        V.ravel(),
        W.ravel(),
        colors=colors,
        length=scale,
        normalize=True,
        alpha=Vc.arrow_alpha_3d,
        linewidth=Vc.arrow_width_3d,
        arrow_length_ratio=Vc.arrow_ratio_3d,
    )
    return _build_speed_mappable(norm)



def draw_current_xy(ax, field_xy, bounds, norm=None):
    if norm is None:
        norm = _build_speed_norm(float(np.max(field_xy["speed"])))
    scene = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    base = scene / field_xy["X"].shape[0] / 1.8
    speed = np.maximum(field_xy["speed"], 1e-9)
    u = field_xy["u"] / speed * base * Vc.arrow_scale_2d / 10.0
    v = field_xy["v"] / speed * base * Vc.arrow_scale_2d / 10.0
    return ax.quiver(
        field_xy["X"],
        field_xy["Y"],
        u,
        v,
        field_xy["speed"],
        cmap=Vc.cmap,
        norm=norm,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        alpha=Vc.arrow_alpha_2d,
        width=Vc.arrow_width_2d,
        zorder=2,
    )



def draw_current_xz(ax, field_xz, bounds, norm=None):
    if norm is None:
        norm = _build_speed_norm(float(np.max(field_xz["speed"])))
    scene = max(bounds[1] - bounds[0], bounds[5] - bounds[4])
    base = scene / field_xz["X"].shape[0] / 1.8
    speed = np.maximum(field_xz["speed"], 1e-9)
    u = field_xz["u"] / speed * base * Vc.arrow_scale_2d / 10.0
    w = field_xz["w"] / speed * base * Vc.arrow_scale_2d / 10.0
    return ax.quiver(
        field_xz["X"],
        field_xz["Z"],
        u,
        w,
        field_xz["speed"],
        cmap=Vc.cmap,
        norm=norm,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        alpha=Vc.arrow_alpha_2d,
        width=Vc.arrow_width_2d,
        zorder=2,
    )



def _draw_speed_background(ax, X, Y, speed, norm=None):
    if norm is None:
        norm = _build_speed_norm(float(np.max(speed)))
    levels = np.linspace(norm.vmin, norm.vmax, int(Vc.speed_background_levels))
    clipped = np.clip(speed, norm.vmin, norm.vmax)
    return ax.contourf(
        X,
        Y,
        clipped,
        levels=levels,
        cmap=Vc.cmap,
        norm=norm,
        alpha=Vc.speed_background_alpha,
        antialiased=True,
        zorder=0,
    )



def _normalized_xy_vectors(field_xy, bounds):
    scene = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    base = scene / field_xy["X"].shape[0] / 1.8
    speed = np.maximum(field_xy["speed"], 1e-9)
    u = field_xy["u"] / speed * base * Vc.arrow_scale_2d / 10.0
    v = field_xy["v"] / speed * base * Vc.arrow_scale_2d / 10.0
    return u, v



def _normalized_xz_vectors(field_xz, bounds):
    scene = max(bounds[1] - bounds[0], bounds[5] - bounds[4])
    base = scene / field_xz["X"].shape[0] / 1.8
    speed = np.maximum(field_xz["speed"], 1e-9)
    u = field_xz["u"] / speed * base * Vc.arrow_scale_2d / 10.0
    w = field_xz["w"] / speed * base * Vc.arrow_scale_2d / 10.0
    return u, w



def draw_current_xy_speed_background(ax, field_xy, bounds, norm=None):
    bg = _draw_speed_background(ax, field_xy["X"], field_xy["Y"], field_xy["speed"], norm=norm)
    u, v = _normalized_xy_vectors(field_xy, bounds)
    ax.quiver(
        field_xy["X"],
        field_xy["Y"],
        u,
        v,
        color=Vc.arrow_color_2d_background,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        alpha=Vc.arrow_alpha_2d_background,
        width=Vc.arrow_width_2d_background,
        zorder=2,
    )
    return bg



def draw_current_xz_speed_background(ax, field_xz, bounds, norm=None):
    bg = _draw_speed_background(ax, field_xz["X"], field_xz["Z"], field_xz["speed"], norm=norm)
    u, w = _normalized_xz_vectors(field_xz, bounds)
    ax.quiver(
        field_xz["X"],
        field_xz["Z"],
        u,
        w,
        color=Vc.arrow_color_2d_background,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        alpha=Vc.arrow_alpha_2d_background,
        width=Vc.arrow_width_2d_background,
        zorder=2,
    )
    return bg


# ============================================================================
# Trajectory rendering
# ============================================================================


def _plot_traj(ax, traj: np.ndarray, color: str, label: str, kind: str, end_marker=False):
    if len(traj) < 2:
        return

    if kind == "3d":
        xs, ys = traj[:, 0], traj[:, 1]
        zs = traj[:, 2]
        ax.plot(xs, ys, zs, color=color, linewidth=Vc.traj_width_3d, alpha=0.92, label=label, zorder=5)
        idx = _traj_dot_indices(len(traj))
        if idx:
            ax.scatter(xs[idx], ys[idx], zs[idx], c=color, s=Vc.pos_size * 0.45, alpha=0.60, edgecolors="none", zorder=6)
        if end_marker:
            ax.scatter(xs[-1], ys[-1], zs[-1], c=color, s=Vc.pos_size * 0.90, marker="P", edgecolors="#1C1C1C", linewidths=1.0, zorder=7)
        return

    if kind == "xy":
        xs, ys = traj[:, 0], traj[:, 1]
    elif kind == "xz":
        xs, ys = traj[:, 0], traj[:, 2]
    else:
        raise ValueError(f"Unsupported trajectory kind: {kind}")

    ax.plot(xs, ys, color=color, linewidth=Vc.traj_width_2d, alpha=0.92, label=label, zorder=5)
    idx = _traj_dot_indices(len(traj))
    if idx:
        ax.scatter(xs[idx], ys[idx], c=color, s=Vc.pos_size * 0.35, alpha=0.60, edgecolors="none", zorder=6)
    if end_marker:
        ax.scatter(xs[-1], ys[-1], c=color, s=Vc.pos_size * 0.80, marker="P", edgecolors="#1C1C1C", linewidths=0.9, zorder=7)



def _draw_start_target_3d(ax, env):
    ax.scatter(*env.start, c=Vc.c_start_fc, s=Vc.start_size, marker="o", edgecolors=Vc.c_start_ec, linewidths=2.2, zorder=8, label="Start")
    ax.scatter(*env.target, c=Vc.c_target_fc, s=Vc.target_size, marker="*", edgecolors=Vc.c_target_ec, linewidths=1.5, zorder=8, label="Target")



def _draw_start_target_xy(ax, env):
    ax.scatter(env.start[0], env.start[1], c=Vc.c_start_fc, s=Vc.start_size * 0.85, marker="o", edgecolors=Vc.c_start_ec, linewidths=2.0, zorder=8, label="Start")
    ax.scatter(env.target[0], env.target[1], c=Vc.c_target_fc, s=Vc.target_size * 0.85, marker="*", edgecolors=Vc.c_target_ec, linewidths=1.4, zorder=8, label="Target")



def _draw_start_target_xz(ax, env):
    ax.scatter(env.start[0], env.start[2], c=Vc.c_start_fc, s=Vc.start_size * 0.85, marker="o", edgecolors=Vc.c_start_ec, linewidths=2.0, zorder=8, label="Start")
    ax.scatter(env.target[0], env.target[2], c=Vc.c_target_fc, s=Vc.target_size * 0.85, marker="*", edgecolors=Vc.c_target_ec, linewidths=1.4, zorder=8, label="Target")



def _draw_collision_points(ax3d, ax_xy, ax_xz, collision_points):
    if not collision_points:
        return
    bound_pts = np.asarray(collision_points.get("boundary", []), dtype=np.float32)
    obs_pts = np.asarray(collision_points.get("obstacle", []), dtype=np.float32)

    for pts, color, label in [
        (bound_pts, Vc.c_col_bound, "Boundary Hit"),
        (obs_pts, Vc.c_col_obs, "Obstacle Hit"),
    ]:
        if pts.size == 0:
            continue
        if ax3d is not None:
            ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=Vc.hit_size, marker="X", edgecolors="white", linewidths=0.8, zorder=9, label=label)
        if ax_xy is not None:
            ax_xy.scatter(pts[:, 0], pts[:, 1], c=color, s=Vc.hit_size, marker="X", edgecolors="white", linewidths=0.8, zorder=9, label=label)
        if ax_xz is not None:
            ax_xz.scatter(pts[:, 0], pts[:, 2], c=color, s=Vc.hit_size, marker="X", edgecolors="white", linewidths=0.8, zorder=9, label=label)



def _get_vortex_center(v):
    return v["center"] if isinstance(v, dict) else v



def _draw_vortices_3d(ax, vortices):
    if not vortices:
        return
    xs, ys, zs = zip(*[_get_vortex_center(v) for v in vortices])
    ax.scatter(xs, ys, zs, c=Vc.c_vortex_fc, s=Vc.start_size, marker="D", edgecolors=Vc.c_vortex_ec, linewidths=2.0, zorder=10, label="Vortex Center")



def _draw_vortices_xy(ax, vortices):
    if not vortices:
        return
    centers = [_get_vortex_center(v) for v in vortices]
    xs, ys = zip(*[(c[0], c[1]) for c in centers])
    ax.scatter(xs, ys, c=Vc.c_vortex_fc, s=Vc.start_size * 0.8, marker="D", edgecolors=Vc.c_vortex_ec, linewidths=1.8, zorder=10, label="Vortex Center")



def _draw_vortices_xz(ax, vortices):
    if not vortices:
        return
    centers = [_get_vortex_center(v) for v in vortices]
    xs, zs = zip(*[(c[0], c[2]) for c in centers])
    ax.scatter(xs, zs, c=Vc.c_vortex_fc, s=Vc.start_size * 0.8, marker="D", edgecolors=Vc.c_vortex_ec, linewidths=1.8, zorder=10, label="Vortex Center")



def _add_legend(ax, fontsize=9, loc="upper left"):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    uniq = {}
    for handle, label in zip(handles, labels):
        if label not in uniq:
            uniq[label] = handle
    ax.legend(
        list(uniq.values()),
        list(uniq.keys()),
        loc=loc,
        fontsize=fontsize,
        framealpha=0.82,
        edgecolor="#CCCCCC",
        borderpad=0.5,
        handlelength=1.4,
        handletextpad=0.5,
    )


# ============================================================================
# Figure builders
# ============================================================================


def _build_three_panel_figure():
    fig = plt.figure(figsize=Vc.comparison_size, facecolor="white")
    gs = gridspec.GridSpec(1, 3, left=0.04, right=0.97, bottom=0.12, top=0.88, wspace=0.28)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_xy = fig.add_subplot(gs[0, 1])
    ax_xz = fig.add_subplot(gs[0, 2])
    return fig, ax3d, ax_xy, ax_xz



def _build_single_3d_figure():
    fig = plt.figure(figsize=Vc.single_3d_size, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    return fig, ax



def _mean_plane_refs(traj_arrays, env):
    if traj_arrays:
        stacked = np.vstack(traj_arrays)
        return float(np.mean(stacked[:, 2])), float(np.mean(stacked[:, 1]))
    z_ref = float((env.start[2] + env.target[2]) * 0.5)
    y_ref = float((env.start[1] + env.target[1]) * 0.5)
    return z_ref, y_ref




def _draw_environment_layers(ax3d, ax_xy, ax_xz, env):
    _draw_obstacles_3d(ax3d, env.obstacles)
    _draw_obstacles_xy(ax_xy, env.obstacles)
    _draw_obstacles_xz(ax_xz, env.obstacles)
    _draw_start_target_3d(ax3d, env)
    _draw_start_target_xy(ax_xy, env)
    _draw_start_target_xz(ax_xz, env)
    vortices = getattr(env.current, "vortices", getattr(env, "vortices", []))
    _draw_vortices_3d(ax3d, vortices)
    _draw_vortices_xy(ax_xy, vortices)
    _draw_vortices_xz(ax_xz, vortices)



# ============================================================================
# Aggregate metric figures
# ============================================================================


def save_metric_curve(series_list, save_path, title, ylabel, percent=False, ylim=None, figsize=None):
    if not series_list:
        return None

    fig, ax = plt.subplots(figsize=figsize or Vc.reward_size)
    for index, item in enumerate(series_list):
        color = item.get("color") or get_algorithm_color(item["label"], index)
        x = np.asarray(item["x"])
        raw = np.asarray(item["raw"])
        smooth = np.asarray(item["smooth"])
        ax.plot(x, raw, color=color, alpha=Vc.metric_line_alpha_raw)
        ax.plot(x, smooth, color=color, linewidth=Vc.metric_line_width, alpha=Vc.metric_line_alpha_smooth, label=item["label"])

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if percent:
        ax.set_ylim(*(ylim or (-5, 105)))
    elif ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", alpha=0.30)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=Vc.dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path



def save_metric_boxplot(box_items, save_path, title, ylabel, figsize=None):
    if not box_items:
        return None

    fig, ax = plt.subplots(figsize=figsize or Vc.metric_box_size)
    labels = [item["label"] for item in box_items]
    data = [np.asarray(item["values"], dtype=np.float32) for item in box_items]
    colors = [item.get("color") or get_algorithm_color(item["label"], idx) for idx, item in enumerate(box_items)]

    plot = ax.boxplot(
        data,
        patch_artist=True,
        labels=labels,
        showfliers=True,
        flierprops=dict(marker="d", markerfacecolor=Vc.box_flier_color, markeredgecolor="none", markersize=5, alpha=0.9),
        medianprops=dict(color=Vc.box_median_color, linewidth=1.2),
        boxprops=dict(linewidth=1.2, color=Vc.box_edge_color, alpha=0.9),
        whiskerprops=dict(linewidth=1.2, color=Vc.box_edge_color),
        capprops=dict(linewidth=1.2, color=Vc.box_edge_color),
    )
    for patch, color in zip(plot["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.yaxis.grid(True, linestyle="--", alpha=Vc.box_grid_alpha)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=Vc.dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# Current-field figure
# ============================================================================


def visualize_current_environment(env, save_path=None, title="Ocean Current Field", field=None, z_ref=None, y_ref=None):
    field = field if field is not None else (sample_current_field(env.current, env.env_bound, Vc.grid_3d) if getattr(env, "current", None) is not None else None)
    norm = _build_speed_norm(getattr(Vc, "global_v_max", 1.0))
    sm = _build_speed_mappable(norm) if field is not None else None
    field_xy = extract_xy_slice(field, z_ref) if field is not None else None
    field_xz = extract_xz_slice(field, y_ref) if field is not None else None

    fig, ax3d, ax_xy, ax_xz = _build_three_panel_figure()

    if field is not None:
        draw_current_3d(ax3d, field, env.env_bound, norm=norm)
    _draw_obstacles_3d(ax3d, env.obstacles)
    _draw_start_target_3d(ax3d, env)
    _draw_vortices_3d(ax3d, getattr(env.current, "vortices", getattr(env, "vortices", [])))
    set_equal_axis(ax3d, env.env_bound)
    ax3d.view_init(elev=28, azim=-50)
    ax3d.set_title("3D Current Field", fontsize=12, pad=8)

    if field_xy is not None:
        if getattr(Vc, "show_speed_background", True):
            draw_current_xy_speed_background(ax_xy, field_xy, env.env_bound, norm=norm)
        else:
            draw_current_xy(ax_xy, field_xy, env.env_bound, norm=norm)
    _draw_obstacles_xy(ax_xy, env.obstacles)
    _draw_start_target_xy(ax_xy, env)
    _draw_vortices_xy(ax_xy, getattr(env.current, "vortices", getattr(env, "vortices", [])))
    _setup_xy_axis(ax_xy, env, None if field_xy is None else field_xy["z_ref"])

    if field_xz is not None:
        if getattr(Vc, "show_speed_background", True):
            draw_current_xz_speed_background(ax_xz, field_xz, env.env_bound, norm=norm)
        else:
            draw_current_xz(ax_xz, field_xz, env.env_bound, norm=norm)
    _draw_obstacles_xz(ax_xz, env.obstacles)
    _draw_start_target_xz(ax_xz, env)
    _draw_vortices_xz(ax_xz, getattr(env.current, "vortices", getattr(env, "vortices", [])))
    _setup_xz_axis(ax_xz, env, None if field_xz is None else field_xz["y_ref"])

    if getattr(Vc, "colorbar", True) and sm is not None:
        cax = fig.add_axes([0.15, 0.04, 0.70, 0.025])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label("Ocean Current Speed (m/s)", fontsize=10)
        cb.ax.tick_params(labelsize=9)

    _add_legend(ax3d, fontsize=9, loc="upper left")
    _add_legend(ax_xy, fontsize=8, loc="upper left")
    _add_legend(ax_xz, fontsize=8, loc="best")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.97)

    fig_3d, ax_single_3d = _build_single_3d_figure()
    if field is not None:
        draw_current_3d(ax_single_3d, field, env.env_bound, norm=norm)
    _draw_obstacles_3d(ax_single_3d, env.obstacles)
    _draw_start_target_3d(ax_single_3d, env)
    _draw_vortices_3d(ax_single_3d, getattr(env.current, "vortices", getattr(env, "vortices", [])))
    set_equal_axis(ax_single_3d, env.env_bound)
    ax_single_3d.view_init(elev=28, azim=-50)
    ax_single_3d.set_title("3D Current Field", fontsize=12, pad=8)
    if getattr(Vc, "colorbar", True) and sm is not None:
        cb_single = fig_3d.colorbar(sm, ax=ax_single_3d, shrink=0.7, pad=0.08, orientation="horizontal")
        cb_single.set_label("Ocean Current Speed (m/s)", fontsize=10)
    _add_legend(ax_single_3d, fontsize=9, loc="upper left")
    fig_3d.suptitle(title, fontsize=13, fontweight="bold", y=0.95)

    if save_path is not None:
        p = Path(save_path)
        fig.savefig(p, dpi=Vc.dpi, bbox_inches="tight")
        fig_3d.savefig(p.with_name(f"{p.stem}_3D{p.suffix}"), dpi=Vc.dpi, bbox_inches="tight")
        plt.close(fig)
        plt.close(fig_3d)
        return None

    return fig, fig_3d


# ============================================================================
# Single / multi trajectory figures
# ============================================================================


def _build_summary_lines(trajectories):
    lines = []
    for algo_name, info in trajectories.items():
        ep = int(info.get("episode", -1)) if info.get("episode") is not None else -1
        reward = float(info.get("reward", np.nan))
        path_len = float(info.get("path_len", np.nan))
        time_s = float(info.get("time_s", np.nan))
        success_text = "Yes" if bool(info.get("success", False)) else "No"
        if ep >= 0:
            lines.append(f"{algo_name}: Ep {ep:04d} | Reward {reward:+.2f} | Path {path_len:.2f} m | Time {time_s:.2f} s | Success {success_text}")
        else:
            lines.append(f"{algo_name}: Reward {reward:+.2f} | Path {path_len:.2f} m | Time {time_s:.2f} s | Success {success_text}")
    return lines



def _save_trajectory_3d_figure(env, ordered_items, mode, save_path, field=None, title=None, summary_lines=None):
    if not ordered_items:
        return None

    fig, ax = _build_single_3d_figure()
    norm = _build_speed_norm(getattr(Vc, "global_v_max", 1.0))
    sm = _build_speed_mappable(norm) if field is not None else None

    if field is not None:
        draw_current_3d(ax, field, env.env_bound, norm=norm)
    _draw_obstacles_3d(ax, env.obstacles)
    _draw_start_target_3d(ax, env)
    _draw_vortices_3d(ax, getattr(env.current, "vortices", getattr(env, "vortices", [])))

    for index, (algo_name, info) in enumerate(ordered_items):
        traj = np.asarray(info.get("traj", []), dtype=np.float32)
        if len(traj) < 2:
            continue
        _plot_traj(ax, traj, color=get_algorithm_color(algo_name, index), label=algo_name, kind="3d", end_marker=True)

    set_equal_axis(ax, env.env_bound)
    ax.view_init(elev=28, azim=-50)
    ax.set_title("3D Trajectory Comparison", fontsize=12, pad=8)
    if getattr(Vc, "colorbar", True) and sm is not None:
        cb = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.08, orientation="horizontal")
        cb.set_label("Ocean Current Speed (m/s)", fontsize=10)
    _add_legend(ax, fontsize=9, loc="upper left")

    fig.suptitle(_safe_title(title, f"[{str(mode).upper()}] 3D Trajectory Comparison"), fontsize=12, fontweight="bold", y=0.95)
    if summary_lines:
        fig.text(0.5, 0.90, "\n".join(summary_lines), ha="center", va="top", fontsize=8.6, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D0D7DE", alpha=0.78))

    fig.savefig(save_path, dpi=Vc.dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path



def save_algorithm_comparison_figure(env, trajectories, mode, save_path, field=None, title=None, save_3d_path=None):
    if not trajectories:
        return None

    ordered_items = [(name, info) for name, info in trajectories.items() if len(np.asarray(info.get("traj", []))) >= 2]
    if not ordered_items:
        return None

    traj_arrays = [np.asarray(info["traj"], dtype=np.float32) for _, info in ordered_items]
    z_ref, y_ref = _mean_plane_refs(traj_arrays, env)
    norm = _build_speed_norm(getattr(Vc, "global_v_max", 1.0))
    summary_lines = _build_summary_lines(dict(ordered_items))

    fig, ax3d, ax_xy, ax_xz = _build_three_panel_figure()
    sm, field_xy, field_xz = (None, None, None)
    if field is not None:
        sm = _build_speed_mappable(norm)
        draw_current_3d(ax3d, field, env.env_bound, norm=norm)
        field_xy = extract_xy_slice(field, z_ref)
        field_xz = extract_xz_slice(field, y_ref)
        if getattr(Vc, "show_speed_background", True):
            draw_current_xy_speed_background(ax_xy, field_xy, env.env_bound, norm=norm)
            draw_current_xz_speed_background(ax_xz, field_xz, env.env_bound, norm=norm)
        else:
            draw_current_xy(ax_xy, field_xy, env.env_bound, norm=norm)
            draw_current_xz(ax_xz, field_xz, env.env_bound, norm=norm)

    _draw_environment_layers(ax3d, ax_xy, ax_xz, env)

    for index, (algo_name, info) in enumerate(ordered_items):
        color = get_algorithm_color(algo_name, index)
        traj = np.asarray(info["traj"], dtype=np.float32)
        _plot_traj(ax3d, traj, color=color, label=algo_name, kind="3d", end_marker=True)
        _plot_traj(ax_xy, traj, color=color, label=algo_name, kind="xy", end_marker=True)
        _plot_traj(ax_xz, traj, color=color, label=algo_name, kind="xz", end_marker=True)

    set_equal_axis(ax3d, env.env_bound)
    ax3d.view_init(elev=28, azim=-50)
    ax3d.set_title("3D Trajectory Comparison", fontsize=12, pad=8)
    _setup_xy_axis(ax_xy, env, None if field_xy is None else field_xy["z_ref"])
    _setup_xz_axis(ax_xz, env, None if field_xz is None else field_xz["y_ref"])

    if getattr(Vc, "colorbar", True) and sm is not None:
        cax = fig.add_axes([0.15, 0.04, 0.70, 0.025])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label("Ocean Current Speed (m/s)", fontsize=10)
        cb.ax.tick_params(labelsize=9)

    _add_legend(ax3d, fontsize=9, loc="upper left")
    _add_legend(ax_xy, fontsize=8, loc="upper left")
    _add_legend(ax_xz, fontsize=8, loc="best")

    fig.suptitle(
        _safe_title(title, f"[{str(mode).upper()}] Algorithm Trajectory Comparison"),
        fontsize=12,
        fontweight="bold",
        y=0.975,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F4F8", edgecolor="#B0BEC5", alpha=0.85),
    )
    if summary_lines:
        fig.text(0.5, 0.925, "\n".join(summary_lines), ha="center", va="top", fontsize=8.8, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D0D7DE", alpha=0.78))

    fig.savefig(save_path, dpi=Vc.dpi, bbox_inches="tight")
    plt.close(fig)

    if save_3d_path is not None:
        _save_trajectory_3d_figure(env, ordered_items, mode, save_3d_path, field=field, title=title, summary_lines=summary_lines)

    return save_path



def save_episode_combo_figure(env, traj, collision_points, ep_reward, time_s, success, path_len, mode, save_path, field=None, ep=None):
    title = f"[{str(mode).upper()}] Episode {int(ep):04d}" if ep is not None else f"[{str(mode).upper()}] Episode Trajectory"
    trajectories = {
        "trajectory": {
            "traj": np.asarray(traj, dtype=np.float32),
            "episode": ep,
            "reward": ep_reward,
            "path_len": path_len,
            "time_s": time_s,
            "success": success,
        }
    }

    fig, ax3d, ax_xy, ax_xz = _build_three_panel_figure()
    traj = np.asarray(traj, dtype=np.float32)
    if len(traj) == 0:
        plt.close(fig)
        return None

    z_ref, y_ref = _mean_plane_refs([traj], env)
    norm = _build_speed_norm(getattr(Vc, "global_v_max", 1.0))
    sm = None
    field_xy = None
    field_xz = None
    if field is not None:
        sm = _build_speed_mappable(norm)
        draw_current_3d(ax3d, field, env.env_bound, norm=norm)
        field_xy = extract_xy_slice(field, z_ref)
        field_xz = extract_xz_slice(field, y_ref)
        if getattr(Vc, "show_speed_background", True):
            draw_current_xy_speed_background(ax_xy, field_xy, env.env_bound, norm=norm)
            draw_current_xz_speed_background(ax_xz, field_xz, env.env_bound, norm=norm)
        else:
            draw_current_xy(ax_xy, field_xy, env.env_bound, norm=norm)
            draw_current_xz(ax_xz, field_xz, env.env_bound, norm=norm)

    _draw_environment_layers(ax3d, ax_xy, ax_xz, env)
    color = _traj_color(mode)
    _plot_traj(ax3d, traj, color=color, label="Trajectory", kind="3d", end_marker=True)
    _plot_traj(ax_xy, traj, color=color, label="Trajectory", kind="xy", end_marker=True)
    _plot_traj(ax_xz, traj, color=color, label="Trajectory", kind="xz", end_marker=True)
    _draw_collision_points(ax3d, ax_xy, ax_xz, collision_points)

    set_equal_axis(ax3d, env.env_bound)
    ax3d.view_init(elev=28, azim=-50)
    ax3d.set_title("3D Trajectory", fontsize=12, pad=8)
    _setup_xy_axis(ax_xy, env, None if field_xy is None else field_xy["z_ref"])
    _setup_xz_axis(ax_xz, env, None if field_xz is None else field_xz["y_ref"])

    if getattr(Vc, "colorbar", True) and sm is not None:
        cax = fig.add_axes([0.15, 0.04, 0.70, 0.025])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label("Ocean Current Speed (m/s)", fontsize=10)
        cb.ax.tick_params(labelsize=9)

    _add_legend(ax3d, fontsize=9, loc="upper left")
    _add_legend(ax_xy, fontsize=8, loc="upper left")
    _add_legend(ax_xz, fontsize=8, loc="best")

    success_text = "Yes" if bool(success) else "No"
    fig.suptitle(
        f"{title}   |   Reward: {float(ep_reward):+.2f}   |   Time: {float(time_s):.1f} s   |   Path: {float(path_len):.1f} m   |   Success: {success_text}",
        fontsize=12,
        fontweight="bold",
        y=0.97,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F4F8", edgecolor="#B0BEC5", alpha=0.85),
    )
    fig.savefig(save_path, dpi=Vc.dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


__all__ = [
    "sample_current_field",
    "extract_xy_slice",
    "extract_xz_slice",
    "draw_current_3d",
    "draw_current_xy",
    "draw_current_xz",
    "draw_current_xy_speed_background",
    "draw_current_xz_speed_background",
    "save_metric_curve",
    "save_metric_boxplot",
    "save_episode_combo_figure",
    "visualize_current_environment",
    "save_algorithm_comparison_figure",
    "get_algorithm_color",
]
