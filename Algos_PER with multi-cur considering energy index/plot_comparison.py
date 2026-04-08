from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from config import FileAddress, MapConfig, VisualizationConfig as Vc
from env import AUVEnv
from ocean_current import build_current
from visualization import (
    sample_current_field,
    save_algorithm_comparison_figure,
    save_metric_boxplot,
    save_metric_curve,
)


# ============================================================================
# Data loading helpers
# ============================================================================


def _build_env(scene=None):
    current = None
    if MapConfig.current_mode == "real":
        current = build_current(getattr(MapConfig, "center_name", None))
    return AUVEnv(obstacle_name=scene, current=current)


def _rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(max(1, int(window)), min_periods=1).mean()


def _series_or_default(df: pd.DataFrame, column: str, default_value: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default_value)
    return pd.Series(default_value, index=df.index, dtype=float)


def _read_summary_csv(algo_name: str, mode: str) -> pd.DataFrame:
    path = FileAddress.summary_csv_for(algo_name, mode)
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return df

    if "algorithm" not in df.columns:
        df["algorithm"] = algo_name
    if "mode" not in df.columns:
        df["mode"] = mode
    if "episode" in df.columns:
        df = df.sort_values("episode").reset_index(drop=True)
    return df


def load_summary_data(algo_names, mode: str = "train") -> pd.DataFrame:
    frames = []
    for algo_name in algo_names:
        df = _read_summary_csv(algo_name, mode)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _resolve_trajectory_path(row: pd.Series, algo_name: str, mode: str, episode: int) -> Path:
    value = row.get("trajectory_csv")
    if pd.notna(value) and str(value).strip():
        return Path(str(value))
    return FileAddress.traj_csv_for(algo_name, mode, episode)


# ============================================================================
# Aggregate metric reports
# ============================================================================


def generate_metric_reports(algo_names, mode: str = "train"):
    df_all = load_summary_data(algo_names, mode=mode)
    if df_all.empty:
        return []

    save_dir = FileAddress.comparison_dir(mode, algo_names)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 直接从 config (Vc) 中读取固定的窗口大小
    if str(mode).lower() == "train":
        reward_window = int(Vc.reward_window_train)
        success_window = int(Vc.success_window_train)
    else:
        reward_window = int(Vc.reward_window_eval)
        success_window = int(Vc.success_window_eval)

    reward_series = []
    success_series = []
    energy_items = []

    available_algorithms = []
    for index, algo_name in enumerate(algo_names):
        df_algo = df_all[df_all["algorithm"] == algo_name].copy()
        if df_algo.empty:
            continue

        available_algorithms.append(algo_name)
        episodes = df_algo["episode"].to_numpy(dtype=int)

        rewards = _series_or_default(df_algo, "episode_reward", 0.0)
        reward_series.append(
            {
                "label": algo_name,
                "x": episodes,
                "raw": rewards.to_numpy(dtype=np.float32),
                # 使用 reward_window
                "smooth": _rolling(rewards, reward_window).to_numpy(dtype=np.float32),
                "color": Vc.get_algorithm_color(algo_name, index),
            }
        )

        success = _series_or_default(df_algo, "episode_success", 0.0)
        success_series.append(
            {
                "label": algo_name,
                "x": episodes,
                "raw": (success * 100.0).to_numpy(dtype=np.float32),
                # 使用 success_window
                "smooth": (_rolling(success, success_window) * 100.0).to_numpy(dtype=np.float32),
                "color": Vc.get_algorithm_color(algo_name, index),
            }
        )

        energy_df = df_algo.copy()
        if "episode_success" in energy_df.columns and (energy_df["episode_success"] == 1).any():
            energy_df = energy_df[energy_df["episode_success"] == 1]
        values = _series_or_default(energy_df, "episode_energy_index", np.nan).dropna().to_numpy(dtype=np.float32)
        if len(values) > 0:
            energy_items.append(
                {
                    "label": algo_name,
                    "values": values,
                    "color": Vc.get_algorithm_color(algo_name, index),
                }
            )

    if not available_algorithms:
        return []

    title_suffix = "Comparison" if len(available_algorithms) > 1 else f"({available_algorithms[0]})"
    outputs = []

    reward_path = FileAddress.comparison_png(mode, algo_names, f"{mode}_reward_curve")
    if reward_series:
        save_metric_curve(
            reward_series,
            save_path=str(reward_path),
            title=f"{str(mode).capitalize()} Reward {title_suffix}",
            ylabel="Reward",
            percent=False,
            figsize=Vc.reward_size,
        )
        outputs.append(str(reward_path))

    success_path = FileAddress.comparison_png(mode, algo_names, f"{mode}_success_curve")
    if success_series:
        save_metric_curve(
            success_series,
            save_path=str(success_path),
            title=f"{str(mode).capitalize()} Success Rate {title_suffix}",
            ylabel="Success Rate (%)",
            percent=True,
            figsize=Vc.reward_size,
        )
        outputs.append(str(success_path))

    energy_path = FileAddress.comparison_png(mode, algo_names, f"{mode}_energy_boxplot")
    if energy_items:
        save_metric_boxplot(
            energy_items,
            save_path=str(energy_path),
            title=f"{str(mode).capitalize()} Energy Distribution {title_suffix}",
            ylabel="Energy Consumption Index",
            figsize=Vc.metric_box_size,
        )
        outputs.append(str(energy_path))

    return outputs


# ============================================================================
# Episode trajectory reports
# ============================================================================


def _collect_episode_trajectories(df_all: pd.DataFrame, algo_names, mode: str):
    if df_all.empty or "episode" not in df_all.columns:
        return {}

    episode_map: dict[int, dict[str, dict]] = {}
    for algo_name in algo_names:
        df_algo = df_all[df_all["algorithm"] == algo_name].copy()
        if df_algo.empty:
            continue

        for _, row in df_algo.iterrows():
            episode = int(row["episode"])
            traj_path = _resolve_trajectory_path(row, algo_name, mode, episode)
            if not traj_path.exists():
                continue

            traj_df = pd.read_csv(traj_path)
            required_cols = {"x", "y", "z"}
            if not required_cols.issubset(set(traj_df.columns)):
                continue

            episode_map.setdefault(episode, {})[algo_name] = {
                "algorithm": algo_name,
                "episode": episode,
                "traj": traj_df[["x", "y", "z"]].to_numpy(dtype=np.float32),
                "reward": float(row.get("episode_reward", np.nan)),
                "path_len": float(row.get("episode_path_m", np.nan)),
                "time_s": float(row.get("episode_time_s", np.nan)),
                "success": bool(row.get("episode_success", 0)),
                "trajectory_csv": str(traj_path),
            }
    return episode_map



def generate_episode_trajectory_reports(algo_names, mode: str = "train", scene=None):
    df_all = load_summary_data(algo_names, mode=mode)
    if df_all.empty:
        return []

    episode_map = _collect_episode_trajectories(df_all, algo_names, mode)
    if not episode_map:
        return []

    env = _build_env(scene=scene)
    field = None
    if getattr(env, "current_on", False) and getattr(env, "current", None) is not None:
        field = sample_current_field(env.current, env.env_bound, Vc.grid_3d)

    manifest_rows = []
    generated_count = 0

    pbar = tqdm(sorted(episode_map), desc=f"Plotting {mode} trajectories")
    for episode in pbar:
        trajectories = episode_map[episode]
        if not trajectories:
            continue
        # ====================================================================
        # [新增] 1. 在进度条上方，安全地打印多行历史记录（类似 _build_summary_lines 的效果）
        # ====================================================================
        tqdm.write(f"\n>>> Episode {episode:04d} Summary:")
        for algo_name, info in trajectories.items():
            reward = float(info.get("reward", np.nan))
            path_len = float(info.get("path_len", np.nan))
            time_s = float(info.get("time_s", np.nan))
            success_text = "Yes" if bool(info.get("success", False)) else "No"
            
            # 使用 tqdm.write 逐行输出，不会打乱底部的进度条
            tqdm.write(f"  - {algo_name}: Reward {reward:+.2f} | Path {path_len:.2f} m | Time {time_s:.2f} s | Success {success_text}")

        combo_path = FileAddress.comparison_episode_combo_png(mode, algo_names, episode)
        single_3d_path = FileAddress.comparison_episode_3d_png(mode, algo_names, episode)
        title = f"[{str(mode).upper()}] Episode {episode:04d} Trajectory Comparison"

        saved = save_algorithm_comparison_figure(
            env=env,
            trajectories=trajectories,
            mode=mode,
            save_path=str(combo_path),
            field=field,
            title=title,
            save_3d_path=str(single_3d_path),
        )
        if saved is None:
            continue

        generated_count += 1
        for algo_name, info in trajectories.items():
            manifest_rows.append(
                {
                    "episode": episode,
                    "algorithm": algo_name,
                    "success": int(bool(info.get("success", False))),
                    "episode_reward": float(info.get("reward", np.nan)),
                    "episode_path_m": float(info.get("path_len", np.nan)),
                    "episode_time_s": float(info.get("time_s", np.nan)),
                    "trajectory_csv": info.get("trajectory_csv", ""),
                    "combo_png": str(combo_path),
                    "single_3d_png": str(single_3d_path),
                }
            )

    if generated_count == 0:
        return []

    manifest_path = FileAddress.comparison_csv(mode, algo_names, f"{mode}_episode_trajectory_manifest")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    return [
        str(FileAddress.comparison_episode_combo_dir(mode, algo_names)),
        str(FileAddress.comparison_episode_3d_dir(mode, algo_names)),
        str(manifest_path),
    ]


# ============================================================================
# Public entry
# ============================================================================


def generate_experiment_reports(algo_names, mode: str = "train", scene=None):
    outputs = []
    outputs.extend(generate_metric_reports(algo_names, mode=mode))
    outputs.extend(generate_episode_trajectory_reports(algo_names, mode=mode, scene=scene))
    return outputs


if __name__ == "__main__":
    mode = str(MapConfig.mode)
    outputs = generate_experiment_reports(["TD3-PER", "TD3"], mode="train", scene=getattr(MapConfig, "scene", None))
    if outputs:
        print(f"\n[{mode.upper()}] Generated experiment reports:")
        for output in outputs:
            print(f"  - {output}")