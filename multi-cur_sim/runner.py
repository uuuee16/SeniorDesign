from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FileAddress, MapConfig, NetworkConfig, VisualizationConfig
from visualization import (
    sample_current_field,
    save_episode_combo_figure,
    visualize_current_environment,
)


class Runner:
    def __init__(self, env, agent, mode: str, map_cfg=MapConfig, net_cfg=NetworkConfig, vis_cfg=VisualizationConfig):
        self.env = env
        self.agent = agent
        self.mode = mode
        self.map_cfg = map_cfg
        self.net_cfg = net_cfg
        self.vis_cfg = vis_cfg
        self.current_field_snapshot = None
        FileAddress.make_dirs()


    def _reset_episode(self, ep: int):
        obs, info = self.env.reset(options={})
        self._capture_current_snapshot(ep)
        return obs, info

    def _capture_current_snapshot(self, ep: int):
        self.current_field_snapshot = None
        if not (self.env.current_on and self.env.current is not None):
            return

        self.current_field_snapshot = sample_current_field(
            self.env.current,
            self.env.env_bound,
            self.vis_cfg.grid_3d,
        )

        if not self.vis_cfg.save_current_fig:
            return

        z_ref = float((self.env.start[2] + self.env.target[2]) * 0.5)
        y_ref = float((self.env.start[1] + self.env.target[1]) * 0.5)
        title = f"[{self.mode.upper()}] Episode {ep:04d} current snapshot"
        if getattr(self.env, "sim_time", None) is not None:
            title += f" | {self.env.sim_time}"
        visualize_current_environment(
            self.env,
            save_path=str(FileAddress.current_ep_png(self.mode, ep)),
            title=title,
            field=self.current_field_snapshot,
            z_ref=z_ref,
            y_ref=y_ref,
        )

    def _blank_update_row(self, step: int, sim_time):
        return {
            "step": step,
            "sim_time": sim_time,
            "update_index": np.nan,
            "critic1_loss": np.nan,
            "critic2_loss": np.nan,
            "actor_loss": np.nan,
            "actor_updated": False,
            "target_q_mean": np.nan,
            "q1_mean": np.nan,
            "q2_mean": np.nan,
        }

    def _run_episode(self, ep: int, max_steps: int):
        obs, _ = self._reset_episode(ep)
        traj = [self.env.pos.copy()]
        collision_points = {"boundary": [], "obstacle": []}
        step_rows = []
        update_rows = []

        ep_reward = 0.0
        path_len = 0.0
        prev_pos = self.env.pos.copy()
        final_info = None

        for step in range(1, max_steps + 1):
            action = np.asarray(
                self.agent.select_action(obs, evaluate=(self.mode != "train")),
                dtype=np.float32,
            )
            next_obs, reward, done, _, info = self.env.step(action)

            update_info = None
            if self.mode == "train":
                self.agent.memory.store_transition(obs, action, reward, next_obs, done)
                update_info = self.agent.update()

            applied_cur = np.zeros(3, dtype=np.float32)
            if info.get("current_applied") is not None:
                applied_cur = np.asarray(info["current_applied"], dtype=np.float32)

            actual_yaw_rate = float(action[1]) * self.map_cfg.max_yaw_rate
            actual_pitch_rate = float(action[2]) * self.map_cfg.max_pitch_rate
            step_disp = float(np.linalg.norm(self.env.pos - prev_pos))
            # actual_speed = step_disp / max(self.map_cfg.dt, 1e-8)
            actual_speed = float(action[0]) * self.map_cfg.max_body_speed
            path_len += step_disp
            prev_pos = self.env.pos.copy()
            traj.append(self.env.pos.copy())

            if info["out"]:
                collision_points["boundary"].append(self.env.pos.copy())
            if info["hit"]:
                collision_points["obstacle"].append(self.env.pos.copy())

            step_row = {
                "step": step,
                "t_s": step * self.map_cfg.dt,
                "sim_time": info.get("sim_time"),
                "x": float(self.env.pos[0]),
                "y": float(self.env.pos[1]),
                "z": float(self.env.pos[2]),
                "yaw": float(self.env.yaw),
                "pitch": float(self.env.pitch),
                "actual_speed": actual_speed,
                "actual_yaw_rate": actual_yaw_rate,
                "actual_pitch_rate": actual_pitch_rate,
                "actual_cur_vel_x": float(applied_cur[0]),
                "actual_cur_vel_y": float(applied_cur[1]),
                "actual_cur_vel_z": float(applied_cur[2]),
                "reward": float(reward),
                "dist2target": float(info["dist"]),
                "nearest_obstacle_dist": float(info["obs_dist"]),
                "step_terminated": bool(done),
                "hit_boundary": bool(info["out"]),
                "hit_obstacle": bool(info["hit"]),
                "goal_flag": bool(info["goal"]),
                "warning_flag": bool(info["warn"]),
                "termination_reason": info.get("termination_reason", "running"),
                "episode_success": 0,
            }
            for k, v in info["reward_part"].items():
                step_row[k] = float(v)
            step_rows.append(step_row)

            update_row = self._blank_update_row(step, info.get("sim_time"))
            if update_info is not None:
                update_row.update(update_info)
                update_row["step"] = step
                update_row["sim_time"] = info.get("sim_time")
            update_rows.append(update_row)

            ep_reward += float(reward)
            obs = next_obs
            final_info = info
            if done:
                break

        success = int(final_info is not None and final_info["goal"])
        for row in step_rows:
            row["episode_success"] = success

        return {
            "traj": np.asarray(traj, dtype=np.float32),
            "step": pd.DataFrame(step_rows),
            "update": pd.DataFrame(update_rows),
            "reward": ep_reward,
            "step_num": len(step_rows),
            "time": len(step_rows) * self.map_cfg.dt,
            "path_len": path_len,
            "success": success,
            "collision_points": collision_points,
        }

    def _save_episode_data(self, result, ep: int):
        pd.DataFrame(result["traj"], columns=["x", "y", "z"]).to_csv(FileAddress.traj_csv(self.mode, ep), index=False)
        result["step"].to_csv(FileAddress.step_csv(self.mode, ep), index=False)
        result["update"].to_csv(FileAddress.update_csv(self.mode, ep), index=False)

        if self.vis_cfg.save_ep_fig:
            save_episode_combo_figure(
                env=self.env,
                traj=result["traj"],
                collision_points=result["collision_points"],
                ep_reward=result["reward"],
                time_s=result["time"],
                success=result["success"],
                path_len=result["path_len"],
                mode=self.mode,
                save_path=str(FileAddress.ep_png(self.mode, ep)),
                field=self.current_field_snapshot,
                ep=ep,
            )

    def _save_summary(self, records):
        df = pd.DataFrame(records)
        df.to_csv(FileAddress.summary_csv(self.mode), index=False)

        suc = df[df["episode_success"] == 1]
        stat = pd.DataFrame([
            {
                "total_episodes": int(len(df)),
                "success_count": int(len(suc)),
                "success_rate_percent": float(len(suc) / max(len(df), 1) * 100.0),
                "shortest_path_m": float(suc["episode_path_m"].min()) if len(suc) else np.nan,
                "shortest_time_s": float(suc["episode_time_s"].min()) if len(suc) else np.nan,
                "average_path_m": float(suc["episode_path_m"].mean()) if len(suc) else np.nan,
                "average_time_s": float(suc["episode_time_s"].mean()) if len(suc) else np.nan,
                "max_reward": float(df["episode_reward"].max()) if len(df) else np.nan,
                "mean_reward": float(df["episode_reward"].mean()) if len(df) else np.nan,
            }
        ])
        stat.to_csv(FileAddress.stat_csv(self.mode), index=False)

        win = self.vis_cfg.reward_window_train if self.mode == "train" else self.vis_cfg.reward_window_eval
        win = max(1, int(win))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.vis_cfg.reward_size, sharex=True)

        ax1.plot(df["episode"], df["episode_reward"], alpha=0.45, label="Episode reward")
        ax1.plot(
            df["episode"],
            df["episode_reward"].rolling(win, min_periods=1).mean(),
            linewidth=2.0,
            label=f"Reward rolling mean ({win})",
        )
        ax1.set_ylabel("Reward")
        ax1.set_title(f"{self.mode.capitalize()} reward and success rate")
        ax1.grid(True, linestyle="--", alpha=0.35)
        ax1.legend(loc="best")

        rolling_success = df["episode_success"].rolling(win, min_periods=1).mean() * 100.0
        ax2.plot(df["episode"], df["episode_success"] * 100.0, alpha=0.25, label="Episode success")
        ax2.plot(df["episode"], rolling_success, linewidth=2.0, label=f"Success rate rolling mean ({win})")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Success rate (%)")
        ax2.set_ylim(-5, 105)
        ax2.grid(True, linestyle="--", alpha=0.35)
        ax2.legend(loc="best")

        fig.tight_layout()
        fig.savefig(FileAddress.curve_png(self.mode), dpi=self.vis_cfg.dpi, bbox_inches="tight")
        plt.close(fig)

    def run(self, episodes: int, max_steps: int):
        records = []
        for ep in range(1, episodes + 1):
            result = self._run_episode(ep, max_steps)
            self._save_episode_data(result, ep)
            
            print(
                f"[{self.mode.upper()}] Episode {ep}/{episodes} | "
                f"Reward: {result['reward']:.2f} | "
                f"Steps: {result['step_num']} | "
                f"Time: {result['time']:.2f}s | "
                f"Path: {result['path_len']:.2f}m | "
                f"Success: {bool(result['success'])}"
            )

            records.append(
                {
                    "episode": ep,
                    "episode_reward": result["reward"],
                    "episode_step_num": result["step_num"],
                    "episode_time_s": result["time"],
                    "episode_path_m": result["path_len"],
                    "episode_success": result["success"],
                }
            )

            if self.mode == "train" and ep % self.net_cfg.save_gap == 0:
                self.agent.save_weights(episode=ep, include_target=False)

        if self.mode == "train":
            self.agent.save_weights(episode=None, include_target=False)

        self._save_summary(records)
        return records


def train_agent(env, agent, total_episodes=MapConfig.train_eps, max_episode_steps=MapConfig.max_steps):
    return Runner(env, agent, "train").run(total_episodes, max_episode_steps)


def evaluate_agent(env, agent, eval_episodes=MapConfig.eval_eps, max_episode_steps=MapConfig.max_steps):
    return Runner(env, agent, "eval").run(eval_episodes, max_episode_steps)
