from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


class TrajectoryMetrics:
    def compute(self, trajectory: pd.DataFrame) -> Dict[str, float]:
        """从同步轨迹表计算任务相关物理指标。"""
        if trajectory.empty:
            return {"nan_count": 0.0, "frame_count": 0.0}
        metrics = {
            "nan_count": float(trajectory.isna().sum().sum()),
            "frame_count": float(len(trajectory)),
            "duration": float(trajectory["sim_time"].max() - trajectory["sim_time"].min()),
        }
        if "base_z" in trajectory:
            metrics["jump_height"] = float(trajectory["base_z"].max() - trajectory["base_z"].iloc[0])
        if "roll" in trajectory:
            metrics["max_abs_roll"] = float(trajectory["roll"].abs().max())
        if "pitch" in trajectory:
            metrics["max_abs_pitch"] = float(trajectory["pitch"].abs().max())
            metrics["body_pitch_within_limit"] = metrics["max_abs_pitch"]
        if "roll" in trajectory and "pitch" in trajectory:
            squared_tilt = trajectory["roll"].astype(float) ** 2 + trajectory["pitch"].astype(float) ** 2
            metrics["orientation"] = float(np.exp(-squared_tilt / 0.25).mean())
            metrics["roll_limit"] = metrics["max_abs_roll"]
            metrics["pitch_limit"] = metrics["max_abs_pitch"]
        if "energy" in trajectory:
            metrics["energy"] = float(trajectory["energy"].sum())
        contact_columns = [name for name in ("contact_fl", "contact_fr", "contact_rl", "contact_rr")
                           if name in trajectory.columns]
        if contact_columns:
            metrics["mean_contact_count"] = float(trajectory[contact_columns].astype(bool).sum(axis=1).mean())
            front_contact = trajectory[[name for name in ("contact_fl", "contact_fr")
                                        if name in trajectory]].astype(bool).any(axis=1)
            rear_contact = trajectory[[name for name in ("contact_rl", "contact_rr")
                                       if name in trajectory]].astype(bool).any(axis=1)
            metrics["front_contact_fraction"] = float(front_contact.mean())
            metrics["front_leg_off_ground_ratio"] = float((~front_contact).mean())
            metrics["rear_leg_off_ground_ratio"] = float((~rear_contact).mean())
            upright = ((trajectory.get("roll", 0.0).abs() <= 0.8) &
                       (trajectory.get("pitch", 0.0).abs() <= 1.0))
            rear_stand = rear_contact & ~front_contact & upright
            front_stand = front_contact & ~rear_contact & upright
            dt = float(trajectory["sim_time"].diff().dropna().median()) if len(trajectory) > 1 else 0.0

            def longest_duration(mask: pd.Series) -> float:
                """计算一个布尔姿态条件的最长连续持续时间。"""
                groups = (mask != mask.shift(fill_value=False)).cumsum()
                longest = int(mask.groupby(groups).sum().max()) if len(mask) else 0
                return float(longest * dt)

            metrics["rear_stand_duration"] = longest_duration(rear_stand)
            metrics["rear_leg_stand_duration"] = metrics["rear_stand_duration"]
            metrics["front_leg_stand_duration"] = longest_duration(front_stand)
            metrics["stable_stand_duration"] = max(
                metrics["rear_leg_stand_duration"], metrics["front_leg_stand_duration"])
        if "command" in trajectory and "base_vx" in trajectory:
            def command_x(value: object) -> float:
                """从 rollout 命令向量中读取前向速度目标。"""
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                return float(value[0]) if isinstance(value, (list, tuple)) and value else 0.0
            target = trajectory["command"].map(command_x).to_numpy(dtype=float)
            actual = trajectory["base_vx"].to_numpy(dtype=float)
            error = np.abs(target - actual)
            metrics["tracking_error"] = float(error.mean())
            metrics["tracking_lin_vel"] = float(np.exp(-np.square(target - actual) / 0.25).mean())
            if contact_columns:
                gated_tracking = np.exp(-np.square(target - actual) / 0.25)
                rear_mask = rear_stand.to_numpy(dtype=bool)
                front_mask = front_stand.to_numpy(dtype=bool)
                metrics["rear_leg_walk_velocity_tracking"] = (
                    float(gated_tracking[rear_mask].mean()) if rear_mask.any() else 0.0)
                metrics["front_leg_walk_velocity_tracking"] = (
                    float(gated_tracking[front_mask].mean()) if front_mask.any() else 0.0)
                metrics["walking_speed_tracking"] = max(
                    metrics["rear_leg_walk_velocity_tracking"],
                    metrics["front_leg_walk_velocity_tracking"])
                rear_tracking = pd.Series(rear_mask & (gated_tracking >= 0.7), index=trajectory.index)
                front_tracking = pd.Series(front_mask & (gated_tracking >= 0.7), index=trajectory.index)
                metrics["rear_leg_walk_completion"] = longest_duration(rear_tracking)
                metrics["front_leg_walk_completion"] = longest_duration(front_tracking)
        if "body_collision" in trajectory:
            collision_frames = float(trajectory["body_collision"].astype(bool).sum())
            metrics["body_collision_count"] = collision_frames
            metrics["forbidden_collisions"] = collision_frames
        force_column = next((name for name in ("forbidden_contact_force_max", "penalized_contact_force_max")
                             if name in trajectory), None)
        if force_column:
            metrics["forbidden_contact_force"] = float(trajectory[force_column].astype(float).max())
            metrics["forbidden_body_contact"] = metrics["forbidden_contact_force"]
        if "fall" in trajectory:
            metrics["fall_rate"] = float(trajectory["fall"].astype(bool).mean())
        if "termination_reason" in trajectory:
            abnormal = float((trajectory["termination_reason"].fillna("").astype(str) == "reset").sum())
            metrics["abnormal_terminations"] = abnormal
            metrics["episode_survival"] = 1.0 if abnormal == 0 else 0.0
        return metrics
