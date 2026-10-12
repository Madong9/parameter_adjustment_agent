from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from .velocity import heading_forward_velocity


class TrajectoryMetrics:
    @staticmethod
    def _nonfinite_count(value) -> int:
        """递归检查标量和嵌套关节数组，NaN、Inf 和缺失值均属于无效证据。"""
        if isinstance(value, (list, tuple, np.ndarray)):
            return sum(TrajectoryMetrics._nonfinite_count(item) for item in value)
        if value is None:
            return 1
        if isinstance(value, (float, int, np.number)):
            return int(not np.isfinite(value))
        return 0

    @staticmethod
    def _limit_violations(trajectory, values, limits, paired=False) -> float:
        """使用轨迹携带的真实模型限位计算越限；缺失或形状错误返回未知。"""
        if values not in trajectory or limits not in trajectory:
            return float('nan')
        total = 0
        try:
            for raw, bound in zip(trajectory[values], trajectory[limits]):
                actual = np.asarray(raw, dtype=float)
                limit = (np.stack([np.asarray(pair, dtype=float) for pair in bound])
                         if paired else np.asarray(bound, dtype=float))
                if actual.ndim != 1 or not len(actual) or not np.isfinite(actual).all() or not np.isfinite(limit).all():
                    return float('nan')
                if paired:
                    if limit.shape != (len(actual), 2) or (limit[:, 0] > limit[:, 1]).any():
                        return float('nan')
                    total += int(((actual < limit[:, 0] - 1e-6) | (actual > limit[:, 1] + 1e-6)).sum())
                else:
                    if limit.shape != actual.shape or (limit <= 0).any():
                        return float('nan')
                    total += int((np.abs(actual) > limit + 1e-6).sum())
        except (TypeError, ValueError):
            return float('nan')
        return float(total)

    @staticmethod
    def _jump_metrics(trajectory, contact_columns) -> Dict[str, float]:
        """从每次实际腾空前的离地时序和落地首帧计算保守跳跃指标。"""
        result = {'takeoff_time_spread': float('nan'), 'landing_pitch_abs': float('nan')}
        if len(contact_columns) != 4 or 'pitch' not in trajectory:
            return result
        contacts = trajectory[contact_columns].to_numpy(dtype=bool)
        times = trajectory['sim_time'].to_numpy(dtype=float)
        if not np.isfinite(times).all() or trajectory.get(
                'termination_reason', pd.Series('', index=trajectory.index)).eq('reset').any():
            return result
        airborne = ~contacts.any(axis=1)
        starts = np.flatnonzero(airborne & ~np.r_[False, airborne[:-1]])
        spreads, pitches = [], []
        previous_end = 0
        for start in starts:
            end = int(start)
            while end < len(airborne) and airborne[end]:
                end += 1
            departures = []
            for foot in range(4):
                grounded = np.flatnonzero(contacts[previous_end:start, foot])
                if not len(grounded):
                    return result
                departures.append(times[previous_end + int(grounded[-1]) + 1])
            spreads.append(max(departures) - min(departures))
            if end >= len(airborne):
                return result
            pitches.append(abs(float(trajectory['pitch'].iloc[end])))
            previous_end = end
        if spreads and np.isfinite(pitches).all():
            result.update(takeoff_time_spread=float(max(spreads)), landing_pitch_abs=float(max(pitches)))
        return result

    def compute(self, trajectory: pd.DataFrame) -> Dict[str, float]:
        """从同步轨迹表计算任务相关物理指标。"""
        if trajectory.empty:
            return {"nan_count": float('nan'), "frame_count": 0.0}
        metrics = {
            "nan_count": float(sum(self._nonfinite_count(value) for row in trajectory.itertuples(index=False, name=None) for value in row)),
            "frame_count": float(len(trajectory)),
            "duration": float(trajectory["sim_time"].max() - trajectory["sim_time"].min()),
        }
        metrics['joint_limit_violations'] = self._limit_violations(
            trajectory, 'joint_positions', 'joint_position_limits', paired=True)
        metrics['torque_limit_violations'] = self._limit_violations(
            trajectory, 'joint_torques', 'joint_torque_limits')
        if 'joint_soft_position_limits' in trajectory:
            metrics['joint_soft_limit_violations'] = self._limit_violations(
                trajectory, 'joint_positions', 'joint_soft_position_limits', paired=True)
        if "base_z" in trajectory:
            metrics["jump_height"] = float(trajectory["base_z"].max() - trajectory["base_z"].iloc[0])
        if "roll" in trajectory:
            metrics["max_abs_roll"] = float(trajectory["roll"].abs().max())
            # 欧拉角在 +/-pi 处回绕；完整侧翻必须依据展开后的净旋转，
            # 不能用最大姿态角代替，也不能把来回摆动的路程累计成一圈。
            angles = trajectory["roll"].to_numpy(dtype=float)
            if np.isfinite(angles).all() and not trajectory.get(
                    "termination_reason", pd.Series("", index=trajectory.index)).eq("reset").any():
                unwrapped = np.unwrap(angles)
                metrics["final_roll_angle"] = float(abs(unwrapped[-1] - unwrapped[0]))
        if "base_wx" in trajectory:
            rates = trajectory["base_wx"].to_numpy(dtype=float)
            if np.isfinite(rates).all():
                metrics["max_roll_velocity"] = float(np.abs(rates).max())
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
            metrics.update(self._jump_metrics(trajectory, contact_columns))
            metrics["mean_contact_count"] = float(trajectory[contact_columns].astype(bool).sum(axis=1).mean())
            front_contact = trajectory[[name for name in ("contact_fl", "contact_fr")
                                        if name in trajectory]].astype(bool).any(axis=1)
            rear_contact = trajectory[[name for name in ("contact_rl", "contact_rr")
                                       if name in trajectory]].astype(bool).any(axis=1)
            metrics["front_contact_fraction"] = float(front_contact.mean())
            metrics["front_leg_off_ground_ratio"] = float((~front_contact).mean())
            metrics["rear_leg_off_ground_ratio"] = float((~rear_contact).mean())
            unknown_pose = pd.Series(float('nan'), index=trajectory.index)
            upright = ((trajectory.get("roll", unknown_pose).abs() <= 0.8) &
                       (trajectory.get("pitch", unknown_pose).abs() <= 1.0))
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
            if len(contact_columns) == 4 and "roll" in trajectory and "pitch" in trajectory:
                all_contact = trajectory[contact_columns].astype(bool).all(axis=1)
                airborne = ~trajectory[contact_columns].astype(bool).any(axis=1)
                metrics["feet_air_time"] = float(airborne.sum() * dt)
                # 落地验收只看最后一次腾空后的恢复窗口，初始站立不贡献分数。
                flight = np.flatnonzero(airborne.to_numpy())
                recovery_start = int(flight[-1]) + 1 if len(flight) else len(trajectory)
                recovery = all_contact.iloc[recovery_start:].astype(float) * np.exp(
                    -4.0 * (trajectory["roll"].iloc[recovery_start:] ** 2 +
                            trajectory["pitch"].iloc[recovery_start:] ** 2))
                metrics["landing_stability"] = float(recovery.mean()) if len(recovery) else 0.0
                if len(flight):
                    metrics["stable_stand_duration"] = longest_duration(
                        (all_contact & upright).iloc[recovery_start:])
                else:
                    metrics["stable_stand_duration"] = max(
                        metrics["stable_stand_duration"], longest_duration(all_contact & upright))
            forward = heading_forward_velocity(trajectory)
            for prefix, mask in (("front_leg", front_stand), ("rear_leg", rear_stand)):
                values = forward[mask.to_numpy(dtype=bool)]
                if len(values) and np.isfinite(values).all():
                    metrics[prefix + "_forward_speed"] = float(values.mean())
                elif not len(values):
                    metrics[prefix + "_forward_speed"] = 0.0
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
