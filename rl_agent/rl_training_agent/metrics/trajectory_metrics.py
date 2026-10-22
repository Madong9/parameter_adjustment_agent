from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from .velocity import heading_forward_velocity


class TrajectoryMetrics:
    @staticmethod
    def _front_flip_metrics(trajectory):
        """从完整 xyzw 四元数序列验收单次正向腾空旋转，缺证据和重置保持未知。"""
        from ..utils.rotation import quaternion_increment_numpy
        result = {'final_pitch_angle': float('nan'), 'pitch_axis_error': float('nan')}
        quaternion_columns = ['quat_x', 'quat_y', 'quat_z', 'quat_w']
        contact_columns = ['contact_fl', 'contact_fr', 'contact_rl', 'contact_rr']
        if not all(name in trajectory for name in quaternion_columns + contact_columns):
            return result
        reasons = trajectory.get('termination_reason', pd.Series('', index=trajectory.index))
        if (len(trajectory) < 2 or reasons.eq('reset').any() or
                reasons.iloc[:-1].eq('timeout').any()):
            return result
        times = trajectory['sim_time'].to_numpy(dtype=float)
        differences = np.diff(times)
        if not np.isfinite(times).all() or (differences <= 0).any() or differences.max() > 2 * np.median(differences) + 1e-6:
            return result
        quaternions = trajectory[quaternion_columns].to_numpy(dtype=float)
        rotation, valid = quaternion_increment_numpy(quaternions[:-1], quaternions[1:])
        if not valid.all():
            return result
        raw_contacts = trajectory[contact_columns]
        if raw_contacts.isna().any().any():
            return result
        contacts = raw_contacts.to_numpy(dtype=bool)
        airborne = ~contacts.any(axis=1)
        support = np.flatnonzero(contacts.all(axis=1))
        if not len(support):
            return dict(final_pitch_angle=0., pitch_axis_error=0.)
        first_support = int(support[0])
        starts = np.flatnonzero(airborne & ~np.r_[False, airborne[:-1]])
        starts = starts[starts > first_support]
        # 以首次全足支撑作为起跳来源；多次动作起跳不能拼凑一次前翻。
        if not len(starts):
            return dict(final_pitch_angle=0., pitch_axis_error=0.)
        if len(starts) != 1 or starts[0] == 0 or airborne[-1]:
            return dict(final_pitch_angle=0., pitch_axis_error=0.)
        start = int(starts[0])
        end = start
        while end < len(airborne) and airborne[end]:
            end += 1
        if not contacts[start - 1].any() or not contacts[end:].all(axis=1).any():
            return dict(final_pitch_angle=0., pitch_axis_error=0.)
        segment = rotation[start - 1:end]
        magnitude = float(np.linalg.norm(segment, axis=1).sum())
        return {'final_pitch_angle': float(segment[:, 1].sum()),
                'pitch_axis_error': float(np.linalg.norm(segment[:, [0, 2]], axis=1).sum() / magnitude) if magnitude > 1e-8 else 0.}

    @staticmethod
    def _pilot_episode_segments(trajectory):
        """只接受有同步采集标记的完整控制步证据，切分原生重置而不跨边界积分。"""
        required = ['sim_time', 'quat_x', 'quat_y', 'quat_z', 'quat_w', 'roll', 'pitch',
                    'contact_fl', 'contact_fr', 'contact_rl', 'contact_rr',
                    'termination_reason', 'episode_id', 'episode_step', 'terminal_state_preserved']
        if len(trajectory) < 2 or not all(name in trajectory for name in required):
            return None
        try:
            times = trajectory['sim_time'].to_numpy(dtype=float)
            differences = np.diff(times)
            if (not np.isfinite(times).all() or (differences <= 0).any() or
                    not np.allclose(differences, np.median(differences), rtol=1e-4, atol=1e-8)):
                return None
            pose = trajectory[['quat_x', 'quat_y', 'quat_z', 'quat_w', 'roll', 'pitch']].to_numpy(dtype=float)
            with np.errstate(over='ignore', invalid='ignore'):
                norms = np.linalg.norm(pose[:, :4], axis=1)
            if (not np.isfinite(pose).all() or not np.isfinite(norms).all() or
                    (norms <= 1e-8).any()):
                return None
            for name in ('contact_fl', 'contact_fr', 'contact_rl', 'contact_rr', 'terminal_state_preserved'):
                values = trajectory[name]
                if (values.isna().any() or not values.isin([True, False]).all() or
                        not (pd.api.types.is_bool_dtype(values) or pd.api.types.is_numeric_dtype(values))):
                    return None
            reasons = trajectory['termination_reason']
            if not reasons.isin(['', 'reset', 'timeout']).all():
                return None
            terminal = reasons.isin(['reset', 'timeout']).to_numpy(dtype=bool)
            preserved = trajectory['terminal_state_preserved'].to_numpy(dtype=bool)
            if not np.array_equal(terminal, preserved):
                return None
            episode_ids = trajectory['episode_id'].to_numpy(dtype=float)
            episode_steps = trajectory['episode_step'].to_numpy(dtype=float)
            if (not np.isfinite(episode_ids).all() or not np.isfinite(episode_steps).all() or
                    (episode_ids < 0).any() or (episode_steps < 1).any() or
                    (episode_ids != np.floor(episode_ids)).any() or
                    (episode_steps != np.floor(episode_steps)).any()):
                return None
            boundaries = terminal[:-1]
            if (not np.array_equal(np.diff(episode_ids), boundaries.astype(int)) or
                    not np.array_equal(np.diff(episode_steps)[~boundaries],
                                       np.ones((~boundaries).sum())) or
                    (episode_steps[1:][boundaries] != 1).any()):
                return None
            end_indices = list(np.flatnonzero(terminal) + 1)
            if not end_indices or end_indices[-1] != len(trajectory):
                end_indices.append(len(trajectory))
            starts = [0] + end_indices[:-1]
            segments = [trajectory.iloc[start:end].copy() for start, end in zip(starts, end_indices)]
            from ..utils.rotation import quaternion_increment_numpy
            for segment in segments:
                quaternion = segment[['quat_x', 'quat_y', 'quat_z', 'quat_w']].to_numpy(dtype=float)
                if len(quaternion) > 1:
                    _, valid = quaternion_increment_numpy(quaternion[:-1], quaternion[1:])
                    if not valid.all():
                        return None
            return segments
        except (ValueError, TypeError, OverflowError):
            return None

    @staticmethod
    def _pilot_front_flip_metrics(episode, dt):
        """记录一个 episode 首次起跳的有限局部进度，尚未落地也不伪装成缺失采样。"""
        from ..utils.rotation import quaternion_increment_numpy
        result = {'final_pitch_angle': 0., 'pitch_axis_error': 0.,
                  'landing_stability': 0., 'stable_stand_duration': 0.}
        contacts = episode[['contact_fl', 'contact_fr', 'contact_rl', 'contact_rr']].to_numpy(dtype=bool)
        support = np.flatnonzero(contacts.all(axis=1))
        if not len(support) or len(episode) < 2:
            return result
        airborne = ~contacts.any(axis=1)
        starts = np.flatnonzero(airborne & ~np.r_[False, airborne[:-1]])
        starts = starts[starts > support[0]]
        if not len(starts):
            return result
        start = int(starts[0])
        end = start
        while end < len(episode) and airborne[end]:
            end += 1
        quaternions = episode[['quat_x', 'quat_y', 'quat_z', 'quat_w']].to_numpy(dtype=float)
        rotation, _ = quaternion_increment_numpy(quaternions[:-1], quaternions[1:])
        segment = rotation[start - 1:end]
        magnitude = float(np.linalg.norm(segment, axis=1).sum())
        result['final_pitch_angle'] = float(segment[:, 1].sum())
        result['pitch_axis_error'] = (float(np.linalg.norm(segment[:, [0, 2]], axis=1).sum() / magnitude)
                                      if magnitude > 1e-8 else 0.)
        recovery_end = int(starts[1]) if len(starts) > 1 else len(episode)
        recovery = episode.iloc[end:recovery_end]
        if not recovery.empty:
            all_contact = recovery[['contact_fl', 'contact_fr', 'contact_rl', 'contact_rr']].astype(bool).all(axis=1)
            result['landing_stability'] = float((all_contact.astype(float) * np.exp(
                -4. * (recovery['roll'] ** 2 + recovery['pitch'] ** 2))).mean())
            upright = all_contact & (recovery['roll'].abs() <= .8) & (recovery['pitch'].abs() <= 1.)
            groups = (upright != upright.shift(fill_value=False)).cumsum()
            result['stable_stand_duration'] = float(upright.groupby(groups).sum().max() * dt)
        return result

    def compute_pilot(self, trajectory: pd.DataFrame) -> Dict[str, float]:
        """仅用可证明同步的分段前翻证据检查早期学习，最终验收继续调用严格 compute。"""
        try:
            metrics = self.compute(trajectory)
        except (KeyError, TypeError, ValueError, OverflowError):
            metrics = {'nan_count': float('nan'), 'frame_count': float(len(trajectory))}
        metrics.update(final_pitch_angle=float('nan'), pitch_axis_error=float('nan'),
                       pilot_evidence_valid=0., pilot_episode_count=0., pilot_reset_count=0.)
        episodes = self._pilot_episode_segments(trajectory)
        if episodes is None:
            return metrics
        dt = float(np.median(np.diff(trajectory['sim_time'].to_numpy(dtype=float))))
        progress = [self._pilot_front_flip_metrics(episode, dt) for episode in episodes]
        for name in progress[0]:
            metrics[name] = float(np.mean([item[name] for item in progress]))
        metrics.update(pilot_evidence_valid=1., pilot_episode_count=float(len(episodes)),
                       pilot_reset_count=float(trajectory['termination_reason'].eq('reset').sum()))
        return metrics

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
        if all(name in trajectory for name in ('quat_x', 'quat_y', 'quat_z', 'quat_w')):
            metrics.update(self._front_flip_metrics(trajectory))
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
