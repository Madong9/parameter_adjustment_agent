from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..schemas.task import TaskSpec
from ..utils.io import write_json
from .frame_sampler import FrameSampler


class SynchronizedEvidenceBuilder:
    """从完整同步轨迹构建不含奖励和 PPO 指标的行为证据。"""

    command_names = ("lin_vel_x", "lin_vel_y", "ang_vel_yaw", "heading")
    contact_names = ("front_left", "front_right", "rear_left", "rear_right")

    @staticmethod
    def _numbers(value: Any) -> List[float]:
        """把 Parquet 中的列表或数组安全转换为有限浮点数列表。"""
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            return []
        result: List[float] = []
        for item in value:
            try:
                number = float(item)
            except (TypeError, ValueError):
                number = 0.0
            result.append(number if np.isfinite(number) else 0.0)
        return result

    @classmethod
    def _command(cls, row: pd.Series) -> Dict[str, float]:
        """按训练环境约定把 command 向量映射为具名物理量。"""
        values = cls._numbers(row.get("command", []))
        return {name: round(values[index], 5) if index < len(values) else 0.0
                for index, name in enumerate(cls.command_names)}

    @classmethod
    def _force_norms(cls, row: pd.Series) -> Dict[str, float]:
        """计算四个足端三轴接触力向量的模长。"""
        raw = row.get("contact_forces", [])
        if isinstance(raw, np.ndarray):
            raw = raw.tolist()
        vectors = raw if isinstance(raw, (list, tuple)) else []
        result: Dict[str, float] = {}
        for index, name in enumerate(cls.contact_names):
            vector = cls._numbers(vectors[index]) if index < len(vectors) else []
            result[name] = round(float(np.linalg.norm(vector)), 4) if vector else 0.0
        return result

    @staticmethod
    def _longest_run(mask: pd.Series, frame_times: pd.Series) -> Dict[str, Any]:
        """计算布尔条件在完整轨迹上的最长连续区间。"""
        best_start = best_end = None
        current_start = None
        values = mask.astype(bool).tolist()
        for index, active in enumerate(values + [False]):
            if active and current_start is None:
                current_start = index
            elif not active and current_start is not None:
                end = index - 1
                if best_start is None or end - current_start > best_end - best_start:
                    best_start, best_end = current_start, end
                current_start = None
        if best_start is None or best_end is None:
            return {"frames": 0, "start_frame": None, "end_frame": None, "duration_seconds": 0.0}
        if len(frame_times) > 1:
            step = float(frame_times.diff().dropna().median())
        else:
            step = 0.0
        return {
            "frames": int(best_end - best_start + 1),
            "start_frame": int(best_start),
            "end_frame": int(best_end),
            "duration_seconds": round(float(frame_times.iloc[best_end] - frame_times.iloc[best_start] + step), 4),
        }

    @classmethod
    def _frame_record(cls, row: pd.Series, events: Sequence[str]) -> Dict[str, Any]:
        """生成单帧的紧凑同步物理证据。"""
        contacts = {name: bool(row.get(column, False)) for name, column in zip(
            cls.contact_names, ("contact_fl", "contact_fr", "contact_rl", "contact_rr"))}
        forbidden_force = row.get("forbidden_contact_force_max",
                                  row.get("penalized_contact_force_max", None))
        if forbidden_force is not None and not pd.isna(forbidden_force):
            forbidden_force = round(float(forbidden_force), 4)
        else:
            forbidden_force = None
        return {
            "frame": int(row.get("video_frame", 0)),
            "time_seconds": round(float(row.get("sim_time", 0.0)), 4),
            "events": list(events),
            "command": cls._command(row),
            "measured_velocity_mps": {
                "x": round(float(row.get("base_vx", 0.0)), 5),
                "y": round(float(row.get("base_vy", 0.0)), 5),
                "z": round(float(row.get("base_vz", 0.0)), 5),
            },
            "pose": {
                "base_height_m": round(float(row.get("base_z", 0.0)), 5),
                "roll_rad": round(float(row.get("roll", 0.0)), 5),
                "pitch_rad": round(float(row.get("pitch", 0.0)), 5),
            },
            "feet_contact": contacts,
            "feet_contact_force_norm_n": cls._force_norms(row),
            "forbidden_body_contact_force_max_n": forbidden_force,
            "body_collision": bool(row.get("body_collision", False)),
            "fall": bool(row.get("fall", False)),
            "foot_slip_mps": round(float(row.get("foot_slip", 0.0)), 5),
        }

    @staticmethod
    def _event_map(events: Iterable[Dict[str, Any]]) -> Dict[int, List[str]]:
        """把事件列表转换为按帧索引分组的名称映射。"""
        result: Dict[int, List[str]] = {}
        for event in events:
            result.setdefault(int(event["frame"]), []).append(str(event["name"]))
        return result

    @classmethod
    def annotations(cls, trajectory: pd.DataFrame,
                    events: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, str]]:
        """为接触图生成含 command、接触力和安全标志的紧凑标注。"""
        event_map = cls._event_map(events)
        result: Dict[int, Dict[str, str]] = {}
        for _, row in trajectory.iterrows():
            frame = int(row["video_frame"])
            command = cls._command(row)
            forces = cls._force_norms(row)
            contacts = "".join("1" if bool(row.get(name, False)) else "0" for name in
                               ("contact_fl", "contact_fr", "contact_rl", "contact_rr"))
            body_force = row.get("forbidden_contact_force_max",
                                 row.get("penalized_contact_force_max", None))
            body_force_text = "NA" if body_force is None or pd.isna(body_force) else "%.2fN" % float(body_force)
            result[frame] = {
                "frame_time": "%d  t=%.2fs" % (frame, float(row["sim_time"])),
                "event_phase": ",".join(event_map.get(frame, [])) or "-",
                "command_target": "cmd vx/vy/yaw %.2f/%.2f/%.2f" %
                                  (command["lin_vel_x"], command["lin_vel_y"], command["ang_vel_yaw"]),
                "measured_velocity": "vel x/y/z %.2f/%.2f/%.2f" %
                                     (float(row.get("base_vx", 0.0)), float(row.get("base_vy", 0.0)),
                                      float(row.get("base_vz", 0.0))),
                "pose": "z/r/p %.2f/%.2f/%.2f" %
                        (float(row.get("base_z", 0.0)), float(row.get("roll", 0.0)),
                         float(row.get("pitch", 0.0))),
                "feet_contacts": "feet FL/FR/RL/RR %s" % contacts,
                "foot_forces": "foot N %.1f/%.1f/%.1f/%.1f" % tuple(forces.values()),
                "safety_flags": "body=%s(%s) fall=%s slip=%.2f" %
                                (bool(row.get("body_collision", False)), body_force_text,
                                 bool(row.get("fall", False)), float(row.get("foot_slip", 0.0))),
            }
        return result

    @staticmethod
    def critical_indices(frame_count: int, events: Iterable[Dict[str, Any]],
                         radius: int = 3, maximum: int = 48) -> List[int]:
        """选择关键事件连续窗口，并限制接触图的最大帧数。"""
        indices = FrameSampler.event_indices(frame_count, [int(event["frame"]) for event in events], radius)
        if len(indices) <= maximum:
            return indices
        positions = np.linspace(0, len(indices) - 1, maximum, dtype=int)
        return sorted({indices[int(position)] for position in positions})

    @staticmethod
    def multiview_indices(frame_count: int, events: Iterable[Dict[str, Any]], maximum: int = 12) -> List[int]:
        """合并全程均匀帧和关键事件帧，生成多视角对照索引。"""
        values = set(FrameSampler.uniform_indices(frame_count, 6))
        values.update(int(event["frame"]) for event in events)
        ordered = sorted(index for index in values if 0 <= index < frame_count)
        if len(ordered) <= maximum:
            return ordered
        positions = np.linspace(0, len(ordered) - 1, maximum, dtype=int)
        return sorted({ordered[int(position)] for position in positions})

    def build(self, task: TaskSpec, trajectory: pd.DataFrame, events: List[Dict[str, Any]],
              output: Path, metadata: Optional[Dict[str, Any]] = None) -> Path:
        """扫描完整轨迹并写入可供视觉评论员引用的同步行为证据。"""
        if trajectory.empty:
            raise ValueError("behavior evidence requires a non-empty trajectory")
        contact_columns = [name for name in ("contact_fl", "contact_fr", "contact_rl", "contact_rr")
                           if name in trajectory]
        contacts = trajectory[contact_columns].astype(bool) if len(contact_columns) == 4 else pd.DataFrame(
            False, index=trajectory.index, columns=("contact_fl", "contact_fr", "contact_rl", "contact_rr"))
        upright = (trajectory["roll"].abs() <= 0.8) & (trajectory["pitch"].abs() <= 1.0)
        rear_support_stand = contacts[["contact_rl", "contact_rr"]].any(axis=1) & ~contacts[
            ["contact_fl", "contact_fr"]].any(axis=1) & upright
        front_support_stand = contacts[["contact_fl", "contact_fr"]].any(axis=1) & ~contacts[
            ["contact_rl", "contact_rr"]].any(axis=1) & upright
        all_air = ~contacts.any(axis=1)
        command_matrix = np.array([self._numbers(value)[:4] + [0.0] * max(0, 4 - len(self._numbers(value)[:4]))
                                   for value in trajectory["command"]], dtype=float)
        target_x = command_matrix[:, 0]
        measured_x = trajectory["base_vx"].astype(float).to_numpy()
        event_map = self._event_map(events)
        centers = sorted(set(int(event["frame"]) for event in events))
        window_indices = FrameSampler.event_indices(len(trajectory), centers, radius=3)
        if len(window_indices) > 64:
            positions = np.linspace(0, len(window_indices) - 1, 64, dtype=int)
            window_indices = sorted({window_indices[int(position)] for position in positions})
        evidence = {
            "evidence_version": 1,
            "scope": "同步物理行为证据；不包含 reward、PPO、loss 或训练分数",
            "task": {
                "task_id": task.task_id,
                "instruction": task.original_instruction,
                "required_behaviors": [item.dict() for item in task.required_behaviors],
                "forbidden_behaviors": [item.dict() for item in task.forbidden_behaviors],
                "visual_requirements": task.visual_evaluation_requirements,
            },
            "coverage": {
                "frame_count": int(len(trajectory)),
                "analyzed_frame_count": int(len(trajectory)),
                "start_time_seconds": round(float(trajectory["sim_time"].min()), 4),
                "end_time_seconds": round(float(trajectory["sim_time"].max()), 4),
                "all_frames_scanned_for_upright_and_contacts": True,
            },
            "command_tracking": {
                "command_vector_order": list(self.command_names),
                "target_lin_vel_x_mps": {
                    "mean": round(float(target_x.mean()), 5), "min": round(float(target_x.min()), 5),
                    "max": round(float(target_x.max()), 5),
                },
                "measured_base_vx_mps": {
                    "mean": round(float(measured_x.mean()), 5), "min": round(float(measured_x.min()), 5),
                    "max": round(float(measured_x.max()), 5),
                },
                "absolute_tracking_error_mps": {
                    "mean": round(float(np.abs(measured_x - target_x).mean()), 5),
                    "p95": round(float(np.percentile(np.abs(measured_x - target_x), 95)), 5),
                },
                "world_frame_displacement_m": {
                    "x": round(float(trajectory["base_x"].iloc[-1] - trajectory["base_x"].iloc[0]), 5),
                    "y": round(float(trajectory["base_y"].iloc[-1] - trajectory["base_y"].iloc[0]), 5),
                    "note": "世界坐标位移；command 和 base_vx/base_vy 为机体坐标，不可直接按符号比较",
                },
            },
            "continuous_behavior_scan": {
                "rear_support_stand_definition": "至少一个后足接触、两个前足均不接触，且姿态未越过安全界限",
                "rear_support_stand_candidate_frames": int(rear_support_stand.sum()),
                "longest_rear_support_stand_run": self._longest_run(
                    rear_support_stand, trajectory["sim_time"]),
                "front_support_stand_definition": "至少一个前足接触、两个后足均不接触，且姿态未越过安全界限",
                "front_support_stand_candidate_frames": int(front_support_stand.sum()),
                "longest_front_support_stand_run": self._longest_run(
                    front_support_stand, trajectory["sim_time"]),
                # 保留旧字段供历史评论模板兼容，语义明确为“后足支撑”。
                "rear_stand_definition": "至少一个后足接触、两个前足均不接触，且姿态未越过安全界限",
                "rear_stand_candidate_frames": int(rear_support_stand.sum()),
                "longest_rear_stand_run": self._longest_run(
                    rear_support_stand, trajectory["sim_time"]),
                "airborne_frames": int(all_air.sum()),
                "longest_airborne_run": self._longest_run(all_air, trajectory["sim_time"]),
                "front_contact_fraction": round(float(contacts[["contact_fl", "contact_fr"]].any(axis=1).mean()), 5),
                "rear_contact_fraction": round(float(contacts[["contact_rl", "contact_rr"]].any(axis=1).mean()), 5),
            },
            "safety_scan": {
                "body_collision_flag_definition": (
                    "max(contact force norm over every non-foot rigid body) > 0.1 N"),
                "body_collision_frames": [int(value) for value in trajectory.loc[
                    trajectory.get("body_collision", False).astype(bool), "video_frame"].tolist()]
                if "body_collision" in trajectory else [],
                "fall_frames": [int(value) for value in trajectory.loc[
                    trajectory.get("fall", False).astype(bool), "video_frame"].tolist()]
                if "fall" in trajectory else [],
                "max_penalized_link_contact_force_n": (
                    round(float(trajectory["penalized_contact_force_max"].max()), 5)
                    if "penalized_contact_force_max" in trajectory else None),
                "max_termination_link_contact_force_n": (
                    round(float(trajectory["termination_contact_force_max"].max()), 5)
                    if "termination_contact_force_max" in trajectory else None),
                "max_all_nonfoot_contact_force_n": (
                    round(float(trajectory["nonfoot_contact_force_max"].max()), 5)
                    if "nonfoot_contact_force_max" in trajectory else None),
                "max_forbidden_body_contact_force_n": (
                    round(float(trajectory["forbidden_contact_force_max"].max()), 5)
                    if "forbidden_contact_force_max" in trajectory else
                    (round(float(trajectory["penalized_contact_force_max"].max()), 5)
                     if "penalized_contact_force_max" in trajectory else None)),
                "max_foot_slip_mps": round(float(trajectory["foot_slip"].max()), 5),
                "max_foot_slip_frame": int(trajectory.loc[trajectory["foot_slip"].idxmax(), "video_frame"]),
                "sensor_configuration": {
                    "collision_force_threshold_n": (metadata or {}).get(
                        "body_collision_force_threshold_n", 0.1),
                    "penalized_contact_patterns": (metadata or {}).get(
                        "penalized_contact_patterns", []),
                    "termination_contact_patterns": (metadata or {}).get(
                        "termination_contact_patterns", []),
                    "body_collision_scope": (metadata or {}).get(
                        "body_collision_scope", "legacy_configured_contact_links"),
                    "nonfoot_rigid_body_count": (metadata or {}).get(
                        "nonfoot_rigid_body_count", None),
                },
            },
            "vertical_motion": {
                "peak_upward_velocity_mps": round(float(trajectory["base_vz"].max()), 5),
                "peak_upward_velocity_frame": int(trajectory.loc[trajectory["base_vz"].idxmax(), "video_frame"]),
                "base_height_range_m": [round(float(trajectory["base_z"].min()), 5),
                                        round(float(trajectory["base_z"].max()), 5)],
                "airborne_frames": int(all_air.sum()),
            },
            "event_windows": [self._frame_record(trajectory.iloc[index], event_map.get(
                int(trajectory.iloc[index]["video_frame"]), [])) for index in window_indices],
        }
        write_json(output, evidence)
        return output
