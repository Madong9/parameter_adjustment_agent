"""将模型生成的验收指标名称规范为项目可计算的物理指标。"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List

from ..schemas.task import TaskSpec


EXPLICIT_ALIASES = {
    "front_leg_walk_duration": "front_leg_walk_completion",
    "rear_leg_walk_duration": "rear_leg_walk_completion",
    "body_contact_force": "forbidden_body_contact",
    "forbidden_contact_force": "forbidden_body_contact",
    "walking_velocity_tracking": "walking_speed_tracking",
    "velocity_tracking": "tracking_lin_vel",
    "stand_duration": "stable_stand_duration",
    "body_pitch": "body_pitch_within_limit",
    "body_roll": "roll_limit",
}


def _token(value: str) -> str:
    """把指标名称转换为稳定的小写下划线形式。"""
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def canonical_metric_name(name: str, supported: Iterable[str]) -> str:
    """根据显式别名和物理语义把一个指标映射到受支持名称。"""
    supported_names = set(supported)
    normalized = _token(name)
    if normalized in supported_names:
        return normalized
    alias = EXPLICIT_ALIASES.get(normalized)
    if alias in supported_names:
        return alias
    if "contact" in normalized and ("force" in normalized or "collision" in normalized):
        return "forbidden_body_contact" if "forbidden_body_contact" in supported_names else normalized
    if "pitch" in normalized:
        return "body_pitch_within_limit" if "body_pitch_within_limit" in supported_names else "pitch_limit"
    if "roll" in normalized:
        return "roll_limit" if "roll_limit" in supported_names else normalized
    side = "front" if "front" in normalized else "rear" if "rear" in normalized else ""
    if "stand" in normalized and ("duration" in normalized or "time" in normalized):
        candidate = "%s_leg_stand_duration" % side if side else "stable_stand_duration"
        return candidate if candidate in supported_names else normalized
    if "walk" in normalized and ("duration" in normalized or "completion" in normalized or "time" in normalized):
        candidate = "%s_leg_walk_completion" % side if side else "stable_stand_duration"
        return candidate if candidate in supported_names else normalized
    if "tracking" in normalized or ("speed" in normalized and "error" not in normalized):
        candidate = "%s_leg_walk_velocity_tracking" % side if side else "walking_speed_tracking"
        return candidate if candidate in supported_names else "tracking_error"
    return normalized


def normalize_task_metrics(task: TaskSpec, supported: Iterable[str]) -> List[Dict[str, str]]:
    """原地规范任务的成功与安全指标，并返回可审计的名称映射。"""
    mappings: List[Dict[str, str]] = []
    for threshold in list(task.success_metrics) + list(task.safety_constraints):
        original = threshold.name
        canonical = canonical_metric_name(original, supported)
        threshold.name = canonical
        if canonical != original:
            mappings.append({"original": original, "canonical": canonical})
    return mappings
