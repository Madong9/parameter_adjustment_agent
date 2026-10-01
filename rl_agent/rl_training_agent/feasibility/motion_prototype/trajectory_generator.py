"""为跳跃/特技生成仅供分析的语义阶段骨架。"""
from __future__ import annotations

import re
from typing import List

from ...schemas.agent_workflow import TaskIntentSpec
from .schema import (MotionTrajectoryPhase, MotionType, TrajectoryPhaseName,
                     TrajectoryPrototype)


class TrajectoryPrototypeGenerator:
    """编译跳跃/空翻阶段，不生成位姿数值、关节轨迹或控制输入。"""

    @staticmethod
    def generate(intent: TaskIntentSpec, motion_type: MotionType) -> TrajectoryPrototype:
        """基于动作类别创建标注假设的阶段模板，等待专用规划器提供真实轨迹。"""
        if motion_type not in (MotionType.JUMP, MotionType.ACROBATIC):
            raise ValueError("trajectory template supports JUMP or ACROBATIC only")
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:秒|s\b|seconds?)",
                          intent.original_instruction, re.IGNORECASE)
        duration = float(match.group(1)) if match else 2.0
        if not 0.5 <= duration <= 30.0:
            raise ValueError("jump/acrobatic duration must be within [0.5, 30] seconds")
        if motion_type == MotionType.ACROBATIC:
            names = [TrajectoryPhaseName.PRELOAD, TrajectoryPhaseName.TAKEOFF,
                     TrajectoryPhaseName.FLIGHT, TrajectoryPhaseName.ROTATION,
                     TrajectoryPhaseName.LANDING, TrajectoryPhaseName.RECOVERY]
        else:
            names = [TrajectoryPhaseName.PRELOAD, TrajectoryPhaseName.TAKEOFF,
                     TrajectoryPhaseName.FLIGHT, TrajectoryPhaseName.LANDING,
                     TrajectoryPhaseName.RECOVERY]
        weights = ([0.18, 0.16, 0.18, 0.20, 0.13, 0.15] if
                   motion_type == MotionType.ACROBATIC else
                   [0.20, 0.16, 0.24, 0.20, 0.20])
        phases: List[MotionTrajectoryPhase] = []
        cursor = 0.0
        for index, (name, weight) in enumerate(zip(names, weights)):
            end = duration if index == len(names) - 1 else cursor + duration * weight
            phases.append(MotionTrajectoryPhase(
                name=name, start=cursor, end=end,
                semantic_goal="等待专用任务规划器提供可验证的阶段目标",
            ))
            cursor = end
        assumptions = [
            "阶段时长比例是文档模板，不是 Go2 实测轨迹",
            "没有生成起跳速度、飞行高度、旋转角、落地冲击或关节轨迹",
            "当前不允许该模板升级为 PHYSICS_VALIDATED",
        ]
        if match is None:
            assumptions.append("用户未指定持续时间；采用 2 秒仅作为阶段模板假设")
        return TrajectoryPrototype(
            action=intent.action_name, motion_type=motion_type, duration=duration,
            phases=phases, assumptions=assumptions,
        )
