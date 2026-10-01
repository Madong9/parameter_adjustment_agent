"""将语义阶段编译为基于真实模型的末端脚端和基座目标。"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .schema import MotionPrototype, RobotMotionTarget


class RobotMotionTargetBuilder:
    """优先使用显式笛卡尔目标，简单 Go2 站立则从标称关节姿态做真实 FK。"""

    @classmethod
    def build(cls, prototype: MotionPrototype, robot_model: Any) -> List[Tuple[str, Dict[str, Any]]]:
        """展开每个阶段的 RobotMotionTarget，绝不从语义文本猜关节角。"""
        result: List[Tuple[str, Dict[str, Any]]] = []
        for phase in prototype.phases:
            if phase.robot_targets:
                for target in phase.robot_targets:
                    result.append((phase.name, target.dict()))
                continue
            if phase.target_poses:
                for pose in phase.target_poses:
                    target = RobotMotionTarget(
                        time_seconds=phase.duration_seconds,
                        feet={pose.frame: pose.position},
                        base_height=float(robot_model.get("base_initial_height", 0.0)),
                        base_orientation_xyzw=(pose.quaternion_xyzw or [0.0, 0.0, 0.0, 1.0]),
                    )
                    result.append((phase.name, target.dict()))
                continue
            if str(robot_model.get("robot_name", "")).lower() != "go2":
                if not robot_model.get("_allow_mock_targets"):
                    raise ValueError("当前机器人没有显式 RobotMotionTarget 或标称目标规划器")
                positions = {frame: [0.0, 0.0, 0.0]
                             for frame in robot_model.get("foot_frames", [])}
                if not positions:
                    raise ValueError("Mock target fixture also requires configured foot frames")
                prototype.notes.append("仅 Mock：非 Go2 动作没有配置确定性目标规划器")
            else:
                from ..robot_models.go2 import Go2RobotModel
                try:
                    positions = Go2RobotModel.nominal_feet_positions(
                        robot_model["_descriptor"], phase.body_goal)
                except Exception as exc:
                    if not robot_model.get("_allow_mock_targets"):
                        raise
                    positions = {frame: [0.0, 0.0, 0.0]
                                 for frame in robot_model.get("foot_frames", [])}
                    if not positions:
                        raise ValueError("Mock target fixture also requires configured foot frames")
                    prototype.notes.append("仅 Mock 目标回退：%s" % str(exc)[:200])
            target = RobotMotionTarget(
                time_seconds=phase.duration_seconds,
                feet=positions,
                base_height=float(robot_model.get("base_initial_height", 0.0)),
            )
            phase.robot_targets.append(target)
            result.append((phase.name, target.dict()))
        return result
