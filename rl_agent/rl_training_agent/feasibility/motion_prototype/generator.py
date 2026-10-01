"""生成或校验不包含关节角的高层动作原型。"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from ...schemas.agent_workflow import TaskIntentSpec
from .schema import MotionPhase, MotionPrototype


class MotionPrototypeGenerator:
    """优先校验语义模型输出；离线时使用显式标记的确定性模板。"""

    FORBIDDEN_KEYS = {"joint_angle", "joint_angles", "joint_position", "joint_positions",
                      "torque", "torques", "policy", "control_signal"}

    @classmethod
    def _reject_low_level_values(cls, value: Any) -> None:
        """递归拒绝混入原型的低层控制量。"""
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in cls.FORBIDDEN_KEYS:
                    raise ValueError("motion prototype must not contain low-level field: %s" % key)
                cls._reject_low_level_values(child)
        elif isinstance(value, list):
            for child in value:
                cls._reject_low_level_values(child)

    @staticmethod
    def _requested_duration(text: str) -> Optional[float]:
        """从用户原文中提取明确的秒数，不猜测无单位数字。"""
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:秒|s\b|seconds?)", text, re.IGNORECASE)
        return float(match.group(1)) if match else None

    @classmethod
    def deterministic(cls, intent: TaskIntentSpec) -> MotionPrototype:
        """在离线演练或模型不可用时创建有显式假设的语义阶段。"""
        text = (intent.original_instruction + " " + intent.action_name + " " +
                intent.normalized_goal).lower()
        duration = cls._requested_duration(intent.original_instruction)
        notes = []
        if duration is None:
            duration = 3.0
            notes.append("未指定持续时间；3 秒仅用于低成本预检，不构成任务验收目标。")
        duration = min(30.0, max(0.1, duration))
        if any(token in text for token in ("后腿", "hind_leg", "rear_leg")):
            action = "hind_leg_stand_walk" if any(
                token in text for token in ("行走", "走路", "walk")) else "hind_leg_stand"
            phases = [
                MotionPhase(name="stand_up", duration_seconds=min(2.0, duration),
                            body_goal={"torso": "upright", "front_feet": "lift",
                                       "hind_feet": "support"},
                            constraints=["avoid fall", "respect joint limits"]),
            ]
            remaining = duration - phases[0].duration_seconds
            if remaining > 0.0:
                phases.append(MotionPhase(name="balance_hold", duration_seconds=remaining,
                                          body_goal={"torso": "upright", "front_feet": "lift",
                                                     "hind_feet": "support"},
                                          constraints=["keep center of mass stable", "avoid forbidden contact"]))
        elif any(token in text for token in ("前腿站立", "前足站立", "front_leg_stand")):
            action = "front_leg_stand"
            phases = [MotionPhase(name="stand_up", duration_seconds=min(2.0, duration),
                                  body_goal={"torso": "upright", "front_feet": "support",
                                             "hind_feet": "lift"},
                                  constraints=["avoid fall", "respect joint limits"])]
            remaining = duration - phases[0].duration_seconds
            if remaining > 0.0:
                phases.append(MotionPhase(name="balance_hold", duration_seconds=remaining,
                                          body_goal={"torso": "upright", "hind_feet": "lift"},
                                          constraints=["keep center of mass stable"]))
        elif any(token in text for token in ("站立", "stand", "保持平衡", "balance")):
            action = "stable_stand"
            rise = min(1.0, duration)
            phases = [MotionPhase(name="stand_up", duration_seconds=rise,
                                  body_goal={"torso": "upright", "feet": "support"},
                                  constraints=["avoid fall", "respect joint limits"])]
            remaining = duration - rise
            if remaining > 0.0:
                phases.append(MotionPhase(name="balance_hold", duration_seconds=remaining,
                                          body_goal={"torso": "upright", "feet": "support"},
                                          constraints=["keep center of mass stable", "avoid forbidden contact"]))
        elif any(token in text for token in ("行走", "走路", "倒退", "前进", "walk", "locomotion")):
            action = intent.action_name or "locomotion"
            direction = "backward" if any(token in text for token in ("倒退", "后退", "向后", "backward", "reverse")) else "forward_or_user_command"
            phases = [MotionPhase(name="locomotion", duration_seconds=duration,
                                  body_goal={"base_motion": direction, "feet": "alternating_contact"},
                                  constraints=["track only configured velocity command", "avoid fall", "respect joint limits"])]
        else:
            action = intent.action_name or "custom_motion"
            phases = [MotionPhase(name="task_motion", duration_seconds=duration,
                                  body_goal={"goal": intent.normalized_goal},
                                  constraints=["avoid fall", "respect joint limits"])]
        return MotionPrototype(robot=intent.robot, action=action, phases=phases,
                               source="deterministic_fallback", notes=notes)

    @classmethod
    def generate(cls, intent: TaskIntentSpec, proposal: Optional[Any] = None) -> MotionPrototype:
        """验证 Provider 的语义 JSON，未提供时返回确定性原型。"""
        if proposal is None:
            return cls.deterministic(intent)
        raw: Dict[str, Any] = proposal.dict() if hasattr(proposal, "dict") else dict(proposal)
        cls._reject_low_level_values(raw)
        prototype = MotionPrototype.parse_obj(raw)
        if prototype.robot != intent.robot:
            raise ValueError("motion prototype robot does not match selected robot")
        return prototype
