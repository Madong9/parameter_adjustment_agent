"""从结构化 TaskIntentSpec 生成不含策略的低维动态动作原型。"""
from __future__ import annotations

import re
from typing import Optional, Tuple

from ...schemas.agent_workflow import TaskIntentSpec
from .schema import (DynamicMotionPhase, DynamicMotionPrototype, GaitPattern,
                     GaitType, MotionType, VelocityTrajectory, FootTrajectory)


class DynamicMotionPrototypeGenerator:
    """使用受限规则分类动作并编译动态速度阶段，不生成关节目标。"""

    LOCOMOTION_TERMS = ("走", "行走", "跑", "倒退", "后退", "前进", "walk", "run",
                        "locomotion", "gait", "步态")
    STATIC_TERMS = ("站立", "站着", "保持平衡", "站稳", "stand", "balance", "静止")

    @classmethod
    def classify(cls, intent: TaskIntentSpec) -> MotionType:
        """按动作语义规则分类；未知文本不会被猜成可验证动态动作。"""
        text = cls._task_text(intent)
        if any(term in text for term in ("机械臂", "抓取", "抓住", "grasp", "manipulat", "arm")):
            return MotionType.MANIPULATION
        if any(term in text for term in (
                "后空翻", "前空翻", "空翻", "翻跟头", "backflip", "frontflip",
                "somersault", "aerial", "翻滚")):
            return MotionType.ACROBATIC
        if any(term in text for term in ("跳", "jump", "takeoff", "起跳")):
            return MotionType.JUMP
        if any(term in text for term in (
                "单腿", "单足", "倒立", "handstand", "one-leg", "single-leg", "inverted",
                "前腿站立", "前足站立", "后腿站立", "后足站立", "front_leg_stand",
                "hind_leg_stand", "rear_leg_stand", "front_leg_walk", "hind_leg_walk")):
            return MotionType.BALANCE
        if any(term in text for term in ("转身", "转向", "左转", "右转", "turn", "rotate")):
            return MotionType.TURNING
        if any(term in text for term in cls.LOCOMOTION_TERMS):
            return MotionType.LOCOMOTION
        if any(term in text for term in ("平衡", "balance")):
            return MotionType.BALANCE
        if any(term in text for term in cls.STATIC_TERMS):
            return MotionType.STATIC_POSE
        return MotionType.UNKNOWN

    @staticmethod
    def _task_text(intent: TaskIntentSpec) -> str:
        """汇总用户原文和结构化动作字段供确定性分类。"""
        return " ".join((intent.original_instruction, intent.action_name,
                         intent.normalized_goal, " ".join(intent.required_behaviors))).lower()

    @staticmethod
    def _duration(intent: TaskIntentSpec) -> Tuple[float, bool]:
        """只提取带秒单位的持续时间；未说明时返回明确标记的 5 秒预检假设。"""
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:秒|s\b|seconds?)",
                          intent.original_instruction, re.IGNORECASE)
        return (float(match.group(1)), False) if match else (5.0, True)

    @staticmethod
    def _x_velocity(intent: TaskIntentSpec, text: str) -> Tuple[float, Optional[str]]:
        """读取 TaskIntent 的速度或原文速度，并确定后退方向。"""
        speed = intent.target_velocity
        if speed is None:
            match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:m\s*/\s*s|米每秒|米/秒)",
                              intent.original_instruction, re.IGNORECASE)
            if match:
                speed = float(match.group(1))
        backward = any(token in text for token in ("倒退", "后退", "向后", "backward", "reverse"))
        if speed is None:
            # This is only a low-speed feasibility probe, not a claimed user target.
            return (-0.3 if backward else 0.3), "未指定速度；使用 0.3 m/s 低速预检假设"
        speed = float(speed)
        if backward:
            speed = -abs(speed)
        return speed, None

    @classmethod
    def generate(cls, intent: TaskIntentSpec) -> DynamicMotionPrototype:
        """将运动任务编译为准备、运动、停止三个低维阶段。"""
        motion_type = cls.classify(intent)
        duration, assumed_duration = cls._duration(intent)
        if duration > 30.0:
            raise ValueError("用户指定时长超过动态预检原型上限 30 秒；拒绝静默截短")
        if duration <= 0.0:
            raise ValueError("动态动作持续时间必须大于零")
        duration = max(0.1, duration)
        notes = []
        if assumed_duration:
            notes.append("未指定持续时间；5 秒仅用于低成本预检，不构成任务验收时长。")

        if motion_type == MotionType.LOCOMOTION:
            text = cls._task_text(intent)
            velocity, speed_note = cls._x_velocity(intent, text)
            if speed_note:
                notes.append(speed_note)
            if abs(velocity) > 1.0:
                notes.append("目标速度绝对值超过 1.0 m/s；需先由本地环境命令范围检查确认。")
            prepare = min(0.5, duration / 4.0)
            stop = min(0.5, duration / 4.0)
            move_end = duration - stop
            phases = [
                DynamicMotionPhase(name="stand_prepare", start=0.0, end=prepare,
                                   target_velocity={"x": 0.0}),
                DynamicMotionPhase(name="locomotion", start=prepare, end=move_end,
                                   target_velocity={"x": velocity}),
                DynamicMotionPhase(name="stop", start=move_end, end=duration,
                                   target_velocity={"x": 0.0}),
            ]
            requested_gait = intent.constraints.get("gait_type", "TROT")
            if isinstance(requested_gait, GaitType):
                requested_gait = requested_gait.value
            gait_type = GaitType(str(requested_gait).upper())
            gait_offsets = intent.constraints.get("gait_phase_offsets")
            if gait_offsets is None:
                gait_offsets = GaitPattern.default_phase_offsets(gait_type)
            gait = GaitPattern(
                type=gait_type,
                frequency=float(intent.constraints.get("gait_frequency", 2.0)),
                duty_factor=float(intent.constraints.get("duty_factor", 0.5)),
                phase_offsets=gait_offsets,
            )
            requested_step_length = intent.constraints.get("step_length")
            step_length = (float(requested_step_length) if requested_step_length is not None
                           else max(0.02, abs(velocity) / gait.frequency))
            requested_swing_height = intent.constraints.get("swing_height", 0.08)
            foot_trajectory = FootTrajectory(
                swing_height=float(requested_swing_height), step_length=step_length,
                step_period=1.0 / gait.frequency,
                direction_x=-1.0 if velocity < 0.0 else 1.0,
                duty_factor=gait.duty_factor,
                phase_offset=dict(gait.phase_offsets),
            )
            velocity_trajectory = VelocityTrajectory(
                duration=duration,
                target_linear_velocity={"x": velocity},
            )
            if not any(token in text for token in ("trot", "对角小跑", "小跑")):
                notes.append("用户未指定步态类型；采用 2 Hz、占空比 0.5 的 TROT 预检原型。")
        elif motion_type == MotionType.TURNING:
            text = cls._task_text(intent)
            direction = -1.0 if any(term in text for term in ("左转", "向左", "turn left")) else 1.0
            yaw_rate = float(intent.constraints.get("target_yaw_rate", 0.5))
            if not any(term in text for term in ("左", "右", "left", "right")):
                notes.append("未指定转向方向；按向右/正偏航 0.5 rad/s 做预检假设。")
            prepare = min(0.5, duration / 4.0)
            stop = min(0.5, duration / 4.0)
            phases = [
                DynamicMotionPhase(name="turn_prepare", start=0.0, end=prepare,
                                   target_velocity={"yaw": 0.0}),
                DynamicMotionPhase(name="turn", start=prepare, end=duration - stop,
                                   target_velocity={"yaw": direction * abs(yaw_rate)}),
                DynamicMotionPhase(name="turn_stop", start=duration - stop, end=duration,
                                   target_velocity={"yaw": 0.0}),
            ]
            gait = None
            foot_trajectory = None
            velocity_trajectory = VelocityTrajectory(
                duration=duration,
                target_angular_velocity={"yaw": direction * abs(yaw_rate)},
            )
        else:
            phase_name = "static_pose" if motion_type == MotionType.STATIC_POSE else "unsupported_motion"
            phases = [DynamicMotionPhase(name=phase_name, start=0.0, end=duration)]
            notes.append("该 MotionType 不由动态四足步态预检控制器执行。")
            gait = None
            foot_trajectory = None
            velocity_trajectory = None

        return DynamicMotionPrototype(
            robot=intent.robot, action=intent.action_name, motion_type=motion_type,
            duration=duration, phases=phases, velocity=velocity_trajectory, gait=gait,
            foot_trajectory=foot_trajectory,
            constraints=["maintain_height", "avoid_slip", "avoid_fall", "avoid_joint_limit",
                         "track_requested_body_velocity"], notes=notes,
        )
