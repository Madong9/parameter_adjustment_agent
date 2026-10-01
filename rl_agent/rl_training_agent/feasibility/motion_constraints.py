"""定义从任务意图编译得到的通用运动约束协议。"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, root_validator, validator

from ..schemas.agent_workflow import TaskIntentSpec
from .motion_prototype.dynamic_generator import DynamicMotionPrototypeGenerator
from .motion_prototype.schema import MotionType
from .motion_prototype.trajectory_generator import TrajectoryPrototypeGenerator


_FORBIDDEN_CONTROL_KEYS = {
    "joint", "joints", "joint_angle", "joint_angles", "joint_positions",
    "torque", "torques", "actuator", "actuator_command", "policy", "actions",
}


def _reject_control_fields(value: Any) -> None:
    """递归拒绝藏在嵌套目标中的关节、力矩或策略指令。"""
    if isinstance(value, dict):
        if {str(key).lower() for key in value} & _FORBIDDEN_CONTROL_KEYS:
            raise ValueError("MotionConstraintSpec cannot contain joint, torque or policy commands")
        for item in value.values():
            _reject_control_fields(item)
    elif isinstance(value, list):
        for item in value:
            _reject_control_fields(item)


class MotionConstraintPhase(BaseModel):
    """描述一个动作阶段的目标、接触和约束，不包含底层控制量。"""

    name: str
    start: float
    end: float
    goals: Dict[str, Any] = Field(default_factory=dict)
    active_contacts: List[str] = Field(default_factory=list)
    forbidden_contacts: List[str] = Field(default_factory=list)
    optional_contacts: List[str] = Field(default_factory=list)
    transition_conditions: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)

    class Config:
        """拒绝 LLM 添加协议之外的未审查字段。"""

        extra = "forbid"

    @validator("start", "end")
    def finite_time(cls, value: float) -> float:
        """保证阶段边界为有限值。"""
        if not math.isfinite(float(value)):
            raise ValueError("phase time must be finite")
        return float(value)

    @root_validator
    def validate_phase(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """拒绝非法时序和藏在目标映射中的关节/力矩命令。"""
        if float(values.get("end", 0.0)) <= float(values.get("start", 0.0)):
            raise ValueError("phase end must be after start")
        _reject_control_fields(values.get("goals") or {})
        required = set(values.get("active_contacts") or [])
        forbidden = set(values.get("forbidden_contacts") or [])
        optional = set(values.get("optional_contacts") or [])
        if required & forbidden or required & optional or forbidden & optional:
            raise ValueError("phase contact requirements conflict")
        return values


class MotionConstraintSpec(BaseModel):
    """统一表示 LLM 动作语义经本地校验后的规划输入。"""

    robot: str
    action: str
    motion_type: MotionType
    duration: float
    phases: List[MotionConstraintPhase]
    required_planners: List[str]
    global_constraints: List[str] = Field(default_factory=list)
    hard_constraints: List[str] = Field(default_factory=list)
    soft_preferences: List[str] = Field(default_factory=list)
    task_requirements: List[str] = Field(default_factory=list)
    acceptance_source: str = "task_spec.json"
    allowed_assumptions: List[str] = Field(default_factory=list)
    source: str = "task_intent_compiler"

    class Config:
        """禁止额外控制字段，确保 LLM 只描述目标与约束。"""

        extra = "forbid"

    @validator("duration")
    def bounded_duration(cls, value: float) -> float:
        """限制训练前探针的时长预算。"""
        if not math.isfinite(float(value)) or not 0.0 < float(value) <= 30.0:
            raise ValueError("constraint duration must be within (0, 30] seconds")
        return float(value)

    @validator("required_planners")
    def known_planners(cls, value: List[str]) -> List[str]:
        """只接受本地注册表能够识别的规划器名称。"""
        allowed = {"gait", "com", "contact_schedule", "base_pose", "end_effector"}
        if not value or set(value) - allowed:
            raise ValueError("required_planners contains unknown planner")
        return list(dict.fromkeys(value))

    @root_validator
    def contiguous_phases(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """要求阶段连续覆盖整个动作时长。"""
        phases = values.get("phases") or []
        duration = values.get("duration")
        if not phases or abs(phases[0].start) > 1.0e-6:
            raise ValueError("constraint phases must start at zero")
        for index, phase in enumerate(phases):
            if index and abs(phase.start - phases[index - 1].end) > 1.0e-6:
                raise ValueError("constraint phases must be contiguous")
        if duration is not None and abs(phases[-1].end - float(duration)) > 1.0e-6:
            raise ValueError("constraint phases must cover duration")
        return values


class MotionConstraintCompiler:
    """把已校验的 TaskIntentSpec 编译为确定性规划器输入。"""

    @staticmethod
    def _duration(intent: TaskIntentSpec, default: float) -> float:
        """从用户原文提取秒数，未指定时使用显式记录的探针默认值。"""
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:秒|s\b|seconds?)",
                          intent.original_instruction, re.IGNORECASE)
        return float(match.group(1)) if match else default

    @classmethod
    def compile(cls, intent: TaskIntentSpec,
                motion_type: Optional[MotionType] = None) -> MotionConstraintSpec:
        """根据动作类别生成阶段、规划器需求和允许假设。"""
        kind = motion_type or DynamicMotionPrototypeGenerator.classify(intent)
        assumptions = list(intent.assumptions)
        common = ["avoid_fall", "respect_joint_limits", "respect_actuator_limits"]

        if kind == MotionType.LOCOMOTION:
            prototype = DynamicMotionPrototypeGenerator.generate(intent)
            phases = [
                MotionConstraintPhase(
                    name=phase.name, start=phase.start, end=phase.end,
                    goals={"base_velocity": dict(phase.target_velocity),
                           "gait": prototype.gait.dict() if prototype.gait else None},
                    active_contacts=["scheduled_by_gait"],
                    constraints=list(phase.constraints),
                ) for phase in prototype.phases
            ]
            assumptions.extend(prototype.notes)
            required = ["gait", "contact_schedule", "base_pose", "end_effector"]
            duration = prototype.duration
        elif kind in (MotionType.JUMP, MotionType.ACROBATIC):
            prototype = TrajectoryPrototypeGenerator.generate(intent, kind)
            phases = [
                MotionConstraintPhase(
                    name=phase.name.value, start=phase.start, end=phase.end,
                    goals={"semantic_goal": phase.semantic_goal,
                           "angular_momentum_required": kind == MotionType.ACROBATIC},
                    active_contacts=(["FL", "FR", "RL", "RR"]
                                     if phase.name.value in ("PRELOAD", "TAKEOFF", "LANDING", "RECOVERY")
                                     else []),
                    constraints=["bounded_landing_impact"] if phase.name.value == "LANDING" else [],
                ) for phase in prototype.phases
            ]
            assumptions.extend(prototype.assumptions)
            required = ["com", "contact_schedule", "base_pose", "end_effector"]
            duration = prototype.duration
        elif kind == MotionType.BALANCE:
            duration = cls._duration(intent, 3.0)
            text = " ".join((intent.original_instruction, intent.action_name)).lower()
            contacts = ["FL", "FR", "RL", "RR"]
            if any(token in text for token in ("单腿", "单足", "single-leg", "one-leg")):
                contacts = ["candidate_single_support"]
            elif any(token in text for token in ("后腿", "后足", "hind", "rear")):
                contacts = ["RL", "RR"]
            elif any(token in text for token in ("前腿", "前足", "front")):
                contacts = ["FL", "FR"]
            phases = [MotionConstraintPhase(
                name="balance_hold", start=0.0, end=duration,
                goals={"center_of_mass": "inside_support_region",
                       "base_orientation": "task_aligned"},
                active_contacts=contacts,
                constraints=["static_or_dynamic_balance", "avoid_slip"],
            )]
            if set(contacts) in ({"FL", "FR"}, {"RL", "RR"}):
                lifted = sorted({"FL", "FR", "RL", "RR"} - set(contacts))
                phases = [
                    MotionConstraintPhase(
                        name="prepare", start=0, end=0.10 * duration,
                        active_contacts=["FL", "FR", "RL", "RR"],
                        goals={"base_pose": "nominal"}),
                    MotionConstraintPhase(
                        name="com_transfer", start=0.10 * duration, end=0.30 * duration,
                        active_contacts=["FL", "FR", "RL", "RR"],
                        goals={"center_of_mass": "toward_target_support"}),
                    MotionConstraintPhase(
                        name="unload_and_lift", start=0.30 * duration, end=0.52 * duration,
                        active_contacts=contacts, optional_contacts=lifted,
                        goals={"end_effector": "unload_then_lift_non_support_feet"},
                        transition_conditions=["non_support_contact_force_below_threshold"],
                        constraints=["continuous_state", "avoid_slip"]),
                    MotionConstraintPhase(
                        name="balance_hold", start=0.52 * duration, end=duration,
                        active_contacts=contacts, forbidden_contacts=lifted,
                        goals={"center_of_mass": "balance_with_target_support",
                               "base_orientation": "task_aligned"},
                        constraints=["avoid_slip", "respect_actuator_limits"]),
                ]
                moving = any(token in text for token in ("走", "前进", "倒退", "walk", "move"))
                if moving:
                    phases[-1].name = "support_walk"
                    phases[-1].active_contacts = []
                    phases[-1].optional_contacts = list(contacts)
                    phases[-1].goals["base_motion"] = intent.normalized_goal
                    phases[-1].constraints.append("at_least_one_target_support_contact")
                assumptions.append(
                    "预检阶段时序为有界候选；准备阶段允许四足接触，目标保持阶段才要求非支撑足离地。"
                    "切换条件是待验证约束，不代表已有事件触发控制器；任务验收时长以 task_spec 为准。")
            required = ["com", "contact_schedule", "base_pose", "end_effector"]
        elif kind == MotionType.MANIPULATION:
            duration = cls._duration(intent, 2.0)
            phases = [MotionConstraintPhase(
                name="approach_and_interact", start=0.0, end=duration,
                goals={"end_effector": intent.constraints.get(
                    "manipulation_prototype", "requires_validated_cartesian_target")},
                active_contacts=["FL", "FR", "RL", "RR"],
                constraints=["collision_free", "maintain_support"],
            )]
            required = ["com", "contact_schedule", "base_pose", "end_effector"]
        elif kind == MotionType.TURNING:
            duration = cls._duration(intent, 5.0)
            phases = [MotionConstraintPhase(
                name="turn", start=0.0, end=duration,
                goals={"base_angular_velocity": intent.constraints.get(
                    "target_yaw_rate", 0.5)},
                active_contacts=["scheduled_by_turning_gait"],
                constraints=["avoid_slip", "maintain_height"],
            )]
            required = ["gait", "contact_schedule", "base_pose", "end_effector"]
            assumptions.append("未指定转向步态时只记录 0.5 rad/s 语义探针，不生成关节控制")
        else:
            duration = cls._duration(intent, 2.0)
            phases = [MotionConstraintPhase(
                name="hold" if kind == MotionType.STATIC_POSE else "unresolved",
                start=0.0, end=duration,
                goals={"base_pose": "upright"} if kind == MotionType.STATIC_POSE else
                      {"semantic_goal": intent.normalized_goal},
                active_contacts=["FL", "FR", "RL", "RR"] if kind == MotionType.STATIC_POSE else [],
                constraints=["maintain_pose"] if kind == MotionType.STATIC_POSE else [],
            )]
            required = (["com", "contact_schedule", "base_pose", "end_effector"]
                        if kind == MotionType.STATIC_POSE else ["base_pose"])

        return MotionConstraintSpec(
            robot=intent.robot, action=intent.action_name, motion_type=kind,
            duration=duration, phases=phases, required_planners=required,
            global_constraints=common,
            hard_constraints=common,
            soft_preferences=["smooth_motion", "low_energy"],
            task_requirements=list(intent.required_behaviors) + list(intent.forbidden_behaviors),
            allowed_assumptions=list(dict.fromkeys(item for item in assumptions if item)),
        )
