"""定义任务可行性检查的稳定输入输出协议。"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, root_validator, validator


class FeasibilityStatus(str, Enum):
    """区分纯能力、Mock 结果和已通过真实物理 rollout 的结论。"""

    CAPABILITY_SUPPORTED = "CAPABILITY_SUPPORTED"
    SUPPORTED = "CAPABILITY_SUPPORTED"
    MOCK_VALIDATED = "MOCK_VALIDATED"
    INCONCLUSIVE = "INCONCLUSIVE"
    PHYSICS_VALIDATED = "PHYSICS_VALIDATED"
    PHYSICS_FAILED = "PHYSICS_FAILED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    CONDITIONAL = "CONDITIONAL"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    UNSUPPORTED = "UNSUPPORTED"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _missing_(cls, value: object) -> Optional["FeasibilityStatus"]:
        """读取旧产物中的 SUPPORTED 字符串并兼容映射为能力通过。"""
        if value == "SUPPORTED":
            return cls.CAPABILITY_SUPPORTED
        return None


class ValidationLevel(str, Enum):
    """区分能力、Mock、静态物理、动态物理与训练前置条件。"""

    CAPABILITY_ONLY = "CAPABILITY_ONLY"
    MOCK_VALIDATED = "MOCK_VALIDATED"
    STATIC_PHYSICS_VALIDATED = "STATIC_PHYSICS_VALIDATED"
    DYNAMIC_PHYSICS_VALIDATED = "DYNAMIC_PHYSICS_VALIDATED"
    TRAINING_READY = "TRAINING_READY"
    PHYSICS_FAILED = "PHYSICS_FAILED"


class FeasibilityLevel(str, Enum):
    """区分从语言理解到真实物理预检的证据层级；不表示策略已学会。"""

    LEVEL_0_LANGUAGE = "LEVEL_0_LANGUAGE"
    LEVEL_1_CAPABILITY = "LEVEL_1_CAPABILITY"
    LEVEL_2_KINEMATIC = "LEVEL_2_KINEMATIC"
    LEVEL_3_STATIC_DYNAMICS = "LEVEL_3_STATIC_DYNAMICS"
    LEVEL_4_TRAJECTORY = "LEVEL_4_TRAJECTORY"
    LEVEL_5_PHYSICS = "LEVEL_5_PHYSICS"
    LEVEL_6_OPTIONAL_RL_PROBE = "LEVEL_6_OPTIONAL_RL_PROBE"


class StageStatus(str, Enum):
    """子阶段的证据状态；UNKNOWN/INCONCLUSIVE 不得折算为通过。"""

    PASSED = "PASSED"
    FAILED = "FAILED"
    CONDITIONAL = "CONDITIONAL"
    UNKNOWN = "UNKNOWN"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_RUN = "NOT_RUN"


class ValidationStageReport(BaseModel):
    """保存单个验证子阶段的状态、后端、指标和来源。"""

    stage: str
    status: StageStatus
    reason: str = ""
    backend: str = "not_run"
    metrics: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[str] = Field(default_factory=list)


class DynamicFeasibilityReport(BaseModel):
    """聚合动态轨迹、Isaac Gym 指标和本次验证范围。"""

    task: str
    motion_type: str
    backend: str
    duration: Optional[float] = None
    gait: Optional[Dict[str, Any]] = None
    success: Optional[bool] = None
    validation_level: str = ValidationLevel.CAPABILITY_ONLY.value
    trajectory: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    violations: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    reason: str = ""


class FeasibilityCheck(BaseModel):
    """保存一项确定性检查的结果和可追溯依据。"""

    name: str
    status: FeasibilityStatus
    summary: str
    evidence: List[str] = Field(default_factory=list)


class CapabilityReport(BaseModel):
    """记录纯确定性机器人和环境能力检查结果。"""

    status: FeasibilityStatus
    required_capabilities: List[str] = Field(default_factory=list)
    checks: List[FeasibilityCheck] = Field(default_factory=list)
    missing_requirements: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)


class FeasibilityReport(BaseModel):
    """汇总能力、运动原型、IK 和短时仿真验证结果。"""

    status: FeasibilityStatus
    backend: Optional[str] = None
    evidence_mode: str = "REAL"
    validation_level: str = "CAPABILITY_ONLY"
    task: str
    action_type: Optional[str] = None
    confidence: float = 0.0
    confidence_basis: str = "证据覆盖度，不是策略成功概率"
    feasibility_level: FeasibilityLevel = FeasibilityLevel.LEVEL_0_LANGUAGE
    stage_reports: List[ValidationStageReport] = Field(default_factory=list)
    kinematic_report: Dict[str, Any] = Field(default_factory=dict)
    static_dynamics_report: Dict[str, Any] = Field(default_factory=dict)
    trajectory_report: Dict[str, Any] = Field(default_factory=dict)
    physics_report: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    recommended_next_step: str = ""
    required_capabilities: List[str] = Field(default_factory=list)
    checks: List[FeasibilityCheck] = Field(default_factory=list)
    missing_requirements: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    capability_check: Dict[str, Any] = Field(default_factory=dict)
    capability_report: Dict[str, Any] = Field(default_factory=dict)
    motion_prototype: Optional[Dict[str, Any]] = None
    motion_type: Optional[str] = None
    motion_report: Dict[str, Any] = Field(default_factory=dict)
    motion_constraint_spec: Dict[str, Any] = Field(default_factory=dict)
    planning_report: Dict[str, Any] = Field(default_factory=dict)
    whole_body_report: Dict[str, Any] = Field(default_factory=dict)
    dynamic_report: Optional[Dict[str, Any]] = None
    training_readiness: Optional[Dict[str, Any]] = None
    training_admission: Optional[Dict[str, Any]] = None
    ik_result: Optional[Dict[str, Any]] = None
    ik_report: Dict[str, Any] = Field(default_factory=dict)
    simulation_result: Optional[Dict[str, Any]] = None
    simulation_report: Dict[str, Any] = Field(default_factory=dict)
    robot_model: Dict[str, Any] = Field(default_factory=dict)
    converted_model: bool = False

    @validator("confidence")
    def confidence_is_probability_bounded(cls, value: float) -> float:
        """限制证据覆盖度字段范围；调用方仍需读取 confidence_basis 避免误读为成功率。"""
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        return float(value)


class CompleteFeasibilityReport(FeasibilityReport):
    """提供显式 overall_status 名称的完整流水线报告。"""

    overall_status: FeasibilityStatus

    @root_validator(pre=True)
    def mirror_overall_status(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """保证简版 status 与完整报告 overall_status 始终一致。"""
        values = dict(values)
        if "overall_status" not in values:
            values["overall_status"] = values.get("status", FeasibilityStatus.UNKNOWN)
        if "status" not in values:
            values["status"] = values["overall_status"]
        if values["status"] != values["overall_status"]:
            raise ValueError("status and overall_status must be identical")
        return values


def training_ready(validation_level: str, reward_config: Any,
                   evaluation_metrics: Any) -> bool:
    """仅当动态物理预检、奖励配置和确定性评估指标都存在时标记可训练。"""
    has_reward = bool(reward_config)
    has_evaluation = bool(evaluation_metrics)
    return (str(validation_level) == ValidationLevel.DYNAMIC_PHYSICS_VALIDATED.value and
            has_reward and has_evaluation)
