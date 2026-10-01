"""定义平衡动作数值规划、接触力和逆动力学的结构化报告。"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator


class BalanceTrajectorySample(BaseModel):
    """保存一个时刻的基座、质心、足端、接触和全身求解结果。"""

    time: float
    phase: str
    base_position: List[float]
    base_orientation_xyzw: List[float]
    com_position: List[float]
    feet: Dict[str, List[float]]
    contacts: Dict[str, bool]
    contact_forces: Dict[str, List[float]] = Field(default_factory=dict)
    joint_positions: Dict[str, float] = Field(default_factory=dict)
    joint_velocities: Dict[str, float] = Field(default_factory=dict)
    joint_accelerations: Dict[str, float] = Field(default_factory=dict)
    joint_torques: Dict[str, float] = Field(default_factory=dict)
    ik_residual: Optional[float] = None
    wrench_residual: Optional[float] = None
    base_dynamics_residual: Optional[float] = None

    @validator("time")
    def time_is_finite(cls, value: float) -> float:
        """拒绝无效时间戳。"""
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError("sample time must be finite and non-negative")
        return float(value)

    @validator("base_position", "com_position")
    def vectors_have_three_values(cls, value: List[float]) -> List[float]:
        """要求三维位置向量均为有限数。"""
        if len(value) != 3 or not all(math.isfinite(float(item)) for item in value):
            raise ValueError("position vectors must contain three finite values")
        return [float(item) for item in value]

    @validator("base_orientation_xyzw")
    def quaternion_is_valid(cls, value: List[float]) -> List[float]:
        """要求基座四元数为有限非零向量。"""
        if len(value) != 4 or not all(math.isfinite(float(item)) for item in value):
            raise ValueError("base quaternion must contain four finite values")
        norm = math.sqrt(sum(float(item) ** 2 for item in value))
        if norm <= 1.0e-9:
            raise ValueError("base quaternion must be non-zero")
        return [float(item) / norm for item in value]


class BalanceMotionPlan(BaseModel):
    """汇总六项数值规划结果；成功仅表示可进入物理 rollout。"""

    status: str
    backend: str = "pinocchio_scipy_balance_planner"
    robot: str
    action: str
    duration: float
    dt: float
    support_legs: List[str]
    lifted_legs: List[str]
    moving: bool = False
    samples: List[BalanceTrajectorySample] = Field(default_factory=list)
    available_solvers: List[str] = Field(default_factory=list)
    violations: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)
    reason: str = ""

    @validator("duration", "dt")
    def positive_finite_values(cls, value: float) -> float:
        """保证规划时长和采样周期为有限正数。"""
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError("duration and dt must be finite and positive")
        return float(value)
