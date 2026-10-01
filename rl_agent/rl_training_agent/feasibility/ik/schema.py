"""定义逆运动学求解结果。"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class IKResult(BaseModel):
    """返回 IK 状态、关节位置以及失败或跳过原因。"""

    status: str
    success: Optional[bool] = None
    backend: str = "pinocchio"
    joint_positions: Dict[str, float] = Field(default_factory=dict)
    violations: List[str] = Field(default_factory=list)
    reason: str = ""
    iterations: int = 0
    residual: Optional[float] = None
    error: Optional[float] = None
    self_collision_checked: bool = False
    base_height: Optional[float] = None
    base_orientation_xyzw: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    duration_seconds: float = 0.0
