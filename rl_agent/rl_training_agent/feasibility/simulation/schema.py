"""定义短时动力学验证结果。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SimulationReport(BaseModel):
    """保存短时物理仿真状态、持续时间、违规和指标。"""

    status: str
    success: Optional[bool] = None
    backend: str = "mujoco"
    validated: bool = False
    validation_level: str = "CAPABILITY_ONLY"
    converted_model: bool = False
    model_source: str = ""
    duration: float = 0.0
    fall: Optional[bool] = None
    violations: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    self_collision_checked: bool = False
    reason: str = ""
