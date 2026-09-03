from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, root_validator, validator

from .task import MetricThreshold


class RewardTerm(BaseModel):
    name: str
    implementation: str
    purpose: str
    weight: float
    parameters: Dict[str, Any] = Field(default_factory=dict)
    active_phases: List[str] = Field(default_factory=lambda: ["all"])
    activation_condition: Optional[str] = None
    normalization: str = "none"
    expected_raw_range: Optional[Tuple[float, float]] = None
    expected_training_trend: str = "increase"
    dependencies: List[str] = Field(default_factory=list)
    failure_modes_addressed: List[str] = Field(default_factory=list)
    reward_hacking_risks: List[str] = Field(default_factory=list)

    @validator("weight")
    def finite_weight(cls, value: float) -> float:
        """校验奖励权重为有限数值。"""
        if not math.isfinite(value):
            raise ValueError("reward weight must be finite")
        return value

    @validator("parameters", pre=True)
    def object_parameters(cls, value: Any) -> Dict[str, Any]:
        """校验奖励参数使用 JSON 对象结构。"""
        if not isinstance(value, dict):
            raise ValueError("reward parameters must be an object")
        return value

    @validator("expected_raw_range")
    def ordered_range(cls, value: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """校验数值范围的下界不大于上界。"""
        if value is not None and value[0] > value[1]:
            raise ValueError("expected range must be ordered")
        return value


class TerminationSpec(BaseModel):
    name: str
    condition: str
    is_timeout: bool = False
    enabled: bool = True


class CurriculumStage(BaseModel):
    name: str
    start_iteration: int
    end_iteration: int
    parameter_changes: Dict[str, Any] = Field(default_factory=dict)

    @validator("start_iteration", "end_iteration")
    def nonnegative_boundary(cls, value: int) -> int:
        """校验课程阶段边界不是负迭代数。"""
        if value < 0:
            raise ValueError("curriculum boundaries must be nonnegative")
        return value

    @root_validator
    def ordered_boundaries(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """校验课程阶段起点不晚于终点。"""
        start = values.get("start_iteration")
        end = values.get("end_iteration")
        if start is not None and end is not None and start > end:
            raise ValueError("curriculum start_iteration must not exceed end_iteration")
        return values


class RewardConflict(BaseModel):
    terms: List[str]
    description: str
    mitigation: str


class RewardPlan(BaseModel):
    task_id: str
    version: int
    parent_version: Optional[int] = None
    design_rationale: List[str]
    terms: List[RewardTerm]
    terminations: List[TerminationSpec] = Field(default_factory=list)
    curriculum: List[CurriculumStage] = Field(default_factory=list)
    success_metrics: List[MetricThreshold] = Field(default_factory=list)
    known_conflicts: List[RewardConflict] = Field(default_factory=list)
    expected_learning_stages: List[str] = Field(default_factory=list)

    @validator("terms")
    def unique_terms(cls, terms: List[RewardTerm]) -> List[RewardTerm]:
        """校验奖励计划中的奖励项名称互不重复。"""
        names = [term.name for term in terms]
        if len(names) != len(set(names)):
            raise ValueError("reward term names must be unique")
        return terms
