from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, validator


class MetricThreshold(BaseModel):
    name: str
    operator: Literal[">", ">=", "<", "<=", "=="]
    value: float
    unit: str = ""
    aggregation: str = "mean"
    required: bool = True


class BehaviorRequirement(BaseModel):
    name: str
    description: str
    phase: str = "all"


class ForbiddenBehavior(BaseModel):
    name: str
    description: str
    severity: Literal["warning", "hard_constraint"] = "hard_constraint"


class TaskPhase(BaseModel):
    name: str
    description: str
    entry_condition: str = ""
    exit_condition: str = ""


class TrainingBudget(BaseModel):
    max_total_iterations: int = 12000
    max_wall_time_seconds: int = 86400
    max_reward_revisions: int = 3

    @validator("max_total_iterations", "max_wall_time_seconds")
    def positive(cls, value: int) -> int:
        """校验数值必须为正数。"""
        if value <= 0:
            raise ValueError("training budget must be positive")
        return value


class TaskSpec(BaseModel):
    task_id: str
    robot: str
    task_name: str
    original_instruction: str
    normalized_description: str
    initial_state: str
    required_behaviors: List[BehaviorRequirement]
    forbidden_behaviors: List[ForbiddenBehavior]
    phases: List[TaskPhase]
    required_observations: List[str]
    required_sensors: List[str]
    success_metrics: List[MetricThreshold]
    safety_constraints: List[MetricThreshold]
    training_budget: TrainingBudget
    visual_evaluation_requirements: List[str]
    unsupported_requirements: List[str] = Field(default_factory=list)
    derivable_requirements: List[str] = Field(default_factory=list)
