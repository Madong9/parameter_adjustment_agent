from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class RewardStatistics(BaseModel):
    name: str
    raw_mean: float
    raw_std: float
    raw_min: float
    raw_max: float
    weight: float
    weighted_mean: float
    weighted_std: float
    contribution_ratio: float
    nonzero_ratio: float
    saturation_ratio: float
    per_phase: Dict[str, float] = Field(default_factory=dict)
    episode_sum: float = 0.0


class MetricSummary(BaseModel):
    name: str
    value: float
    unit: str = ""
    passed: Optional[bool] = None


class PPOStatistics(BaseModel):
    iteration: int = 0
    mean_reward: float = 0.0
    value_loss: float = 0.0
    surrogate_loss: float = 0.0
    learning_rate: float = 0.0
    mean_noise_std: float = 0.0
    kl: Optional[float] = None
    fps: float = 0.0


class EvaluationResult(BaseModel):
    hard_constraints_passed: bool
    task_metrics_passed: bool
    visual_alignment_passed: bool
    completed: bool
    metrics: List[MetricSummary] = Field(default_factory=list)
    violations: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)

