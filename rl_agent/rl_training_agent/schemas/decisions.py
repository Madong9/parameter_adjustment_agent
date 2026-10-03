from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, validator, root_validator


class DiagnosisItem(BaseModel):
    category: str
    finding: str
    severity: Literal["info", "warning", "critical"] = "warning"


class EvidenceItem(BaseModel):
    source: str
    metric: str
    value: Any
    interpretation: str


class RewardChange(BaseModel):
    term: str
    action: Literal["add", "remove", "update"]
    changes: Dict[str, Any] = Field(default_factory=dict)
    rationale: str


class CurriculumChange(BaseModel):
    stage: str
    changes: Dict[str, Any]
    rationale: str


class TrainingDiagnosis(BaseModel):
    diagnosis: List[DiagnosisItem]
    evidence: List[EvidenceItem]
    decision: Literal["continue", "revise_reward", "revise_curriculum", "restart", "rollback", "complete", "human_review", "failed"]
    confidence: float
    reward_changes: List[RewardChange] = Field(default_factory=list)
    curriculum_changes: List[CurriculumChange] = Field(default_factory=list)
    expected_effects: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    checkpoint_strategy: Literal["continue_from_current", "continue_from_parent", "restart_from_scratch"]

    @validator("confidence")
    def unit_interval(cls, value: float) -> float:
        """校验数值位于零到一的闭区间。"""
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        return value


class GPTTrainingDiagnosis(TrainingDiagnosis):
    """真实 GPT 诊断必须对未完成的训练给出具体建议；兼容旧历史结构。"""

    @root_validator
    def failed_training_needs_suggestions(cls, values):
        if values.get('decision') != 'complete' and not any(str(s).strip() for s in values.get('suggestions', [])):
            raise ValueError('训练未完成时必须给出非空 suggestions，不能只有失败结论')
        return values
