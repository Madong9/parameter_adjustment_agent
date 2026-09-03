from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, root_validator, validator


class VisualPhaseResult(BaseModel):
    phase: str
    score: float
    evidence_frames: List[int] = Field(default_factory=list)
    notes: str = ""

    @validator("score")
    def discrete_score(cls, value: float) -> float:
        """校验阶段分数只能取允许的离散值。"""
        if value not in (0.0, 0.5, 1.0):
            raise ValueError("phase score must be 0.0, 0.5, or 1.0")
        return value


class VisualFailure(BaseModel):
    failure_mode: str
    evidence_frames: List[int]
    description: str


class VisualEvidenceFinding(BaseModel):
    """记录一个原不确定问题如何被同步证据确认、否定或保留。"""

    name: str
    status: Literal["confirmed", "refuted", "uncertain"]
    source: str
    evidence_frames: List[int] = Field(default_factory=list)
    evidence: str

    @root_validator(pre=True)
    def recover_evidence_text(cls, values):
        """兼容模型把证据正文命名为 notes/description，避免完整视觉结论因字段别名被丢弃。"""
        data = dict(values or {})
        if not str(data.get("evidence", "")).strip():
            fallback = data.get("notes") or data.get("description") or data.get("source")
            if fallback:
                data["evidence"] = str(fallback)
        return data


class VisualBehaviorReport(BaseModel):
    visual_success: bool
    alignment_score: float
    confidence: float
    summary: str
    phase_results: List[VisualPhaseResult]
    failure_modes: List[VisualFailure] = Field(default_factory=list)
    unintended_behaviors: List[str] = Field(default_factory=list)
    evidence_frames: List[int] = Field(default_factory=list)
    evidence_findings: List[VisualEvidenceFinding] = Field(default_factory=list)
    uncertain_items: List[str] = Field(default_factory=list)
    requires_human_review: bool = False

    @validator("alignment_score", "confidence")
    def unit_interval(cls, value: float) -> float:
        """校验数值位于零到一的闭区间。"""
        if not 0.0 <= value <= 1.0:
            raise ValueError("value must be within [0, 1]")
        return value
