"""定义奖励经验 Agent 使用的证据、资格和输出协议。"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator


ExperienceOutcome = Literal["SUCCESS", "VERIFIED_FAILURE"]
EligibilityOutcome = Literal["SUCCESS", "VERIFIED_FAILURE", "INCONCLUSIVE"]


class EvidenceSource(BaseModel):
    """描述实验目录中的一个真实文件及其不可变内容摘要。"""

    source: str
    sha256: str


class RewardChangeEvidence(BaseModel):
    """记录初始和最终奖励配置之间的单项确定性差异。"""

    reward_name: str
    before: Optional[Any] = None
    after: Optional[Any] = None
    iteration: Optional[int] = None
    reward_version: Optional[int] = None
    evidence_ids: List[str] = Field(default_factory=list)


class RewardPlanSnapshot(BaseModel):
    """指向奖励版本父子链中的一个真实计划、manifest 和可选修订审计。"""

    experiment_id: str
    parent_experiment_id: Optional[str] = None
    reward_version: int
    iteration: Optional[int] = None
    evidence_ids: List[str] = Field(default_factory=list)


class RewardExperienceEvidence(BaseModel):
    """保存由真实实验文件构建、供资格门控和总结使用的紧凑证据包。"""

    task_id: str
    task: Dict[str, Any]
    initial_reward: Dict[str, Any]
    final_reward: Dict[str, Any]
    reward_history: List[RewardPlanSnapshot] = Field(default_factory=list)
    reward_changes: List[RewardChangeEvidence] = Field(default_factory=list)
    training_result: Dict[str, Any] = Field(default_factory=dict)
    visual_evaluation: Dict[str, Any] = Field(default_factory=dict)
    numeric_evaluation: Dict[str, Any] = Field(default_factory=dict)
    failure_cases: List[str] = Field(default_factory=list)
    git_commit: str = "unknown"
    config_hash: str = "unknown"
    evidence_index: Dict[str, EvidenceSource] = Field(default_factory=dict)


class ExperienceEligibility(BaseModel):
    """记录实验是否满足写入奖励经验和情景记忆的确定性门槛。"""

    eligible: bool
    reason: str
    outcome: EligibilityOutcome
    checks: Dict[str, bool] = Field(default_factory=dict)


class EvidenceStatement(BaseModel):
    """表示必须引用一个或多个真实来源的观察、假设或限制。"""

    statement: str
    evidence_ids: List[str] = Field(default_factory=list)

    @validator("statement")
    def nonempty_statement(cls, value: str) -> str:
        """拒绝空的经验陈述。"""
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("statement must not be empty")
        return cleaned


class RewardChangeSummary(BaseModel):
    """将本地计算的奖励差异与模型归纳的同期行为变化组合。"""

    reward_name: str
    before: Optional[Any] = None
    after: Optional[Any] = None
    iteration: Optional[int] = None
    reward_version: Optional[int] = None
    observed_behavior_change: str = ""
    evidence_ids: List[str] = Field(default_factory=list)


class RewardChangeObservation(BaseModel):
    """记录某个奖励项与同期行为观察之间的显式、非因果关联。"""

    reward_name: str
    reward_version: Optional[int] = None
    observed_behavior_change: str
    evidence_ids: List[str] = Field(default_factory=list)


class ScopedPattern(BaseModel):
    """保存严格限定在本机器人、动作和环境范围内的成功或失败模式。"""

    statement: str
    applicability: Dict[str, str]
    evidence_ids: List[str] = Field(default_factory=list)


class RewardEvolution(BaseModel):
    """记录由证据构建器提供的不可由模型改写的奖励版本和差异。"""

    initial_design: Dict[str, Any]
    final_design: Dict[str, Any]
    changes: List[RewardChangeSummary] = Field(default_factory=list)


class RewardExperienceNarrative(BaseModel):
    """模型可生成的受限叙述，不包含任务、奖励数值或评估数据字段。"""

    observed_facts: List[EvidenceStatement] = Field(default_factory=list)
    hypotheses: List[EvidenceStatement] = Field(default_factory=list)
    reward_change_observations: List[RewardChangeObservation] = Field(default_factory=list)
    successful_patterns: List[str] = Field(default_factory=list)
    failure_patterns: List[str] = Field(default_factory=list)
    limitations: List[EvidenceStatement] = Field(default_factory=list)


class RewardExperience(BaseModel):
    """表示通过证据校验后可写入情景记忆的单次实验经验。"""

    experience_id: str
    outcome: ExperienceOutcome
    task_context: Dict[str, Any]
    reward_evolution: RewardEvolution
    evidence_index: Dict[str, EvidenceSource] = Field(default_factory=dict)
    observed_facts: List[EvidenceStatement] = Field(default_factory=list)
    hypotheses: List[EvidenceStatement] = Field(default_factory=list)
    successful_patterns: List[ScopedPattern] = Field(default_factory=list)
    failure_patterns: List[ScopedPattern] = Field(default_factory=list)
    limitations: List[EvidenceStatement] = Field(default_factory=list)
    confidence: float = 0.5

    @validator("confidence")
    def confidence_interval(cls, value: float) -> float:
        """限制最终经验可信度在零到一之间。"""
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        return value
