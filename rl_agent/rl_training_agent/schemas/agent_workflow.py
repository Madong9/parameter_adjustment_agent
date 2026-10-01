"""定义多 Agent 协作、奖励审查和长期记忆使用的结构化协议。"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator

from .rewards import RewardPlan
from .task import TaskSpec


class TaskIntentSpec(BaseModel):
    """表示任务理解 Agent 从自然语言中提取的动作意图。"""

    original_instruction: str
    robot: str
    action_name: str
    normalized_goal: str
    target_velocity: Optional[float] = None
    terrain: str = "默认平地"
    required_behaviors: List[str] = Field(default_factory=list)
    forbidden_behaviors: List[str] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    ambiguities: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    retrieval_keywords: List[str] = Field(default_factory=list)

    @validator("original_instruction", "robot", "action_name", "normalized_goal")
    def nonempty_core_fields(cls, value: str) -> str:
        """保证任务意图的核心字段不是空字符串。"""
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("task intent core fields must not be empty")
        return cleaned


class TaskRewardBundle(BaseModel):
    """封装奖励设计 Agent 必须返回的任务和奖励候选。"""

    task_spec: TaskSpec
    reward_plans: List[RewardPlan]
    reward_hacking_risks: List[str] = Field(default_factory=list)
    termination_suggestions: List[str] = Field(default_factory=list)

    @validator("reward_plans")
    def nonempty_reward_plans(cls, value: List[RewardPlan]) -> List[RewardPlan]:
        """拒绝没有任何奖励候选的模型回复。"""
        if not value:
            raise ValueError("reward design must contain at least one candidate")
        return value


class RewardCandidateReview(BaseModel):
    """记录单个奖励候选的准入结果和可审计拒绝原因。"""

    candidate_index: int
    approved: bool
    omissions: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)


class RewardReviewReport(BaseModel):
    """记录逐候选奖励审查结果以及兼容旧调用方的汇总字段。"""

    approved: bool
    omissions: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    reward_hacking_risks: List[str] = Field(default_factory=list)
    checked_candidates: int = 0
    candidate_results: List[RewardCandidateReview] = Field(default_factory=list)
    passed_candidate_indexes: List[int] = Field(default_factory=list)
    rejected_candidate_indexes: List[int] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class LongTermMemoryRecord(BaseModel):
    """表示经过确定性门控后允许长期复用的一条情景记忆。"""

    memory_id: str
    memory_type: Literal["episodic"] = "episodic"
    task_id: str
    robot: str
    action_name: str
    normalized_goal: str
    outcome: str
    final_state: str = "COMPLETED"
    reward_version: int
    reward_terms: List[Dict[str, Any]] = Field(default_factory=list)
    reward_diff: Dict[str, Any] = Field(default_factory=dict)
    reward_experience: Optional[Dict[str, Any]] = None
    lessons: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    deterministic_metrics: Dict[str, Any] = Field(default_factory=dict)
    visual_conclusion: Dict[str, Any] = Field(default_factory=dict)
    evidence_sources: List[str] = Field(default_factory=list)
    environment_fingerprint: str = "unknown"
    simulation_platform: str = "unitree_rl_gym / Isaac Gym"
    environment_version: str = "unknown"
    git_commit: str = "unknown"
    checkpoint: str = ""
    seed_count: int = 0
    consistent_across_seeds: bool = False
    failure_pattern: Optional[str] = None
    confidence: float = 1.0
    status: Literal["active", "archived"] = "active"
    archive_reason: Optional[str] = None
    access_count: int = 0
    last_accessed_at: Optional[str] = None
    created_at: str
    updated_at: Optional[str] = None

    @validator("confidence")
    def confidence_interval(cls, value: float) -> float:
        """保证长期记忆可信度位于零到一之间。"""
        if not 0.0 <= value <= 1.0:
            raise ValueError("memory confidence must be within [0, 1]")
        return value


class WorkingMemorySnapshot(BaseModel):
    """保存当前任务随闭环不断更新、任务结束后仍可审计的工作记忆。"""

    task_id: str
    state: str
    loop_round: int = 0
    reward_version: int = 0
    used_iterations: int = 0
    remaining_iterations: int = 0
    used_revisions: int = 0
    remaining_revisions: int = 0
    latest_evaluation: Dict[str, Any] = Field(default_factory=dict)
    latest_diagnosis: Dict[str, Any] = Field(default_factory=dict)
    updated_at: str


class SemanticMemoryRecord(BaseModel):
    """表示由多条独立情景证据升级形成的通用规律。"""

    semantic_id: str
    memory_type: Literal["semantic"] = "semantic"
    rule_key: str
    statement: str
    robot_scope: List[str] = Field(default_factory=list)
    action_scope: List[str] = Field(default_factory=list)
    supporting_memory_ids: List[str] = Field(default_factory=list)
    supporting_task_ids: List[str] = Field(default_factory=list)
    counterevidence_memory_ids: List[str] = Field(default_factory=list)
    evidence_count: int = 0
    confidence: float = 0.0
    status: Literal["candidate", "active", "archived", "superseded"] = "candidate"
    archive_reason: Optional[str] = None
    superseded_by: Optional[str] = None
    created_at: str
    updated_at: str

    @validator("confidence")
    def semantic_confidence_interval(cls, value: float) -> float:
        """保证语义记忆可信度位于零到一之间。"""
        if not 0.0 <= value <= 1.0:
            raise ValueError("semantic confidence must be within [0, 1]")
        return value


class ProceduralMemoryRecord(BaseModel):
    """记录提示词、Schema、校验规则和闭环策略的可复现版本。"""

    procedure_id: str
    memory_type: Literal["procedural"] = "procedural"
    version: str
    prompt_versions: Dict[str, str] = Field(default_factory=dict)
    schema_hashes: Dict[str, str] = Field(default_factory=dict)
    validation_rules: List[str] = Field(default_factory=list)
    workflow_states: List[str] = Field(default_factory=list)
    diagnosis_policy: Dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str
