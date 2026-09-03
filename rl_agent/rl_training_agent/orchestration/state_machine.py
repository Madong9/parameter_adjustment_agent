from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from ..utils.io import read_json, utc_now, write_json


class AgentState(str, Enum):
    RECEIVED = "RECEIVED"
    ENVIRONMENT_INSPECTED = "ENVIRONMENT_INSPECTED"
    TASK_DESIGNED = "TASK_DESIGNED"
    REWARD_CANDIDATES_CREATED = "REWARD_CANDIDATES_CREATED"
    CONFIGS_COMPILED = "CONFIGS_COMPILED"
    VALIDATED = "VALIDATED"
    SMOKE_TRAINING = "SMOKE_TRAINING"
    CANDIDATE_SCREENING = "CANDIDATE_SCREENING"
    FULL_TRAINING = "FULL_TRAINING"
    ROLLOUT_COLLECTING = "ROLLOUT_COLLECTING"
    VISUAL_EVALUATING = "VISUAL_EVALUATING"
    NUMERIC_EVALUATING = "NUMERIC_EVALUATING"
    DIAGNOSING = "DIAGNOSING"
    CONTINUE_TRAINING = "CONTINUE_TRAINING"
    REVISE_REWARD = "REVISE_REWARD"
    REVISE_CURRICULUM = "REVISE_CURRICULUM"
    ROLLBACK = "ROLLBACK"
    RESTART = "RESTART"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class StateRecord(BaseModel):
    state: AgentState
    updated_at: str
    history: List[Dict[str, str]] = Field(default_factory=list)
    context: Dict[str, object] = Field(default_factory=dict)


class PersistentStateMachine:
    def __init__(self, path: Path):
        """初始化 PersistentStateMachine 实例及其运行依赖。"""
        self.path = path
        if path.exists():
            self.record = StateRecord.parse_obj(read_json(path))
        else:
            self.record = StateRecord(state=AgentState.RECEIVED, updated_at=utc_now(),
                                      history=[{"state": AgentState.RECEIVED.value, "at": utc_now()}])
            self._save()

    def transition(self, state: AgentState, context: Optional[Dict[str, object]] = None) -> None:
        """更新当前状态、追加历史并原子落盘。"""
        now = utc_now()
        self.record.state = state
        self.record.updated_at = now
        self.record.history.append({"state": state.value, "at": now})
        if context:
            self.record.context.update(context)
        self._save()

    def _save(self) -> None:
        """原子保存当前状态机记录。"""
        write_json(self.path, self.record)

