from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from ..utils.io import read_json, utc_now, write_json
from ..observability.events import JsonlEventRecorder


class AgentState(str, Enum):
    RECEIVED = "RECEIVED"
    ENVIRONMENT_INSPECTED = "ENVIRONMENT_INSPECTED"
    TASK_UNDERSTANDING = "TASK_UNDERSTANDING"
    TASK_FEASIBILITY_CHECK = "TASK_FEASIBILITY_CHECK"
    MOTION_PROTOTYPE_GENERATING = "MOTION_PROTOTYPE_GENERATING"
    STATIC_MOTION_VALIDATION = "STATIC_MOTION_VALIDATION"
    DYNAMIC_MOTION_VALIDATION = "DYNAMIC_MOTION_VALIDATION"
    RAG_RETRIEVING = "RAG_RETRIEVING"
    CONTEXT_BUILDING = "CONTEXT_BUILDING"
    PROMPT_COMPILING = "PROMPT_COMPILING"
    REWARD_DESIGNING = "REWARD_DESIGNING"
    REWARD_REVIEWING = "REWARD_REVIEWING"
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
    MEMORY_CURATING = "MEMORY_CURATING"
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
    completed_operations: List[str] = Field(default_factory=list)


ALLOWED_TRANSITIONS = {
    AgentState.RECEIVED: {AgentState.ENVIRONMENT_INSPECTED},
    AgentState.ENVIRONMENT_INSPECTED: {AgentState.TASK_UNDERSTANDING},
    AgentState.TASK_UNDERSTANDING: {AgentState.TASK_FEASIBILITY_CHECK},
    AgentState.TASK_FEASIBILITY_CHECK: {
        AgentState.RAG_RETRIEVING, AgentState.CONTEXT_BUILDING,
        AgentState.MOTION_PROTOTYPE_GENERATING,
        AgentState.HUMAN_REVIEW, AgentState.FAILED,
    },
    AgentState.MOTION_PROTOTYPE_GENERATING: {
        AgentState.STATIC_MOTION_VALIDATION, AgentState.DYNAMIC_MOTION_VALIDATION,
        AgentState.RAG_RETRIEVING, AgentState.CONTEXT_BUILDING,
        AgentState.HUMAN_REVIEW, AgentState.FAILED,
    },
    AgentState.STATIC_MOTION_VALIDATION: {
        AgentState.RAG_RETRIEVING, AgentState.CONTEXT_BUILDING,
        AgentState.HUMAN_REVIEW, AgentState.FAILED,
    },
    AgentState.DYNAMIC_MOTION_VALIDATION: {
        AgentState.RAG_RETRIEVING, AgentState.CONTEXT_BUILDING,
        AgentState.HUMAN_REVIEW, AgentState.FAILED,
    },
    AgentState.RAG_RETRIEVING: {AgentState.CONTEXT_BUILDING},
    AgentState.CONTEXT_BUILDING: {AgentState.PROMPT_COMPILING},
    AgentState.PROMPT_COMPILING: {AgentState.REWARD_DESIGNING},
    AgentState.REWARD_DESIGNING: {AgentState.TASK_DESIGNED},
    AgentState.TASK_DESIGNED: {AgentState.REWARD_REVIEWING, AgentState.HUMAN_REVIEW},
    AgentState.REWARD_REVIEWING: {AgentState.REWARD_CANDIDATES_CREATED, AgentState.HUMAN_REVIEW},
    AgentState.REWARD_CANDIDATES_CREATED: {AgentState.CONFIGS_COMPILED, AgentState.FAILED},
    AgentState.CONFIGS_COMPILED: {AgentState.VALIDATED, AgentState.HUMAN_REVIEW},
    AgentState.VALIDATED: {AgentState.SMOKE_TRAINING, AgentState.HUMAN_REVIEW},
    AgentState.SMOKE_TRAINING: {AgentState.CANDIDATE_SCREENING, AgentState.FAILED},
    AgentState.CANDIDATE_SCREENING: {AgentState.FULL_TRAINING, AgentState.FAILED},
    AgentState.FULL_TRAINING: {AgentState.ROLLOUT_COLLECTING, AgentState.FAILED},
    AgentState.ROLLOUT_COLLECTING: {AgentState.VISUAL_EVALUATING, AgentState.FAILED},
    AgentState.VISUAL_EVALUATING: {AgentState.NUMERIC_EVALUATING, AgentState.HUMAN_REVIEW},
    AgentState.NUMERIC_EVALUATING: {AgentState.DIAGNOSING},
    AgentState.DIAGNOSING: {
        AgentState.CONTINUE_TRAINING, AgentState.REVISE_REWARD, AgentState.REVISE_CURRICULUM,
        AgentState.ROLLBACK, AgentState.RESTART, AgentState.MEMORY_CURATING,
        AgentState.HUMAN_REVIEW, AgentState.FAILED, AgentState.COMPLETED,
    },
    # 决策已经持久化后，编译、预算检查或训练启动仍可能安全阻塞。
    # 这些状态必须能够正常收束到人工复核，不能以非法状态跳转崩溃。
    AgentState.CONTINUE_TRAINING: {AgentState.FULL_TRAINING, AgentState.HUMAN_REVIEW},
    AgentState.REVISE_REWARD: {AgentState.FULL_TRAINING, AgentState.HUMAN_REVIEW},
    AgentState.REVISE_CURRICULUM: {AgentState.FULL_TRAINING, AgentState.HUMAN_REVIEW},
    AgentState.ROLLBACK: {AgentState.FULL_TRAINING, AgentState.HUMAN_REVIEW},
    AgentState.RESTART: {AgentState.FULL_TRAINING, AgentState.HUMAN_REVIEW},
    AgentState.MEMORY_CURATING: {AgentState.COMPLETED, AgentState.FAILED},
    AgentState.HUMAN_REVIEW: {AgentState.ROLLOUT_COLLECTING},
    AgentState.COMPLETED: set(),
    AgentState.FAILED: set(),
}


class PersistentStateMachine:
    def __init__(self, path: Path):
        """初始化 PersistentStateMachine 实例及其运行依赖。"""
        self.path = path
        self.events = JsonlEventRecorder(path.parent / "events.jsonl")
        if path.exists():
            self.record = StateRecord.parse_obj(read_json(path))
        else:
            self.record = StateRecord(state=AgentState.RECEIVED, updated_at=utc_now(),
                                      history=[{"state": AgentState.RECEIVED.value, "at": utc_now()}])
            self._save()

    def transition(self, state: AgentState, context: Optional[Dict[str, object]] = None,
                   operation_id: Optional[str] = None) -> bool:
        """校验并原子提交状态变化；重复操作键直接返回而不重复写入。"""
        if operation_id and operation_id in self.record.completed_operations:
            return False
        current = self.record.state
        if state == current:
            if operation_id:
                self.record.completed_operations.append(operation_id)
                self._save()
            return False
        if state not in ALLOWED_TRANSITIONS.get(current, set()):
            raise ValueError("illegal state transition: %s -> %s" % (current.value, state.value))
        now = utc_now()
        self.record.state = state
        self.record.updated_at = now
        self.record.history.append({"state": state.value, "at": now})
        if context:
            self.record.context.update(context)
        if operation_id:
            self.record.completed_operations.append(operation_id)
        self._save()
        self.events.emit("state_transition", {
            "from": current.value, "to": state.value, "operation_id": operation_id,
            "context": context or {},
        })
        return True

    def start_new_run(self, run_id: str) -> None:
        """显式开启同一稳定 task_id 的新运行，并保留历史审计边界。"""
        if self.record.state == AgentState.RECEIVED and not self.record.context.get("run_id"):
            self.record.context["run_id"] = run_id
            self._save()
            self.events.emit("run_started", {"run_id": run_id})
            return
        now = utc_now()
        self.record.state = AgentState.RECEIVED
        self.record.updated_at = now
        self.record.history.append({"state": AgentState.RECEIVED.value, "at": now,
                                    "reason": "new_run", "run_id": run_id})
        self.record.context = {"run_id": run_id}
        self.record.completed_operations = []
        self._save()
        self.events.emit("run_started", {"run_id": run_id})

    def _save(self) -> None:
        """原子保存当前状态机记录。"""
        write_json(self.path, self.record)
