from __future__ import annotations

from .state_machine import AgentState


RECOVERABLE_STATES = {
    AgentState.RECEIVED, AgentState.ENVIRONMENT_INSPECTED, AgentState.TASK_DESIGNED,
    AgentState.REWARD_CANDIDATES_CREATED, AgentState.CONFIGS_COMPILED, AgentState.VALIDATED,
    AgentState.SMOKE_TRAINING, AgentState.CANDIDATE_SCREENING, AgentState.FULL_TRAINING,
    AgentState.ROLLOUT_COLLECTING, AgentState.VISUAL_EVALUATING, AgentState.NUMERIC_EVALUATING,
    AgentState.DIAGNOSING, AgentState.CONTINUE_TRAINING, AgentState.REVISE_REWARD,
    AgentState.REVISE_CURRICULUM, AgentState.ROLLBACK, AgentState.RESTART,
}


def can_resume(state: AgentState) -> bool:
    """判断指定状态是否允许从持久化现场恢复。"""
    return state in RECOVERABLE_STATES
