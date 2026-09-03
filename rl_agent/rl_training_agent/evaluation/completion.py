from __future__ import annotations

from ..schemas.metrics import EvaluationResult


def completion_decision(result: EvaluationResult) -> str:
    """依据确定性评估结果给出完成或恢复决策。"""
    if result.conflicts:
        return "human_review"
    if result.completed:
        return "complete"
    if not result.hard_constraints_passed:
        return "revise_reward"
    return "continue"

