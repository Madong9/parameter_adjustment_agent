from __future__ import annotations

from typing import Any, Dict

from ..schemas.rewards import RewardPlan


def effective_stage(plan: RewardPlan, iteration: int) -> Dict[str, Any]:
    """返回指定迭代下生效的课程参数。"""
    changes: Dict[str, Any] = {}
    for stage in plan.curriculum:
        if stage.start_iteration <= iteration <= stage.end_iteration:
            changes.update(stage.parameter_changes)
    return changes
