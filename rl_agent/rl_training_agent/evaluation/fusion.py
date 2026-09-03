from __future__ import annotations

from typing import Dict


def candidate_score(metrics: Dict[str, float]) -> float:
    """按安全优先原则计算候选综合分数。"""
    if metrics.get("hard_constraints_passed", 0.0) < 0.5:
        return float("-inf")
    return (0.35 * metrics.get("success_rate", 0.0) +
            0.25 * metrics.get("task_score", 0.0) +
            0.15 * metrics.get("visual_alignment", 0.0) +
            0.15 * metrics.get("ppo_stability", 0.0) -
            0.05 * metrics.get("energy", 0.0) -
            0.05 * metrics.get("reward_hacking_score", 0.0))

