from __future__ import annotations

from typing import Dict, List


class RolloutSelector:
    def select(self, rollouts: List[Dict[str, float]], final: bool = False) -> Dict[str, Dict[str, float]]:
        """按任务、安全、姿态、滑动和能耗分层选择 rollout。"""
        if not rollouts:
            return {}
        ordered = sorted(rollouts, key=lambda item: item.get("task_score", 0.0))
        result = {"worst": ordered[0], "median": ordered[len(ordered) // 2], "best": ordered[-1]}
        if final:
            for label, metric, reverse in (("worst_posture", "posture_error", True),
                                           ("max_foot_slip", "foot_slip", True),
                                           ("max_energy", "energy", True),
                                           ("safety_violation", "safety_violations", True)):
                result[label] = sorted(rollouts, key=lambda item: item.get(metric, 0.0), reverse=reverse)[0]
        return result

