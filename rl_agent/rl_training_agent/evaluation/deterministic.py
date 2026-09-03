from __future__ import annotations

import math
import operator
from typing import Dict, List

from ..schemas.metrics import EvaluationResult, MetricSummary
from ..schemas.task import MetricThreshold, TaskSpec
from ..schemas.visual import VisualBehaviorReport


OPERATORS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq}


class DeterministicEvaluator:
    def _check(self, threshold: MetricThreshold, metrics: Dict[str, float]) -> MetricSummary:
        """用任务阈值检查一项确定性指标。"""
        value = metrics.get(threshold.name, float("nan"))
        passed = math.isfinite(value) and OPERATORS[threshold.operator](value, threshold.value)
        return MetricSummary(name=threshold.name, value=value, unit=threshold.unit, passed=passed)

    def evaluate(self, task: TaskSpec, metrics: Dict[str, float], visual: VisualBehaviorReport) -> EvaluationResult:
        """执行视觉与确定性证据的联合评估。"""
        safety = [self._check(item, metrics) for item in task.safety_constraints]
        task_results = [self._check(item, metrics) for item in task.success_metrics]
        implicit_safety = (metrics.get("nan_count", 0.0) == 0.0 and
                           metrics.get("joint_limit_violations", 0.0) == 0.0 and
                           metrics.get("torque_limit_violations", 0.0) == 0.0 and
                           metrics.get("forbidden_collisions", 0.0) == 0.0 and
                           metrics.get("abnormal_terminations", 0.0) == 0.0)
        hard = implicit_safety and all(item.passed for item in safety)
        task_passed = bool(task_results) and all(item.passed for item in task_results)
        visual_passed = visual.visual_success and visual.alignment_score >= 0.7 and not visual.requires_human_review
        conflicts: List[str] = []
        if visual_passed != task_passed:
            conflicts.append("visual_numeric_disagreement")
        completed = hard and task_passed and visual_passed and not conflicts
        violations = [item.name for item in safety if not item.passed]
        if not implicit_safety:
            violations.append("implicit_simulation_safety")
        return EvaluationResult(hard_constraints_passed=hard, task_metrics_passed=task_passed,
                                visual_alignment_passed=visual_passed, completed=completed,
                                metrics=safety + task_results, violations=violations, conflicts=conflicts)

