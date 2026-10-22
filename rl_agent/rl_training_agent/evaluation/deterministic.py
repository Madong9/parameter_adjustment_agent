from __future__ import annotations

import math
import operator
from typing import Dict, List

from ..schemas.metrics import EvaluationResult, MetricSummary
from ..schemas.task import MetricThreshold, TaskSpec
from ..schemas.visual import VisualBehaviorReport
from ..utils.units import is_dimensionless_unit


OPERATORS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq}


class DeterministicEvaluator:
    def _check(self, threshold: MetricThreshold, metrics: Dict[str, float]) -> MetricSummary:
        """用任务阈值检查一项确定性指标。"""
        value = metrics.get(threshold.name, float("nan"))
        scores = {"tracking_lin_vel", "front_leg_walk_velocity_tracking",
                  "rear_leg_walk_velocity_tracking", "walking_speed_tracking", "orientation", "landing_stability", "pitch_axis_error"}
        incompatible = (threshold.name in scores and threshold.unit != "score"
                        and not is_dimensionless_unit(threshold.unit))
        incompatible |= (threshold.name in {"front_leg_forward_speed", "rear_leg_forward_speed"}
                         and threshold.unit != "m/s")
        physical_units = {"final_roll_angle": "rad", "final_pitch_angle": "rad", "max_roll_velocity": "rad/s", "feet_air_time": "s",
                          "takeoff_time_spread": "s", "landing_pitch_abs": "rad"}
        incompatible |= (threshold.name in physical_units and threshold.unit != physical_units[threshold.name])
        valid = isinstance(value, (int, float)) and math.isfinite(value)
        passed = not incompatible and valid and OPERATORS[threshold.operator](value, threshold.value)
        if not valid:
            value = float('nan')
        unit = "1" if threshold.name in scores else threshold.unit
        return MetricSummary(name=threshold.name, value=value, unit=unit, passed=passed)

    def evaluate(self, task: TaskSpec, metrics: Dict[str, float], visual: VisualBehaviorReport) -> EvaluationResult:
        """执行视觉与确定性证据的联合评估。"""
        safety = [self._check(item, metrics) for item in task.safety_constraints]
        task_results = [self._check(item, metrics) for item in task.success_metrics]
        mandatory = ('nan_count', 'joint_limit_violations', 'torque_limit_violations',
                     'forbidden_collisions', 'abnormal_terminations')
        implicit_safety = all(isinstance(metrics.get(name), (int, float)) and
                              math.isfinite(metrics[name]) and metrics[name] == 0.0
                              for name in mandatory)
        counterfactual = metrics.get('counterfactual_passed')
        counterfactual_passed = (counterfactual is True or
            (type(counterfactual) in (int, float) and math.isfinite(counterfactual) and counterfactual == 1))
        if 'counterfactual_passed' not in metrics and metrics.get('counterfactual_applicable') is not True:
            counterfactual_passed = True
        hard = implicit_safety and counterfactual_passed and all(item.passed for item in safety)
        task_passed = bool(task_results) and all(item.passed for item in task_results)
        visual_passed = visual.visual_success and visual.alignment_score >= 0.7 and not visual.requires_human_review
        conflicts: List[str] = []
        if visual_passed != task_passed:
            conflicts.append("visual_numeric_disagreement")
        completed = hard and task_passed and visual_passed and not conflicts
        violations = [item.name for item in safety if not item.passed]
        if not implicit_safety:
            violations.append("implicit_simulation_safety")
            violations.extend("safety_evidence:%s" % name for name in mandatory
                              if not isinstance(metrics.get(name), (int, float)) or
                              not math.isfinite(metrics[name]) or metrics[name] != 0.0)
        if not counterfactual_passed:
            violations.append("reward_hacking_counterfactual")
        return EvaluationResult(hard_constraints_passed=hard, task_metrics_passed=task_passed,
                                visual_alignment_passed=visual_passed, completed=completed,
                                metrics=safety + task_results, violations=violations, conflicts=conflicts)
