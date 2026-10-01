"""检查由可信动力学后端提供的关节力矩和执行器限值。"""
from __future__ import annotations

import math
from typing import Dict, Optional

from ..schema import StageStatus, ValidationStageReport


class TorqueFeasibilityValidator:
    """不自行推断接触力；只比较同名关节的真实力矩与明确限值。"""

    @staticmethod
    def validate(required_torque: Optional[Dict[str, float]],
                 torque_limits: Optional[Dict[str, float]], backend: str = "provided_dynamics"
                 ) -> ValidationStageReport:
        """验证所提供力矩样本是否全部位于 actuator effort 限值内。"""
        if not required_torque or not torque_limits:
            return ValidationStageReport(
                stage="torque_feasibility", status=StageStatus.UNKNOWN,
                backend="not_available",
                reason="缺少真实 required torque 或 actuator torque limit；没有假设静力模型",
            )
        missing, violations = [], []
        ratios: Dict[str, float] = {}
        for joint, torque in required_torque.items():
            limit = torque_limits.get(joint)
            if limit is None or not math.isfinite(float(limit)) or float(limit) <= 0.0:
                missing.append(joint)
                continue
            if not math.isfinite(float(torque)):
                violations.append("non_finite_torque:%s" % joint)
                continue
            ratio = abs(float(torque)) / float(limit)
            ratios[joint] = ratio
            if ratio > 1.0 + 1.0e-8:
                violations.append("torque_limit:%s" % joint)
        status = (StageStatus.FAILED if violations else
                  StageStatus.CONDITIONAL if missing else StageStatus.PASSED)
        return ValidationStageReport(
            stage="torque_feasibility", status=status, backend=backend,
            reason="力矩超过执行器限制" if violations else
                   "部分关节缺少 effort 限值" if missing else "力矩样本均在 actuator limit 内",
            metrics={"torque_ratio_by_joint": ratios,
                     "max_torque_ratio": max(ratios.values(), default=None),
                     "missing_limits": missing, "violations": violations},
            evidence=["required torque input", "actuator torque limits"],
        )
