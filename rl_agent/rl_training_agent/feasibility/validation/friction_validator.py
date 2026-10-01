"""针对显式接触力样本执行库仑摩擦锥必要条件检查。"""
from __future__ import annotations

import math
from typing import Dict, Optional

from ..schema import StageStatus, ValidationStageReport


class FrictionFeasibilityValidator:
    """只有实际接触力与摩擦系数都存在时才给出通过/失败结论。"""

    @staticmethod
    def validate(contact_force: Optional[Dict[str, float]], friction_coefficient: Optional[float],
                 backend: str = "provided_contact_model") -> ValidationStageReport:
        """检查 sqrt(Fx²+Fy²) <= μFz 且法向力为正。"""
        if not contact_force or friction_coefficient is None:
            return ValidationStageReport(
                stage="friction_cone", status=StageStatus.UNKNOWN,
                backend="not_available", reason="缺少可信接触力或摩擦系数；不生成假想接触数据",
            )
        try:
            fx, fy, fz = (float(contact_force[key]) for key in ("x", "y", "z"))
            mu = float(friction_coefficient)
        except (KeyError, TypeError, ValueError):
            return ValidationStageReport(
                stage="friction_cone", status=StageStatus.INCONCLUSIVE,
                backend=backend, reason="接触力必须提供有限 x/y/z 分量和摩擦系数",
            )
        if not all(math.isfinite(value) for value in (fx, fy, fz, mu)) or mu < 0.0:
            return ValidationStageReport(
                stage="friction_cone", status=StageStatus.INCONCLUSIVE,
                backend=backend, reason="接触力或摩擦系数不是有效有限数值",
            )
        tangential = math.hypot(fx, fy)
        capacity = mu * fz
        feasible = fz > 0.0 and tangential <= capacity + 1.0e-9
        return ValidationStageReport(
            stage="friction_cone", status=StageStatus.PASSED if feasible else StageStatus.FAILED,
            backend=backend,
            reason="接触力在摩擦锥内" if feasible else
                   "法向力非正或切向力超过 μFz 摩擦锥",
            metrics={"tangential_force": tangential, "normal_force": fz,
                     "friction_capacity": capacity, "friction_coefficient": mu},
            evidence=["contact force sample", "surface friction coefficient"],
        )
