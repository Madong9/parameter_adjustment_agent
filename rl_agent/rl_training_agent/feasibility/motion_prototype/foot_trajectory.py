"""按参数化步态采样足端笛卡尔参考，不进行 IK 或策略控制。"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

from .schema import FootTrajectory, GaitPattern


class FootTrajectoryGenerator:
    """把步频、相位和步幅转为四个足端目标位置。"""

    LEG_ORDER = ("FL", "FR", "RL", "RR")

    @classmethod
    def sample(cls, nominal_feet: Dict[str, Sequence[float]], gait: GaitPattern,
               trajectory: FootTrajectory, time_seconds: float) -> Dict[str, List[float]]:
        """生成指定时刻的局部足端笛卡尔目标，结果不含任何关节/执行器量。"""
        if not math.isfinite(float(time_seconds)) or time_seconds < 0.0:
            raise ValueError("foot trajectory sample time must be finite and non-negative")
        if set(nominal_feet) != set(cls.LEG_ORDER):
            raise ValueError("nominal feet must define FL, FR, RL and RR")
        offsets = trajectory.phase_offset or gait.phase_offsets
        if set(offsets) != set(cls.LEG_ORDER):
            raise ValueError("foot trajectory requires phase offsets for four legs")
        result: Dict[str, List[float]] = {}
        for leg in cls.LEG_ORDER:
            origin = [float(value) for value in nominal_feet[leg]]
            if len(origin) != 3 or not all(math.isfinite(value) for value in origin):
                raise ValueError("nominal foot coordinates must be three finite values")
            cycle = (float(time_seconds) / trajectory.step_period +
                     float(offsets[leg])) % 1.0
            if cycle < trajectory.duty_factor:
                # 支撑相在局部机身坐标系中向后扫掠，为机身前进留出相对步长。
                progress = cycle / trajectory.duty_factor
                stride_x = (trajectory.direction_x * trajectory.step_length *
                            (0.5 - progress))
                lift = 0.0
            else:
                progress = (cycle - trajectory.duty_factor) / (1.0 - trajectory.duty_factor)
                stride_x = (trajectory.direction_x * trajectory.step_length *
                            (-0.5 + progress))
                lift = trajectory.swing_height * math.sin(math.pi * progress)
            result[leg] = [origin[0] + stride_x, origin[1], origin[2] + lift]
        return result
