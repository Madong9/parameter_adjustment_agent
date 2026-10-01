"""从显式关节参考样本估算速度/加速度并检查位置、速度、力矩和连续性。"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator

from ..schema import StageStatus, ValidationStageReport


class JointTrajectoryPoint(BaseModel):
    """保存一个时刻的关节位置和可选真实力矩样本。"""

    time: float
    positions: Dict[str, float]
    torques: Dict[str, float] = Field(default_factory=dict)

    @validator("time")
    def time_must_be_finite(cls, value: float) -> float:
        """拒绝非有限的轨迹采样时间。"""
        if not math.isfinite(float(value)):
            raise ValueError("trajectory time must be finite")
        return float(value)


class JointTrajectory(BaseModel):
    """描述由本地 IK/规划器生成的关节参考点，不接受策略隐式输出。"""

    points: List[JointTrajectoryPoint]
    source: str = "local_kinematic_planner"

    @validator("points")
    def points_have_consistent_joint_names(cls, value: List[JointTrajectoryPoint]) -> List[JointTrajectoryPoint]:
        """要求至少两个时间递增且关节集合一致的采样点。"""
        if len(value) < 2:
            raise ValueError("joint trajectory requires at least two points")
        names = set(value[0].positions)
        if not names:
            raise ValueError("joint trajectory points must have positions")
        for index, point in enumerate(value):
            if set(point.positions) != names:
                raise ValueError("joint trajectory joint names must be consistent")
            if index and point.time <= value[index - 1].time:
                raise ValueError("joint trajectory times must be strictly increasing")
        return value


class TrajectoryValidator:
    """执行关节空间轨迹的有限差分和模型限值检查。"""

    @staticmethod
    def validate(trajectory: JointTrajectory, joint_limits: Dict[str, object],
                 torque_limits: Optional[Dict[str, float]] = None) -> ValidationStageReport:
        """检查 q、有限差分 qdot/qddot、可用 torque limit 及轨迹连续性。"""
        points = trajectory.points
        joint_names = sorted(points[0].positions)
        violations: List[str] = []
        unknowns: List[str] = []
        max_abs_position: Dict[str, float] = {name: 0.0 for name in joint_names}
        max_abs_velocity: Dict[str, float] = {name: 0.0 for name in joint_names}
        max_abs_acceleration: Dict[str, float] = {name: 0.0 for name in joint_names}
        velocity_samples: List[Dict[str, float]] = []
        velocity_times: List[float] = []
        for point in points:
            for name in joint_names:
                position = float(point.positions[name])
                if not math.isfinite(position):
                    violations.append("non_finite_position:%s" % name)
                    continue
                max_abs_position[name] = max(max_abs_position[name], abs(position))
                raw_limit = joint_limits.get(name)
                if raw_limit is None:
                    unknowns.append("position_limit:%s" % name)
                    continue
                limit = raw_limit.dict() if hasattr(raw_limit, "dict") else dict(raw_limit)
                lower, upper = limit.get("lower"), limit.get("upper")
                if lower is None or upper is None:
                    unknowns.append("position_limit:%s" % name)
                elif position < float(lower) - 1.0e-8 or position > float(upper) + 1.0e-8:
                    violations.append("joint_position_limit:%s" % name)
        for index in range(len(points) - 1):
            dt = points[index + 1].time - points[index].time
            sample: Dict[str, float] = {}
            for name in joint_names:
                velocity = (float(points[index + 1].positions[name]) -
                            float(points[index].positions[name])) / dt
                sample[name] = velocity
                max_abs_velocity[name] = max(max_abs_velocity[name], abs(velocity))
                raw_limit = joint_limits.get(name)
                limit = (raw_limit.dict() if hasattr(raw_limit, "dict") else
                         dict(raw_limit) if raw_limit is not None else {})
                velocity_limit = limit.get("velocity")
                if velocity_limit is None:
                    unknowns.append("velocity_limit:%s" % name)
                elif abs(velocity) > float(velocity_limit) + 1.0e-8:
                    violations.append("joint_velocity_limit:%s" % name)
            velocity_samples.append(sample)
            velocity_times.append((points[index].time + points[index + 1].time) * 0.5)
        for index in range(len(velocity_samples) - 1):
            dt = velocity_times[index + 1] - velocity_times[index]
            if dt <= 0.0:
                violations.append("trajectory_time_discontinuity")
                continue
            for name in joint_names:
                acceleration = ((velocity_samples[index + 1][name] -
                                 velocity_samples[index][name]) / dt)
                max_abs_acceleration[name] = max(max_abs_acceleration[name], abs(acceleration))
                raw_limit = joint_limits.get(name)
                limit = (raw_limit.dict() if hasattr(raw_limit, "dict") else
                         dict(raw_limit) if raw_limit is not None else {})
                acceleration_limit = limit.get("acceleration")
                if acceleration_limit is None:
                    unknowns.append("acceleration_limit:%s" % name)
                elif abs(acceleration) > float(acceleration_limit) + 1.0e-8:
                    violations.append("joint_acceleration_limit:%s" % name)
        if torque_limits is None:
            torque_limits = {}
            for name, raw_limit in joint_limits.items():
                limit = raw_limit.dict() if hasattr(raw_limit, "dict") else dict(raw_limit)
                if limit.get("effort") is not None:
                    torque_limits[str(name)] = float(limit["effort"])
        torque_samples = 0
        for point in points:
            for name, torque in point.torques.items():
                torque_samples += 1
                limit = torque_limits.get(name)
                if limit is None:
                    unknowns.append("torque_limit:%s" % name)
                elif not math.isfinite(float(torque)) or abs(float(torque)) > float(limit) + 1.0e-8:
                    violations.append("joint_torque_limit:%s" % name)
        if not torque_samples:
            unknowns.append("torque_samples")
        violations = sorted(set(violations))
        unknowns = sorted(set(unknowns))
        status = (StageStatus.FAILED if violations else
                  StageStatus.CONDITIONAL if unknowns else StageStatus.PASSED)
        return ValidationStageReport(
            stage="joint_trajectory", status=status,
            reason=("轨迹包含关节限位/连续性违规" if violations else
                    "轨迹差分已计算，但缺少加速度/力矩等完整验收证据" if unknowns else
                    "位置、速度、加速度和力矩限制均已检查"),
            backend="deterministic_finite_difference",
            metrics={"point_count": len(points), "duration_seconds": points[-1].time - points[0].time,
                     "max_abs_position": max_abs_position,
                     "max_abs_velocity": max_abs_velocity,
                     "max_abs_acceleration": max_abs_acceleration,
                     "torque_sample_count": torque_samples,
                     "unknown_checks": unknowns, "violations": violations,
                     "trajectory_source": trajectory.source},
            evidence=["explicit JointTrajectory points", "RobotModel joint limits"],
        )
