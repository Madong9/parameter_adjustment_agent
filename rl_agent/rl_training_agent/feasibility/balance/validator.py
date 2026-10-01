"""把数值平衡计划转换为 Unitree Isaac Gym 可执行的短时关节参考。"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List

from ..ik.schema import IKResult
from ..motion_prototype.schema import MotionPhase, MotionPrototype
from ..simulation.schema import SimulationReport
from .schema import BalanceMotionPlan


class IsaacGymBalanceValidator:
    """复用现有 PPO 环境验证静态双腿支撑计划，不训练或加载策略。"""

    def __init__(self, simulation_validator: Any, max_targets: int = 8):
        """保存底层 Isaac Gym validator 并限制预检目标数量。"""
        self.simulation_validator = simulation_validator
        self.max_targets = max(2, int(max_targets))
        self.backend = str(getattr(simulation_validator, "backend", "unavailable"))

    def validate(self, robot_model: Any, plan: BalanceMotionPlan) -> SimulationReport:
        """按时间抽样关节参考，并调用同一 Unitree PPO 环境执行短时 rollout。"""
        if plan.status != "READY_FOR_PHYSICS" or not plan.samples:
            return self._unavailable("六项平衡数值规划未就绪，禁止启动 Isaac Gym rollout")
        if plan.moving:
            return self._unavailable(
                "动态双腿支撑行走尚缺少角动量/单足支撑控制，静态 validator 不会冒充动态物理验证")
        if not hasattr(self.simulation_validator, "validate_targets"):
            return self._unavailable("当前仿真 adapter 不支持平衡关节参考序列")

        selected = self._select_samples(plan)
        rollout_duration = min(float(plan.duration), 5.0)
        duration = max(0.02, rollout_duration / float(len(selected)))
        phases: List[MotionPhase] = []
        rows: List[Dict[str, Any]] = []
        base_height = float(self._value(robot_model, "base_initial_height", 0.0))
        for index, sample in enumerate(selected):
            phase_name = "balance_%03d" % index
            phases.append(MotionPhase(
                name=phase_name, duration_seconds=duration,
                body_goal={"feet": "support"},
                constraints=["follow_balance_plan", "avoid_fall", "respect_joint_limits"],
            ))
            ik_result = IKResult(
                status="SOLVED", success=True, backend=plan.backend,
                joint_positions=dict(sample.joint_positions),
                residual=sample.ik_residual,
                error=sample.ik_residual,
                base_height=base_height + float(sample.base_position[2]),
                base_orientation_xyzw=list(sample.base_orientation_xyzw),
                duration_seconds=duration,
                reason="来自浮动基座 IK、接触力优化和逆动力学已审查计划",
            )
            rows.append({
                "phase": phase_name,
                "target": {
                    "time_seconds": duration,
                    "base_height": ik_result.base_height,
                    "base_orientation_xyzw": list(sample.base_orientation_xyzw),
                    "expected_contacts": dict(sample.contacts),
                },
                "result": ik_result,
            })
        prototype = MotionPrototype(
            robot=plan.robot, action=plan.action, phases=phases,
            source="deterministic_balance_plan",
            notes=["不训练 policy；只在 Unitree PPO 环境执行平衡关节参考序列"],
        )
        try:
            validator = copy.copy(self.simulation_validator)
            if hasattr(validator, "max_seconds"):
                validator.max_seconds = rollout_duration
            if hasattr(validator, "max_steps"):
                validator.max_steps = max(
                    int(validator.max_steps), int(math.ceil(rollout_duration / 0.01)))
            report = validator.validate_targets(
                self._runtime_model(robot_model), prototype, rows)
        except Exception as exc:
            return self._unavailable("平衡 Isaac Gym adapter 异常：%s" % str(exc)[:400])
        report.metrics.update({
            "balance_plan_consumed": True,
            "balance_target_count": len(rows),
            "balance_support_legs": list(plan.support_legs),
            "balance_lifted_legs": list(plan.lifted_legs),
            "balance_numeric_metrics": dict(plan.metrics),
            "policy_trained": False,
        })
        return report

    def _select_samples(self, plan: BalanceMotionPlan) -> List[Any]:
        """保留首尾、阶段边界并均匀补点，避免只验证最终姿态。"""
        samples = list(plan.samples)
        if len(samples) <= self.max_targets:
            return samples
        indexes = {0, len(samples) - 1}
        for index in range(1, len(samples)):
            if samples[index].phase != samples[index - 1].phase:
                indexes.update((index - 1, index))
        if len(indexes) < self.max_targets:
            denominator = max(1, self.max_targets - 1)
            indexes.update(round(step * (len(samples) - 1) / denominator)
                           for step in range(self.max_targets))
        ordered = sorted(indexes)
        if len(ordered) > self.max_targets:
            positions = [round(step * (len(ordered) - 1) / (self.max_targets - 1))
                         for step in range(self.max_targets)]
            ordered = [ordered[position] for position in positions]
        return [samples[index] for index in ordered]

    @staticmethod
    def _value(robot_model: Any, key: str, default: Any = None) -> Any:
        """兼容 RobotModel 与 runtime 字典读取。"""
        if isinstance(robot_model, dict):
            return robot_model.get(key, default)
        return getattr(robot_model, key, default)

    @staticmethod
    def _runtime_model(robot_model: Any) -> Any:
        """把 RobotModel 转换为底层 validator 使用的可序列化字典。"""
        return robot_model.runtime_dict() if hasattr(robot_model, "runtime_dict") else robot_model

    def _unavailable(self, reason: str) -> SimulationReport:
        """返回保守未运行结论，不把 adapter 缺口记为物理失败。"""
        return SimulationReport(
            status="UNAVAILABLE", success=None, backend=self.backend,
            validated=False, validation_level="CAPABILITY_ONLY", reason=reason,
        )
