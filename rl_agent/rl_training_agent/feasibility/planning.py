"""按运动约束分派确定性规划器，并保留每个组件的证据边界。"""
from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from ..schemas.agent_workflow import TaskIntentSpec
from .motion_constraints import MotionConstraintSpec
from .balance.planner import BalanceMotionPlanner
from .motion_prototype.dynamic_generator import DynamicMotionPrototypeGenerator
from .motion_prototype.schema import MotionType


class PlannerComponentReport(BaseModel):
    """记录单个确定性规划器的输出，不把生成成功当作物理成功。"""

    planner: str
    status: str
    backend: str
    output: Dict[str, Any] = Field(default_factory=dict)
    limitations: List[str] = Field(default_factory=list)


class MotionPlanningReport(BaseModel):
    """聚合规划器输出，并指出进入全身求解前仍缺少的组件。"""

    status: str
    motion_type: MotionType
    components: List[PlannerComponentReport] = Field(default_factory=list)
    missing_planners: List[str] = Field(default_factory=list)
    solver_requirements: List[str] = Field(default_factory=list)
    available_solvers: List[str] = Field(default_factory=list)
    balance_plan: Dict[str, Any] = Field(default_factory=dict)
    limitations: List[str] = Field(default_factory=list)


class DeterministicMotionPlannerRegistry:
    """为每类目标调用受限本地规划器；未知规划器不会由 LLM 临时创造。"""

    SUPPORTED = {"gait", "com", "contact_schedule", "base_pose", "end_effector"}

    def plan(self, spec: MotionConstraintSpec, intent: TaskIntentSpec,
             robot_model: Any = None) -> MotionPlanningReport:
        """执行所有声明规划器，并以最保守组件确定规划就绪状态。"""
        components: List[PlannerComponentReport] = []
        missing: List[str] = []
        balance_plan = None
        if spec.motion_type == MotionType.BALANCE and robot_model is not None:
            balance_plan = BalanceMotionPlanner().plan(spec, intent, robot_model)
        for name in spec.required_planners:
            handler = getattr(self, "_plan_" + name, None)
            if name not in self.SUPPORTED or handler is None:
                missing.append(name)
                continue
            component = (self._balance_component(name, balance_plan)
                         if balance_plan is not None and name in
                         {"com", "contact_schedule", "base_pose", "end_effector"}
                         else handler(spec, intent))
            components.append(component)
            if component.status == "UNAVAILABLE":
                missing.append(name)

        limitations = [item for component in components for item in component.limitations]
        status = ("INCONCLUSIVE" if missing or any(
            component.status in ("INCONCLUSIVE", "UNAVAILABLE") for component in components)
                  else "READY_FOR_SOLVER")
        return MotionPlanningReport(
            status=status, motion_type=spec.motion_type, components=components,
            missing_planners=sorted(set(missing)),
            solver_requirements=self._solver_requirements(spec.motion_type),
            available_solvers=(list(balance_plan.available_solvers) if balance_plan else []),
            balance_plan=(balance_plan.dict() if balance_plan else {}),
            limitations=list(dict.fromkeys(limitations)),
        )


    @staticmethod
    def _balance_component(name: str, plan: Any) -> PlannerComponentReport:
        """从一次共享的数值平衡规划中提取各组件报告，避免重复求解。"""
        ready = plan.status == "READY_FOR_PHYSICS"
        status = "GENERATED" if ready else "INCONCLUSIVE"
        sample_count = len(plan.samples)
        outputs = {
            "com": {"sample_count": sample_count,
                    "max_base_dynamics_residual": plan.metrics.get(
                        "max_base_dynamics_residual")},
            "contact_schedule": {"sample_count": sample_count,
                                 "support_legs": list(plan.support_legs),
                                 "moving": bool(plan.moving)},
            "base_pose": {"sample_count": sample_count,
                          "max_ik_residual": plan.metrics.get("max_ik_residual")},
            "end_effector": {"sample_count": sample_count,
                             "lifted_legs": list(plan.lifted_legs)},
        }
        return PlannerComponentReport(
            planner=name, status=status, backend=plan.backend,
            output=outputs[name],
            limitations=[] if ready else [plan.reason] + list(plan.limitations),
        )
    @staticmethod
    def _plan_gait(spec: MotionConstraintSpec,
                   intent: TaskIntentSpec) -> PlannerComponentReport:
        """为普通位移生成参数化四足步态；转向保持未实现而不复用直线步态。"""
        if spec.motion_type != MotionType.LOCOMOTION:
            return PlannerComponentReport(
                planner="gait", status="UNAVAILABLE",
                backend="deterministic_gait_registry",
                limitations=["当前只有直线 LOCOMOTION 步态规划器；转向需要旋转足端轨迹"],
            )
        prototype = DynamicMotionPrototypeGenerator.generate(intent)
        return PlannerComponentReport(
            planner="gait", status="GENERATED",
            backend="parameterized_quadruped_gait",
            output={"dynamic_motion_prototype": prototype.dict()},
            limitations=["参数化步态是开环规划结果，不是 policy"],
        )

    @staticmethod
    def _plan_com(spec: MotionConstraintSpec,
                  _intent: TaskIntentSpec) -> PlannerComponentReport:
        """生成质心约束目标；不伪造优化后的数值质心轨迹。"""
        targets = [{"phase": phase.name,
                    "target": phase.goals.get("center_of_mass", "maintain_nominal_com")}
                   for phase in spec.phases]
        status = ("GENERATED" if spec.motion_type in
                  (MotionType.STATIC_POSE, MotionType.LOCOMOTION) else "INCONCLUSIVE")
        return PlannerComponentReport(
            planner="com", status=status, backend="constraint_template",
            output={"targets": targets},
            limitations=[] if status == "GENERATED" else
            ["已生成质心约束，但尚未运行浮动基座质心轨迹优化"],
        )

    @staticmethod
    def _plan_contact_schedule(spec: MotionConstraintSpec,
                               _intent: TaskIntentSpec) -> PlannerComponentReport:
        """从阶段和步态语义构造接触时序，保留候选接触的不确定性。"""
        schedule = [{"phase": phase.name, "start": phase.start, "end": phase.end,
                     "active_contacts": list(phase.active_contacts),
                     "forbidden_contacts": list(phase.forbidden_contacts),
                     "optional_contacts": list(phase.optional_contacts),
                     "transition_conditions": list(phase.transition_conditions)}
                    for phase in spec.phases]
        unresolved = any(any("candidate" in contact or "scheduled_by_turning" in contact
                             for contact in phase.active_contacts)
                         for phase in spec.phases)
        return PlannerComponentReport(
            planner="contact_schedule",
            status="INCONCLUSIVE" if unresolved else "GENERATED",
            backend="deterministic_phase_schedule",
            output={"schedule": schedule},
            limitations=["候选/转向接触仍需专用规划器解析"] if unresolved else [],
        )

    @staticmethod
    def _plan_base_pose(spec: MotionConstraintSpec,
                        _intent: TaskIntentSpec) -> PlannerComponentReport:
        """提取基座姿态、速度与角动量目标，不生成浮动基座控制。"""
        targets = [{"phase": phase.name,
                    "base_pose": phase.goals.get("base_pose"),
                    "base_orientation": phase.goals.get("base_orientation"),
                    "base_velocity": phase.goals.get("base_velocity"),
                    "base_angular_velocity": phase.goals.get("base_angular_velocity"),
                    "angular_momentum_required": phase.goals.get(
                        "angular_momentum_required", False)}
                   for phase in spec.phases]
        hard = spec.motion_type in (
            MotionType.BALANCE, MotionType.JUMP, MotionType.ACROBATIC, MotionType.TURNING)
        return PlannerComponentReport(
            planner="base_pose", status="INCONCLUSIVE" if hard else "GENERATED",
            backend="semantic_base_target_compiler",
            output={"targets": targets},
            limitations=["需要浮动基座轨迹优化"] if hard else [],
        )

    @staticmethod
    def _plan_end_effector(spec: MotionConstraintSpec,
                           _intent: TaskIntentSpec) -> PlannerComponentReport:
        """汇总足端或末端目标；缺少机器人末端模型时保持不确定。"""
        if spec.motion_type == MotionType.LOCOMOTION:
            return PlannerComponentReport(
                planner="end_effector", status="GENERATED",
                backend="foot_trajectory_generator",
                output={"mode": "quadruped_foot_trajectory"},
                limitations=["数值足端采样在 IsaacGymDynamicValidator 中逐步完成"],
            )
        targets = [{"phase": phase.name,
                    "end_effector": phase.goals.get("end_effector"),
                    "contacts": list(phase.active_contacts)}
                   for phase in spec.phases]
        hard = spec.motion_type in (
            MotionType.MANIPULATION, MotionType.JUMP, MotionType.ACROBATIC,
            MotionType.BALANCE, MotionType.TURNING)
        return PlannerComponentReport(
            planner="end_effector", status="INCONCLUSIVE" if hard else "GENERATED",
            backend="cartesian_constraint_compiler",
            output={"targets": targets},
            limitations=["需要专用末端/足端轨迹优化与碰撞检查"] if hard else [],
        )

    @staticmethod
    def _solver_requirements(motion_type: MotionType) -> List[str]:
        """返回各动作进入可信物理验证前必须具备的全身求解能力。"""
        common = ["pinocchio_ik", "joint_limit_check", "isaacgym_rollout"]
        requirements = {
            MotionType.STATIC_POSE: common + ["static_support_check"],
            MotionType.LOCOMOTION: common + ["trajectory_check"],
            MotionType.TURNING: common + ["floating_base_ik", "trajectory_optimization"],
            MotionType.BALANCE: common + [
                "floating_base_ik", "contact_force_optimization", "inverse_dynamics"],
            MotionType.JUMP: common + [
                "floating_base_ik", "inverse_dynamics", "contact_force_optimization",
                "trajectory_optimization"],
            MotionType.ACROBATIC: common + [
                "floating_base_ik", "centroidal_dynamics", "inverse_dynamics",
                "contact_force_optimization", "trajectory_optimization"],
            MotionType.MANIPULATION: common + [
                "end_effector_ik", "collision_check", "trajectory_optimization"],
            MotionType.UNKNOWN: ["clarified_motion_constraints"],
        }
        return requirements[motion_type]
