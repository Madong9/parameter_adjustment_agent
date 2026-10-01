"""聚合全身运动求解能力，严格区分可用后端与尚未实现的求解器。"""
from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from .motion_constraints import MotionConstraintSpec
from .motion_prototype.schema import MotionType
from .planning import MotionPlanningReport


class WholeBodySolveReport(BaseModel):
    """记录全身求解阶段的后端覆盖度；它本身不是物理通过结论。"""

    status: str
    backend: str = "whole_body_solver_registry"
    available_solvers: List[str] = Field(default_factory=list)
    missing_solvers: List[str] = Field(default_factory=list)
    solver_plan: List[Dict[str, Any]] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class WholeBodyMotionSolver:
    """根据动作需求组合真实可用求解器，缺失能力时保守停止。"""

    def assess(self, spec: MotionConstraintSpec, planning: MotionPlanningReport,
               ik_backend: str, robot_model: Any,
               physics_backend: str = "unavailable") -> WholeBodySolveReport:
        """生成求解执行计划，不用固定基座 IK 冒充全身动力学。"""
        available = {"joint_limit_check", "trajectory_check"}
        available.update(planning.available_solvers)
        if str(physics_backend).lower() == "isaacgym":
            available.add("isaacgym_rollout")
        if str(ik_backend).lower() == "pinocchio":
            available.add("pinocchio_ik")
        if getattr(robot_model, "limits", None) or (
                isinstance(robot_model, dict) and robot_model.get("limits")):
            available.add("static_support_check")

        required = set(planning.solver_requirements)
        missing = sorted(required - available)
        plan = []
        if "pinocchio_ik" in required:
            plan.append({
                "stage": "kinematics", "backend": str(ik_backend),
                "mode": ("fixed_base_multi_frame_ik"
                         if spec.motion_type == MotionType.LOCOMOTION else "task_specific_ik"),
            })
        if any(name in required for name in (
                "floating_base_ik", "inverse_dynamics", "centroidal_dynamics",
                "contact_force_optimization", "trajectory_optimization")):
            plan.append({
                "stage": "whole_body_dynamics",
                "backend": (planning.balance_plan.get("backend", "not_available")
                            if spec.motion_type == MotionType.BALANCE else "not_available"),
                "required": sorted(required & {
                    "floating_base_ik", "inverse_dynamics", "centroidal_dynamics",
                    "contact_force_optimization", "trajectory_optimization"}),
            })
        if "isaacgym_rollout" in required:
            plan.append({
                "stage": "physics_validation", "backend": "isaacgym",
                "mode": "execute_only_after_all_required_solver_outputs_exist",
            })

        planner_ready = planning.status == "READY_FOR_SOLVER"
        # 直线 locomotion、普通静态姿态和数值平衡计划可进入对应物理执行器；
        # 其他动作必须等完整专用求解器，不能因注册表检查通过而形成物理结论。
        executable_now = (
            spec.motion_type in (MotionType.LOCOMOTION, MotionType.STATIC_POSE, MotionType.BALANCE)
            and planner_ready and not missing
        )
        status = "READY_FOR_PHYSICS" if executable_now else "INCONCLUSIVE"
        limitations = list(planning.limitations)
        if missing:
            limitations.append("缺少全身求解后端：" + ", ".join(missing))
        if spec.motion_type not in (MotionType.LOCOMOTION, MotionType.STATIC_POSE, MotionType.BALANCE):
            limitations.append("当前动作没有专用的完整全身求解与 Isaac Gym 执行器")
        return WholeBodySolveReport(
            status=status,
            available_solvers=sorted(available),
            missing_solvers=missing,
            solver_plan=plan,
            limitations=list(dict.fromkeys(limitations)),
        )
