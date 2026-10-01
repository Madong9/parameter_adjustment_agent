"""验证通用运动约束、确定性规划注册表与全身求解门控。"""
from __future__ import annotations

import pytest

from rl_training_agent.environment.inspector import EnvironmentInspector
from rl_training_agent.feasibility.agent import FeasibilityPipeline
from rl_training_agent.feasibility.motion_constraints import (
    MotionConstraintCompiler, MotionConstraintPhase,
)
from rl_training_agent.feasibility.motion_prototype.schema import MotionType
from rl_training_agent.feasibility.planning import DeterministicMotionPlannerRegistry
from rl_training_agent.feasibility.whole_body import WholeBodyMotionSolver
from rl_training_agent.schemas.agent_workflow import TaskIntentSpec
from rl_training_agent.settings import load_settings


def _intent(text: str, action: str = "task", velocity=None) -> TaskIntentSpec:
    """构造无需模型调用的已结构化任务意图。"""
    return TaskIntentSpec(
        original_instruction=text, robot="go2", action_name=action,
        normalized_goal=text, target_velocity=velocity,
    )


def test_constraint_phase_rejects_llm_joint_or_torque_commands():
    """LLM 只能声明目标与约束，不能把关节位置藏在通用协议中。"""
    with pytest.raises(ValueError, match="cannot contain"):
        MotionConstraintPhase(
            name="bad", start=0.0, end=1.0,
            goals={"joint_positions": {"hip": 0.2}},
        )
    with pytest.raises(ValueError, match="cannot contain"):
        MotionConstraintPhase(
            name="bad", start=0.0, end=1.0,
            goals={"torque": {"hip": 12.0}},
        )


def test_locomotion_compiles_to_all_required_deterministic_planners():
    """普通行走生成步态、接触、基座和足端规划，但仍不包含 policy。"""
    intent = _intent("Go2 倒退走 0.3m/s 5秒", "backward_walk", 0.3)
    spec = MotionConstraintCompiler.compile(intent)
    planning = DeterministicMotionPlannerRegistry().plan(spec, intent)

    assert spec.motion_type == MotionType.LOCOMOTION
    assert set(spec.required_planners) == {
        "gait", "contact_schedule", "base_pose", "end_effector"}
    assert planning.status == "READY_FOR_SOLVER"
    gait = next(item for item in planning.components if item.planner == "gait")
    prototype = gait.output["dynamic_motion_prototype"]
    assert prototype["velocity"]["target_linear_velocity"]["x"] == pytest.approx(-0.3)
    assert "policy" not in prototype and "torques" not in prototype


def test_jump_declares_whole_body_requirements_without_fake_solution():
    """跳跃会列出逆动力学和接触优化缺口，不能仅凭阶段模板进入物理验证。"""
    intent = _intent("Go2 原地跳跃 2秒", "jump")
    spec = MotionConstraintCompiler.compile(intent)
    planning = DeterministicMotionPlannerRegistry().plan(spec, intent)
    solve = WholeBodyMotionSolver().assess(
        spec, planning, "pinocchio", {"limits": {"hip": {"lower": -1, "upper": 1}}},
        physics_backend="isaacgym")

    assert spec.motion_type == MotionType.JUMP
    assert [phase.name for phase in spec.phases] == [
        "PRELOAD", "TAKEOFF", "FLIGHT", "LANDING", "RECOVERY"]
    assert solve.status == "INCONCLUSIVE"
    assert {"floating_base_ik", "inverse_dynamics", "contact_force_optimization",
            "trajectory_optimization"} <= set(solve.missing_solvers)


def test_pipeline_exposes_constraints_planners_and_solver_gaps():
    """主报告始终携带三层结构，并保持 Mock/缺失求解器的保守结论。"""
    settings = load_settings()
    manifest = EnvironmentInspector(settings.training_root).inspect("go2")
    report = FeasibilityPipeline.mocked(settings.training_root).assess(
        _intent("Go2 后腿站立 3秒", "hind_leg_stand"), manifest)

    assert report.motion_constraint_spec["motion_type"] == MotionType.BALANCE.value
    assert report.planning_report["status"] == "READY_FOR_SOLVER"
    assert report.whole_body_report["status"] == "INCONCLUSIVE"
    assert "floating_base_ik" in report.whole_body_report["available_solvers"]
    assert "isaacgym_rollout" in report.whole_body_report["missing_solvers"]
    assert report.validation_level == "CAPABILITY_ONLY"
