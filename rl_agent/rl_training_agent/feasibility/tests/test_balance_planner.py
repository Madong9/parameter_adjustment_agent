"""验证通用平衡动作的六项确定性规划及物理后端分级。"""
from __future__ import annotations

from rl_training_agent.feasibility.balance.planner import BalanceMotionPlanner
from rl_training_agent.feasibility.balance.validator import IsaacGymBalanceValidator
from rl_training_agent.feasibility.motion_constraints import MotionConstraintCompiler
from rl_training_agent.feasibility.robot_models.loader import RobotModelLoader
from rl_training_agent.feasibility.simulation.schema import SimulationReport
from rl_training_agent.schemas.agent_workflow import TaskIntentSpec
from rl_training_agent.settings import load_settings


def _intent(text: str, action: str) -> TaskIntentSpec:
    """构造无需 LLM 的结构化 Go2 平衡任务。"""
    return TaskIntentSpec(
        original_instruction=text, robot="go2", action_name=action,
        normalized_goal=text,
    )


def _model():
    """从相对路径注册表加载真实 Go2 URDF descriptor。"""
    settings = load_settings()
    return RobotModelLoader(settings.training_root).load_robot_model("go2")


def test_front_leg_static_balance_produces_six_numeric_solver_outputs():
    """双前腿静态支撑应生成质心、接触、基座、足端、IK、接触力和逆动力学证据。"""
    intent = _intent("Go2两只前腿站立3秒", "front_leg_stand")
    spec = MotionConstraintCompiler.compile(intent)
    plan = BalanceMotionPlanner(dt=0.2, max_samples=21).plan(spec, intent, _model())

    assert plan.status == "READY_FOR_PHYSICS"
    assert len(plan.samples) >= 3
    assert {"contact_schedule", "com_trajectory", "base_pose_trajectory",
            "end_effector_trajectory", "floating_base_ik",
            "contact_force_optimization", "inverse_dynamics"} <= set(plan.available_solvers)
    assert set(plan.support_legs) == {"FL", "FR"}
    assert plan.metrics["contact_force_failure_count"] == 0
    assert all(sample.joint_positions for sample in plan.samples)


def test_front_leg_walking_is_rejected_when_single_support_dynamics_are_infeasible():
    """单前足阶段若接触力或基座动力学不满足约束，不能伪装成物理就绪。"""
    intent = _intent("Go2两只前腿站立前走", "front_leg_walk")
    spec = MotionConstraintCompiler.compile(intent)
    plan = BalanceMotionPlanner(dt=0.2, max_samples=21).plan(spec, intent, _model())

    assert plan.status == "INCONCLUSIVE"
    assert plan.moving is True
    assert any(item.startswith("contact_force_qp_infeasible_samples")
               for item in plan.violations)
    assert "floating_base_dynamics_residual_exceeded" in plan.violations


class _RecordingValidator:
    """记录传入的关节参考，用于验证适配逻辑而不提供真实物理证据。"""

    backend = "mock-isaacgym"

    def __init__(self):
        """初始化调用记录。"""
        self.rows = []

    def validate_targets(self, _model, _prototype, rows):
        """返回明确标记的 Mock 成功报告。"""
        self.rows = list(rows)
        return SimulationReport(
            status="PASSED", success=True, backend=self.backend,
            validated=False, validation_level="MOCK_VALIDATED",
            metrics={"policy_trained": False}, reason="mock only",
        )


def test_balance_adapter_consumes_plan_without_upgrading_mock_to_physics():
    """适配器必须消费计划，但 Mock 后端永远不能升级为真实物理通过。"""
    intent = _intent("Go2两只前腿站立3秒", "front_leg_stand")
    plan = BalanceMotionPlanner(dt=0.2, max_samples=21).plan(
        MotionConstraintCompiler.compile(intent), intent, _model())
    backend = _RecordingValidator()
    report = IsaacGymBalanceValidator(backend, max_targets=6).validate(_model(), plan)

    assert report.metrics["balance_target_count"] > 0
    assert report.backend == "mock-isaacgym"
    assert report.validated is False
    assert report.validation_level == "MOCK_VALIDATED"
    assert report.metrics["balance_plan_consumed"] is True
    assert report.metrics["policy_trained"] is False


def test_forbidden_rear_leg_text_does_not_override_front_leg_goal_metrics():
    """禁止后腿接触属于安全约束，不得把双前腿目标错误分类成后腿任务。"""
    from rl_training_agent.feasibility.capability_checker import CapabilityChecker

    settings = load_settings()
    intent = TaskIntentSpec(
        original_instruction="Go2两只前腿站立", robot="go2",
        action_name="front_leg_stand", normalized_goal="两只前腿支撑站立",
        forbidden_behaviors=["禁止后腿接触地面"],
    )
    manifest = {
        "robot": "go2", "robots": ["go2"], "command_space": [],
        "reward_variables": [
            {"name": "base_quat", "available_to_policy": True},
        ],
        "evaluation_metrics": ["front_leg_stand_duration", "fall_rate"],
    }
    report = CapabilityChecker(settings.training_root).assess(intent, manifest)
    metric_check = next(item for item in report.checks if item.name == "evaluation_metrics")

    assert "front_leg_stand_duration" in metric_check.evidence
    assert not any("rear" in item for item in metric_check.evidence)
