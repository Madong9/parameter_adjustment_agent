"""覆盖真实资产能力检查和显式 Mock 的 Task Feasibility Pipeline。"""
from pathlib import Path

from rl_training_agent.environment.inspector import EnvironmentInspector
from rl_training_agent.feasibility.agent import FeasibilityPipeline
from rl_training_agent.feasibility.capability_checker import CapabilityChecker
from rl_training_agent.feasibility.ik.mock import MockIKSolver
from rl_training_agent.feasibility.motion_prototype.generator import MotionPrototypeGenerator
from rl_training_agent.feasibility.motion_prototype.schema import (
    MotionPhase, MotionPrototype, RobotMotionTarget, TargetPose,
)
from rl_training_agent.feasibility.schema import FeasibilityStatus
from rl_training_agent.feasibility.simulation.mujoco_validator import MockMuJoCoValidator
from rl_training_agent.schemas.agent_workflow import TaskIntentSpec
from rl_training_agent.settings import load_settings
from rl_training_agent.orchestration.state_machine import AgentState, PersistentStateMachine


def _manifest():
    """读取当前仓库真实 Go2 能力清单作为能力检查证据。"""
    return EnvironmentInspector(load_settings().training_root).inspect("go2")


def _intent(text, action="forward_locomotion"):
    """构造任务理解输出，不依赖外部 LLM。"""
    return TaskIntentSpec(original_instruction=text, robot="go2", action_name=action,
                          normalized_goal=text, required_behaviors=["保持平衡"])


def test_go2_backward_walk_one_meter_per_second_is_capability_supported():
    """确认 Go2 的现有命令空间、关节和指标支持倒退 1m/s 的尝试。"""
    assessment = CapabilityChecker(load_settings().training_root).assess(
        _intent("Go2 倒退走 1m/s", "backward_locomotion"), _manifest())
    assert assessment.status == FeasibilityStatus.SUPPORTED
    assert any(item.name == "locomotion_command" and item.status == FeasibilityStatus.SUPPORTED
               for item in assessment.checks)


def test_go2_arm_grasp_is_unsupported_by_configured_actuators():
    """确认缺少机械臂/夹爪控制关节时拒绝 Go2 抓取任务。"""
    assessment = CapabilityChecker(load_settings().training_root).assess(
        _intent("Go2 机械臂抓取物体", "arm_grasp"), _manifest())
    assert assessment.status == FeasibilityStatus.UNSUPPORTED
    assert any("夹爪" in item or "机械臂" in item for item in assessment.missing_requirements)


def test_standing_task_generates_semantic_motion_phases_without_joint_angles():
    """确认站立任务能生成语义阶段而不输出关节角。"""
    prototype = MotionPrototypeGenerator.deterministic(_intent("Go2 保持站立 5 秒", "stable_stand"))
    assert [item.name for item in prototype.phases] == ["stand_up", "balance_hold"]
    assert sum(item.duration_seconds for item in prototype.phases) == 5.0
    assert all(not hasattr(item, "joint_positions") for item in prototype.phases)


def test_mock_ik_solver_returns_explicit_mock_success():
    """确认 Mock IK 接口可被测试替换且明确标注 mock 后端。"""
    result = MockIKSolver().solve_ik("missing.urdf", {"frame": "foot", "position": [0, 0, 0.1]})
    assert result.success is True and result.backend == "mock"


def test_mock_mujoco_validator_returns_explicit_mock_success():
    """确认 Mock MuJoCo 接口输出结构化短时 rollout 指标。"""
    intent = _intent("Go2 保持站立", "stable_stand")
    prototype = MotionPrototypeGenerator.deterministic(intent)
    ik = MockIKSolver().solve_ik({}, {"frame": "foot", "position": [0, 0, 0.1]})
    report = MockMuJoCoValidator().validate({}, prototype, ik)
    assert report.success is True and report.backend == "mock"
    assert "min_height" in report.metrics


def test_complete_pipeline_runs_with_injected_mock_ik_and_mujoco():
    """确认能力、原型、Mock IK 和 Mock 仿真可以串成完整流水线。"""
    def prototype_provider(intent):
        """提供明确标记为 mock 的末端目标，不依赖 LLM 或物理假设。"""
        return MotionPrototype(
            robot=intent.robot, action=intent.action_name, source="mock",
            phases=[MotionPhase(name="balance_hold", duration_seconds=1.0,
                                body_goal={"torso": "upright", "feet": "support"},
                                robot_targets=[RobotMotionTarget(
                                    time_seconds=1.0,
                                    feet={frame: [0.0, 0.0, -0.3]
                                          for frame in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")},
                                    base_height=0.42,
                                )])],
        )

    pipeline = FeasibilityPipeline(
        load_settings().training_root, ik_solver=MockIKSolver(),
        simulation_validator=MockMuJoCoValidator(), prototype_provider=prototype_provider)
    report = pipeline.assess(_intent("Go2 保持站立", "stable_stand"), _manifest())
    assert report.status == FeasibilityStatus.SUPPORTED
    assert report.evidence_mode == "MOCK"
    assert report.ik_result["backend"] == "mock"
    assert len(report.ik_result["joint_positions"]) == 1
    assert len(report.ik_result["target_results"]) == 1
    assert report.simulation_result["backend"] == "mock"
    assert report.simulation_result["metrics"]["target_count"] == 1


def test_production_adapters_fail_open_as_conditional_when_assets_or_dependencies_missing():
    """确认真实依赖不可用时报告条件状态，而非伪造 Mock 通过。"""
    root = load_settings().training_root
    pipeline = FeasibilityPipeline(root, prototype_provider=lambda intent: {
        "robot": intent.robot, "action": "stand", "phases": [{
            "name": "balance", "duration_seconds": 1.0,
            "body_goal": {"torso": "upright"}, "target_poses": [{
                "frame": "FL_foot", "position": [0.0, 0.0, 0.1]}]}],
    })
    report = pipeline.assess(_intent("Go2 保持站立", "stable_stand"), _manifest())
    assert report.evidence_mode == "REAL"
    assert report.status == FeasibilityStatus.CONDITIONAL
    assert report.overall_status == report.status
    assert report.simulation_result["status"] in ("UNAVAILABLE", "SKIPPED", "FAILED")


def test_unsupported_pipeline_short_circuits_before_motion_or_ik():
    """确认硬件不支持的动作在调用动作阶段 Provider 前即停止。"""
    called = []
    pipeline = FeasibilityPipeline(
        load_settings().training_root, ik_solver=MockIKSolver(),
        simulation_validator=MockMuJoCoValidator(),
        prototype_provider=lambda intent: called.append(intent))
    report = pipeline.assess(_intent("Go2 机械臂抓取物体", "arm_grasp"), _manifest())
    assert report.status == FeasibilityStatus.UNSUPPORTED
    assert not called and report.motion_prototype is None


def test_feasibility_state_routes_supported_conditional_and_unsupported(tmp_path):
    """验证状态机支持通过、人工复核和失败三种可行性分支及幂等键。"""
    def enter_check(filename):
        """把新状态机推进至动作可行性检查状态。"""
        machine = PersistentStateMachine(tmp_path / filename)
        machine.transition(AgentState.ENVIRONMENT_INSPECTED)
        machine.transition(AgentState.TASK_UNDERSTANDING)
        assert machine.transition(AgentState.TASK_FEASIBILITY_CHECK,
                                  operation_id="feasibility-check-run-1")
        assert not machine.transition(AgentState.TASK_FEASIBILITY_CHECK,
                                      operation_id="feasibility-check-run-1")
        return machine

    supported = enter_check("supported.json")
    supported.transition(AgentState.RAG_RETRIEVING)
    assert supported.record.state == AgentState.RAG_RETRIEVING
    conditional = enter_check("conditional.json")
    conditional.transition(AgentState.HUMAN_REVIEW)
    assert conditional.record.state == AgentState.HUMAN_REVIEW
    unsupported = enter_check("unsupported.json")
    unsupported.transition(AgentState.FAILED)
    assert unsupported.record.state == AgentState.FAILED
