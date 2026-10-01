"""验证通用动作准入、候选证据隔离、真实环境门控与预算恢复。"""
import json

import pytest

from rl_training_agent.feasibility.admission import TrainingAdmissionPolicy
from rl_training_agent.feasibility.schema import CompleteFeasibilityReport
from rl_training_agent.settings import Settings


def probe_report(**overrides):
    """构造已运行但跟踪失败的候选证据，仅用于本地测试。"""
    capability = {"status": "CAPABILITY_SUPPORTED", "missing_requirements": [],
                  "checks": [{"name": name, "status": "CAPABILITY_SUPPORTED"}
                             for name in ("robot_model", "joints_and_actuators",
                                          "orientation_observation", "evaluation_metrics")]}
    data = dict(
        status="PHYSICS_FAILED", task="Go2前腿站立", motion_type="BALANCE",
        backend="isaacgym", evidence_mode="REAL", validation_level="PHYSICS_FAILED",
        capability_report=capability, motion_constraint_spec={"robot": "go2"},
        robot_model={"robot_name": "go2", "model_status": "AVAILABLE"},
        simulation_report={"backend": "isaacgym", "status": "FAILED", "validated": True,
                           "success": False, "duration": 0.38,
                           "violations": ["actuator_settled_tracking_error_exceeded"]})
    data.update(overrides)
    return CompleteFeasibilityReport(**data)


def test_tracking_failure_allows_capped_exploration_without_changing_evidence():
    """参考跟踪失败允许探索，但物理等级、失败指标和总预算必须保持准确。"""
    report = probe_report()
    original = report.json()
    admission = TrainingAdmissionPolicy().assess(report, Settings())
    assert admission.decision == "ALLOW_BUDGETED_EXPLORATION"
    assert admission.probe_status == "REFERENCE_TRACKING_FAILED"
    assert admission.max_iterations == 3000
    assert admission.max_revisions == 1
    assert report.json() == original


def test_strict_mode_keeps_failed_probe_blocked():
    """旧严格模式仍要求目标物理探针通过。"""
    admission = TrainingAdmissionPolicy().assess(
        probe_report(), Settings(feasibility_admission_mode="strict"))
    assert admission.decision == "NEEDS_REVIEW"


@pytest.mark.parametrize("mode", ["MOCK", "MIXED"])
def test_mock_or_mixed_cannot_enter_real_training(mode):
    """Mock 和混合结果不能借用真实后端名称获得生产准入。"""
    result = TrainingAdmissionPolicy().assess(probe_report(evidence_mode=mode), Settings())
    assert not result.allowed


@pytest.mark.parametrize("missing", ["evaluation_metrics", "joints_and_actuators", "robot_model"])
def test_missing_required_capability_check_blocks_exploration(missing):
    """缺少指标、机器人或执行器证据时禁止探索。"""
    report = probe_report()
    report.capability_report["checks"] = [
        check for check in report.capability_report["checks"] if check["name"] != missing]
    assert not TrainingAdmissionPolicy().assess(report, Settings()).allowed


def test_unsupported_capability_is_rejected_not_reclassified_as_candidate_failure():
    """确定硬件缺失应拒绝，不能用候选跟踪失败放行。"""
    report = probe_report(capability_report={"status": "UNSUPPORTED"})
    assert TrainingAdmissionPolicy().assess(report, Settings()).decision == "REJECT"


def test_unknown_optimization_needs_independent_real_environment_evidence():
    """优化失败需额外真实环境健康证据；默认站姿通过不升级目标物理等级。"""
    report = probe_report(status="CONDITIONAL", validation_level="CAPABILITY_ONLY",
                          simulation_report={})
    policy = TrainingAdmissionPolicy()
    assert not policy.assess(report, Settings()).allowed
    health = {"backend": "isaacgym", "validated": True, "success": True, "duration": 1.0}
    decision = policy.assess(report, Settings(), environment_health=health)
    assert decision.decision == "ALLOW_BUDGETED_EXPLORATION"
    assert decision.probe_status == "OPTIMIZATION_INCONCLUSIVE"
    assert report.validation_level == "CAPABILITY_ONLY"
    health["backend"] = "mock-isaacgym"
    assert not policy.assess(report, Settings(), environment_health=health).allowed


def test_non_finite_rollout_requires_independent_environment_health():
    """NaN/Inf 类目标 rollout 不能自身证明环境健康，健康基线通过后才允许探索。"""
    report = probe_report()
    report.simulation_report["violations"] = ["non_finite_joint_state", "non_finite_torque"]
    policy = TrainingAdmissionPolicy()
    blocked = policy.assess(report, Settings())
    assert blocked.decision == "NEEDS_REVIEW"
    assert blocked.probe_status == "RUNTIME_INVALID"
    health = {"backend": "isaacgym", "validated": True, "success": True, "duration": 1.0}
    allowed = policy.assess(report, Settings(), environment_health=health)
    assert allowed.decision == "ALLOW_BUDGETED_EXPLORATION"
    assert report.validation_level == "PHYSICS_FAILED"


def test_exploration_budget_reserves_equal_training_share_for_revision():
    """首轮多种子训练不得耗尽声明给一次奖励修订的探索额度。"""
    from rl_training_agent.orchestration.budget import BudgetTracker
    budget = BudgetTracker(max_iterations=3000, used_iterations=900,
                           max_revisions=1, reserve_for_revisions=True)
    first_per_seed = budget.per_seed_allocation(1500, 3)
    assert first_per_seed == 350
    budget.consume_iterations(first_per_seed * 3)
    budget.consume_revision()
    revision_per_seed = budget.per_seed_allocation(1000, 3)
    assert revision_per_seed == 350


def test_real_physics_pass_keeps_normal_budget():
    """物理通过使用原有训练预算，探索上限不影响它。"""
    report = probe_report(status="PHYSICS_VALIDATED", validation_level="STATIC_PHYSICS_VALIDATED",
                          simulation_report={"backend": "isaacgym", "validated": True,
                                             "success": True, "duration": 1.0})
    admission = TrainingAdmissionPolicy().assess(report, Settings())
    assert admission.decision == "ALLOW_TRAINING"
    assert admission.max_iterations == 12000


def test_budgeted_exploration_reaches_training_after_compilation(tmp_path, monkeypatch):
    """通过真实训练编排分支验证第二道门和预算；外部推理、GPU 训练均由测试替身替代。"""
    from rl_training_agent.feasibility.agent import FeasibilityPipeline
    from rl_training_agent.orchestration.orchestrator import TrainingOrchestrator
    from rl_training_agent.providers.mock_provider import MockLLMReasoningProvider

    monkeypatch.setattr(FeasibilityPipeline, "assess", lambda *args: probe_report(
        motion_type="LOCOMOTION", task="向前走"))
    settings = Settings(experiment_root=str(tmp_path / "experiments"),
                        artifact_root=str(tmp_path / "artifacts"),
                        rag_enabled=False, memory_enabled=False,
                        exploration_max_iterations=80, exploration_max_revisions=0,
                        smoke_iterations=2, screening_iterations=5)
    orchestrator = TrainingOrchestrator(settings, MockLLMReasoningProvider(1))
    seen = []

    def record_training(task, task_dir, state, candidates, budget):
        """捕获真正训练调用之前的预算和报告，不启动 PPO。"""
        assert budget.max_iterations == 80 and budget.max_revisions == 0
        assert task.training_budget.max_total_iterations == 80
        readiness = json.loads((task_dir / "training_readiness.json").read_text())
        assert readiness["ready"] is True
        assert readiness["physics_validated"] is False
        assert readiness["validation_level"] == "PHYSICS_FAILED"
        context = json.loads((task_dir / "context_snapshot.json").read_text())
        assert context["training_admission"]["decision"] == "ALLOW_BUDGETED_EXPLORATION"
        seen.append(task_dir)
        return {}

    monkeypatch.setattr(orchestrator, "_real_train", record_training)
    monkeypatch.setattr(orchestrator, "_evaluate", lambda *args: {"training_invoked": True})
    result = orchestrator.train("机器狗稳定向前行走", "go2", dry_run=False)
    assert result["training_invoked"] and len(seen) == 1
    saved = json.loads((seen[0] / "feasibility_report.json").read_text())
    assert saved["status"] == "PHYSICS_FAILED"


def test_budgeted_exploration_blocks_budget_too_small_for_reserved_revision(tmp_path, monkeypatch):
    """就绪门必须覆盖候选筛选、首轮种子和已声明修订轮次的最小额度。"""
    from rl_training_agent.feasibility.agent import FeasibilityPipeline
    from rl_training_agent.orchestration.orchestrator import TrainingOrchestrator
    from rl_training_agent.providers.mock_provider import MockLLMReasoningProvider

    monkeypatch.setattr(FeasibilityPipeline, "assess", lambda *args: probe_report(
        motion_type="LOCOMOTION", task="向前走"))
    settings = Settings(experiment_root=str(tmp_path / "experiments"),
                        artifact_root=str(tmp_path / "artifacts"),
                        rag_enabled=False, memory_enabled=False,
                        exploration_max_iterations=6, exploration_max_revisions=1,
                        smoke_iterations=1, screening_iterations=2)
    orchestrator = TrainingOrchestrator(settings, MockLLMReasoningProvider(1))
    monkeypatch.setattr(orchestrator, "_real_train", lambda *args: pytest.fail(
        "预算不足时不应启动真实训练"))
    result = orchestrator.train("机器狗稳定向前行走", "go2", dry_run=False)
    assert result["state"] == "HUMAN_REVIEW"
    readiness = json.loads((settings.experiments_path / result["task_id"] /
                            result["training_readiness"]).read_text())
    assert readiness["minimum_training_iterations"] == 8
    assert readiness["ready"] is False


def test_resume_budget_cannot_expand_persisted_exploration_limit(tmp_path):
    """恢复时不能通过改大配置或摘要剩余额度突破原准入预算。"""
    from rl_training_agent.orchestration.orchestrator import TrainingOrchestrator
    from rl_training_agent.orchestration.budget import BudgetTracker
    from rl_training_agent.orchestration.state_machine import PersistentStateMachine
    from rl_training_agent.providers.mock_provider import MockLLMReasoningProvider
    from rl_training_agent.schemas.task import TaskSpec
    from rl_training_agent.utils.io import write_json

    settings = Settings(exploration_max_iterations=100, exploration_max_revisions=1)
    provider = MockLLMReasoningProvider(1)
    orchestrator = TrainingOrchestrator(settings, provider)
    task = TaskSpec.parse_obj(provider.design_task_and_rewards("向前行走", "go2", {})["task_spec"])
    state = PersistentStateMachine(tmp_path / "state.json")
    state.start_new_run("test-run")
    admission = TrainingAdmissionPolicy().assess(probe_report(), settings)
    admission.run_id = "test-run"
    write_json(tmp_path / "training_admission.json", admission)
    write_json(tmp_path / "acceptance_contract.json", {
        "run_id": "test-run", "robot": task.robot,
        "instruction": task.original_instruction,
        "task_identity": orchestrator._task_acceptance_identity(task),
        "success_metrics": [item.dict() for item in task.success_metrics],
        "safety_constraints": [item.dict() for item in task.safety_constraints],
    })
    budget = BudgetTracker(max_iterations=12000, used_iterations=95, max_revisions=9)
    orchestrator._apply_persisted_admission_budget(tmp_path, task, state, budget, False)
    assert budget.max_iterations == 100 and budget.used_iterations == 95
    assert budget.max_revisions == 1
    with pytest.raises(RuntimeError, match="budget exhausted"):
        budget.consume_iterations(6)
    admission.run_id = "old-run"
    write_json(tmp_path / "training_admission.json", admission)
    with pytest.raises(RuntimeError, match="run_id"):
        orchestrator._apply_persisted_admission_budget(tmp_path, task, state, budget, False)


@pytest.mark.parametrize("failure", ["crash", "timeout", "invalid_json"])
def test_health_worker_fault_never_becomes_runtime_evidence(monkeypatch, failure):
    """原生崩溃、超时和无效报告均为基础设施未知，不得放行探索。"""
    import subprocess
    from types import SimpleNamespace
    from rl_training_agent.feasibility.admission import check_environment_health

    def failed_worker(*args, **kwargs):
        """模拟隔离进程失败，禁止测试实际启动 GPU。"""
        if failure == "timeout":
            raise subprocess.TimeoutExpired("worker", 60)
        return SimpleNamespace(returncode=-11 if failure == "crash" else 0,
                               stdout="__DYNAMIC_FEASIBILITY_REPORT__invalid")

    monkeypatch.setattr(subprocess, "run", failed_worker)
    health = check_environment_health("../unitree_rl_gym", "go2")
    assert health["validated"] is False and health["success"] is None
    report = probe_report(simulation_report={})
    assert not TrainingAdmissionPolicy().assess(report, Settings(), environment_health=health).allowed


def test_balance_constraints_separate_prepare_and_goal_contacts():
    """准备阶段允许四足着地，目标阶段禁止非支撑足接触，避免全局条件自相矛盾。"""
    from rl_training_agent.feasibility.motion_constraints import MotionConstraintCompiler
    from rl_training_agent.schemas.agent_workflow import TaskIntentSpec
    intent = TaskIntentSpec(original_instruction="Go2两只前腿站立3秒", robot="go2",
                            action_name="front_leg_stand", normalized_goal="前腿站立",
                            forbidden_behaviors=["保持阶段禁止后腿触地"])
    spec = MotionConstraintCompiler.compile(intent)
    assert set(spec.phases[0].active_contacts) == {"FL", "FR", "RL", "RR"}
    assert set(spec.phases[-1].active_contacts) == {"FL", "FR"}
    assert set(spec.phases[-1].forbidden_contacts) == {"RL", "RR"}
    assert spec.phases[2].transition_conditions
    assert intent.forbidden_behaviors[0] in spec.task_requirements
    assert intent.forbidden_behaviors[0] not in spec.global_constraints


def test_nested_control_injection_and_conflicting_contacts_are_rejected():
    """通用协议校验应递归检查指令，并拒绝同一足端既必需又禁止接触。"""
    from rl_training_agent.feasibility.motion_constraints import MotionConstraintPhase
    with pytest.raises(ValueError, match="cannot contain"):
        MotionConstraintPhase(name="bad", start=0, end=1,
                              goals={"nested": [{"torques": [1, 2]}]})
    with pytest.raises(ValueError, match="contact requirements conflict"):
        MotionConstraintPhase(name="bad", start=0, end=1,
                              active_contacts=["FL"], forbidden_contacts=["FL"])


def test_acceptance_contract_rejects_changed_threshold_on_resume(tmp_path):
    """调整验收阈值必须建立新运行，不能借用原准入恢复训练。"""
    from rl_training_agent.orchestration.orchestrator import TrainingOrchestrator
    from rl_training_agent.orchestration.state_machine import PersistentStateMachine
    from rl_training_agent.providers.mock_provider import MockLLMReasoningProvider
    from rl_training_agent.schemas.task import TaskSpec
    from rl_training_agent.utils.io import write_json
    task = TaskSpec.parse_obj(MockLLMReasoningProvider(1).design_task_and_rewards(
        "向前走", "go2", {})["task_spec"])
    state = PersistentStateMachine(tmp_path / "state.json")
    state.start_new_run("contract-run")
    write_json(tmp_path / "acceptance_contract.json", {
        "run_id": "contract-run", "robot": task.robot,
        "instruction": task.original_instruction,
        "task_identity": TrainingOrchestrator._task_acceptance_identity(task),
        "success_metrics": [item.dict() for item in task.success_metrics],
        "safety_constraints": [item.dict() for item in task.safety_constraints],
    })
    TrainingOrchestrator._validate_acceptance_contract(tmp_path, task, state)
    task.success_metrics[0].value += 0.1
    with pytest.raises(RuntimeError, match="验收合同不一致"):
        TrainingOrchestrator._validate_acceptance_contract(tmp_path, task, state)


def test_acceptance_contract_rejects_changed_task_goal(tmp_path):
    """即使指标相同，改变动作目标也不能复用旧运行与旧 checkpoint。"""
    from rl_training_agent.orchestration.orchestrator import TrainingOrchestrator
    from rl_training_agent.orchestration.state_machine import PersistentStateMachine
    from rl_training_agent.providers.mock_provider import MockLLMReasoningProvider
    from rl_training_agent.schemas.task import TaskSpec
    from rl_training_agent.utils.io import write_json
    task = TaskSpec.parse_obj(MockLLMReasoningProvider(1).design_task_and_rewards(
        "向前走", "go2", {})["task_spec"])
    state = PersistentStateMachine(tmp_path / "state.json")
    state.start_new_run("goal-run")
    write_json(tmp_path / "acceptance_contract.json", {
        "run_id": "goal-run", "robot": task.robot,
        "instruction": task.original_instruction,
        "task_identity": TrainingOrchestrator._task_acceptance_identity(task),
        "success_metrics": [item.dict() for item in task.success_metrics],
        "safety_constraints": [item.dict() for item in task.safety_constraints],
    })
    task.task_name = "front_leg_stand"
    task.normalized_description = "前腿站立五秒"
    with pytest.raises(RuntimeError, match="改变任务动作"):
        TrainingOrchestrator._validate_acceptance_contract(tmp_path, task, state)
