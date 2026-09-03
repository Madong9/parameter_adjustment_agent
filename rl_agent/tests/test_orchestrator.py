import json

import pytest

from rl_training_agent.environment.inspector import EnvironmentInspector
from rl_training_agent.environment.metric_registry import normalize_task_metrics
from rl_training_agent.orchestration.orchestrator import TrainingOrchestrator
from rl_training_agent.orchestration.state_machine import AgentState, PersistentStateMachine
from rl_training_agent.providers.mock_provider import MockLLMReasoningProvider
from rl_training_agent.schemas.decisions import DiagnosisItem, EvidenceItem, RewardChange, TrainingDiagnosis
from rl_training_agent.schemas.rewards import RewardPlan
from rl_training_agent.schemas.task import TaskSpec
from rl_training_agent.schemas.visual import VisualBehaviorReport
from rl_training_agent.settings import Settings, load_settings


def test_end_to_end_dry_run(tmp_path):
    """验证“end to end dry run”场景的预期行为。"""
    base = load_settings()
    settings = Settings(**{**base.dict(), "experiment_root": str(tmp_path / "experiments"),
                           "artifact_root": str(tmp_path / "artifacts"), "full_iterations": 300,
                           "screening_iterations": 30, "smoke_iterations": 5,
                           "max_total_iterations": 1000})
    result = TrainingOrchestrator(settings, MockLLMReasoningProvider(3)).train(
        "测试机器狗稳定向前行走", "go2", dry_run=True)
    task_dir = settings.experiments_path / result["task_id"]
    assert result["state"] == "COMPLETED"
    assert (task_dir / "final" / "checkpoint.pt").is_file()
    rollout = task_dir / result["rollout"]
    for name in ("front.mp4", "side.mp4", "overview.mp4", "trajectory.parquet", "rewards.parquet",
                 "metadata.json", "events.json", "contact_sheet_clean.png", "contact_sheet_annotated.png",
                 "contact_sheet_multiview.png", "behavior_evidence.json", "visual_attachment_manifest.json",
                 "visual_report.json", "numeric_summary.json"):
        assert (rollout / name).is_file(), name


def test_state_recovery_and_decision_states(tmp_path):
    """验证“state recovery and decision states”场景的预期行为。"""
    path = tmp_path / "state.json"
    state = PersistentStateMachine(path)
    for value in (AgentState.REVISE_REWARD, AgentState.ROLLBACK, AgentState.RESTART, AgentState.HUMAN_REVIEW):
        state.transition(value)
        assert PersistentStateMachine(path).record.state == value


def test_budget_exhaustion(tmp_path):
    """验证“budget exhaustion”场景的预期行为。"""
    from rl_training_agent.orchestration.budget import BudgetTracker
    budget = BudgetTracker(max_iterations=10, max_revisions=1)
    budget.consume_iterations(10)
    try:
        budget.consume_iterations(1)
        assert False
    except RuntimeError:
        assert True


def test_visual_numeric_conflict_requires_noncompletion():
    """验证“visual numeric conflict requires noncompletion”场景的预期行为。"""
    from rl_training_agent.evaluation.deterministic import DeterministicEvaluator
    from rl_training_agent.schemas.task import TaskSpec
    from rl_training_agent.schemas.visual import VisualBehaviorReport
    task = MockLLMReasoningProvider().design_task_and_rewards(
        "测试机器狗稳定向前行走", "go2",
        EnvironmentInspector(load_settings().training_root).inspect("go2").dict())["task_spec"]
    task = TaskSpec.parse_obj(task)
    visual = VisualBehaviorReport(visual_success=True, alignment_score=1, confidence=1,
                                  summary="visual pass", phase_results=[])
    result = DeterministicEvaluator().evaluate(task,
        {"tracking_error": 1.0, "fall_rate": 0.0, "nan_count": 0,
         "joint_limit_violations": 0, "torque_limit_violations": 0,
         "forbidden_collisions": 0, "abnormal_terminations": 0}, visual)
    assert not result.completed and result.conflicts == ["visual_numeric_disagreement"]


def test_multi_rollout_metrics_use_conservative_worst_case():
    """验证多种子验收不会因一个偶然优秀 rollout 掩盖跌倒或跟踪失败。"""
    design = MockLLMReasoningProvider(1).design_task_and_rewards("向前行走", "go2", {})
    task = TaskSpec.parse_obj(design["task_spec"])
    records = [
        {"metrics": {"tracking_error": 0.1, "fall_rate": 0.0, "energy": 1.0}},
        {"metrics": {"tracking_error": 0.6, "fall_rate": 0.2, "energy": 3.0}},
        {"metrics": {"tracking_error": 0.2, "fall_rate": 0.0, "energy": 2.0}},
    ]
    aggregate = TrainingOrchestrator._aggregate_rollout_metrics(task, records)
    assert aggregate["tracking_error"] == 0.6
    assert aggregate["fall_rate"] == 0.2
    assert aggregate["energy"] == 2.0
    assert aggregate["evaluated_rollout_count"] == 3


def test_provider_failure_leaves_recoverable_state(tmp_path):
    """验证“provider failure leaves recoverable state”场景的预期行为。"""
    class FailedProvider(MockLLMReasoningProvider):
        def design_task_and_rewards(self, instruction, robot, capabilities):
            """依据任务描述和环境能力生成任务规格与奖励候选。"""
            raise RuntimeError("provider unavailable")
    base = load_settings()
    settings = Settings(**{**base.dict(), "experiment_root": str(tmp_path / "experiments"),
                           "artifact_root": str(tmp_path / "artifacts")})
    try:
        TrainingOrchestrator(settings, FailedProvider()).train("failure test", "go2", dry_run=True)
        assert False
    except RuntimeError:
        task_id = TrainingOrchestrator._task_id("failure test", "go2")
        state = json.loads((settings.experiments_path / task_id / "state.json").read_text())
        assert state["state"] == "ENVIRONMENT_INSPECTED"


def test_candidate_selection_rejects_failed_hard_constraints(tmp_path):
    """验证全部候选未通过硬门槛时不会继续完整训练。"""
    candidates = []
    for index in range(2):
        directory = tmp_path / ("candidate-%d" % index)
        (directory / "metrics").mkdir(parents=True)
        (directory / "metrics" / "screening.json").write_text(json.dumps({
            "hard_constraints_passed": False,
            "composite_score": -1.0,
        }))
        candidates.append({"dir": directory})
    with pytest.raises(RuntimeError, match="禁止进入完整训练"):
        TrainingOrchestrator._select_safe_candidate(candidates)


def test_candidate_selection_uses_only_safe_finite_scores(tmp_path):
    """验证候选选择会忽略未通过门槛或非有限得分。"""
    candidates = []
    for index, (passed, score) in enumerate(((True, 0.4), (False, 0.9), (True, 0.7))):
        directory = tmp_path / ("candidate-%d" % index)
        (directory / "metrics").mkdir(parents=True)
        (directory / "metrics" / "screening.json").write_text(json.dumps({
            "hard_constraints_passed": passed,
            "composite_score": score,
        }))
        candidates.append({"dir": directory, "index": index})
    assert TrainingOrchestrator._select_safe_candidate(candidates)["index"] == 2


def test_reward_plan_must_cover_required_task_metrics():
    """验证仅保留速度代理而遗漏任务指标的奖励计划会被拒绝。"""
    design = MockLLMReasoningProvider(1).design_task_and_rewards("向前行走", "go2", {})
    task = TaskSpec.parse_obj(design["task_spec"])
    plan = RewardPlan.parse_obj(design["reward_plans"][0])
    plan.success_metrics = plan.success_metrics[:1]
    with pytest.raises(ValueError, match="misses required success metrics"):
        TrainingOrchestrator._validate_plan_metric_coverage(task, plan)


def test_rear_leg_task_requires_posture_gated_rewards():
    """验证后腿站立任务不能再次退化为普通速度跟踪方案。"""
    design = MockLLMReasoningProvider(1).design_task_and_rewards("向前行走", "go2", {})
    task = TaskSpec.parse_obj(design["task_spec"])
    task.original_instruction = "机器狗后腿站立行走"
    plan = RewardPlan.parse_obj(design["reward_plans"][0])
    with pytest.raises(ValueError, match="posture-gated rewards"):
        TrainingOrchestrator._validate_plan_metric_coverage(task, plan)


def test_rear_leg_plan_normalization_adds_metrics_and_removes_proxy_reward():
    """验证后腿任务会继承验收指标并移除未门控速度奖励。"""
    design = MockLLMReasoningProvider(1).design_task_and_rewards(
        "向前行走", "go2", {"rewards": [{"name": "tracking_lin_vel"}]})
    task = TaskSpec.parse_obj(design["task_spec"])
    task.original_instruction = "机器狗后腿站立行走"
    plan = RewardPlan.parse_obj(design["reward_plans"][0])
    plan.success_metrics = []
    adjustments = TrainingOrchestrator._normalize_plan_for_task(task, plan)
    assert {item.name for item in plan.success_metrics} == {item.name for item in task.success_metrics}
    assert "tracking_lin_vel" not in {item.name for item in plan.terms}
    assert any("移除后腿任务冲突奖励" in item for item in adjustments)


def test_front_leg_support_instruction_corrects_task_and_injects_gated_rewards():
    """验证“用前腿站立”会被解释为前足支撑，并补充前腿门控奖励。"""
    design = MockLLMReasoningProvider(1).design_task_and_rewards(
        "向前行走", "go2", {"rewards": [{"name": "tracking_lin_vel"}]})
    task = TaskSpec.parse_obj(design["task_spec"])
    task.original_instruction = "机器狗两只前腿站立走路"
    task.required_behaviors[0].name = "front_leg_lifted_posture"
    task.required_behaviors[0].description = "保持两只前腿离地"
    task_adjustments = TrainingOrchestrator._normalize_task_for_instruction(task)
    plan = RewardPlan.parse_obj(design["reward_plans"][0])
    plan.success_metrics = []
    plan_adjustments = TrainingOrchestrator._normalize_plan_for_task(task, plan)
    names = {item.name for item in plan.terms}
    assert task.required_behaviors[0].name == "front_leg_support_posture"
    assert {"front_leg_stand", "front_leg_walk"} <= names
    assert "tracking_lin_vel" not in names
    assert task_adjustments and any("补充前腿" in item for item in plan_adjustments)


def test_generated_metric_aliases_are_normalized_before_capability_validation():
    """验证模型使用近义指标时会映射到可计算物理量，而不是误入人工审核。"""
    settings = load_settings()
    inspector = EnvironmentInspector(settings.training_root)
    manifest = inspector.inspect("go2")
    design = MockLLMReasoningProvider(1).design_task_and_rewards(
        "机器狗用前腿支撑行走", "go2", manifest.dict())
    task = TaskSpec.parse_obj(design["task_spec"])
    task.success_metrics[0].name = "front_leg_walk_duration"
    task.safety_constraints[0].name = "body_contact_force"
    mappings = normalize_task_metrics(task, manifest.evaluation_metrics)
    unsupported, _ = inspector.validate_task(task, manifest)
    assert task.success_metrics[0].name == "front_leg_walk_completion"
    assert task.safety_constraints[0].name == "forbidden_body_contact"
    assert not [item for item in unsupported if item.startswith("evaluation_metric:")]
    assert len(mappings) == 2


def test_failed_evaluation_revises_reward_and_rechecks_until_completed(tmp_path):
    """验证视觉未达标会触发可执行奖励修订、续训和第二轮联合验收。"""
    class RevisingProvider(MockLLMReasoningProvider):
        """在第一轮制造视觉失败并给出一次确定性的奖励修订。"""

        def __init__(self):
            """初始化视觉和诊断调用计数。"""
            super().__init__(1)
            self.visual_calls = 0
            self.diagnosis_calls = 0

        def critique_visual_behavior(self, task, files):
            """第一轮返回未通过，第二轮返回通过。"""
            self.visual_calls += 1
            if self.visual_calls == 1:
                return VisualBehaviorReport(
                    visual_success=False, alignment_score=0.4, confidence=0.9,
                    summary="第一轮动作姿态尚未达到目标。", phase_results=[])
            return super().critique_visual_behavior(task, files)

        def diagnose_training(self, payload):
            """第一轮提高速度跟踪奖励，第二轮确认联合验收结果。"""
            self.diagnosis_calls += 1
            if self.diagnosis_calls == 1:
                assert payload["capabilities"]["registered_rewards"]
                return TrainingDiagnosis(
                    diagnosis=[DiagnosisItem(
                        category="动作对齐", finding="速度跟踪奖励不足", severity="warning")],
                    evidence=[EvidenceItem(
                        source="visual", metric="alignment_score", value=0.4,
                        interpretation="动作仍需优化")],
                    decision="revise_reward", confidence=0.9,
                    reward_changes=[RewardChange(
                        term="tracking_lin_vel", action="update",
                        changes={"weight_multiplier": 1.1}, rationale="增强目标速度跟踪")],
                    expected_effects=["提高动作对齐度"], risks=["能耗可能增加"],
                    checkpoint_strategy="continue_from_current")
            return super().diagnose_training(payload)

    base = load_settings()
    settings = Settings(**{
        **base.dict(), "experiment_root": str(tmp_path / "experiments"),
        "artifact_root": str(tmp_path / "artifacts"), "num_reward_candidates": 1,
        "smoke_iterations": 5, "screening_iterations": 10, "full_iterations": 100,
        "mid_iterations": 20, "max_total_iterations": 500,
        "max_reward_revisions": 2, "evaluation_seeds": [1], "rollouts_per_seed": 1,
    })
    provider = RevisingProvider()
    result = TrainingOrchestrator(settings, provider).train(
        "测试机器狗稳定向前行走", "go2", dry_run=True)
    task_dir = settings.experiments_path / result["task_id"]
    history = json.loads((task_dir / "loop_history.json").read_text())
    lineage = json.loads((task_dir / "lineage.json").read_text())
    assert result["state"] == "COMPLETED"
    assert result["loop_rounds"] == 2 and result["used_revisions"] == 1
    assert result["reward_version"] == 2
    assert [item["decision"] for item in history] == ["revise_reward", "complete"]
    assert len(lineage["edges"]) == 1
    assert lineage["nodes"][-1]["result"] == "completed"


def test_initial_full_training_is_capped_by_remaining_budget(tmp_path):
    """验证任务预算较小时会缩短完整训练，而不是在启动阶段直接超预算崩溃。"""
    base = load_settings()
    settings = Settings(**{
        **base.dict(), "experiment_root": str(tmp_path / "experiments"),
        "artifact_root": str(tmp_path / "artifacts"), "num_reward_candidates": 1,
        "smoke_iterations": 5, "screening_iterations": 10, "full_iterations": 100,
        "max_total_iterations": 50, "evaluation_seeds": [7], "rollouts_per_seed": 1,
    })
    result = TrainingOrchestrator(settings, MockLLMReasoningProvider(1)).train(
        "测试机器狗稳定向前行走", "go2", dry_run=True)
    task_dir = settings.experiments_path / result["task_id"]
    assert result["state"] == "COMPLETED"
    assert result["used_iterations"] == 50
    assert list(task_dir.glob("candidates/*/checkpoints/seed_7/model_40.pt"))
