import json

import pytest

from rl_training_agent.environment.inspector import EnvironmentInspector
from rl_training_agent.environment.metric_registry import normalize_task_metrics
from rl_training_agent.orchestration.budget import BudgetTracker
from rl_training_agent.orchestration.orchestrator import TrainingOrchestrator
from rl_training_agent.orchestration.state_machine import AgentState, PersistentStateMachine
from rl_training_agent.providers.mock_provider import MockLLMReasoningProvider
from rl_training_agent.providers.errors import ProviderTimeout
from rl_training_agent.rewards.validator import RewardValidationError
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


def test_real_dynamic_run_stops_before_reward_design_when_probe_is_not_validated(tmp_path):
    """验证动态行走必须由真实短 rollout 通过后才进入奖励设计/PPO。"""
    class CountingProvider(MockLLMReasoningProvider):
        """记录奖励设计是否越过可行性前置门控。"""

        def __init__(self):
            """初始化调用标志。"""
            super().__init__(1)
            self.reward_design_called = False

        def design_task_and_rewards(self, instruction, robot, capabilities):
            """标记真实奖励设计阶段是否被执行。"""
            self.reward_design_called = True
            return super().design_task_and_rewards(instruction, robot, capabilities)

    base = load_settings()
    settings = Settings(**{**base.dict(), "experiment_root": str(tmp_path / "experiments"),
                           "artifact_root": str(tmp_path / "artifacts"),
                           "feasibility_admission_mode": "strict"})
    provider = CountingProvider()
    result = TrainingOrchestrator(settings, provider).train(
        "Go2 倒退走 1m/s 0.2秒", "go2", dry_run=False)
    task_dir = settings.experiments_path / result["task_id"]
    feasibility = json.loads((task_dir / "feasibility_report.json").read_text(encoding="utf-8"))

    assert result["state"] in ("HUMAN_REVIEW", "FAILED")
    assert feasibility["motion_type"] == "LOCOMOTION"
    assert feasibility["validation_level"] in ("CAPABILITY_ONLY", "PHYSICS_FAILED")
    assert feasibility["dynamic_report"] is not None
    assert not provider.reward_design_called
    assert not (task_dir / "design_response.json").exists()


def test_orchestrator_trains_only_candidates_approved_by_reviewer(tmp_path):
    """验证奖励审查拒绝中间候选后，其他候选仍会编译并完成演练。"""
    class MixedCandidateProvider(MockLLMReasoningProvider):
        """为回归测试注入一个非法奖励项，其余候选保持有效。"""

        def design_task_and_rewards(self, instruction, robot, capabilities):
            """生成标准候选后，只破坏第二个候选以覆盖逐项过滤。"""
            bundle = super().design_task_and_rewards(instruction, robot, capabilities)
            bundle["reward_plans"][1]["terms"][0]["name"] = "unregistered_test_reward"
            return bundle

    base = load_settings()
    settings = Settings(**{**base.dict(), "experiment_root": str(tmp_path / "experiments"),
                           "artifact_root": str(tmp_path / "artifacts"), "full_iterations": 300,
                           "screening_iterations": 30, "smoke_iterations": 5,
                           "max_total_iterations": 1000})
    result = TrainingOrchestrator(settings, MixedCandidateProvider(3)).train(
        "测试机器狗稳定向前行走", "go2", dry_run=True)
    task_dir = settings.experiments_path / result["task_id"]
    review = json.loads((task_dir / "reward_review.json").read_text(encoding="utf-8"))
    candidate_root = task_dir / "candidates"

    assert result["state"] == "COMPLETED"
    assert review["passed_candidate_indexes"] == [1, 3]
    assert review["rejected_candidate_indexes"] == [2]
    assert any("candidate-01" in path.name for path in candidate_root.iterdir())
    assert not any("candidate-02" in path.name for path in candidate_root.iterdir())
    assert any("candidate-03" in path.name for path in candidate_root.iterdir())


def test_state_recovery_and_decision_states(tmp_path):
    """验证状态机拒绝非法跳转，并使用操作键幂等提交合法决策状态。"""
    path = tmp_path / "state.json"
    state = PersistentStateMachine(path)
    with pytest.raises(ValueError, match="illegal state transition"):
        state.transition(AgentState.REVISE_REWARD)
    path_to_diagnosis = (
        AgentState.ENVIRONMENT_INSPECTED, AgentState.TASK_UNDERSTANDING,
        AgentState.TASK_FEASIBILITY_CHECK,
        AgentState.CONTEXT_BUILDING, AgentState.PROMPT_COMPILING,
        AgentState.REWARD_DESIGNING, AgentState.TASK_DESIGNED,
        AgentState.REWARD_REVIEWING, AgentState.REWARD_CANDIDATES_CREATED,
        AgentState.CONFIGS_COMPILED, AgentState.VALIDATED, AgentState.SMOKE_TRAINING,
        AgentState.CANDIDATE_SCREENING, AgentState.FULL_TRAINING,
        AgentState.ROLLOUT_COLLECTING, AgentState.VISUAL_EVALUATING,
        AgentState.NUMERIC_EVALUATING, AgentState.DIAGNOSING,
    )
    for value in path_to_diagnosis:
        state.transition(value)
    assert state.transition(AgentState.REVISE_REWARD, operation_id="round-1-decision")
    assert not state.transition(AgentState.REVISE_REWARD, operation_id="round-1-decision")
    assert PersistentStateMachine(path).record.state == AgentState.REVISE_REWARD


@pytest.mark.parametrize("decision_state", [
    AgentState.CONTINUE_TRAINING,
    AgentState.REVISE_REWARD,
    AgentState.REVISE_CURRICULUM,
    AgentState.ROLLBACK,
    AgentState.RESTART,
])
def test_decision_state_can_finish_as_human_review_after_safe_block(tmp_path, decision_state):
    """验证决策落盘后预算或编译阻塞可以正常收束，而不是非法跳转崩溃。"""
    state = PersistentStateMachine(tmp_path / (decision_state.value + ".json"))
    path_to_diagnosis = (
        AgentState.ENVIRONMENT_INSPECTED, AgentState.TASK_UNDERSTANDING,
        AgentState.TASK_FEASIBILITY_CHECK,
        AgentState.CONTEXT_BUILDING, AgentState.PROMPT_COMPILING,
        AgentState.REWARD_DESIGNING, AgentState.TASK_DESIGNED,
        AgentState.REWARD_REVIEWING, AgentState.REWARD_CANDIDATES_CREATED,
        AgentState.CONFIGS_COMPILED, AgentState.VALIDATED, AgentState.SMOKE_TRAINING,
        AgentState.CANDIDATE_SCREENING, AgentState.FULL_TRAINING,
        AgentState.ROLLOUT_COLLECTING, AgentState.VISUAL_EVALUATING,
        AgentState.NUMERIC_EVALUATING, AgentState.DIAGNOSING,
    )
    for value in path_to_diagnosis:
        state.transition(value)
    state.transition(decision_state)
    assert state.transition(AgentState.HUMAN_REVIEW, {"reason": "预算耗尽"})
    assert state.record.state == AgentState.HUMAN_REVIEW


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


def test_checkpoint_restore_recognizes_real_and_dry_run_seed_directories(tmp_path):
    """验证恢复闭环时既识别真实训练 seed 后缀，也识别离线演练 seed 目录。"""
    dry = tmp_path / "checkpoints" / "seed_1"
    real = tmp_path / "Jul01_demo-seed-2"
    collision = tmp_path / "Jul01_demo-seed-10"
    for directory, iteration in ((dry, 100), (real, 200), (collision, 999)):
        directory.mkdir(parents=True)
        (directory / ("model_%d.pt" % iteration)).write_text("checkpoint", encoding="utf-8")
    assert TrainingOrchestrator._checkpoint_for_seed(tmp_path, 1).name == "model_100.pt"
    assert TrainingOrchestrator._checkpoint_for_seed(tmp_path, 2).name == "model_200.pt"


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
        assert state["state"] == "REWARD_DESIGNING"


def test_visual_provider_failure_resumes_from_checkpoint_and_cached_rollout(tmp_path):
    """验证视觉服务故障后可复用 checkpoint 和 rollout 完成闭环，而不是从头训练。"""
    class VisualFailureProvider(MockLLMReasoningProvider):
        """在视觉评论阶段模拟一次网页 Provider 超时。"""

        def critique_visual_behavior(self, task, files):
            """抛出可恢复的视觉 Provider 超时。"""
            raise ProviderTimeout("视觉回复暂时不可用")

    base = load_settings()
    settings = Settings(**{
        **base.dict(), "experiment_root": str(tmp_path / "experiments"),
        "artifact_root": str(tmp_path / "artifacts"), "num_reward_candidates": 1,
        "smoke_iterations": 5, "screening_iterations": 10, "full_iterations": 100,
        "max_total_iterations": 300, "evaluation_seeds": [1, 2], "rollouts_per_seed": 1,
    })
    first = TrainingOrchestrator(settings, VisualFailureProvider(1)).train(
        "测试视觉故障恢复稳定向前行走", "go2", dry_run=True)
    task_dir = settings.experiments_path / first["task_id"]
    rollout_path = task_dir / first["rollout"]
    rollout_mtime = (rollout_path / "trajectory.parquet").stat().st_mtime_ns
    assert first["state"] == "HUMAN_REVIEW"
    resumed = TrainingOrchestrator(settings, MockLLMReasoningProvider(1)).resume(
        first["task_id"], dry_run=True)
    assert resumed["state"] == "COMPLETED"
    assert resumed["selected_experiment"] == first["selected_experiment"]
    assert (rollout_path / "trajectory.parquet").stat().st_mtime_ns == rollout_mtime


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


def test_backward_speed_is_compiled_into_command_stage_and_tracking_reward():
    """验证明确倒退速度不会只停留在提示词中，而会成为真实 Unitree 命令范围。"""
    design = MockLLMReasoningProvider(1).design_task_and_rewards(
        "向前行走", "go2", {"rewards": [{"name": "tracking_lin_vel"}]})
    task = TaskSpec.parse_obj(design["task_spec"])
    task.original_instruction = "机器狗倒着走路速度1m/s"
    task.normalized_description = "控制 Go2 以 1 m/s 向后行走"
    task.success_metrics[0].name = "tracking_lin_vel"
    task.success_metrics[0].required = True
    plan = RewardPlan.parse_obj(design["reward_plans"][0])
    plan.curriculum = []
    tracking = next(item for item in plan.terms if item.name == "tracking_lin_vel")
    tracking.weight = 0.0
    tracking.active_phases = []
    adjustments = TrainingOrchestrator._normalize_plan_for_task(task, plan)
    assert len(plan.curriculum) == 1
    changes = plan.curriculum[0].parameter_changes
    assert changes["lin_vel_x"] == [-1.0, -1.0]
    assert changes["lin_vel_y"] == [0.0, 0.0]
    assert changes["ang_vel_yaw"] == [0.0, 0.0]
    assert tracking.weight == 1.0 and tracking.active_phases == ["all"]
    assert any("确定性任务命令" in item for item in adjustments)


def test_backward_command_ignores_opposite_direction_in_forbidden_description():
    """验证“禁止向前”不会抵消用户原始指令中明确的倒退速度。"""
    design = MockLLMReasoningProvider(1).design_task_and_rewards(
        "向前行走", "go2", {"rewards": [{"name": "tracking_lin_vel"}]})
    task = TaskSpec.parse_obj(design["task_spec"])
    task.original_instruction = "倒着走直线，速度大小是1m/s"
    task.normalized_description = "以 1 m/s 倒退，禁止向前行走或偏离直线"
    assert TrainingOrchestrator._explicit_locomotion_command(task) == -1.0
    plan = RewardPlan.parse_obj(design["reward_plans"][0])
    TrainingOrchestrator._normalize_plan_for_task(task, plan)
    assert all(stage.parameter_changes["lin_vel_x"] == [-1.0, -1.0]
               for stage in plan.curriculum)


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


def test_fixed_speed_motion_does_not_require_zero_command_counterfactual():
    """验证固定速度动作不因未要求的零命令停车能力而被判定奖励投机。"""
    task = TaskSpec.parse_obj({
        "task_id": "task-fixed", "robot": "go2", "task_name": "backward_walk",
        "original_instruction": "倒着走直线，速度大小是1m/s",
        "normalized_description": "以固定1m/s速度连续倒退",
        "initial_state": "standing",
        "required_behaviors": [{
            "name": "backward_locomotion", "description": "保持连续倒退步态",
        }],
        "forbidden_behaviors": [], "phases": [], "required_observations": [],
        "required_sensors": [],
        "success_metrics": [{
            "name": "tracking_error", "operator": "<=", "value": 0.15, "unit": "m/s",
        }],
        "safety_constraints": [], "training_budget": {},
        "visual_evaluation_requirements": [],
    })
    conditional = task.copy(deep=True)
    conditional.original_instruction = "根据速度指令倒着走，零速度时停下，最大速度1m/s"

    assert not TrainingOrchestrator._counterfactual_applicable(task)
    assert TrainingOrchestrator._counterfactual_applicable(conditional)


def test_failed_revision_compile_does_not_consume_revision_budget(tmp_path, monkeypatch):
    """验证安全编译失败不会占用一次并未真正生成的奖励修订机会。"""
    orchestrator = TrainingOrchestrator(load_settings(), MockLLMReasoningProvider(1))
    budget = BudgetTracker(max_iterations=100, max_revisions=3)

    def reject_revision(*args, **kwargs):
        """模拟奖励计划在创建新版本前被本地安全校验拒绝。"""
        raise RewardValidationError("测试编译拒绝")

    monkeypatch.setattr(orchestrator, "_compile_revision", reject_revision)
    with pytest.raises(RewardValidationError, match="测试编译拒绝"):
        orchestrator._train_revision(
            None, tmp_path, None, {}, None, 1, True, budget)
    assert budget.used_revisions == 0
