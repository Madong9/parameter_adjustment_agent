"""前翻任务语义、轴向能力和接触单位的离线回归，不读取真实实验。"""
import json

import pytest

from rl_training_agent.agents.task_semantics import TaskSemanticValidator
from rl_training_agent.environment.metric_registry import canonical_metric_name
from rl_training_agent.feasibility.agent import FeasibilityPipeline
from rl_training_agent.feasibility.motion_prototype.dynamic_generator import DynamicMotionPrototypeGenerator
from rl_training_agent.orchestration.orchestrator import TrainingOrchestrator
from rl_training_agent.providers.mock_provider import MockLLMReasoningProvider
from rl_training_agent.schemas.agent_workflow import TaskIntentSpec
from rl_training_agent.schemas.task import MetricThreshold, TaskPhase, TaskSpec
from rl_training_agent.settings import Settings, load_settings
from rl_training_agent.utils.task_semantics import ForwardFlipKind, classify_forward_flip


def intent(instruction, goal=None):
    """构造保留用户原文的任务理解。"""
    return TaskIntentSpec(original_instruction=instruction, robot="go2", action_name="motion",
                          normalized_goal=goal or instruction, required_behaviors=[goal or instruction])


def task(instruction):
    """构造不包含额外转轴要求的最小任务合同。"""
    result = TaskSpec.parse_obj(MockLLMReasoningProvider(1).design_task_and_rewards(
        "稳定站立", "go2", {})["task_spec"])
    result.original_instruction = instruction
    result.task_name = result.normalized_description = instruction
    result.required_behaviors = []
    result.phases = [TaskPhase(name="FLIGHT", description="腾空旋转")]
    result.success_metrics = []
    return result


def settings(tmp_path):
    """隔离所有实验、产物和长期记忆，限制离线迭代数。"""
    return Settings(**{**load_settings().dict(), "experiment_root": str(tmp_path / "experiments"),
                      "artifact_root": str(tmp_path / "artifacts"), "memory_enabled": False,
                      "num_reward_candidates": 1, "smoke_iterations": 1,
                      "screening_iterations": 2, "full_iterations": 3})


class CountingProvider(MockLLMReasoningProvider):
    """提供明确任务理解，并统计奖励请求以检测门控位置。"""

    def __init__(self, invalid_contact=False):
        """初始化设计次数和可选的接触单位错误。"""
        super().__init__(1)
        self.reward_calls = 0
        self.invalid_contact = invalid_contact

    def understand_task(self, instruction, robot):
        """使用原文生成最小理解，避免模型猜测接触方式。"""
        return intent(instruction)

    def design_task_and_rewards(self, instruction, robot, capabilities):
        """按需在普通站立任务中模拟最新实验的力指标计数单位错误。"""
        self.reward_calls += 1
        design = super().design_task_and_rewards(instruction, robot, capabilities)
        if self.invalid_contact:
            design["task_spec"]["safety_constraints"] = [
                {"name": "forbidden_body_contact", "operator": "<=", "value": 0,
                 "unit": "count", "required": True}]
        return design


@pytest.mark.parametrize("instruction,expected", [
    ("向前翻跟头", ForwardFlipKind.AMBIGUOUS),
    ("forward somersault", ForwardFlipKind.AMBIGUOUS),
    ("向前翻滚，不起跳", ForwardFlipKind.GROUND),
    ("允许机体触地的前滚翻后站起", ForwardFlipKind.GROUND),
    ("forward roll", ForwardFlipKind.GROUND),
    ("起跳腾空前空翻后足端落地", ForwardFlipKind.AERIAL),
    ("front flip from flat ground", ForwardFlipKind.AERIAL),
    ("起跳向前翻跟头", ForwardFlipKind.AERIAL),
    ("向前腾空翻跟头", ForwardFlipKind.AERIAL),
    ("向前起跳翻跟头", ForwardFlipKind.AERIAL),
    ("向前主动起跳腾空翻跟头", ForwardFlipKind.AERIAL),
    ("向前在空中完成一次翻跟头", ForwardFlipKind.AERIAL),
    ("朝前腾空翻筋斗，足端落地后站稳", ForwardFlipKind.AERIAL),
    ("往前腾空翻跟头", ForwardFlipKind.AERIAL),
    ("前向腾空翻转", ForwardFlipKind.AERIAL),
    ("向前 翻跟头", ForwardFlipKind.AMBIGUOUS),
    ("贴地前滚翻并腾空前空翻", ForwardFlipKind.AMBIGUOUS),
    ("稳定站立，不做前空翻", ForwardFlipKind.NONE),
    ("stand, not a front flip", ForwardFlipKind.NONE),
    ("后退走，不要向前翻跟头", ForwardFlipKind.NONE),
    ("保持站立，不要向前腾空翻跟头", ForwardFlipKind.NONE),
    ("保持站立，不做向前起跳翻跟头", ForwardFlipKind.NONE),
    ("向后腾空翻跟头", ForwardFlipKind.NONE),
    ("向前走，机器人在空中翻跟头", ForwardFlipKind.NONE),
    ("向前，然后腾空翻跟头", ForwardFlipKind.NONE),
    ("腾空前空翻，而非机体接触地面的前滚翻", ForwardFlipKind.AERIAL),
    ("腾空前空翻，不是沿地面滚动的前滚翻", ForwardFlipKind.AERIAL),
    ("稳定站立，不得以机体沿地面接触滚动的前滚翻", ForwardFlipKind.NONE),
    ("起跳腾空侧翻", ForwardFlipKind.NONE),
])
def test_forward_motion_respects_contact_mode_and_negation(instruction, expected):
    """前翻接触方式及否定词决定路由，平地起跳不能被误认为地面翻滚。"""
    assert classify_forward_flip(instruction) == expected


@pytest.mark.parametrize("instruction,goal", [
    ("腾空前空翻后落地", "腾空侧空翻后落地"),
    ("腾空侧翻后落地", "腾空前空翻后落地"),
    ("贴地前滚翻后站起", "腾空前空翻后落地"),
    ("起跳前空翻后落地", "地面前滚翻后站起"),
    ("保持站立", "腾空前空翻后落地"),
])
def test_intent_rejects_wrong_axis_and_invented_flight(instruction, goal):
    """前向与侧向动作、地面与腾空动作之间的替换都被拒绝。"""
    assert TaskSemanticValidator.assess_intent(intent(instruction, goal)).status == "CONFLICT"


def aerial_forward_intent():
    """复现本轮模型正确理解及否定地面替代动作的表达，不读取实验目录。"""
    return TaskIntentSpec(original_instruction="向前腾空翻跟头", robot="go2",
        action_name="前空翻（Forward Somersault / Front Flip）",
        normalized_goal="完成一次向前腾空翻跟头：主动起跳，在空中绕机体横向轴完成向前翻转，"
                        "足端着地并恢复稳定姿态。该动作属于腾空前空翻，而非机体接触地面的前滚翻。",
        required_behaviors=["全部足端离地的腾空阶段", "绕横向轴进行向前翻转（前空翻）",
                            "空中完成一次完整向前翻转", "足端落地后恢复稳定四足支撑"],
        forbidden_behaviors=["不得以机体沿地面接触滚动的前滚翻代替腾空前空翻",
                             "不得以侧翻代替向前翻转"])


def test_inserted_airborne_modifier_preserves_correct_model_intent():
    """原文与模型的前空翻理解一致，反面说明不混成地面/腾空双重目标。"""
    value = aerial_forward_intent()
    report = TaskSemanticValidator.assess_intent(value)
    assert report.status == "CONSISTENT" and not report.issues
    assert report.forward_flip_family == ForwardFlipKind.AERIAL
    assert classify_forward_flip(value.normalized_goal) == ForwardFlipKind.AERIAL


def test_correct_aerial_paraphrase_passes_persisted_intent_gate(tmp_path):
    """实际编排入口放行正确理解并保存一致报告，不修改旧实验或启动 PPO。"""
    from rl_training_agent.orchestration.state_machine import AgentState, PersistentStateMachine
    configuration = settings(tmp_path)
    root = configuration.experiments_path / "task-correct-aerial"
    root.mkdir(parents=True)
    state = PersistentStateMachine(root / "state.json")
    orchestrator = TrainingOrchestrator(configuration, CountingProvider())
    assert orchestrator._check_task_semantics(root, state, aerial_forward_intent()) is None
    saved = json.loads((root / "intent_semantic_check.json").read_text())
    assert saved["status"] == "CONSISTENT"
    assert saved["forward_flip_family"] == ForwardFlipKind.AERIAL.value
    assert state.record.state == AgentState.RECEIVED
    assert not (root / "blocking_report.json").exists()


@pytest.mark.parametrize("goal", ["腾空侧空翻后落地", "地面前滚翻后站起"])
def test_aerial_paraphrase_still_rejects_wrong_motion(goal):
    """修复等价输入识别不能放行真正的轴向或接触方式替换。"""
    assert TaskSemanticValidator.assess_intent(intent("向前腾空翻跟头", goal)).status == "CONFLICT"


@pytest.mark.parametrize("metric", ["final_roll_angle", "max_roll_velocity"])
def test_front_flip_cannot_use_roll_completion_metric(metric):
    """明确前翻也不能用侧翻角度或纵轴角速度验收，审核不修改原合同。"""
    instruction = "起跳前空翻后落地"
    contract = task(instruction)
    contract.success_metrics = [MetricThreshold(name=metric, operator=">=", value=6.0)]
    before = contract.dict()
    report = TaskSemanticValidator.assess_task(intent(instruction), contract)
    assert report.status == "CONFLICT" and any("roll 轴" in item for item in report.issues)
    assert contract.dict() == before


@pytest.mark.parametrize("metric", ["final_pitch_angle", "net_pitch_progress", "pitch_rotation", "front_flip_completion"])
def test_unsupported_front_flip_metric_is_not_mapped_to_pitch_limit(metric):
    """未知前翻进度继续表现为缺失能力，不能被普通俯仰姿态指标冒充。"""
    assert canonical_metric_name(metric, ["body_pitch_within_limit", "pitch_limit"]) == metric
    assert canonical_metric_name("body_pitch", ["body_pitch_within_limit"]) == "body_pitch_within_limit"


@pytest.mark.parametrize("instruction,clarification", [
    ("向前翻跟头", True),
    ("起跳腾空前空翻后足端落地", False),
    ("贴地前滚翻后站起", False),
])
def test_front_flip_stops_before_reward_requests_and_physics(tmp_path, monkeypatch, instruction, clarification):
    """含糊目标先澄清，明确前翻先报告缺失实现，均不启动奖励模型或 PPO。"""
    configuration = settings(tmp_path)
    provider = CountingProvider()
    orchestrator = TrainingOrchestrator(configuration, provider)
    original = orchestrator.inspect_environment

    def incomplete_manifest(*args):
        """模拟尚未安装前翻后端的旧环境，不能被侧翻能力替代。"""
        manifest = original(*args)
        manifest.rewards = [reward for reward in manifest.rewards if reward.name not in (
            'net_pitch_progress', 'pitch_rotation', 'front_flip_recovery')]
        manifest.evaluation_metrics = [name for name in manifest.evaluation_metrics if name not in (
            'final_pitch_angle', 'pitch_axis_error')]
        return manifest

    monkeypatch.setattr(orchestrator, 'inspect_environment', incomplete_manifest)
    result = orchestrator.train(instruction, "go2", dry_run=True)
    root = configuration.experiments_path / result["task_id"]
    assert result["state"] == "HUMAN_REVIEW"
    assert result["requires_clarification"] is clarification
    assert provider.reward_calls == 0
    assert not (root / "feasibility_report.json").exists()
    assert not (root / "acceptance_contract.json").exists()
    assert not list(root.rglob("model_*.pt"))
    if clarification:
        assert "前滚翻" in result["semantic_check"]["question"]
    else:
        assert result["requires_local_implementation"]
        assert (root / "action_capability_check.json").is_file()
        assert result["capability_gaps"]


def test_contact_unit_error_blocks_before_freezing_and_reward_retry(tmp_path):
    """固定任务合同自身出错时不再浪费奖励重试，也不会冻结该无效合同。"""
    configuration = settings(tmp_path)
    provider = CountingProvider(invalid_contact=True)
    result = TrainingOrchestrator(configuration, provider).train("稳定站立", "go2", dry_run=True)
    root = configuration.experiments_path / result["task_id"]
    assert result["state"] == "HUMAN_REVIEW" and result["requires_task_redesign"]
    assert result["stage"] == "task_contract_check" and "N 单位" in result["reason"]
    assert "forbidden_collisions" in result["reason"]
    assert provider.reward_calls == 1
    assert not (root / "acceptance_contract.json").exists()
    assert not (root / "reward_review_attempt_00.json").exists()
    assert not list(root.rglob("model_*.pt"))
    saved = json.loads((root / "task_spec.json").read_text())
    assert saved["safety_constraints"][0]["unit"] == "count"


@pytest.mark.parametrize("name,unit", [("forbidden_collisions", "count"), ("forbidden_body_contact", "N")])
def test_valid_contact_contract_preserves_original_threshold(tmp_path, name, unit):
    """合法计数和力合同分别通过，不修改名称、阈值或单位。"""
    from rl_training_agent.orchestration.state_machine import AgentState, PersistentStateMachine
    configuration = settings(tmp_path)
    orchestrator = TrainingOrchestrator(configuration, CountingProvider())
    root = configuration.experiments_path / "task-valid"
    root.mkdir(parents=True)
    state = PersistentStateMachine(root / "state.json")
    # 正常合同无需触发状态变化，使用初始状态验证无副作用。
    contract = task("稳定站立")
    contract.safety_constraints = [MetricThreshold(name=name, unit=unit, operator="<=", value=0.0)]
    before = contract.dict()
    assert orchestrator._check_task_contract(root, state, contract) is None
    assert contract.dict() == before and state.record.state == AgentState.RECEIVED


@pytest.mark.parametrize("instruction,motion_type", [
    ("起跳前空翻后落地", "ACROBATIC"), ("贴地前滚翻后站起", "GROUND_ROLL")])
def test_standalone_feasibility_reports_front_flip_gap_without_loading_model(instruction, motion_type):
    """直接调用预检也不把侧翻奖励能力当成前翻训练能力。"""
    pipeline = object.__new__(FeasibilityPipeline)
    pipeline.stage_callback = None
    report = pipeline.assess(intent(instruction), {
        "rewards": [{"name": "net_roll_progress"}], "evaluation_metrics": ["final_roll_angle"]})
    assert report.status.value == "CONDITIONAL" and report.validation_level == "CAPABILITY_ONLY"
    assert report.motion_type == motion_type
    if motion_type == "ACROBATIC":
        assert any("net_pitch_progress" in item for item in report.missing_requirements)
        assert any("final_pitch_angle" in item for item in report.missing_requirements)
    else:
        assert any("planned_body_contacts" in item for item in report.missing_requirements)


def test_negated_front_flip_keeps_standing_classification():
    """禁止前翻的普通站立目标仍能走原来的静态姿态分支。"""
    from rl_training_agent.feasibility.motion_prototype.schema import MotionType
    assert DynamicMotionPrototypeGenerator.classify(intent("稳定站立，不做前空翻")) == MotionType.STATIC_POSE


def test_new_run_clears_stale_contract_and_capability_reports(tmp_path):
    """新运行不能展示上一次接触合同错误或动作能力报告。"""
    for name in ("action_capability_check.json", "task_contract_check.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    TrainingOrchestrator._clear_stale_run_outputs(tmp_path)
    assert not (tmp_path / "action_capability_check.json").exists()
    assert not (tmp_path / "task_contract_check.json").exists()
