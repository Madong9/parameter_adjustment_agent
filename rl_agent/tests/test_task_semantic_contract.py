"""回归测试动作歧义、模型语义漂移和训练前确定性门控。"""
import json

import pytest

from rl_training_agent.agents.task_semantics import TaskSemanticValidator
from rl_training_agent.feasibility.agent import FeasibilityPipeline
from rl_training_agent.feasibility.motion_constraints import MotionConstraintCompiler
from rl_training_agent.feasibility.motion_prototype.dynamic_generator import DynamicMotionPrototypeGenerator
from rl_training_agent.feasibility.motion_prototype.schema import MotionType
from rl_training_agent.orchestration.orchestrator import TrainingOrchestrator
from rl_training_agent.providers.mock_provider import MockLLMReasoningProvider
from rl_training_agent.schemas.agent_workflow import TaskIntentSpec
from rl_training_agent.schemas.task import MetricThreshold, TaskPhase, TaskSpec
from rl_training_agent.settings import Settings, load_settings
from rl_training_agent.utils.task_semantics import RollMotionKind, classify_roll_motion, is_side_flip_text


def make_intent(instruction, goal=None):
    """构造原文与模型理解可以独立变化的最小任务输入。"""
    return TaskIntentSpec(original_instruction=instruction, robot="go2", action_name="custom_motion",
                          normalized_goal=goal or instruction, required_behaviors=[goal or instruction])


def make_task(instruction, description=None):
    """从离线任务结构构造固定合同，不访问真实实验或推理服务。"""
    task = TaskSpec.parse_obj(MockLLMReasoningProvider(1).design_task_and_rewards(
        "稳定站立", "go2", {})["task_spec"])
    task.original_instruction = instruction
    task.normalized_description = description or instruction
    task.task_name = "custom_motion"
    task.required_behaviors = []
    task.phases = [TaskPhase(name="execute", description=description or instruction)]
    return task


@pytest.mark.parametrize("instruction,expected", [
    ("Go2侧向翻滚", RollMotionKind.AMBIGUOUS),
    ("向左侧翻一圈", RollMotionKind.AMBIGUOUS),
    ("向左贴地侧滚一圈后站起", RollMotionKind.GROUND_ROLL),
    ("地面侧向翻滚，不起跳", RollMotionKind.GROUND_ROLL),
    ("侧向翻滚，不腾空", RollMotionKind.GROUND_ROLL),
    ("lateral roll, not an aerial side flip", RollMotionKind.GROUND_ROLL),
    ("侧向翻滚，不是侧空翻", RollMotionKind.AMBIGUOUS),
    ("Go2侧空翻一圈后落地", RollMotionKind.AERIAL_SIDE_FLIP),
    ("起跳腾空侧翻后落地", RollMotionKind.AERIAL_SIDE_FLIP),
    ("从平坦地面起跳做侧空翻", RollMotionKind.AERIAL_SIDE_FLIP),
    ("侧向翻跟头", RollMotionKind.AERIAL_SIDE_FLIP),
    ("side flip and land", RollMotionKind.AERIAL_SIDE_FLIP),
    ("贴地侧滚并腾空侧翻", RollMotionKind.AMBIGUOUS),
    ("稳定站立，不做侧空翻", RollMotionKind.NONE),
    ("倒退走，不要侧向翻滚", RollMotionKind.NONE),
    ("Go2 倒退走 0.3m/s", RollMotionKind.NONE),
])
def test_roll_motion_classification_preserves_positive_goal(instruction, expected):
    """否定动作不触发配方，含糊目标也不被悄悄升级为腾空动作。"""
    assert classify_roll_motion(instruction) == expected


def test_original_instruction_wins_over_generated_task_name():
    """奖励归一化只信原始指令，后续模型空翻命名不能覆盖贴地目标。"""
    assert not is_side_flip_text(("贴地侧滚后站起", "aerial side flip"))
    assert not is_side_flip_text(("侧向翻滚", "侧空翻"))
    assert is_side_flip_text(("起跳侧空翻", "ground roll"))


def test_intent_reports_ground_to_flight_drift():
    """原文是地面侧滚时，任务理解新增腾空条件必须拒绝。"""
    report = TaskSemanticValidator.assess_intent(make_intent("贴地侧滚后站起", "腾空侧翻后落地"))
    assert report.status == "CONFLICT"
    assert "起跳/腾空" in report.issues[0]


def test_intent_rejects_invented_side_flip():
    """普通站立不能由模型替换为侧空翻。"""
    assert TaskSemanticValidator.assess_intent(make_intent("保持站立", "侧空翻后落地")).status == "CONFLICT"


def test_negated_side_flip_does_not_change_standing_type():
    """禁止事项中的空翻不能让站立任务走特技规划分支。"""
    intent = make_intent("稳定站立，不做侧空翻")
    assert TaskSemanticValidator.assess_intent(intent).status == "CONSISTENT"
    assert DynamicMotionPrototypeGenerator.classify(intent) == MotionType.STATIC_POSE


def test_task_contract_rejects_airborne_metric_for_ground_roll():
    """地面侧滚不得通过添加必选腾空指标而变成空翻验收。"""
    instruction = "贴地侧滚后站起"
    task = make_task(instruction)
    task.success_metrics.append(MetricThreshold(name="feet_air_time", operator=">=", value=0.5))
    report = TaskSemanticValidator.assess_task(make_intent(instruction), task)
    assert report.status == "CONFLICT"
    assert any("全足离地" in issue for issue in report.issues)


def test_ground_contact_conflict_does_not_weaken_frozen_safety_contract():
    """接触合同与动作冲突时只阻塞，不能自动豁免非足端接触。"""
    instruction = "贴地侧滚后站起"
    task = make_task(instruction)
    task.safety_constraints.append(MetricThreshold(name="forbidden_collisions", operator="==", value=0))
    before = task.dict()
    report = TaskSemanticValidator.assess_task(make_intent(instruction), task)
    assert report.status == "CONFLICT"
    assert any("接触阶段" in issue for issue in report.issues)
    assert task.dict() == before


def test_explicit_side_flip_with_flight_contract_passes_semantic_check():
    """明确腾空目标且合同具有执行飞行阶段时只通过语义门控。"""
    instruction = "起跳腾空侧翻后落地"
    task = make_task(instruction)
    task.phases = [TaskPhase(name="FLIGHT", description="保持空中旋转")]
    assert TaskSemanticValidator.assess_task(make_intent(instruction), task).status == "CONSISTENT"


def test_task_identity_mismatch_is_rejected():
    """合同机器人或原始指令不符时不得继续训练。"""
    intent = make_intent("保持站立")
    task = make_task("保持站立")
    task.robot = "g1"
    assert TaskSemanticValidator.assess_task(intent, task).status == "CONFLICT"


def test_ground_roll_has_separate_constraint_plan_without_flight():
    """贴地原型只输出缺失接触合同，不虚构全身求解或腾空阶段。"""
    intent = make_intent("贴地侧滚后站起")
    assert DynamicMotionPrototypeGenerator.classify(intent) == MotionType.GROUND_ROLL
    spec = MotionConstraintCompiler.compile(intent, MotionType.GROUND_ROLL)
    assert spec.motion_type == MotionType.GROUND_ROLL
    assert not any(phase.name in ("TAKEOFF", "FLIGHT") for phase in spec.phases)
    assert "planned_body_contacts_require_explicit_approval" in spec.phases[0].constraints


def test_ambiguous_pipeline_stops_before_model_or_physics():
    """独立调用预检也先澄清歧义，不加载模型或启动 Isaac Gym。"""
    pipeline = object.__new__(FeasibilityPipeline)
    events = []
    pipeline.stage_callback = events.append
    report = pipeline.assess(make_intent("侧向翻滚"), {})
    assert report.status.value == "NEEDS_CLARIFICATION"
    assert report.validation_level == "CAPABILITY_ONLY"
    assert report.motion_type == "UNKNOWN"
    assert events[0]["status"] == "blocked"


def test_ambiguous_orchestrator_stops_before_feasibility_and_reward_design(tmp_path):
    """完整入口输出人工澄清和事件证据，不请求奖励或花费训练额度。"""
    class CountingProvider(MockLLMReasoningProvider):
        """记录不应越过语义门控的奖励调用。"""

        def __init__(self):
            """初始化奖励调用计数。"""
            super().__init__(1)
            self.reward_calls = 0

        def design_task_and_rewards(self, instruction, robot, capabilities):
            """门控失效时记录奖励设计请求。"""
            self.reward_calls += 1
            return super().design_task_and_rewards(instruction, robot, capabilities)

    settings = Settings(**{**load_settings().dict(), "experiment_root": str(tmp_path / "experiments"),
                          "artifact_root": str(tmp_path / "artifacts"), "memory_enabled": False})
    provider = CountingProvider()
    result = TrainingOrchestrator(settings, provider).train("侧向翻滚", "go2", dry_run=True)
    task_dir = settings.experiments_path / result["task_id"]
    assert result["state"] == "HUMAN_REVIEW"
    assert result["requires_clarification"]
    assert provider.reward_calls == 0
    assert not (task_dir / "feasibility_report.json").exists()
    assert not (task_dir / "design_response.json").exists()
    assert not (task_dir / "training_execution.json").exists()
    persisted = json.loads((task_dir / "intent_semantic_check.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "NEEDS_CLARIFICATION"
    events = [json.loads(line) for line in (task_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event.get("event_type") == "task_semantic_check" for event in events)


def test_new_run_removes_old_semantic_reports_but_keeps_checkpoints(tmp_path):
    """新运行清理过时语义终态，不影响候选 checkpoint。"""
    for name in ("intent_semantic_check.json", "task_semantic_check.json", "checkpoint.pt"):
        (tmp_path / name).write_text("old", encoding="utf-8")
    TrainingOrchestrator._clear_stale_run_outputs(tmp_path)
    assert not (tmp_path / "intent_semantic_check.json").exists()
    assert not (tmp_path / "task_semantic_check.json").exists()
    assert (tmp_path / "checkpoint.pt").exists()
