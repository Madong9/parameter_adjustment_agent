"""离线验证奖励提示词绑定当前任务并显式列出必填结构。"""
import json
from pathlib import Path

import pytest

from rl_training_agent.agents.prompt_compiler import RewardPromptCompiler
from rl_training_agent.orchestration.orchestrator import TrainingOrchestrator
from rl_training_agent.schemas.agent_workflow import TaskIntentSpec, TaskRewardBundle


def _section(prompt, name):
    """解析提示词内单行结构化字段。"""
    prefix = name + ": "
    return json.loads(next(line[len(prefix):] for line in prompt.splitlines()
                           if line.startswith(prefix)))


def _compiler():
    """使用真实奖励设计模板构建离线编译器。"""
    template = Path(__file__).parents[1] / "rl_training_agent/prompts/reward_design_agent.md"
    return RewardPromptCompiler(template)


def _intent(robot="go2", instruction="向前腾空翻跟头"):
    """构造不依赖模型或环境的当前任务意图。"""
    return TaskIntentSpec(original_instruction=instruction, robot=robot,
                          action_name="前空翻", normalized_goal=instruction)


@pytest.mark.parametrize("robot,instruction", [
    ("go2", "向前腾空翻跟头"), ("g1", "向前腾空翻跟头"),
    ("go2", "原地跳跃"), ("h1", "向前行走"),
])
def test_default_identity_matches_orchestrator(robot, instruction):
    """缺少显式任务标识时提示词和调度器使用同一稳定身份。"""
    result = _compiler().compile(_intent(robot, instruction), {}, 2)
    identity = _section(result["prompt"], "CURRENT_TASK_IDENTITY")
    assert identity == {"task_id": TrainingOrchestrator._task_id(instruction, robot),
                        "robot": robot, "original_instruction": instruction}


def test_historical_g1_identity_does_not_replace_current_go2():
    """检索历史 G1 前翻记录时当前权威身份仍绑定本次 Go2 请求。"""
    instruction = "向前腾空翻跟头"
    historical_id = TrainingOrchestrator._task_id(instruction, "g1")
    context = {"rag": {"hits": [{"task_id": historical_id, "robot": "g1",
                                   "original_instruction": instruction}]}}
    result = _compiler().compile(_intent(), context, 3)
    prompt = result["prompt"]
    identity = _section(prompt, "CURRENT_TASK_IDENTITY")
    assert identity["task_id"] == TrainingOrchestrator._task_id(instruction, "go2")
    assert identity["robot"] == "go2"
    assert _section(prompt, "CONTEXT_SNAPSHOT") == context
    assert "不得复用为当前身份" in prompt
    assert "所有 reward_plans[*].task_id 也必须一致" in prompt


def test_explicit_task_id_is_authoritative():
    """已分配任务标识可由调用方传入而不被重新计算。"""
    result = _compiler().compile(_intent(), {"task_id": "historical-task"}, 1,
                                 task_id="task-current-review")
    assert _section(result["prompt"], "CURRENT_TASK_IDENTITY")["task_id"] == "task-current-review"


def test_empty_explicit_task_id_is_rejected():
    """拒绝把空的显式标识发布成权威任务身份。"""
    with pytest.raises(ValueError, match="task_id must not be empty"):
        _compiler().compile(_intent(), {}, 1, task_id=" ")


def test_required_fields_include_actual_missing_reply_fields():
    """真实缺失字段、奖励用途和课程迭代边界在简明清单中显式出现。"""
    prompt = _compiler().compile(_intent(), {}, 1)["prompt"]
    fields = _section(prompt, "SCHEMA_REQUIRED_FIELDS")
    assert set(fields["TaskSpec"]) >= {
        "phases", "required_observations", "required_sensors", "safety_constraints",
        "visual_evaluation_requirements", "task_id", "robot", "original_instruction"}
    assert "purpose" in fields["RewardTerm"]
    assert fields["CurriculumStage"] == ["name", "start_iteration", "end_iteration"]
    schema = _section(prompt, "OUTPUT_JSON_SCHEMA")
    assert fields["TaskSpec"] == schema["definitions"]["TaskSpec"]["required"]
    assert "必须是对象列表" in prompt
    example = _section(prompt, "SAFETY_CONSTRAINT_FORMAT_EXAMPLE")
    assert isinstance(example["value"], (int, float))
    assert example["unit"] == "count" and example["operator"] == "<="
    assert "不要为了补齐结构补造阈值" in prompt


def test_required_fields_follow_schema_additions(monkeypatch):
    """Schema 后续新增必填字段时摘要自动更新，避免提示词手工列表漂移。"""
    schema = TaskRewardBundle.schema()
    modified = json.loads(json.dumps(schema))
    modified["definitions"]["TaskSpec"]["required"].append("future_required_evidence")
    monkeypatch.setattr(TaskRewardBundle, "schema", classmethod(lambda cls: modified))
    prompt = _compiler().compile(_intent(), {}, 1)["prompt"]
    assert "future_required_evidence" in _section(prompt, "SCHEMA_REQUIRED_FIELDS")["TaskSpec"]
    assert "future_required_evidence" in _section(prompt, "OUTPUT_JSON_SCHEMA")["definitions"]["TaskSpec"]["required"]
