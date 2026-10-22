"""验证所有奖励设计入口使用来自当前 Schema 的字段类型摘要。"""
import copy
import json
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from rl_training_agent.agents.prompt_compiler import RewardPromptCompiler
from rl_training_agent.agents.schema_contract import render_schema_contract
from rl_training_agent.providers.opencli_chatgpt import OpenCLIChatGPTWebProvider
from rl_training_agent.schemas.agent_workflow import TaskIntentSpec, TaskRewardBundle
from rl_training_agent.schemas.rewards import RewardPlan


def contract_types(prompt):
    """读取单行结构化摘要，以类型语义而非提示词全文作为断言依据。"""
    prefix = 'SCHEMA_FIELD_TYPES: '
    return json.loads(next(line[len(prefix):] for line in prompt.splitlines() if line.startswith(prefix)))


def test_real_contract_distinguishes_advice_from_execution_and_nullable_from_optional():
    """对容易混淆的建议、执行条件、版本和可空范围给出准确类型。"""
    summary = contract_types(render_schema_contract(TaskRewardBundle))
    assert summary['TaskRewardBundle']['termination_suggestions'] == 'optional array<string>'
    assert summary['TaskRewardBundle']['reward_hacking_risks'] == 'optional array<string>'
    assert summary['RewardPlan']['terminations'] == 'optional array<TerminationSpec>'
    assert summary['RewardPlan']['version'] == 'required integer'
    assert summary['RewardPlan']['parent_version'] == 'optional integer | null'
    assert summary['RewardPlan']['design_rationale'] == 'required array<string>'
    assert summary['TaskSpec']['safety_constraints'] == 'required array<MetricThreshold>'
    assert summary['RewardTerm']['expected_raw_range'] == 'optional tuple<number,number> | null'
    assert summary['RewardTerm']['parameters'] == 'optional object'


def test_summary_covers_all_actual_fields_and_future_schema_changes(monkeypatch):
    """摘要覆盖所有必填与默认字段，Schema 新增字段无需再补手工清单。"""
    schema = copy.deepcopy(TaskRewardBundle.schema())
    schema['definitions']['RewardPlan']['properties']['future_advice'] = {'type': 'array', 'items': {'type': 'string'}}
    schema['definitions']['RewardPlan']['required'].append('future_advice')
    monkeypatch.setattr(TaskRewardBundle, 'schema', classmethod(lambda cls: schema))
    summary = contract_types(render_schema_contract(TaskRewardBundle))
    assert summary['RewardPlan']['future_advice'] == 'required array<string>'
    for name, definition in dict(schema['definitions'], TaskRewardBundle=schema).items():
        assert set(summary[name]) == set(definition['properties'])


def test_aliases_nested_optional_items_and_enums_use_live_model_contract():
    """保留字段别名、列表内可空类型和枚举，默认容器不会被误标为 null。"""
    class Child(BaseModel):
        """模拟独立结构化对象。"""
        value: int

    class Reply(BaseModel):
        """模拟别名、枚举、可空容器项目和字典。"""
        advice: List[str] = Field(default_factory=list, alias='notes')
        items: List[Optional[int]]
        child: Optional[Child] = None
        mode: Literal['a', 'b'] = 'a'
        lookup: Dict[str, int] = Field(default_factory=dict)

    summary = contract_types(render_schema_contract(Reply))
    assert summary['Reply']['notes'] == 'optional array<string>'
    assert summary['Reply']['items'] == 'required array<integer | null>'
    assert summary['Reply']['child'] == 'optional Child | null'
    assert summary['Reply']['mode'] == 'optional string enum["a","b"]'
    assert summary['Reply']['lookup'] == 'optional object<string,integer>'


def test_initial_legacy_and_redesign_prompts_all_include_schema_types(monkeypatch):
    """初次、兼容和重设计入口都携带完整类型摘要，无需访问浏览器。"""
    template = Path(__file__).parents[1] / 'rl_training_agent/prompts/reward_design_agent.md'
    intent = TaskIntentSpec(original_instruction='站稳', robot='go2', action_name='stand', normalized_goal='站稳')
    initial = RewardPromptCompiler(template).compile(intent, {}, 3)['prompt']
    assert contract_types(initial) == contract_types(render_schema_contract(TaskRewardBundle))
    provider = OpenCLIChatGPTWebProvider()
    calls = []

    class Reply:
        """最小离线模型返回值，避免实际模型请求。"""
        def dict(self):
            """模拟已经解析的结果。"""
            return {}

    def request(prompt, schema, title):
        """捕获两个真实入口发送的提示词。"""
        calls.append((prompt, schema))
        return Reply()

    monkeypatch.setattr(provider, '_request_model', request)
    provider.design_task_and_rewards('站稳', 'go2', {})
    provider.redesign_rewards({'previous_plan': {'task_id': 'current'}})
    assert len(calls) == 2
    for prompt, schema in calls:
        assert contract_types(prompt) == contract_types(render_schema_contract(schema))
    assert calls[0][1] is TaskRewardBundle and calls[1][1] is RewardPlan
