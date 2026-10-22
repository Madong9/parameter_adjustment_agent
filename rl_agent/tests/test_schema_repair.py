"""验证无损指标结构转换、修复退化防护与任务身份，不访问模型或 GPU。"""
import copy
import json
from typing import List

import pytest
from pydantic import BaseModel, ValidationError

from rl_training_agent.providers.errors import ProviderResponseError
from rl_training_agent.providers.mock_provider import MockLLMReasoningProvider
from rl_training_agent.providers.opencli_chatgpt import OpenCLIChatGPTWebProvider
from rl_training_agent.providers.schema_repair import normalize_metric_comparisons, repair_preservation_errors
from rl_training_agent.schemas.agent_workflow import TaskRewardBundle
from rl_training_agent.schemas.experiments import ConversationHandle
from rl_training_agent.schemas.rewards import CurriculumStage
from rl_training_agent.schemas.task import MetricThreshold
from rl_training_agent.settings import OpenCLISettings


def bundle():
    """构造完整离线候选，并模拟本轮三条格式明确的安全比较式。"""
    payload = MockLLMReasoningProvider(1).design_task_and_rewards('稳定站立', 'go2', {})
    payload['reward_plans'][0]['terms'] = [{'name': 'collision', 'implementation': 'registry:collision',
        'purpose': '惩罚非足端碰撞', 'weight': -1.}]
    payload['task_spec']['safety_constraints'] = ['forbidden_body_contact <= 1 N',
        'joint_limit_violations == 0', 'torque_limit_violations == 0']
    return payload


def test_complete_bundle_with_explicit_safety_strings_needs_no_model_retry(tmp_path, monkeypatch):
    """本轮同类回复无损转换后通过完整 Schema，一次回复即可使用且保留审计记录。"""
    payload = bundle()
    before = copy.deepcopy(payload)
    provider = OpenCLIChatGPTWebProvider(record_dir=tmp_path)
    conversation = ConversationHandle(conversation_id='fixture', title_hint='fixture')
    calls = []
    monkeypatch.setattr(provider, 'new_conversation', lambda *args: conversation)
    monkeypatch.setattr(provider, '_send_model_prompt', lambda *args: calls.append(args) or json.dumps(payload))
    result = provider._request_model('完整任务', TaskRewardBundle, 'fixture')
    assert len(calls) == 1 and payload == before
    assert result.task_spec.safety_constraints[0].dict()['value'] == 1
    assert result.task_spec.safety_constraints[0].unit == 'N'
    assert result.task_spec.safety_constraints[1].unit == ''
    assert len(result.reward_plans[0].terms) == len(payload['reward_plans'][0]['terms'])
    audit = json.loads(next(tmp_path.rglob('schema_normalization_*.json')).read_text())
    assert len(audit['adjustments']) == 3
    assert audit['adjustments'][0]['original'] == 'forbidden_body_contact <= 1 N'


@pytest.mark.parametrize('value,expected', [
    ('forbidden_body_contact<=1N', {'name': 'forbidden_body_contact', 'operator': '<=', 'value': 1., 'unit': 'N'}),
    ('final_pitch_angle >= 6.283 rad', {'name': 'final_pitch_angle', 'operator': '>=', 'value': 6.283, 'unit': 'rad'}),
    ('velocity > -1.2e-3 m/s', {'name': 'velocity', 'operator': '>', 'value': -.0012, 'unit': 'm/s'}),
    ('joint_limit_violations == 0', {'name': 'joint_limit_violations', 'operator': '==', 'value': 0., 'unit': ''}),
])
def test_comparison_conversion_preserves_explicit_value_and_unit(value, expected):
    """有限数值、比较符和显式单位逐项保留，单位缺失保持空而不是推测。"""
    payload = bundle()
    payload['task_spec']['safety_constraints'] = [value]
    result, adjustments = normalize_metric_comparisons(payload, TaskRewardBundle)
    assert result['task_spec']['safety_constraints'] == [expected]
    assert len(adjustments) == 1


@pytest.mark.parametrize('value', ['body_contact <= NaN N', 'body_contact < Infinity',
    'body_contact <= 1e999 N', 'body_contact <= 1 N or torque < 5',
    'body_contact <= 1 N，且无碰撞', '无身体碰撞', 'body_contact <=', 'body_contact = 1 N'])
def test_ambiguous_or_nonfinite_constraints_remain_invalid(value):
    """模糊、多条件、缺数值及非有限比较式保持失败，不补造任务安全阈值。"""
    payload = bundle()
    payload['task_spec']['safety_constraints'] = [value]
    result, adjustments = normalize_metric_comparisons(payload, TaskRewardBundle)
    assert result['task_spec']['safety_constraints'] == [value] and not adjustments
    with pytest.raises(ProviderResponseError):
        OpenCLIChatGPTWebProvider().parse_json_response(json.dumps(payload), TaskRewardBundle)


def test_other_fields_and_missing_contract_are_not_autofilled():
    """转换限定于已有任务指标，缺失安全合同和奖励 purpose 仍由严格 Schema 拒绝。"""
    payload = bundle()
    del payload['task_spec']['safety_constraints']
    del payload['reward_plans'][0]['terms'][0]['purpose']
    before = copy.deepcopy(payload)
    normalized, adjustments = normalize_metric_comparisons(payload, TaskRewardBundle)
    assert normalized == before and not adjustments
    with pytest.raises(ProviderResponseError, match='purpose'):
        OpenCLIChatGPTWebProvider().parse_json_response(json.dumps(payload), TaskRewardBundle)


class Item(BaseModel):
    """具有稳定身份与说明的最小奖励项。"""
    name: str
    purpose: str


class Reply(BaseModel):
    """模拟有必填待修字段、候选数组和已正确字段的结构化回复。"""
    identity: str
    items: List[Item]
    memo: str = 'default'
    approval: bool


@pytest.mark.parametrize('change', ['drop_field', 'drop_items', 'change_identity', 'change_threshold'])
def test_valid_schema_repair_cannot_discard_existing_content(change, monkeypatch):
    """即使新回复通过 Schema，也不能用空候选、改身份或改已合法值来减少错误数。"""
    original = {'identity': 'current-go2', 'items': [{'name': 'rotation', 'purpose': 'net rotation >= 6'}],
                'memo': '完整来源'}
    rejected = dict(copy.deepcopy(original), approval=True)
    if change == 'drop_field':
        del rejected['memo']
    elif change == 'drop_items':
        rejected['items'] = []
    elif change == 'change_identity':
        rejected['identity'] = 'old-g1'
    else:
        rejected['items'][0]['purpose'] = 'net rotation >= 1'
    corrected = dict(copy.deepcopy(original), approval=True)
    responses = iter(map(json.dumps, [original, rejected, corrected]))
    prompts = []
    provider = OpenCLIChatGPTWebProvider(OpenCLISettings(max_retries=2))
    monkeypatch.setattr(provider, 'new_conversation', lambda *args: ConversationHandle(conversation_id='fixture', title_hint='fixture'))

    def send(conversation, prompt, *args):
        """逐轮返回完整来源、退化结果与真正的最小修复，记录每轮修复请求。"""
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(provider, '_send_model_prompt', send)
    assert provider._request_model('原始任务', Reply, 'fixture').dict() == corrected
    assert len(prompts) == 3
    assert json.loads(prompts[2].split('Invalid JSON response to repair:\n')[-1]) == original
    assert 'REJECTED_REPAIR_REGRESSIONS' in prompts[2]


def test_repair_limit_reports_best_candidate_not_degraded_missing_fields(monkeypatch):
    """重试耗尽时报告来源中的实际剩余问题，并说明拒绝退化，不改用残缺结果。"""
    original = {'identity': 'current', 'items': [{'name': 'rotation', 'purpose': '保持原值'}]}
    responses = iter([json.dumps(original), '{}'])
    provider = OpenCLIChatGPTWebProvider(OpenCLISettings(max_retries=1))
    monkeypatch.setattr(provider, 'new_conversation', lambda *args: ConversationHandle(conversation_id='fixture', title_hint='fixture'))
    monkeypatch.setattr(provider, '_send_model_prompt', lambda *args: next(responses))
    with pytest.raises(ProviderResponseError, match='已保留最完整来源') as caught:
        provider._request_model('任务', Reply, 'fixture')
    assert '1 validation error' in str(caught.value)


def test_repair_can_change_only_invalid_value_and_preserve_equivalent_numbers():
    """错误位置可以修复，已合法字段的等价类型表示不会导致假退化。"""
    class Count(BaseModel):
        """模拟一个待修值和一个合法数值。"""
        count: int
        value: float
    original = {'count': 'bad', 'value': '1.0'}
    try:
        Count.parse_obj(original)
    except ValidationError as error:
        errors = error.errors()
    assert not repair_preservation_errors(original, {'count': 2, 'value': 1.}, Count, errors)


def test_repair_cannot_disable_safety_via_an_implicit_default():
    """原始 required 默认为真，不能通过新增 required=false 隐藏已有硬约束。"""
    class Report(BaseModel):
        """用缺失 purpose 触发修复，安全项本身已合法。"""
        safety: List[MetricThreshold]
        purpose: str
    original = {'safety': [{'name': 'forbidden_body_contact', 'operator': '<=', 'value': 1., 'unit': 'N'}]}
    try:
        Report.parse_obj(original)
    except ValidationError as error:
        errors = error.errors()
    changed = copy.deepcopy(original)
    changed['purpose'] = '修复说明'
    changed['safety'][0]['required'] = False
    assert any('required' in error for error in repair_preservation_errors(original, changed, Report, errors))


def test_repair_cannot_append_items_to_a_valid_list():
    """增加未声明的候选或安全指标也属于重设计，不能混入格式修复。"""
    original = {'identity': 'current', 'items': [{'name': 'rotation', 'purpose': '旋转'}]}
    try:
        Reply.parse_obj(original)
    except ValidationError as error:
        errors = error.errors()
    changed = dict(copy.deepcopy(original), approval=True)
    changed['items'].append({'name': 'new', 'purpose': '新增候选'})
    assert repair_preservation_errors(original, changed, Reply, errors)


def test_curriculum_root_error_allows_only_boundary_correction():
    """阶段顺序校验涉及起止两个字段，允许只修正该错误，其他课程语义继续冻结。"""
    class Report(BaseModel):
        """构造包含一个边界颠倒课程的回复。"""
        curriculum: List[CurriculumStage]
    original = {'curriculum': [{'name': 'warmup', 'start_iteration': 100, 'end_iteration': 0}]}
    try:
        Report.parse_obj(original)
    except ValidationError as error:
        errors = error.errors()
    fixed = {'curriculum': [{'name': 'warmup', 'start_iteration': 100, 'end_iteration': 200}]}
    assert not repair_preservation_errors(original, fixed, Report, errors)
    fixed['curriculum'][0]['name'] = 'different-action'
    assert repair_preservation_errors(original, fixed, Report, errors)
