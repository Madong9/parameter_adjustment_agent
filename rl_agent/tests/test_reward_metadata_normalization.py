"""验证奖励版本标签和设计说明的无损格式转换，不访问模型或 GPU。"""
import copy
import json

import pytest
from pydantic import BaseModel, ValidationError

from rl_training_agent.providers.errors import ProviderResponseError
from rl_training_agent.providers.opencli_chatgpt import OpenCLIChatGPTWebProvider
from rl_training_agent.providers.schema_repair import (
    normalize_metric_comparisons,
    normalize_reward_response,
    repair_preservation_errors,
)
from rl_training_agent.schemas.agent_workflow import TaskRewardBundle
from rl_training_agent.schemas.experiments import ConversationHandle
from rl_training_agent.schemas.rewards import RewardPlan

from test_schema_repair import bundle


def metadata_bundle():
    """构造本轮同类三候选回复，执行字段均来自完整离线夹具。"""
    payload, _ = normalize_metric_comparisons(bundle(), TaskRewardBundle)
    template = payload['reward_plans'][0]
    template['terms'][0]['parameters'] = {'force_threshold': 20.0}
    template['curriculum'] = [{
        'name': 'warmup', 'start_iteration': 0, 'end_iteration': 300,
        'parameter_changes': {'command_scale': 0.5},
    }]
    payload['reward_plans'] = [copy.deepcopy(template) for _ in range(3)]
    for index, plan in enumerate(payload['reward_plans'], 1):
        plan['version'] = 'candidate-%02d' % index
        plan['design_rationale'] = '  完整候选 %d 的设计说明。\n保留原始空白与措辞。  ' % index
    return payload


def test_three_candidates_parse_once_without_rewriting_execution_fields(tmp_path, monkeypatch):
    """六处元数据表示错误本地转换，一次发送即可返回且保持全部执行合同。"""
    payload = metadata_bundle()
    before = copy.deepcopy(payload)
    expected = copy.deepcopy(payload)
    for index, plan in enumerate(expected['reward_plans'], 1):
        plan['version'] = index
        plan['design_rationale'] = [plan['design_rationale']]

    provider = OpenCLIChatGPTWebProvider(record_dir=tmp_path)
    conversation = ConversationHandle(conversation_id='fixture', title_hint='fixture')
    calls = []
    monkeypatch.setattr(provider, 'new_conversation', lambda *args: conversation)
    monkeypatch.setattr(provider, '_send_model_prompt',
                        lambda *args: calls.append(args) or json.dumps(payload))

    result = provider._request_model('完整任务合同', TaskRewardBundle, 'fixture')
    assert len(calls) == 1
    assert payload == before
    assert result.dict() == TaskRewardBundle.parse_obj(expected).dict()

    audit = json.loads(next(tmp_path.rglob('schema_normalization_*.json')).read_text())
    assert len(audit['adjustments']) == 6
    for index in range(3):
        for field in ('version', 'design_rationale'):
            adjustment = next(item for item in audit['adjustments']
                              if item['path'] == ['reward_plans', index, field])
            assert adjustment['original'] == before['reward_plans'][index][field]
            assert adjustment['normalized'] == expected['reward_plans'][index][field]


@pytest.mark.parametrize('label,number', [
    ('candidate-1', 1),
    ('candidate-01', 1),
    ('candidate-000000001', 1),
    ('candidate-999999999', 999999999),
])
def test_standalone_plan_normalizes_explicit_version_and_parent_with_audit(label, number):
    """单个 RewardPlan 使用同样的严格转换，原说明与父版本数字逐项保留。"""
    payload = metadata_bundle()['reward_plans'][0]
    payload['version'] = label
    payload['parent_version'] = 'candidate-000000007'
    before = copy.deepcopy(payload)
    normalized, adjustments = normalize_reward_response(payload, RewardPlan)

    assert payload == before
    assert normalized['version'] == number
    assert normalized['parent_version'] == 7
    assert normalized['design_rationale'] == [before['design_rationale']]
    assert RewardPlan.parse_obj(normalized).version == number
    assert {tuple(item['path']) for item in adjustments} == {
        ('version',), ('parent_version',), ('design_rationale',),
    }
    for item in adjustments:
        assert item['original'] == before[item['path'][0]]
        assert item['normalized'] == normalized[item['path'][0]]


@pytest.mark.parametrize('label', [
    'candidate-0', 'candidate-000000000', 'candidate--1', 'candidate-1000000000',
    'candidate-1.0', 'candidate-1e2', 'candidate-01-old', 'candidate-01/candidate-02',
    'candidate-01 ', ' candidate-01', 'Candidate-01', 'v1', 'candidate-１',
])
def test_unknown_ambiguous_zero_or_negative_version_labels_remain_invalid(label):
    """只有正十进制候选标签可转换，不对未知或含糊版本猜测编号。"""
    payload = metadata_bundle()['reward_plans'][0]
    payload['version'] = label
    payload['design_rationale'] = ['已有完整说明']
    before = copy.deepcopy(payload)
    normalized, adjustments = normalize_reward_response(payload, RewardPlan)
    assert normalized == before and payload == before and not adjustments
    with pytest.raises(ProviderResponseError, match='version'):
        OpenCLIChatGPTWebProvider().parse_json_response(json.dumps(payload), RewardPlan)


def test_invalid_parent_version_is_not_guessed():
    """可选父版本的未知标签同样交给 Schema 拒绝，不能改为默认空值。"""
    payload = metadata_bundle()['reward_plans'][0]
    payload.update(version=1, parent_version='candidate-0', design_rationale=['完整说明'])
    normalized, adjustments = normalize_reward_response(payload, RewardPlan)
    assert normalized['parent_version'] == 'candidate-0' and not adjustments
    with pytest.raises(ProviderResponseError, match='parent_version'):
        OpenCLIChatGPTWebProvider().parse_json_response(json.dumps(payload), RewardPlan)


@pytest.mark.parametrize('rationale', ['', '   ', '\n\t'])
def test_empty_rationale_strings_are_not_filled_or_wrapped(rationale):
    """空设计说明继续拒绝，不把空文本包装成已满足必填说明的列表。"""
    payload = metadata_bundle()['reward_plans'][0]
    payload.update(version=1, design_rationale=rationale)
    normalized, adjustments = normalize_reward_response(payload, RewardPlan)
    assert normalized['design_rationale'] == rationale and not adjustments
    with pytest.raises(ProviderResponseError, match='design_rationale'):
        OpenCLIChatGPTWebProvider().parse_json_response(json.dumps(payload), RewardPlan)


def test_normalization_is_idempotent_and_scoped_to_reward_metadata():
    """重复处理不重复审计，任务、奖励参数及未知同名字段均保留原表示。"""
    payload = metadata_bundle()
    payload['task_spec']['version'] = 'candidate-02'
    payload['task_spec']['design_rationale'] = '任务额外字段不是奖励说明'
    payload['extra'] = {'version': 'candidate-03', 'design_rationale': '额外对象'}
    payload['reward_plans'][0]['terms'][0]['parameters'].update(
        version='candidate-04', design_rationale='执行参数不得转换')
    before = copy.deepcopy(payload)
    normalized, adjustments = normalize_reward_response(payload, TaskRewardBundle)
    repeated, repeated_adjustments = normalize_reward_response(normalized, TaskRewardBundle)
    assert len(adjustments) == 6
    assert repeated == normalized and not repeated_adjustments
    assert normalized['task_spec'] == before['task_spec']
    assert normalized['extra'] == before['extra']
    assert normalized['reward_plans'][0]['terms'] == before['reward_plans'][0]['terms']
    assert payload == before


def test_unrelated_schema_does_not_normalize_matching_field_names():
    """转换范围由合同类型确定，不根据其他 Schema 的同名字段扩散规则。"""
    class OtherReply(BaseModel):
        """模拟恰好具有同名字符串字段的独立回复。"""
        version: str
        design_rationale: str

    payload = {'version': 'candidate-01', 'design_rationale': '保持字符串'}
    normalized, adjustments = normalize_reward_response(payload, OtherReply)
    assert normalized == payload and not adjustments
    assert OtherReply.parse_obj(normalized).dict() == payload


@pytest.mark.parametrize('path,replacement', [
    (('reward_plans', 0, 'terms', 0, 'weight'), -7.0),
    (('reward_plans', 0, 'terms', 0, 'parameters', 'force_threshold'), 21.0),
    (('reward_plans', 0, 'terms', 0, 'active_phases'), ['flight']),
    (('reward_plans', 0, 'terms', 0, 'purpose'), '重写目的'),
    (('reward_plans', 0, 'curriculum', 0, 'name'), 'changed'),
    (('reward_plans', 0, 'curriculum', 0, 'parameter_changes', 'command_scale'), 0.9),
    (('reward_plans', 0, 'objective_links'), []),
    (('reward_plans', 0, 'behavior_links'), []),
    (('task_spec', 'success_metrics', 0, 'value'), 999.0),
])
def test_metadata_normalization_keeps_repair_guard_for_execution_values(path, replacement):
    """元数据表示兼容不会放松已有奖励、课程、映射或验收阈值的修复防护。"""
    original = metadata_bundle()
    with pytest.raises(ValidationError) as caught:
        TaskRewardBundle.parse_obj(original)
    errors = caught.value.errors()
    assert len(errors) == 6
    normalized, _ = normalize_reward_response(original, TaskRewardBundle)
    assert not repair_preservation_errors(original, normalized, TaskRewardBundle, errors)

    changed = copy.deepcopy(normalized)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    regressions = repair_preservation_errors(original, changed, TaskRewardBundle, errors)
    assert regressions
    assert any('.'.join(map(str, path)) in item for item in regressions)
