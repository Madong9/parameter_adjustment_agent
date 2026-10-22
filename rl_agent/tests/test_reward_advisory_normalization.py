"""验证说明对象可还原地保存为文本，执行合同继续严格校验。"""
import copy
import json

import pytest
from pydantic import ValidationError

from rl_training_agent.providers.errors import ProviderResponseError
from rl_training_agent.providers.opencli_chatgpt import OpenCLIChatGPTWebProvider
from rl_training_agent.providers.schema_repair import normalize_reward_response, repair_preservation_errors
from rl_training_agent.schemas.agent_workflow import TaskRewardBundle
from rl_training_agent.schemas.experiments import ConversationHandle
from rl_training_agent.schemas.rewards import RewardPlan

from test_schema_repair import bundle


def advisory_bundle():
    """构造完整合同和本次同类终止建议，加入未知键以验证不丢信息。"""
    payload, _ = normalize_reward_response(bundle(), TaskRewardBundle)
    payload['termination_suggestions'] = [
        {'name': 'forbidden_body_contact', 'condition': '非足端身体接触超过安全阈值'},
        {'name': 'joint_limit_violation', 'condition': '关节位置超过限制'},
        {'name': 'torque_limit_violation', 'condition': '力矩超过限制',
         'extra': {'threshold': 1.25, 'verified': False, 'evidence': [None, '待验证']}},
    ]
    return payload


def test_advice_objects_parse_in_one_request_and_preserve_all_information(tmp_path, monkeypatch):
    """三条建议无需重写候选即可解析，审计原值且不自动启用执行终止条件。"""
    payload = advisory_bundle()
    before = copy.deepcopy(payload)
    provider = OpenCLIChatGPTWebProvider(record_dir=tmp_path)
    conversation = ConversationHandle(conversation_id='advice-fixture', title_hint='fixture')
    calls = []
    monkeypatch.setattr(provider, 'new_conversation', lambda *args: conversation)
    monkeypatch.setattr(provider, '_send_model_prompt',
                        lambda *args: calls.append(args) or json.dumps(payload))
    result = provider._request_model('完整任务合同', TaskRewardBundle, 'fixture')
    assert len(calls) == 1 and payload == before
    assert [json.loads(item) for item in result.termination_suggestions] == before['termination_suggestions']
    assert [plan.dict() for plan in result.reward_plans] == [
        RewardPlan.parse_obj(plan).dict() for plan in before['reward_plans']]
    assert result.task_spec.dict() == TaskRewardBundle.parse_obj(
        dict(before, termination_suggestions=[])).task_spec.dict()
    audit = json.loads(next(tmp_path.rglob('schema_normalization_*.json')).read_text())
    assert audit['scope'] == 'lossless_reward_response_types_v3'
    assert len(audit['adjustments']) == 3
    for index, adjustment in enumerate(audit['adjustments']):
        assert adjustment['path'] == ['termination_suggestions', index]
        assert adjustment['original'] == before['termination_suggestions'][index]
        assert json.loads(adjustment['normalized']) == adjustment['original']


@pytest.mark.parametrize('field', ['termination_suggestions', 'reward_hacking_risks'])
def test_only_object_entries_are_serialized_in_place_and_idempotently(field):
    """两个顶层说明列表保留现有文本、项目数量及顺序，重复处理不重复转换。"""
    payload, _ = normalize_reward_response(bundle(), TaskRewardBundle)
    advice = {'name': '保持信息', 'condition': '阈值由原合同确定', 'unknown': {'中文': ['a', 2]}}
    payload[field] = ['已有说明', advice, '  保留空白\n']
    before = copy.deepcopy(payload)
    normalized, adjustments = normalize_reward_response(payload, TaskRewardBundle)
    assert payload == before
    assert normalized[field][0] == before[field][0] and normalized[field][2] == before[field][2]
    assert len(normalized[field]) == len(before[field])
    assert json.loads(normalized[field][1]) == advice
    assert len(adjustments) == 1 and adjustments[0]['path'] == [field, 1]
    repeated, repeated_adjustments = normalize_reward_response(normalized, TaskRewardBundle)
    assert repeated == normalized and not repeated_adjustments


@pytest.mark.parametrize('item', [
    None, ['不是一个说明对象'], {'number': float('nan')}, {'number': float('inf')},
    {'number': -float('inf')}, {'not_json': {1, 2}}, {1: '非字符串键'}, {'tuple': (1, 2)},
])
def test_non_json_or_unexpected_advice_is_not_converted(item):
    """拒绝非 JSON、非有限和不可还原结构，不用字符串转换掩盖未知格式。"""
    payload, _ = normalize_reward_response(bundle(), TaskRewardBundle)
    payload['termination_suggestions'] = [item]
    normalized, adjustments = normalize_reward_response(payload, TaskRewardBundle)
    assert type(normalized['termination_suggestions'][0]) is type(item)
    assert not adjustments
    if item is None or isinstance(item, list) or isinstance(item, dict):
        with pytest.raises(ValidationError):
            TaskRewardBundle.parse_obj(normalized)


def test_executable_terminations_safety_and_nested_names_are_not_serialized():
    """说明兼容不会扩散至执行条件、安全阈值、参数或未知嵌套同名字段。"""
    payload = advisory_bundle()
    payload['reward_plans'][0]['terminations'] = [{'name': 'invalid', 'condition': {'threshold': 0.}}]
    payload['reward_plans'][0]['terms'][0].setdefault('parameters', {})['termination_suggestions'] = [{'threshold': 7}]
    payload['extra'] = {'termination_suggestions': [{'condition': '未知对象'}]}
    payload['task_spec']['safety_constraints'] = [{'name': '不可补造', 'condition': '无数值'}]
    before = copy.deepcopy(payload)
    normalized, adjustments = normalize_reward_response(payload, TaskRewardBundle)
    assert normalized['reward_plans'] == before['reward_plans']
    assert normalized['task_spec'] == before['task_spec']
    assert normalized['extra'] == before['extra']
    assert len(adjustments) == 3
    with pytest.raises(ProviderResponseError, match='condition'):
        OpenCLIChatGPTWebProvider().parse_json_response(json.dumps(payload), TaskRewardBundle)
    standalone = before['reward_plans'][0]
    assert normalize_reward_response(standalone, RewardPlan) == (standalone, [])


@pytest.mark.parametrize('path,replacement', [
    (('reward_plans', 0, 'terms', 0, 'weight'), -8.0),
    (('reward_plans', 0, 'terms'), []),
    (('reward_plans', 0, 'behavior_links'), []),
    (('reward_plans', 0, 'design_rationale', 0), '重写设计说明'),
    (('task_spec', 'safety_constraints', 0, 'value'), 9.0),
    (('reward_hacking_risks', 0), '重写有效风险说明'),
])
def test_advice_conversion_keeps_guard_for_existing_valid_execution_fields(path, replacement):
    """格式规范化后的有效字段仍受防退化检查保护，不能删除安全项或改阈值。"""
    normalized, _ = normalize_reward_response(advisory_bundle(), TaskRewardBundle)
    changed = copy.deepcopy(normalized)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    regressions = repair_preservation_errors(normalized, changed, TaskRewardBundle, [])
    assert any('.'.join(map(str, path)) in item for item in regressions)
