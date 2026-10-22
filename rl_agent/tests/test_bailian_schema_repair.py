"""验证百炼奖励回复使用相同的无损规范和有效字段保护，不调用外部模型。"""
import copy
import json

import pytest

from rl_training_agent.providers.bailian_glm import BailianGLMProvider
from rl_training_agent.providers.errors import ProviderResponseError
from rl_training_agent.providers.schema_repair import normalize_reward_response
from rl_training_agent.schemas.agent_workflow import TaskRewardBundle
from rl_training_agent.settings import BailianSettings

from test_reward_advisory_normalization import advisory_bundle


def scripted_provider(monkeypatch, tmp_path, responses):
    """保留真实请求记录流程，用本地信封代替 HTTP 请求。"""
    provider = BailianGLMProvider(BailianSettings(max_retries=0), tmp_path)
    replies = iter(responses)
    prompts = []

    def completion(body):
        prompts.append(body['messages'][0]['content'])
        reply = next(replies)
        content = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        return json.dumps({'choices': [{'message': {'content': content}}]})

    monkeypatch.setattr(provider, '_http_request', completion)
    return provider, prompts


def test_advisory_objects_need_one_request_and_preserve_audit(monkeypatch, tmp_path):
    """三条终止建议及风险对象不触发整份重试，原文和可还原转换均有记录。"""
    original = advisory_bundle()
    original['reward_hacking_risks'].append({'risk': '潜在漏洞', 'evidence': ['尚未验证', None]})
    before = copy.deepcopy(original)
    provider, prompts = scripted_provider(monkeypatch, tmp_path, [original])
    result = provider.request_model('完整任务合同', TaskRewardBundle, 'advice-fixture')
    normalized, adjustments = normalize_reward_response(original, TaskRewardBundle)
    assert result == TaskRewardBundle.parse_obj(normalized)
    assert original == before and len(prompts) == 1
    assert [json.loads(item) for item in result.termination_suggestions] == original['termination_suggestions']
    audit = json.loads(next(tmp_path.glob('*_schema_normalization.json')).read_text())
    assert audit == {'scope': 'lossless_reward_response_types_v3', 'adjustments': adjustments}
    assert json.loads(next(tmp_path.glob('*_response.txt')).read_text()) == original


def test_minimal_repair_keeps_original_prompt_and_structured_errors(monkeypatch, tmp_path):
    """修复缺失 purpose 时携带原始任务约束，只补充该字段。"""
    repaired = advisory_bundle()
    original = copy.deepcopy(repaired)
    del original['reward_plans'][0]['terms'][0]['purpose']
    provider, prompts = scripted_provider(monkeypatch, tmp_path, [original, repaired])
    result = provider.request_model('原始任务：保留机器人身份及全部安全阈值', TaskRewardBundle, 'repair-fixture')
    assert result.reward_plans[0].terms[0].purpose == repaired['reward_plans'][0]['terms'][0]['purpose']
    assert len(prompts) == 2 and prompts[1].startswith(prompts[0])
    errors = json.loads(prompts[1].split('VALIDATION_ERRORS: ')[1].splitlines()[0])
    assert errors[0]['loc'] == ['reward_plans', 0, 'terms', 0, 'purpose']
    assert json.loads(prompts[1].split('Invalid JSON response to repair:\n')[-1]) == normalize_reward_response(original, TaskRewardBundle)[0]
    assert len(list(tmp_path.glob('*_schema_normalization.json'))) == 2


@pytest.mark.parametrize('path,replacement', [
    (('reward_hacking_risks', 0), '重新改写已有风险'),
    (('task_spec', 'task_id'), 'different-task'),
    (('task_spec', 'safety_constraints', 0, 'value'), 99.),
    (('task_spec', 'safety_constraints', 0, 'required'), False),
])
def test_repair_restores_valid_fields_and_uses_only_missing_purpose(monkeypatch, tmp_path, path, replacement):
    """重写结果中只提取缺失的用途，身份、风险、安全阈值仍使用原始内容。"""
    repaired = advisory_bundle()
    expected, _ = normalize_reward_response(repaired, TaskRewardBundle)
    original = copy.deepcopy(repaired)
    del original['reward_plans'][0]['terms'][0]['purpose']
    target = repaired
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    normalized, _ = normalize_reward_response(repaired, TaskRewardBundle)
    TaskRewardBundle.parse_obj(normalized)
    provider, prompts = scripted_provider(monkeypatch, tmp_path, [original, repaired])
    result = provider.request_model('完整任务', TaskRewardBundle, 'repair-fixture')
    assert result == TaskRewardBundle.parse_obj(expected)
    audit = json.loads(next(tmp_path.glob('*_schema_field_repair.json')).read_text())
    assert any('.'.join(map(str, path)) in item for item in audit['ignored_repair_changes'])
    assert audit['effective_response'] == expected
    assert len(prompts) == 2


def test_repair_cannot_remove_rewards_when_required_purpose_is_missing(monkeypatch, tmp_path):
    original = advisory_bundle()
    del original['reward_plans'][0]['terms'][0]['purpose']
    repaired = copy.deepcopy(original)
    repaired['reward_plans'][0]['terms'] = []
    provider, prompts = scripted_provider(monkeypatch, tmp_path, [original, repaired])
    with pytest.raises(ProviderResponseError, match='拒绝删除或改写有效字段'):
        provider.request_model('完整任务', TaskRewardBundle, 'reject-fixture')
    assert len(prompts) == 2


def test_executable_termination_errors_still_stop_after_one_repair(monkeypatch, tmp_path):
    """说明格式转换不能掩盖真正的执行终止条件错误，也不会无限重试。"""
    original = advisory_bundle()
    original['reward_plans'][0]['terminations'] = [{'name': 'collision', 'condition': {'threshold': 2}}]
    provider, prompts = scripted_provider(monkeypatch, tmp_path, [original, original])
    with pytest.raises(ProviderResponseError, match='condition'):
        provider.request_model('完整任务', TaskRewardBundle, 'invalid-fixture')
    assert len(prompts) == 2


def test_invalid_json_can_still_be_repaired(monkeypatch, tmp_path):
    """没有可解析来源时，保留一次完整 JSON 修复能力。"""
    provider, prompts = scripted_provider(monkeypatch, tmp_path, ['不完整 JSON {', advisory_bundle()])
    result = provider.request_model('完整任务', TaskRewardBundle, 'syntax-fixture')
    assert result.reward_plans and len(prompts) == 2
    assert 'invalid_json' in prompts[1]
