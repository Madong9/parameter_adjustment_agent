"""奖励格式重试只采用错误字段，避免模型重写整份方案造成反复停止。"""
import copy
import json

import pytest
from pydantic import ValidationError

from rl_training_agent.providers.errors import ProviderResponseError
from rl_training_agent.providers.opencli_chatgpt import OpenCLIChatGPTWebProvider
from rl_training_agent.providers.schema_repair import merge_schema_repairs, normalize_reward_response
from rl_training_agent.schemas.agent_workflow import TaskRewardBundle
from rl_training_agent.schemas.experiments import ConversationHandle
from rl_training_agent.schemas.rewards import RewardPlan
from rl_training_agent.settings import OpenCLISettings

from test_schema_repair import bundle
from test_reward_advisory_normalization import advisory_bundle


def missing_purpose_bundle():
    payload, _ = normalize_reward_response(bundle(), TaskRewardBundle)
    del payload['reward_plans'][0]['terms'][0]['purpose']
    return payload


def errors_for(payload, schema=TaskRewardBundle):
    with pytest.raises(ValidationError) as caught:
        schema.parse_obj(payload)
    return caught.value.errors()


def provider_responses(monkeypatch, tmp_path, responses):
    provider = OpenCLIChatGPTWebProvider(OpenCLISettings(max_retries=len(responses) - 1),
                                        record_dir=tmp_path)
    replies = iter(json.dumps(item, ensure_ascii=False) for item in responses)
    prompts = []
    monkeypatch.setattr(provider, 'new_conversation', lambda *args: ConversationHandle(
        conversation_id='field-repair', title_hint='fixture'))

    def send(conversation, prompt, *args):
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr(provider, '_send_model_prompt', send)
    return provider, prompts


def test_reward_retry_recovers_missing_field_without_accepting_rewrites(monkeypatch, tmp_path):
    original = missing_purpose_bundle()
    repaired = copy.deepcopy(original)
    plan = repaired['reward_plans'][0]
    plan['terms'][0]['purpose'] = '惩罚非足端碰撞'
    plan['terms'][0]['weight'] = -9
    plan['behavior_links'] = []
    plan['curriculum'] = []
    plan['design_rationale'] = ['改写有效说明']
    repaired['reward_hacking_risks'] = ['改写有效风险']
    repaired['task_spec']['safety_constraints'][0]['required'] = False
    expected = copy.deepcopy(original)
    expected['reward_plans'][0]['terms'][0]['purpose'] = '惩罚非足端碰撞'
    provider, prompts = provider_responses(monkeypatch, tmp_path, [original, repaired])

    result = provider._request_model('保留完整合同', TaskRewardBundle, 'fixture')

    assert result.dict() == TaskRewardBundle.parse_obj(expected).dict()
    assert len(prompts) == 2
    audit = json.loads(next(tmp_path.rglob('schema_field_repair_*.json')).read_text())
    assert audit['effective_response'] == expected
    assert audit['adjustments'] == [{'path': ['reward_plans', 0, 'terms', 0, 'purpose'],
                                    'normalized': '惩罚非足端碰撞', 'original_present': False}]
    assert any('required' in item for item in audit['ignored_repair_changes'])
    assert not audit['remaining_validation_errors']


def test_partial_field_repair_becomes_next_source(monkeypatch, tmp_path):
    original = missing_purpose_bundle()
    del original['reward_plans'][0]['design_rationale']
    partial = copy.deepcopy(original)
    partial['reward_plans'][0]['terms'][0]['purpose'] = '已补全用途'
    partial['reward_plans'][0]['terms'][0]['weight'] = -9
    final = copy.deepcopy(partial)
    final['reward_plans'][0]['design_rationale'] = ['补全设计说明']
    final['reward_hacking_risks'] = []
    expected = copy.deepcopy(original)
    expected['reward_plans'][0]['terms'][0]['purpose'] = '已补全用途'
    expected['reward_plans'][0]['design_rationale'] = ['补全设计说明']
    provider, prompts = provider_responses(monkeypatch, tmp_path, [original, partial, final])

    result = provider._request_model('保留完整合同', TaskRewardBundle, 'fixture')

    assert result.dict() == TaskRewardBundle.parse_obj(expected).dict()
    retained = json.loads(prompts[2].split('Invalid JSON response to repair:\n')[-1])
    assert retained['reward_plans'][0]['terms'][0] == expected['reward_plans'][0]['terms'][0]
    assert retained['reward_hacking_risks'] == original['reward_hacking_risks']
    assert len(list(tmp_path.rglob('schema_field_repair_*.json'))) == 2


def test_field_repair_preserves_source_and_applied_normalization_audit(monkeypatch, tmp_path):
    original = advisory_bundle()
    del original['reward_plans'][0]['design_rationale']
    repaired = copy.deepcopy(original)
    repaired['reward_plans'][0]['design_rationale'] = '保留完整设计说明'
    repaired['reward_hacking_risks'] = []
    provider, _ = provider_responses(monkeypatch, tmp_path, [original, repaired])
    result = provider._request_model('完整合同', TaskRewardBundle, 'fixture')
    assert [json.loads(item) for item in result.termination_suggestions] == original['termination_suggestions']
    audit = json.loads(next(tmp_path.rglob('schema_field_repair_*.json')).read_text())
    assert len(audit['source_normalizations']) == 3
    assert len(audit['repair_normalizations']) == 1
    normalization = json.loads(next(tmp_path.rglob('schema_normalization_*.json')).read_text())
    assert len(normalization['adjustments']) == 4


def test_item_patch_audits_normalization_of_its_parent_list(monkeypatch, tmp_path):
    original, _ = normalize_reward_response(bundle(), TaskRewardBundle)
    original['reward_plans'][0]['design_rationale'] = [None]
    repaired = copy.deepcopy(original)
    repaired['reward_plans'][0]['design_rationale'] = '完整说明'
    repaired['reward_hacking_risks'] = []
    provider, _ = provider_responses(monkeypatch, tmp_path, [original, repaired])
    result = provider._request_model('完整合同', TaskRewardBundle, 'fixture')
    assert result.reward_plans[0].design_rationale == ['完整说明']
    audit = json.loads(next(tmp_path.rglob('schema_field_repair_*.json')).read_text())
    assert audit['adjustments'][0]['path'] == ['reward_plans', 0, 'design_rationale', 0]
    assert audit['repair_normalizations'][0]['path'] == ['reward_plans', 0, 'design_rationale']
    assert audit['repair_normalizations'][0]['original'] == '完整说明'


@pytest.mark.parametrize('change', ['reorder', 'remove', 'identity'])
def test_array_position_is_not_enough_to_identify_a_repair(change):
    original = missing_purpose_bundle()
    term = copy.deepcopy(original['reward_plans'][0]['terms'][0])
    term.update(name='torque', implementation='registry:torque', purpose='力矩安全')
    original['reward_plans'][0]['terms'].append(term)
    repaired = copy.deepcopy(original)
    terms = repaired['reward_plans'][0]['terms']
    terms[0]['purpose'] = '不能移植的用途'
    if change == 'reorder':
        terms.reverse()
    elif change == 'remove':
        terms.pop()
    else:
        repaired['reward_plans'][0]['version'] += 10

    merged, patches = merge_schema_repairs(original, repaired, TaskRewardBundle, errors_for(original))

    assert merged == original and patches == []


def test_container_validation_does_not_authorize_deleting_existing_rewards():
    original, _ = normalize_reward_response(bundle(), TaskRewardBundle)
    original['reward_plans'][0]['terms'] *= 2  # 名称重复触发整个 terms 的校验错误。
    repaired = copy.deepcopy(original)
    repaired['reward_plans'][0]['terms'] = []
    merged, patches = merge_schema_repairs(original, repaired, TaskRewardBundle, errors_for(original))
    assert merged == original and not patches


@pytest.mark.parametrize('duplicate_identity', [False, True])
def test_missing_or_duplicate_identity_cannot_transfer_fields_between_items(duplicate_identity):
    original, _ = normalize_reward_response(bundle(), TaskRewardBundle)
    if duplicate_identity:
        original['reward_plans'][0]['behavior_links'] = [
            {'behavior': 'standing', 'reward_terms': ['collision'], 'counterexamples': []},
            {'behavior': 'standing', 'reward_terms': ['other'], 'counterexamples': [], 'rationale': '其他解释'},
        ]
        field, missing_field = 'behavior_links', 'rationale'
    else:
        original['reward_plans'][0]['known_conflicts'] = [
            {'terms': ['a', 'b'], 'description': 'A/B'},
            {'terms': ['c', 'd'], 'description': 'C/D', 'mitigation': '降低 C'},
        ]
        field, missing_field = 'known_conflicts', 'mitigation'
    repaired = copy.deepcopy(original)
    items = repaired['reward_plans'][0][field]
    items[0][missing_field] = '原项应有说明'
    items.reverse()
    merged, patches = merge_schema_repairs(original, repaired, TaskRewardBundle, errors_for(original))
    assert merged == original and not patches


def test_unresolved_invalid_field_still_blocks_after_ignored_rewrites(monkeypatch, tmp_path):
    original = missing_purpose_bundle()
    repaired = copy.deepcopy(original)
    repaired['reward_plans'][0]['terms'][0]['purpose'] = None
    repaired['reward_hacking_risks'] = []
    provider, prompts = provider_responses(monkeypatch, tmp_path, [original, repaired])
    with pytest.raises(ProviderResponseError, match='purpose'):
        provider._request_model('保留完整合同', TaskRewardBundle, 'fixture')
    assert len(prompts) == 2
    assert not list(tmp_path.rglob('schema_field_repair_*.json'))


def test_standalone_plan_uses_same_field_repair_and_preserves_inputs():
    original = missing_purpose_bundle()['reward_plans'][0]
    repaired = copy.deepcopy(original)
    repaired['terms'][0].update(purpose='碰撞约束', weight=-99)
    before, after = copy.deepcopy(original), copy.deepcopy(repaired)
    merged, patches = merge_schema_repairs(original, repaired, RewardPlan, errors_for(original, RewardPlan))
    assert RewardPlan.parse_obj(merged).terms[0].purpose == '碰撞约束'
    assert merged['terms'][0]['weight'] == original['terms'][0]['weight']
    assert len(patches) == 1 and original == before and repaired == after
