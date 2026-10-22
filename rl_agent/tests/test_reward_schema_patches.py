"""奖励 JSON 类型错误也只请求出错字段，原始有效合同不交给模型整份覆盖。"""
import copy
import json

import pytest

from rl_training_agent.providers.bailian_glm import BailianGLMProvider
from rl_training_agent.providers.errors import ProviderResponseError
from rl_training_agent.providers.opencli_chatgpt import OpenCLIChatGPTWebProvider
from rl_training_agent.providers.schema_repair import normalize_reward_response
from rl_training_agent.schemas.agent_workflow import TaskRewardBundle
from rl_training_agent.schemas.experiments import ConversationHandle
from rl_training_agent.settings import BailianSettings, OpenCLISettings

from test_reward_advisory_normalization import advisory_bundle


def provider_fixture(kind, original, handler, tmp_path, monkeypatch, retries=1):
    prompts = []

    def send(prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return json.dumps(original, ensure_ascii=False)
        request = json.loads(prompt.split('SCHEMA_PATCH_REQUEST: ')[1].splitlines()[0])
        assert '只返回 RewardSchemaFieldRepair 字段补丁' in prompt
        return json.dumps(handler(request, prompt), ensure_ascii=False)

    if kind == 'opencli':
        provider = OpenCLIChatGPTWebProvider(OpenCLISettings(max_retries=retries), record_dir=tmp_path)
        monkeypatch.setattr(provider, 'new_conversation', lambda *args: ConversationHandle(
            conversation_id='schema-patch', title_hint='fixture'))
        monkeypatch.setattr(provider, '_send_model_prompt', lambda conversation, prompt, *args: send(prompt))
        call = lambda: provider._request_model('完整用户任务', TaskRewardBundle, 'fixture')
    else:
        provider = BailianGLMProvider(BailianSettings(max_retries=0), record_dir=tmp_path)
        monkeypatch.setattr(provider, '_completion', lambda prompt, title: send(prompt))
        call = lambda: provider.request_model('完整用户任务', TaskRewardBundle, 'fixture')
    return call, prompts


@pytest.mark.parametrize('kind', ['opencli', 'bailian'])
@pytest.mark.parametrize('missing', [True, False])
def test_schema_requests_only_missing_or_invalid_field_and_merges_locally(kind, missing, tmp_path, monkeypatch):
    original, _ = normalize_reward_response(advisory_bundle(), TaskRewardBundle)
    if missing:
        del original['reward_plans'][0]['terms'][0]['purpose']
    else:
        original['reward_plans'][0]['terms'][0]['purpose'] = None
    before = copy.deepcopy(original)
    path = ['reward_plans', 0, 'terms', 0, 'purpose']

    def patch(request, prompt):
        assert request['allowed_patches'] == [{'op': 'add' if missing else 'replace', 'path': path}]
        return {'base_sha256': request['base_sha256'], 'patches': [
            {'op': 'add' if missing else 'replace', 'path': path, 'value': '惩罚非足端碰撞'}]}

    call, prompts = provider_fixture(kind, original, patch, tmp_path, monkeypatch)
    result = call()
    expected = copy.deepcopy(original)
    expected['reward_plans'][0]['terms'][0]['purpose'] = '惩罚非足端碰撞'
    assert result == TaskRewardBundle.parse_obj(expected)
    assert original == before and len(prompts) == 2
    assert list(tmp_path.rglob('*schema_error_patches*.json'))


@pytest.mark.parametrize('kind', ['opencli', 'bailian'])
@pytest.mark.parametrize('bad', ['unrelated_field', 'stale_hash'])
def test_schema_patch_cannot_change_valid_fields_or_use_old_source(kind, bad, tmp_path, monkeypatch):
    original, _ = normalize_reward_response(advisory_bundle(), TaskRewardBundle)
    del original['reward_plans'][0]['terms'][0]['purpose']

    def patch(request, prompt):
        if bad == 'unrelated_field':
            edits = [{'op': 'replace', 'path': ['reward_plans', 0, 'terms', 0, 'weight'], 'value': -99}]
        else:
            edits = [{'op': 'add', 'path': ['reward_plans', 0, 'terms', 0, 'purpose'], 'value': '说明'}]
        return {'base_sha256': '0' * 64 if bad == 'stale_hash' else request['base_sha256'], 'patches': edits}

    call, prompts = provider_fixture(kind, original, patch, tmp_path, monkeypatch)
    with pytest.raises(ProviderResponseError):
        call()
    assert len(prompts) == 2


def test_web_schema_repair_keeps_partial_progress_and_recomputes_permissions(tmp_path, monkeypatch):
    original, _ = normalize_reward_response(advisory_bundle(), TaskRewardBundle)
    del original['reward_plans'][0]['terms'][0]['purpose']
    del original['reward_plans'][0]['design_rationale']
    requests = []

    def patch(request, prompt):
        requests.append(request)
        if len(requests) == 1:
            edit = {'op': 'add', 'path': ['reward_plans', 0, 'terms', 0, 'purpose'], 'value': '已补齐用途'}
        else:
            assert len(request['allowed_patches']) == 1
            source = json.loads(prompt.split('Invalid JSON response to repair:\n')[-1])
            assert source['reward_plans'][0]['terms'][0]['purpose'] == '已补齐用途'
            edit = {'op': 'add', 'path': ['reward_plans', 0, 'design_rationale'], 'value': ['保留其他内容']}
        return {'base_sha256': request['base_sha256'], 'patches': [edit]}

    call, prompts = provider_fixture('opencli', original, patch, tmp_path, monkeypatch, retries=2)
    result = call()
    assert len(prompts) == 3 and requests[0]['base_sha256'] != requests[1]['base_sha256']
    assert result.reward_plans[0].terms[0].purpose == '已补齐用途'


@pytest.mark.parametrize('kind', ['opencli', 'bailian'])
def test_legacy_full_response_cannot_discard_valid_rewards_under_container_error(kind, tmp_path, monkeypatch):
    original, _ = normalize_reward_response(advisory_bundle(), TaskRewardBundle)
    original['reward_plans'][0]['terms'] *= 2
    replacement = copy.deepcopy(original)
    replacement['reward_plans'][0]['terms'] = []
    call, prompts = provider_fixture(kind, original, lambda *args: replacement, tmp_path, monkeypatch)
    with pytest.raises(ProviderResponseError):
        call()
    assert len(prompts) == 2
