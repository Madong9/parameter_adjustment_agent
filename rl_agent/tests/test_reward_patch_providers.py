"""各模型入口只请求局部补丁并保留奖励角色路由。"""
from types import SimpleNamespace

import pytest

from rl_training_agent.providers.bailian_glm import BailianGLMProvider
from rl_training_agent.providers.errors import ProviderTimeout
from rl_training_agent.providers.fallback import FallbackProvider
from rl_training_agent.providers.multi_agent import MultiAgentProvider
from rl_training_agent.providers.opencli_chatgpt import OpenCLIChatGPTWebProvider
from rl_training_agent.schemas.reward_repairs import RewardRepairBundle


@pytest.mark.parametrize('provider,method', [(OpenCLIChatGPTWebProvider, '_request_model'),
                                           (BailianGLMProvider, 'request_model')])
def test_model_repair_uses_patch_schema_and_complete_authorization_context(monkeypatch, provider, method):
    instance = provider()
    calls = []
    payload = {'immutable_task': {'task_id': 'current'}, 'targets': [
        {'candidate_index': 2, 'base_sha256': 'a' * 64, 'reward_plan': {'version': 2},
         'allowed_patches': [{'op': 'remove', 'path': ['curriculum', 1, 'parameter_changes', 'bad']}]}]}
    monkeypatch.setattr(instance, method, lambda prompt, schema, title:
                        calls.append((prompt, schema, title)) or RewardRepairBundle(repairs=[]))
    assert instance.repair_reward_plans(payload).repairs == []
    prompt, schema, _ = calls[0]
    assert schema is RewardRepairBundle
    assert '只输出 RewardRepairBundle 字段补丁' in prompt
    assert 'base_sha256' in prompt and 'allowed_patches' in prompt and 'current' in prompt
    assert 'REPAIR_REQUEST: ' in prompt


def test_multiagent_routes_patch_to_reward_role():
    instance = object.__new__(MultiAgentProvider)
    calls = []
    instance.reward_agent = SimpleNamespace(repair_reward_plans=lambda payload: calls.append(payload) or 'patch')
    payload = {'targets': []}
    assert instance.repair_reward_plans(payload) == 'patch' and calls == [payload]


def test_fallback_keeps_same_patch_request_and_never_calls_full_design():
    calls = []
    def unavailable(payload):
        calls.append(('primary', payload))
        raise ProviderTimeout('offline fixture')
    primary = SimpleNamespace(repair_reward_plans=unavailable)
    fallback = SimpleNamespace(repair_reward_plans=lambda payload: calls.append(('fallback', payload)) or 'patch')
    instance = FallbackProvider(primary, fallback)
    payload = {'targets': [{'candidate_index': 1}]}
    assert instance.repair_reward_plans(payload) == 'patch'
    assert calls == [('primary', payload), ('fallback', payload)]
