from types import SimpleNamespace

from rl_training_agent.settings import Settings, load_settings
from rl_training_agent.providers.multi_agent import MultiAgentProvider
from rl_training_agent.providers.registry import ProviderRegistry


def test_production_reward_design_and_revision_use_chatgpt(monkeypatch):
    settings = load_settings()
    assert settings.reward_designer_provider == settings.diagnosis_provider == 'opencli'
    assert Settings().reward_designer_provider == Settings().diagnosis_provider == 'opencli'
    created = []
    calls = []
    def create(self, name, settings, path):
        created.append(name)
        def call(method):
            return lambda *args: calls.append((name, method)) or {'provider': name}
        return SimpleNamespace(design_task_bundle=call('bundle'),
            design_task_and_rewards=call('legacy'), diagnose_training=call('diagnosis'),
            redesign_rewards=call('redesign'))
    monkeypatch.setattr(ProviderRegistry, 'create', create)
    provider = MultiAgentProvider(settings)
    provider.design_task_bundle('compiled prompt')
    provider.design_task_and_rewards('walk', 'go2', {})
    provider.diagnose_training({})
    provider.redesign_rewards({})
    assert calls == [('opencli', 'bundle'), ('opencli', 'legacy'), ('opencli', 'diagnosis'), ('opencli', 'redesign')]
    assert 'bailian' not in created and 'doubao' not in created
