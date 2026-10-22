"""奖励审核重试只调用补丁接口，保留有效候选及上轮局部修复进度。"""
import json
from pathlib import Path
from types import SimpleNamespace

from rl_training_agent.agents.reward_reviewer import RewardReviewAgent
from rl_training_agent.environment.inspector import EnvironmentInspector
from rl_training_agent.orchestration.orchestrator import TrainingOrchestrator
from rl_training_agent.providers.mock_provider import MockLLMReasoningProvider
from rl_training_agent.rewards.compiler import RewardCompiler
from rl_training_agent.schemas.rewards import CurriculumStage
from rl_training_agent.settings import load_settings

from test_failure_reward_redesign import setup_redesign
from test_front_flip_backend import contract
from test_reliability_fixes import settings_for


def test_mixed_candidates_request_only_rejected_fields_and_do_not_modify_good_candidates(tmp_path):
    intent, task, original = contract()
    plans = [original.copy(deep=True) for _ in range(3)]
    for index, plan in enumerate(plans, 1):
        plan.version = index
    plans[2].curriculum = [CurriculumStage(name='learning', start_iteration=0, end_iteration=10,
        parameter_changes={'net_pitch_target': 6.28, 'base_height_target': .3})]
    before = [plan.dict() for plan in plans]
    settings = load_settings()
    manifest = EnvironmentInspector(settings.training_root).inspect('go2')
    capabilities = dict(manifest.dict(), require_objective_links=True)
    reviewer = RewardReviewAgent()
    review = reviewer.review(intent, task, plans, capabilities, ['记录风险'])
    assert review.passed_candidate_indexes == [1, 2]
    captured = []

    def repair(payload):
        captured.append(payload)
        assert [target['candidate_index'] for target in payload['targets']] == [3]
        target = payload['targets'][0]
        assert target['allowed_patches'] == [{
            'op': 'remove', 'path': ['curriculum', 0, 'parameter_changes', 'net_pitch_target']}]
        return {'repairs': [{'candidate_index': 3, 'base_sha256': target['base_sha256'],
            'patches': [{'op': 'remove', 'path': ['curriculum', 0, 'parameter_changes', 'net_pitch_target']}]}]}

    runner = object.__new__(TrainingOrchestrator)
    runner.settings = settings
    runner.provider = SimpleNamespace(repair_reward_plans=repair)
    revised, audit = runner._repair_reward_candidates(task, tmp_path, plans, review, manifest,
                                                    tmp_path / 'repairs', 1)
    assert len(captured) == 1 and len(audit['accepted']) == 1
    assert [plan.dict() for plan in revised[:2]] == before[:2]
    assert [plan.dict() for plan in plans] == before
    assert revised[2].curriculum[0].parameter_changes == {'base_height_target': .3}
    assert reviewer.review(intent, task, revised, capabilities, ['记录风险']).passed_candidate_indexes == [1, 2, 3]
    for index, plan in enumerate(revised):
        RewardCompiler(manifest).compile(plan, tmp_path / ('candidate_%d' % index), task=task)


def test_failure_redesign_retains_each_successful_partial_patch_and_uses_latest_proposal(tmp_path):
    runner, task, selected, diagnosis, manifest, _ = setup_redesign(tmp_path, [])
    runner.settings.reward_design_review_retries = 2
    full_requests, patch_requests = [], []
    proposal = selected['plan'].copy(deep=True)
    proposal.terms[0].weight = 201.
    proposal.success_metrics[0].value *= .5

    def redesign(payload):
        full_requests.append(payload)
        return proposal.copy(deep=True)

    def repair(payload):
        patch_requests.append(payload)
        target = payload['targets'][0]
        if len(patch_requests) == 1:
            edit = {'op': 'replace', 'path': ['terms', 0, 'weight'], 'value': 2.}
        else:
            assert target['reward_plan']['terms'][0]['weight'] == 2.
            assert not any(grant['path'] == ['terms', 0, 'weight'] for grant in target['allowed_patches'])
            edit = {'op': 'replace', 'path': ['success_metrics', 0, 'value'], 'value': task.success_metrics[0].value}
        return {'repairs': [{'candidate_index': 1, 'base_sha256': target['base_sha256'], 'patches': [edit]}]}

    runner.provider = SimpleNamespace(redesign_rewards=redesign, repair_reward_plans=repair)
    revised, _ = runner._redesign_failed_rewards(task, tmp_path, selected, diagnosis, 1, manifest)
    assert len(full_requests) == 1 and len(patch_requests) == 2
    assert revised.terms[0].weight == 2. and revised.success_metrics == task.success_metrics
    assert selected['plan'].terms[0].weight != 2.
    assert revised.version == selected['plan'].version + 1
    assert patch_requests[0]['targets'][0]['base_sha256'] != patch_requests[1]['targets'][0]['base_sha256']
    assert len(list((tmp_path / 'reward_redesign/round_01/field_repairs').glob('patch_audit_*.json'))) == 2


def test_initial_review_repairs_weight_before_compiler_and_never_regenerates_bundle(tmp_path):
    class WeightProvider(MockLLMReasoningProvider):
        designs = 0
        patch_requests = []

        def design_task_and_rewards(self, instruction, robot, capabilities):
            self.designs += 1
            result = super().design_task_and_rewards(instruction, robot, capabilities)
            result['reward_plans'][0]['terms'][0]['weight'] = 201.
            return result

        def repair_reward_plans(self, payload):
            self.patch_requests.append(payload)
            target = payload['targets'][0]
            assert target['allowed_patches'] == [{'op': 'replace', 'path': ['terms', 0, 'weight']}]
            return {'repairs': [{'candidate_index': 1, 'base_sha256': target['base_sha256'],
                                'patches': [{'op': 'replace', 'path': ['terms', 0, 'weight'], 'value': 1.}]}]}

    settings = settings_for(tmp_path)
    provider = WeightProvider(1)
    result = TrainingOrchestrator(settings, provider).train('稳定向前行走', 'go2', True)
    assert result['state'] == 'COMPLETED'
    assert provider.designs == 1 and len(provider.patch_requests) == 1
    root = settings.experiments_path / result['task_id']
    assert json.loads((root / 'reward_repairs/patch_audit_01.json').read_text())['accepted']
    assert result['used_revisions'] == 0
