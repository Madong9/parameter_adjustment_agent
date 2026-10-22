from types import SimpleNamespace

import pytest

from rl_training_agent.agents.reward_reviewer import RewardReviewAgent
from rl_training_agent.environment.inspector import EnvironmentInspector
from rl_training_agent.orchestration.orchestrator import TrainingOrchestrator
from rl_training_agent.providers.mock_provider import MockLLMReasoningProvider
from rl_training_agent.rewards.validator import RewardValidationError
from rl_training_agent.schemas.decisions import TrainingDiagnosis
from rl_training_agent.schemas.rewards import RewardPlan, RewardTerm, RewardObjectiveLink, RewardBehaviorLink
from rl_training_agent.schemas.task import TaskSpec
from rl_training_agent.settings import load_settings
from rl_training_agent.utils.io import read_json


def setup_redesign(tmp_path, proposals):
    settings = load_settings()
    settings.reward_design_review_retries = 1
    manifest = EnvironmentInspector(settings.training_root).inspect('go2')
    bundle = MockLLMReasoningProvider(1).design_task_and_rewards('向前行走', 'go2', manifest.dict())
    task = TaskSpec.parse_obj(bundle['task_spec'])
    plan = RewardPlan.parse_obj(bundle['reward_plans'][0])
    plan.success_metrics = [m.copy(deep=True) for m in task.success_metrics]
    plan.objective_links = [RewardObjectiveLink(metric='tracking_error',
        reward_terms=['tracking_lin_vel'], rationale='追踪目标速度', counterexamples=['原地静止不能通过']),
        RewardObjectiveLink(metric='fall_rate', reward_terms=['orientation'],
        rationale='保持稳定姿态', counterexamples=['倒地不能通过'])]
    plan.behavior_links = [RewardBehaviorLink(behavior=b.name,
        reward_terms=['tracking_lin_vel' if b.name == 'forward_progress' else 'orientation'],
        rationale=b.description, counterexamples=['静止或倒地不代表完成']) for b in task.required_behaviors]
    selected = {'id': 'parent', 'dir': tmp_path / 'parent', 'plan': plan}
    selected['dir'].mkdir()
    diagnosis = TrainingDiagnosis(diagnosis=[], evidence=[], confidence=.9, decision='revise_reward',
        checkpoint_strategy='restart_from_scratch', suggestions=['核对命令与奖励方向，再调整权重'])
    calls = []
    def redesign(payload):
        calls.append(dict(payload))
        result = plan.copy(deep=True)
        result.terms[0].weight *= 1.5
        if proposals[len(calls) - 1] == 'weaken':
            result.success_metrics[0].value *= .5
        return result
    def repair(payload):
        calls.append(dict(payload))
        target = payload['targets'][0]
        edits = []
        for grant in target['allowed_patches']:
            if grant['path'][0] == 'success_metrics' and 'required_value' in grant:
                value = grant['required_value']
                if proposals[len(calls) - 1] == 'weaken':
                    value *= .5
                edits.append(dict(op=grant['op'], path=grant['path'], value=value))
        return {'repairs': [{'candidate_index': target['candidate_index'],
                            'base_sha256': target['base_sha256'], 'patches': edits}]}
    orchestrator = object.__new__(TrainingOrchestrator)
    orchestrator.settings = settings
    orchestrator.provider = SimpleNamespace(redesign_rewards=redesign, repair_reward_plans=repair)
    orchestrator.reward_reviewer = RewardReviewAgent()
    return orchestrator, task, selected, diagnosis, manifest, calls


def test_failure_analysis_is_followed_by_independent_complete_reward_design(tmp_path):
    orchestrator, task, selected, diagnosis, manifest, calls = setup_redesign(tmp_path, ['valid'])
    original = selected['plan'].json()
    revised, audit = orchestrator._redesign_failed_rewards(task, tmp_path, selected, diagnosis, 1, manifest)
    assert len(calls) == 1
    assert calls[0]['failure_analysis']['suggestions'] == diagnosis.suggestions
    assert calls[0]['immutable_task'] == task.dict()
    assert revised.parent_version == selected['plan'].version
    assert revised.version == selected['plan'].version + 1
    assert selected['plan'].json() == original
    assert read_json(tmp_path / 'reward_redesign/round_01/approved_reward_plan.json')['version'] == revised.version
    assert audit


def test_redesign_retries_with_feedback_without_weakening_acceptance(tmp_path):
    orchestrator, task, selected, diagnosis, manifest, calls = setup_redesign(tmp_path, ['weaken', 'valid'])
    revised, _ = orchestrator._redesign_failed_rewards(task, tmp_path, selected, diagnosis, 1, manifest)
    assert len(calls) == 2
    assert '验收指标' in calls[1]['validation_feedback']
    assert revised.success_metrics == task.success_metrics
    assert calls[1]['targets'][0]['reward_plan']['terms'][0]['weight'] == selected['plan'].terms[0].weight * 1.5
    assert revised.terms[0].weight == selected['plan'].terms[0].weight * 1.5
    assert calls[1]['targets'][0]['reward_plan']['version'] == revised.version


def test_repeated_invalid_design_stops_without_creating_training_candidate(tmp_path):
    orchestrator, task, selected, diagnosis, manifest, calls = setup_redesign(tmp_path, ['weaken', 'weaken'])
    with pytest.raises(RewardValidationError, match='补丁|验收指标'):
        orchestrator._redesign_failed_rewards(task, tmp_path, selected, diagnosis, 1, manifest)
    assert len(calls) == 2
    assert not (tmp_path / 'candidates').exists()
    assert not (tmp_path / 'reward_redesign/round_01/approved_reward_plan.json').exists()


@pytest.mark.parametrize('remaining', [0, 100])
def test_failed_policy_respects_budget_and_saves_analysis(tmp_path, remaining):
    from rl_training_agent.orchestration.budget import BudgetTracker
    from rl_training_agent.schemas.metrics import EvaluationResult
    orchestrator, task, selected, diagnosis, manifest, calls = setup_redesign(tmp_path, [])
    selected['checkpoint'] = tmp_path / 'checkpoint.pt'
    diagnosis.decision = 'failed'
    evaluation = EvaluationResult(completed=False, task_metrics_passed=False,
        hard_constraints_passed=False, visual_alignment_passed=False, metrics=[], violations=[], conflicts=[])
    orchestrator._write_working_memory = lambda *args, **kwargs: None
    orchestrator._evaluate_round = lambda *args, **kwargs: {'diagnosis': diagnosis, 'evaluation': evaluation}
    trained = []
    def train(*args, **kwargs):
        trained.append(args[4].decision)
        raise RewardValidationError('模拟审核阻塞，禁止启动 PPO')
    orchestrator._train_revision = train
    orchestrator._finalize_loop = lambda *args: {'state': args[7].value, 'reason': args[8]}
    budget = BudgetTracker(max_iterations=100, used_iterations=100 - remaining, max_revisions=1)
    result = orchestrator._evaluate(task, tmp_path, None, selected, False, budget)
    assert result['state'] == 'HUMAN_REVIEW'
    assert read_json(tmp_path / 'failure_analysis/round_01.json')['suggestions'] == diagnosis.suggestions
    assert trained == (['restart'] if remaining else [])
    assert budget.used_revisions == 0
    assert budget.used_iterations == 100 - remaining


def test_gpt_failure_analysis_requires_actionable_suggestions():
    from rl_training_agent.schemas.decisions import GPTTrainingDiagnosis
    values = dict(diagnosis=[], evidence=[], decision='failed', confidence=.9,
                  checkpoint_strategy='restart_from_scratch')
    with pytest.raises(ValueError, match='suggestions'):
        GPTTrainingDiagnosis(**values)
    assert GPTTrainingDiagnosis(**values, suggestions=['对齐碰撞奖励与验收接触集合']).suggestions
