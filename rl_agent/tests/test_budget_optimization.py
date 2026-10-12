"""验证耗尽预算的可解释提示、无模型恢复及冻结额度，不使用真实模型或训练。"""
import hashlib
from types import SimpleNamespace

import pytest

from rl_training_agent.orchestration.budget import BudgetTracker
from rl_training_agent.orchestration.orchestrator import TrainingOrchestrator
from rl_training_agent.orchestration.state_machine import AgentState, PersistentStateMachine
from rl_training_agent.schemas.agent_workflow import TaskIntentSpec
from rl_training_agent.schemas.decisions import TrainingDiagnosis
from rl_training_agent.schemas.experiments import ExperimentManifest
from rl_training_agent.schemas.metrics import EvaluationResult
from rl_training_agent.schemas.rewards import RewardPlan
from rl_training_agent.schemas.task import MetricThreshold, TaskSpec, TrainingBudget
from rl_training_agent.settings import Settings
from rl_training_agent.utils.io import atomic_write_text, read_json, write_json


def _budget_run(tmp_path, budget, *, dry_run=False):
    """只在临时目录创建合法双种子证据，禁止加载策略或构造真实 Provider。"""
    task_dir = tmp_path / 'task-budget'
    task_dir.mkdir()
    orchestrator = object.__new__(TrainingOrchestrator)
    orchestrator.settings = Settings(experiment_root=str(tmp_path),
        artifact_root=str(tmp_path / 'artifacts'), evaluation_seeds=[1, 2],
        rollouts_per_seed=1, memory_enabled=False, rag_enabled=False,
        max_total_iterations=12000, max_reward_revisions=3,
        exploration_max_iterations=12000, exploration_max_revisions=3)
    orchestrator.store = SimpleNamespace(task_dir=lambda task_id: task_dir,
        candidate_dir=lambda task_id, candidate: task_dir / 'candidates' / candidate)
    orchestrator._write_working_memory = lambda *args, **kwargs: None
    orchestrator._provider_call = lambda *args, **kwargs: pytest.fail('禁止请求真实或昂贵 Provider')
    orchestrator.controller = SimpleNamespace(continue_training=lambda *args, **kwargs:
        pytest.fail('禁止启动训练'))
    state = PersistentStateMachine(task_dir / 'state.json')
    state.start_new_run('run-budget')
    state.record.state = AgentState.HUMAN_REVIEW
    state._save()
    task = TaskSpec(task_id=task_dir.name, robot='go2', task_name='walk',
        original_instruction='follow command', normalized_description='follow command',
        initial_state='standing', required_behaviors=[], forbidden_behaviors=[], phases=[],
        required_observations=[], required_sensors=[], success_metrics=[
            MetricThreshold(name='tracking_error', operator='<=', value=.1)],
        safety_constraints=[], visual_evaluation_requirements=[],
        training_budget=TrainingBudget(max_total_iterations=3000, max_reward_revisions=1))
    write_json(task_dir / 'task_spec.json', task)
    write_json(task_dir / 'evaluation_protocol.json', {'run_id': 'run-budget',
        'dry_run': dry_run, 'seeds': [1, 2], 'rollouts_per_seed': 1,
        'counterfactual_enabled': True})
    write_json(task_dir / 'acceptance_contract.json', {'run_id': 'run-budget', 'robot': 'go2',
        'success_metrics': [item.dict() for item in task.success_metrics], 'safety_constraints': [],
        'task_identity': orchestrator._task_acceptance_identity(task)})
    directory = task_dir / 'candidates' / 'candidate-v3'
    directory.mkdir(parents=True)
    config = directory / 'config.yaml'
    atomic_write_text(config, '{}\n')
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    checkpoints = []
    for seed in [1, 2]:
        checkpoint = directory / 'checkpoints' / ('seed_%s' % seed) / 'model_700.pt'
        atomic_write_text(checkpoint, '仅供来源校验，禁止加载\n')
        checkpoints.append(checkpoint)
    plan = RewardPlan(task_id=task_dir.name, version=3, design_rationale=['临时测试'], terms=[])
    manifest = ExperimentManifest(experiment_id=directory.name, task_id=task_dir.name,
        config_hash=digest, reward_version=3, seed=1, robot='go2', training_command=[],
        training_result='dry_run_completed' if dry_run else 'completed',
        checkpoint=str(checkpoints[0].relative_to(directory)))
    write_json(directory / 'reward_plan.json', plan)
    write_json(directory / 'manifest.json', manifest)
    write_json(directory / 'compile_metadata.json', {'config_hash': digest})
    selected = {'id': directory.name, 'dir': directory, 'plan': plan, 'manifest': manifest,
        'checkpoint': checkpoints[0], 'checkpoints': checkpoints, 'checkpoint_seeds': [1, 2]}
    orchestrator._save_checkpoint_set(task_dir, selected, dry_run)
    history = [{'round': 2, 'experiment_id': selected['id'], 'reward_version': 3,
        'decision': 'human_review', 'completed': False, 'used_iterations': budget.used_iterations}]
    write_json(task_dir / 'loop_history.json', history)
    write_json(task_dir / 'loop_status.json', {'round': 2, 'phase': 'evaluated',
        'used_revisions': budget.used_revisions})
    # 旧摘要没有 max_iterations、remaining_revisions 等新字段，也必须兼容。
    summary = {'task_id': task_dir.name, 'state': 'HUMAN_REVIEW', 'result': 'human_review',
        'reason': '旧预算耗尽提示', 'selected_experiment': selected['id'],
        'checkpoint': 'final/checkpoint.pt', 'config': 'final/config.yaml',
        'reward_version': 3, 'dry_run': dry_run, 'evaluation_seeds': [1, 2],
        'used_iterations': budget.used_iterations, 'remaining_iterations': budget.remaining_iterations,
        'used_revisions': budget.used_revisions, 'max_revisions': budget.max_revisions,
        'loop_rounds': len(history), 'last_decision': 'human_review'}
    write_json(task_dir / 'summary.json', summary)
    orchestrator._persist_training_execution(task_dir, state, budget, phase='trained',
        dry_run=dry_run, experiment_id=selected['id'], parent_experiment=None,
        round_index=1, operations={selected['id'] + ':revision:seed:1': {
            'iterations': budget.used_iterations, 'status': 'completed'}})
    return orchestrator, task_dir, state, task, selected


@pytest.mark.parametrize('iterations,revisions,exhausted', [
    (1950, 0, False), (3000, 0, True), (1950, 1, True), (3000, 1, True)])
def test_budget_summary_has_frozen_limits_without_mutation(iterations, revisions, exhausted):
    """预算只提供训练资源摘要，任何一项到顶都禁止启动自动修订。"""
    budget = BudgetTracker(max_iterations=3000, used_iterations=iterations,
        max_revisions=1, used_revisions=revisions, reserve_for_revisions=True)
    before = budget.dict()
    fields = budget.summary_fields()
    assert fields['budget_exhausted'] is exhausted
    assert fields['max_iterations'] == 3000 and fields['max_revisions'] == 1
    assert fields['remaining_iterations'] == 3000 - iterations
    assert fields['remaining_revisions'] == 1 - revisions
    assert budget.dict() == before


def test_exploration_reservation_allocation_remains_unchanged():
    """候选筛选后剩余 2100 的三种子两轮额度仍为每种子 350，不自动扩容。"""
    budget = BudgetTracker(max_iterations=3000, used_iterations=900,
        max_revisions=1, reserve_for_revisions=True)
    assert budget.per_seed_allocation(1000, 3) == 350
    assert budget.used_iterations == 900


@pytest.mark.parametrize('stage', [AgentState.SMOKE_TRAINING, AgentState.CANDIDATE_SCREENING])
def test_pilot_quality_failure_can_stop_for_review_but_cannot_skip_acceptance(tmp_path, stage):
    """真实 pilot 奖励质量失败可保存人工复核终态，但不允许直接宣称训练完成。"""
    state = PersistentStateMachine(tmp_path / 'state.json')
    state.record.state = stage
    state._save()
    with pytest.raises(ValueError, match='illegal state transition'):
        state.transition(AgentState.COMPLETED)
    assert state.transition(AgentState.HUMAN_REVIEW, {'reason': 'pilot 奖励质量审查失败'})
    saved = read_json(tmp_path / 'state.json')
    assert saved['state'] == 'HUMAN_REVIEW'
    assert saved['context']['reason'] == 'pilot 奖励质量审查失败'


@pytest.mark.parametrize('iterations,revisions,dry_run', [
    (3000, 1, False), (3000, 0, False), (1950, 1, False), (3000, 1, True)])
def test_exhausted_resume_returns_without_evaluation_provider_or_training(
        tmp_path, iterations, revisions, dry_run):
    """重复恢复只能刷新摘要，历史、训练快照、已用预算和阻塞审计均保持原值。"""
    budget = BudgetTracker(max_iterations=3000, used_iterations=iterations,
        max_revisions=1, used_revisions=revisions)
    orchestrator, task_dir, state, task, selected = _budget_run(tmp_path, budget, dry_run=dry_run)
    orchestrator._evaluate = lambda *args, **kwargs: pytest.fail('耗尽恢复不得重新评估或请求模型')
    atomic_write_text(task_dir / 'loop_blocking_error.txt', '保留旧阻塞审计\n')
    preserved = {name: (task_dir / name).read_bytes() for name in [
        'loop_history.json', 'training_execution.json', 'loop_status.json',
        'acceptance_contract.json', 'loop_blocking_error.txt']}
    for _ in range(2):
        result = orchestrator.resume(task_dir.name, dry_run)
        assert result['state'] == 'HUMAN_REVIEW' and result['result'] == 'human_review'
        assert result['reward_version'] == 3 and result['budget_exhausted'] is True
        assert result['resume_allowed'] is False and result['next_training_started'] is False
        assert result['used_iterations'] == iterations and result['max_iterations'] == 3000
        assert result['used_revisions'] == revisions and result['max_revisions'] == 1
        assert '已完成 v3 %s' % ('模拟训练' if dry_run else '奖励训练') in result['reason']
        assert '%s/3000' % iterations in result['reason'] and '%s/1' % revisions in result['reason']
        assert '下一轮未启动训练' in result['reason'] and '不代表验收通过' in result['reason']
        assert '恢复不会重新调用 Provider' in result['reason']
    assert preserved == {name: (task_dir / name).read_bytes() for name in preserved}
    assert read_json(task_dir / 'state.json')['state'] == 'HUMAN_REVIEW'
    assert result['reason'] in (task_dir / 'report.md').read_text()


def test_exhausted_execution_snapshot_overrides_stale_summary(tmp_path):
    """即便摘要仍显示额度可用，也不能退款已落盘的训练消耗后重新调用模型。"""
    budget = BudgetTracker(max_iterations=3000, used_iterations=3000,
        max_revisions=1, used_revisions=1)
    orchestrator, task_dir, state, task, selected = _budget_run(tmp_path, budget)
    summary = read_json(task_dir / 'summary.json')
    summary.update(used_iterations=1950, remaining_iterations=1050, used_revisions=0)
    write_json(task_dir / 'summary.json', summary)
    orchestrator._evaluate = lambda *args, **kwargs: pytest.fail('快照已耗尽')
    result = orchestrator.resume(task_dir.name)
    assert result['used_iterations'] == 3000 and result['used_revisions'] == 1
    assert result['resume_allowed'] is False


def test_cli_resume_reports_exhaustion_without_model_requests(tmp_path, monkeypatch, capsys):
    """CLI 原接口仍返回可读摘要，Provider 只有生命周期关闭且不执行模型请求。"""
    from rl_training_agent import cli
    budget = BudgetTracker(max_iterations=3000, used_iterations=3000,
        max_revisions=1, used_revisions=1)
    orchestrator, task_dir, state, task, selected = _budget_run(tmp_path, budget)
    orchestrator._evaluate = lambda *args, **kwargs: pytest.fail('CLI 耗尽恢复不得重新评估')
    closed = []
    monkeypatch.setattr(cli, 'load_settings', lambda: orchestrator.settings)
    monkeypatch.setattr(cli, '_provider', lambda *args: SimpleNamespace(close=lambda: closed.append(True)))
    monkeypatch.setattr(cli, 'TrainingOrchestrator', lambda *args: orchestrator)
    assert cli.main(['resume', '--task-id', task_dir.name, '--provider', 'mock']) == 0
    output = capsys.readouterr().out
    assert '已完成 v3 奖励训练' in output and '3000/3000' in output and '下一轮未启动训练' in output
    assert '"resume_allowed": false' in output and closed == [True]


def test_non_exhausted_resume_keeps_frozen_caps_and_original_recovery(tmp_path):
    """有额度的 Provider 暂停仍可恢复；更大当前配置不能扩大旧运行的预算。"""
    budget = BudgetTracker(max_iterations=3000, used_iterations=1950, max_revisions=1)
    orchestrator, task_dir, state, task, selected = _budget_run(tmp_path, budget)
    captured = []
    orchestrator._evaluate = lambda *args, **kwargs: captured.append((args[5], kwargs)) or {'state': 'fixture'}
    result = orchestrator.resume(task_dir.name)
    assert result == {'state': 'fixture'} and len(captured) == 1
    restored, options = captured[0]
    assert restored.max_iterations == 3000 and restored.max_revisions == 1
    assert restored.used_iterations == 1950 and restored.used_revisions == 0
    assert options['reuse_first_rollout'] is True
    assert read_json(task_dir / 'state.json')['state'] == 'ROLLOUT_COLLECTING'


@pytest.mark.parametrize('original,goal', [
    ('向左侧向翻滚一圈', '起跳腾空侧翻后落地'),
    ('向左贴地侧滚一圈后站起', '起跳腾空侧翻后落地'),
    ('向左起跳腾空侧翻一圈', '向左贴地侧滚后站起')])
def test_resume_with_budget_refuses_ambiguous_or_changed_motion_without_consumption(
        tmp_path, original, goal):
    """有额度也不能继续含糊或被模型替换的动作，拒绝前不改预算、状态与合同。"""
    budget = BudgetTracker(max_iterations=3000, used_iterations=1950, max_revisions=1)
    orchestrator, task_dir, state, task, selected = _budget_run(tmp_path, budget)
    task.original_instruction = original
    task.normalized_description = goal
    write_json(task_dir / 'task_spec.json', task)
    write_json(task_dir / 'task_intent.json', TaskIntentSpec(original_instruction=original,
        robot='go2', action_name=goal, normalized_goal=goal))
    contract = read_json(task_dir / 'acceptance_contract.json')
    contract['task_identity'] = orchestrator._task_acceptance_identity(task)
    write_json(task_dir / 'acceptance_contract.json', contract)
    preserved = {name: (task_dir / name).read_bytes() for name in [
        'training_execution.json', 'acceptance_contract.json', 'state.json', 'loop_history.json']}
    orchestrator._evaluate = lambda *args, **kwargs: pytest.fail('错误语义不得恢复评估')
    with pytest.raises(RuntimeError, match='原动作语义需要澄清或与验收合同冲突'):
        orchestrator.resume(task_dir.name)
    assert preserved == {name: (task_dir / name).read_bytes() for name in preserved}


def test_exhausted_resume_does_not_redesign_or_check_ambiguous_motion_again(tmp_path, monkeypatch):
    """耗尽时只报告已完成版本与停止原因，不触发需要进一步决策的恢复路径。"""
    budget = BudgetTracker(max_iterations=3000, used_iterations=3000,
        max_revisions=1, used_revisions=1)
    orchestrator, task_dir, state, task, selected = _budget_run(tmp_path, budget)
    monkeypatch.setattr('rl_training_agent.orchestration.orchestrator.TaskSemanticValidator.assess_task',
        lambda *args: pytest.fail('预算早停应先于新的语义恢复审核'))
    result = orchestrator.resume(task_dir.name)
    assert result['resume_allowed'] is False and result['budget_exhausted'] is True


@pytest.mark.parametrize('corruption', ['checkpoint', 'contract'])
def test_exhausted_resume_still_validates_checkpoint_and_acceptance(tmp_path, corruption):
    """短路耗尽恢复不等于放宽证据校验，损坏 checkpoint 或改阈值仍须拒绝。"""
    budget = BudgetTracker(max_iterations=3000, used_iterations=3000,
        max_revisions=1, used_revisions=1)
    orchestrator, task_dir, state, task, selected = _budget_run(tmp_path, budget)
    orchestrator._evaluate = lambda *args, **kwargs: pytest.fail('损坏证据不得评估')
    if corruption == 'checkpoint':
        selected['checkpoints'][1].unlink()
    else:
        changed = read_json(task_dir / 'task_spec.json')
        changed['success_metrics'][0]['value'] = 100
        write_json(task_dir / 'task_spec.json', changed)
    with pytest.raises((RuntimeError, FileNotFoundError)):
        orchestrator.resume(task_dir.name)


@pytest.mark.parametrize('completed,failure_redesign', [(False, True), (False, False), (True, True)])
def test_final_trained_version_is_evaluated_once_even_when_budget_just_exhausted(
        tmp_path, completed, failure_redesign):
    """最后一次训练的首次评估不能被跳过；联合验收通过仍正常完成，否则明确停训。"""
    budget = BudgetTracker(max_iterations=3000, used_iterations=3000,
        max_revisions=1, used_revisions=1)
    orchestrator, task_dir, state, task, selected = _budget_run(tmp_path, budget)
    orchestrator.settings.failure_reward_redesign = failure_redesign
    state.record.state = AgentState.DIAGNOSING
    state._save()
    evaluation = EvaluationResult(completed=completed, hard_constraints_passed=completed,
        task_metrics_passed=completed, visual_alignment_passed=completed)
    diagnosis = TrainingDiagnosis(diagnosis=[], evidence=[], confidence=1,
        decision='complete' if completed else 'continue', checkpoint_strategy='continue_from_current')
    calls = []
    def evaluate_once(*args, **kwargs):
        """只返回本地验收对象，不收集真实 rollout 或请求 Provider。"""
        calls.append(True)
        return {'evaluation': evaluation, 'diagnosis': diagnosis}
    orchestrator._evaluate_round = evaluate_once
    orchestrator._train_revision = lambda *args, **kwargs: pytest.fail('耗尽后禁止下一轮训练')
    result = orchestrator._evaluate(task, task_dir, state, selected, False, budget)
    assert calls == [True]
    assert result['state'] == ('COMPLETED' if completed else 'HUMAN_REVIEW')
    assert result['budget_exhausted'] is True and result['resume_allowed'] is False
    assert result['next_training_started'] is False
    assert result['used_iterations'] == 3000 and result['max_iterations'] == 3000
    if not completed:
        assert '已完成 v3 奖励训练' in result['reason'] and '下一轮未启动训练' in result['reason']
        assert (task_dir / 'failure_analysis' / 'round_01.json').is_file()
