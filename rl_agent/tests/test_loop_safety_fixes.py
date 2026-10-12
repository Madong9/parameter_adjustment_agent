"""验证奖励隔离、启动前预算日志、固定评估覆盖和反事实 fail-closed。"""
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from rl_training_agent.evaluation.deterministic import DeterministicEvaluator
from rl_training_agent.orchestration.budget import BudgetTracker
from rl_training_agent.orchestration.orchestrator import TrainingOrchestrator
from rl_training_agent.orchestration.state_machine import AgentState, PersistentStateMachine
from rl_training_agent.rewards.validator import RewardCodeValidator, RewardValidationError
from rl_training_agent.schemas.decisions import TrainingDiagnosis
from rl_training_agent.schemas.experiments import ExperimentManifest
from rl_training_agent.schemas.metrics import EvaluationResult
from rl_training_agent.schemas.rewards import RewardPlan
from rl_training_agent.schemas.task import TaskSpec, MetricThreshold, TrainingBudget
from rl_training_agent.schemas.visual import VisualBehaviorReport
from rl_training_agent.settings import Settings
from rl_training_agent.training.process_manager import process_identity
from rl_training_agent.utils.io import atomic_write_text, read_json, write_json


def run_fixture(tmp_path, dry_run=False):
    """构造只有临时产物的运行，不加载网页 Provider、凭据或仿真。"""
    task_dir = tmp_path / 'task-audit'
    task_dir.mkdir()
    settings = Settings(experiment_root=str(tmp_path), artifact_root=str(tmp_path / 'artifacts'),
        evaluation_seeds=[1, 2], rollouts_per_seed=1, memory_enabled=False,
        rag_enabled=False, full_iterations=3, mid_iterations=3, max_total_iterations=100)
    orchestrator = object.__new__(TrainingOrchestrator)
    orchestrator.settings = settings
    orchestrator.store = SimpleNamespace(task_dir=lambda task_id: task_dir,
        candidate_dir=lambda task_id, candidate: task_dir / 'candidates' / candidate)
    orchestrator._write_working_memory = lambda *args, **kwargs: None
    state = PersistentStateMachine(task_dir / 'state.json')
    state.start_new_run('run-audit')
    state.record.state = AgentState.DIAGNOSING
    state._save()
    task = TaskSpec(task_id='task-audit', robot='go2', task_name='walk',
        original_instruction='walk according to command', normalized_description='walk according to command',
        initial_state='standing', required_behaviors=[], forbidden_behaviors=[], phases=[],
        required_observations=[], required_sensors=[], success_metrics=[
            MetricThreshold(name='tracking_error', operator='<=', value=.1, unit='m/s')],
        safety_constraints=[], training_budget=TrainingBudget(), visual_evaluation_requirements=[])
    write_json(task_dir / 'task_spec.json', task)
    write_json(task_dir / 'evaluation_protocol.json', {'run_id': 'run-audit', 'dry_run': dry_run,
        'seeds': [1, 2], 'rollouts_per_seed': 1, 'counterfactual_enabled': True})
    write_json(task_dir / 'acceptance_contract.json', {'run_id': 'run-audit', 'robot': 'go2',
        'success_metrics': [item.dict() for item in task.success_metrics], 'safety_constraints': [],
        'task_identity': orchestrator._task_acceptance_identity(task)})
    return orchestrator, task_dir, state, task


def candidate_fixture(orchestrator, task_dir, name='parent', dry_run=False):
    """保存有完整两种子 checkpoint 摘要的候选，不反序列化策略。"""
    directory = task_dir / 'candidates' / name
    directory.mkdir(parents=True)
    config = directory / 'config.yaml'
    atomic_write_text(config, '{}\n')
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    plan = RewardPlan(task_id='task-audit', version=1, design_rationale=['测试'], terms=[])
    checkpoints = []
    for seed in [1, 2]:
        path = directory / 'checkpoints' / ('seed_%d' % seed) / 'model_10.pt'
        atomic_write_text(path, '测试占位，禁止加载\n')
        checkpoints.append(path)
    manifest = ExperimentManifest(experiment_id=name, task_id='task-audit', config_hash=digest,
        reward_version=1, seed=1, robot='go2', training_command=[], training_result='completed',
        checkpoint=str(checkpoints[0].relative_to(directory)))
    for filename, payload in [('reward_plan.json', plan), ('manifest.json', manifest),
                              ('compile_metadata.json', {'config_hash': digest})]:
        write_json(directory / filename, payload)
    selected = dict(id=name, dir=directory, plan=plan, manifest=manifest,
        checkpoints=checkpoints, checkpoint_seeds=[1, 2], checkpoint=checkpoints[0])
    orchestrator._save_checkpoint_set(task_dir, selected, dry_run)
    return selected


@pytest.mark.parametrize('source', [
    "def reward(tensors):\n return [torch.save][0](tensors['x'], '/tmp/forbidden.pt')",
    "def reward(tensors):\n return (torch.sum,)[0](tensors['x'], dim=1)",
    "x=1\ndef reward(tensors):\n return torch.sum(tensors['x'], dim=1)",
    "@torch.sum\ndef reward(tensors):\n return tensors['x'][:,0]",
    "def reward(tensors):\n while True: pass\n return tensors['x'][:,0]",
    "def reward(tensors):\n torch.sum = tensors['x']\n return tensors['x'][:,0]",
])
def test_generated_reward_rejects_indirect_calls_and_impure_top_level(source):
    """所有间接调用、模块副作用、装饰器及非有界函数均在执行前被拒绝。"""
    with pytest.raises(RewardValidationError):
        RewardCodeValidator(['x']).validate_ast(source)


def test_reward_smoke_executes_in_independent_worker(monkeypatch):
    """父进程不执行生成函数，正常张量函数仍能返回有限统计量。"""
    monkeypatch.setattr(RewardCodeValidator, '_tensor_smoke_in_process', lambda *a: pytest.fail('宿主不得执行奖励'))
    result = RewardCodeValidator(['x']).tensor_smoke_test(
        "def reward(tensors):\n return torch.square(tensors['x'][:,0])")
    assert 0 <= result['min'] <= result['max']


def test_reward_worker_allocation_is_bounded():
    """超过进程内存额度的纯表达式会安全拒绝而不拖垮宿主。"""
    with pytest.raises(RewardValidationError):
        RewardCodeValidator(['x']).tensor_smoke_test(
            "def reward(tensors):\n huge='x' * 1073741824\n return tensors['x'][:,0]")


def test_restore_requires_every_original_seed_and_exact_checkpoint(tmp_path):
    """当前配置变更不缩减原种子；缺文件或内容篡改均拒绝恢复。"""
    orchestrator, task_dir, state, task = run_fixture(tmp_path)
    selected = candidate_fixture(orchestrator, task_dir)
    orchestrator.settings.evaluation_seeds = [1]
    restored = orchestrator._restore_selected_candidate(task_dir, 'parent')
    assert restored['checkpoint_seeds'] == [1, 2]
    atomic_write_text(selected['checkpoints'][1], '被修改\n')
    with pytest.raises(RuntimeError, match='摘要'):
        orchestrator._restore_selected_candidate(task_dir, 'parent')
    selected['checkpoints'][1].unlink()
    with pytest.raises(FileNotFoundError, match='种子 2'):
        orchestrator._restore_selected_candidate(task_dir, 'parent')


def test_budget_is_persisted_before_training_and_operation_cannot_replay(tmp_path):
    """训练调用前已有扣费快照，相同阶段键即使中断也不会重复执行。"""
    orchestrator, task_dir, state, task = run_fixture(tmp_path)
    budget = BudgetTracker(max_iterations=100, used_iterations=10, max_revisions=2)
    orchestrator._reserve_training_operation(task_dir, state, budget, 'parent:seed:1', 3,
        dry_run=False, experiment_id='parent', parent_experiment=None)
    saved = read_json(task_dir / 'training_execution.json')
    assert saved['budget']['used_iterations'] == 13
    assert saved['operations']['parent:seed:1']['status'] == 'reserved'
    with pytest.raises(RuntimeError, match='禁止重复'):
        orchestrator._reserve_training_operation(task_dir, state, budget, 'parent:seed:1', 3)
    assert budget.used_iterations == 13


@pytest.mark.parametrize('failure', ['exit', 'interrupt'])
def test_revision_failure_and_interrupt_preserve_budget_and_parent_checkpoint(tmp_path, failure):
    """子进程失败和中断均保留父策略、已用迭代及修订，不重放原阶段。"""
    orchestrator, task_dir, state, task = run_fixture(tmp_path)
    parent = candidate_fixture(orchestrator, task_dir)
    revised = candidate_fixture(orchestrator, task_dir, 'child')
    revised['manifest'].training_result = 'compiled'
    write_json(revised['dir'] / 'manifest.json', revised['manifest'])
    (revised['dir'] / 'checkpoint_set.json').unlink()
    orchestrator._compile_revision = lambda *a, **k: revised
    calls = []
    def train(*args, **kwargs):
        """只检查快照和模拟退出，绝不启动 PPO。"""
        snapshot = read_json(task_dir / 'training_execution.json')
        assert snapshot['budget']['used_iterations'] == 13
        assert snapshot['budget']['used_revisions'] == 1
        calls.append(args)
        if failure == 'interrupt':
            raise KeyboardInterrupt()
        return SimpleNamespace(exit_code=1)
    orchestrator.controller = SimpleNamespace(continue_training=train)
    diagnosis = TrainingDiagnosis(diagnosis=[], evidence=[], decision='continue', confidence=1,
        checkpoint_strategy='continue_from_current')
    budget = BudgetTracker(max_iterations=100, used_iterations=10, max_revisions=2)
    with pytest.raises(KeyboardInterrupt if failure == 'interrupt' else RuntimeError):
        orchestrator._train_revision(task, task_dir, state, parent, diagnosis, 1, False, budget)
    assert state.record.state == AgentState.FULL_TRAINING
    recovered = orchestrator.resume('task-audit')
    assert recovered['state'] == 'HUMAN_REVIEW'
    assert recovered['used_iterations'] == 13 and recovered['used_revisions'] == 1
    assert recovered['selected_experiment'] == 'parent'
    captured = []
    orchestrator._evaluate = lambda *a, **k: captured.append((a[5].dict(), k['start_round'])) or {'state': 'fixture'}
    orchestrator.resume('task-audit')
    assert captured[0][0]['used_iterations'] == 13 and captured[0][0]['used_revisions'] == 1
    assert captured[0][1] == 2
    assert len(calls) == 1


def test_resume_rejects_live_registered_child(tmp_path):
    """进程启动身份仍匹配时不允许另启恢复流程，不读取进程环境。"""
    orchestrator, task_dir, state, task = run_fixture(tmp_path)
    selected = candidate_fixture(orchestrator, task_dir)
    budget = BudgetTracker(max_iterations=100, max_revisions=1)
    orchestrator._reserve_training_operation(task_dir, state, budget, 'parent:operation', 1,
        dry_run=False, experiment_id='parent', parent_experiment=None)
    write_json(selected['dir'] / 'process_identity.json', process_identity(os.getpid()))
    with pytest.raises(RuntimeError, match='仍在运行'):
        orchestrator._load_training_execution(task_dir, state, False)


def test_failed_revision_loop_safely_finishes_human_review(tmp_path):
    """闭环内部的训练失败正常收束，不再因非法状态跳转丢失摘要。"""
    orchestrator, task_dir, state, task = run_fixture(tmp_path)
    parent = candidate_fixture(orchestrator, task_dir)
    child = candidate_fixture(orchestrator, task_dir, 'child')
    orchestrator._compile_revision = lambda *a, **k: child
    orchestrator.controller = SimpleNamespace(continue_training=lambda *a, **k: SimpleNamespace(exit_code=1))
    diagnosis = TrainingDiagnosis(diagnosis=[], evidence=[], decision='continue', confidence=1,
        checkpoint_strategy='continue_from_current')
    evaluation = EvaluationResult(completed=False, task_metrics_passed=False,
        hard_constraints_passed=True, visual_alignment_passed=False)
    orchestrator._evaluate_round = lambda *a, **k: dict(diagnosis=diagnosis, evaluation=evaluation)
    budget = BudgetTracker(max_iterations=100, used_iterations=10, max_revisions=2)
    result = orchestrator._evaluate(task, task_dir, state, parent, False, budget)
    assert result['state'] == 'HUMAN_REVIEW'
    assert result['used_iterations'] == 13 and result['used_revisions'] == 1
    assert read_json(task_dir / 'training_execution.json')['phase'] == 'blocked'


def test_continue_from_parent_never_substitutes_another_seed(tmp_path):
    """缺少某一种子的父策略时禁止续训，不能重复使用另一种子的策略。"""
    orchestrator, task_dir, state, task = run_fixture(tmp_path)
    parent = candidate_fixture(orchestrator, task_dir)
    child = candidate_fixture(orchestrator, task_dir, 'child')
    parent['parent_checkpoints'] = [None, parent['checkpoints'][1]]
    orchestrator._compile_revision = lambda *a, **k: child
    orchestrator.controller = SimpleNamespace(continue_training=lambda *a, **k: pytest.fail('不得续训'))
    diagnosis = TrainingDiagnosis(diagnosis=[], evidence=[], decision='continue', confidence=1,
        checkpoint_strategy='continue_from_parent')
    budget = BudgetTracker(max_iterations=100, used_iterations=10, max_revisions=2)
    with pytest.raises(RuntimeError, match='父 checkpoint 不完整'):
        orchestrator._train_revision(task, task_dir, state, parent, diagnosis, 1, False, budget)
    assert budget.used_iterations == 10


@pytest.mark.parametrize('field,value', [('used_iterations', -1), ('used_iterations', 101),
                                        ('used_revisions', 3), ('used_iterations', '1')])
def test_execution_snapshot_rejects_invalid_budget(tmp_path, field, value):
    """负数、超上限和被类型强转的预算均不能参与阶段恢复。"""
    orchestrator, task_dir, state, task = run_fixture(tmp_path)
    budget = BudgetTracker(max_iterations=100, used_iterations=10, max_revisions=2)
    orchestrator._persist_training_execution(task_dir, state, budget, phase='training',
        dry_run=False, experiment_id='parent', parent_experiment=None)
    saved = read_json(task_dir / 'training_execution.json')
    saved['budget'][field] = value
    write_json(task_dir / 'training_execution.json', saved)
    with pytest.raises(RuntimeError, match='预算'):
        orchestrator._load_training_execution(task_dir, state, False)


@pytest.mark.parametrize('kind', ['string_bool', 'not_dict', 'absolute_path', 'wrong_seed'])
def test_checkpoint_index_rejects_malformed_or_misidentified_entries(tmp_path, kind):
    """checkpoint 清单必须有真实布尔类型、受限路径和对应种子的来源。"""
    orchestrator, task_dir, state, task = run_fixture(tmp_path)
    selected = candidate_fixture(orchestrator, task_dir)
    path = selected['dir'] / 'checkpoint_set.json'
    saved = read_json(path)
    if kind == 'string_bool': saved['dry_run'] = 'false'
    if kind == 'not_dict': saved['entries'][0] = 'bad'
    if kind == 'absolute_path': saved['entries'][0]['path'] = str(selected['checkpoint'])
    if kind == 'wrong_seed': saved['entries'][0]['path'] = saved['entries'][1]['path']
    write_json(path, saved)
    with pytest.raises(RuntimeError, match='checkpoint'):
        orchestrator._restore_selected_candidate(task_dir, 'parent')


@pytest.mark.parametrize('kind', ['missing', 'empty', 'missing_seed', 'duplicate_seed', 'nan', 'inf', 'malformed'])
def test_counterfactual_missing_or_nonfinite_evidence_fails_closed(tmp_path, kind):
    """必需反事实缺文件、缺种子、非有限值或损坏 JSON 都不能放行。"""
    orchestrator, task_dir, state, task = run_fixture(tmp_path)
    round_root = task_dir / 'round'
    round_root.mkdir()
    scenarios = [dict(seed=seed, scenario='zero_command', passed=True, max_base_speed=0., fall_rate=0.) for seed in [1, 2]]
    if kind == 'empty': scenarios = []
    if kind == 'missing_seed': scenarios = scenarios[:1]
    if kind == 'duplicate_seed': scenarios[1]['seed'] = 1
    if kind == 'nan': scenarios[0]['max_base_speed'] = float('nan')
    if kind == 'inf': scenarios[0]['max_base_speed'] = float('inf')
    path = round_root / 'counterfactual_evaluation.json'
    if kind == 'malformed': atomic_write_text(path, '{')
    elif kind != 'missing': write_json(path, dict(applicable=True, dry_run=False, passed=True, scenarios=scenarios))
    assert not orchestrator._counterfactual_evidence(round_root, [1, 2], False)['passed']


def test_deterministic_counterfactual_required_cannot_default_to_success():
    """确定性验收不会把明确必需但缺失的反事实结果默认成通过。"""
    visual = VisualBehaviorReport(visual_success=True, alignment_score=1, confidence=1,
        summary='测试', phase_results=[])
    metrics = dict(nan_count=0., joint_limit_violations=0., torque_limit_violations=0.,
        forbidden_collisions=0., abnormal_terminations=0., counterfactual_applicable=True)
    task = SimpleNamespace(safety_constraints=[], success_metrics=[
        MetricThreshold(name='nan_count', operator='==', value=0)])
    assert not DeterministicEvaluator().evaluate(task, metrics, visual).completed
    metrics['counterfactual_passed'] = float('nan')
    assert not DeterministicEvaluator().evaluate(task, metrics, visual).completed


def test_cached_rollout_rejects_incomplete_original_seed_coverage(tmp_path):
    """缓存样本集合不完整时必须重新采集，而非降低验收种子数量。"""
    orchestrator, task_dir, state, task = run_fixture(tmp_path)
    selected = candidate_fixture(orchestrator, task_dir)
    round_root = selected['dir'] / 'rollouts' / 'round_01'
    round_root.mkdir(parents=True)
    write_json(round_root / 'rollout_metrics.json', {'records': [{'rollout': 'rollout_001', 'seed': 1}],
        'representative_rollout': 'rollout_001'})
    assert orchestrator._cached_round_rollout(round_root, task, selected) is None
