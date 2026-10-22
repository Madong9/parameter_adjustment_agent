"""复现前空翻说明与单位的误判，并贯通任务审核、编译和确定性验收。"""
import json
from pathlib import Path

import pytest

from rl_training_agent.agents.reward_reviewer import RewardReviewAgent
from rl_training_agent.agents.task_semantics import TaskSemanticValidator
from rl_training_agent.environment.inspector import EnvironmentInspector
from rl_training_agent.evaluation.deterministic import DeterministicEvaluator
from rl_training_agent.orchestration.orchestrator import TrainingOrchestrator
from rl_training_agent.orchestration.state_machine import AgentState, PersistentStateMachine
from rl_training_agent.rewards.compiler import RewardCompiler
from rl_training_agent.schemas.rewards import TerminationSpec
from rl_training_agent.schemas.task import MetricThreshold
from rl_training_agent.training.front_flip_contract import resolve_front_flip_contract

from test_forward_flip_contract import CountingProvider, settings
from test_front_flip_backend import contract


@pytest.mark.parametrize('unit', ['dimensionless', 'DIMENSIONLESS', ' dimensionless ',
                                 'unitless', '无量纲', 'ratio', '1', ''])
def test_dimensionless_aliases_pass_both_design_and_acceptance_without_mutation(unit):
    intent, task, _ = contract()
    axis = next(metric for metric in task.success_metrics if metric.name == 'pitch_axis_error')
    axis.operator, axis.value, axis.unit = '<', .5, unit
    before = task.dict()
    assert TaskSemanticValidator.assess_task(intent, task).status == 'CONSISTENT'
    evaluator = DeterministicEvaluator()
    assert evaluator._check(axis, {'pitch_axis_error': .49}).passed
    assert not evaluator._check(axis, {'pitch_axis_error': .5}).passed
    assert not evaluator._check(axis, {}).passed
    assert not evaluator._check(axis, {'pitch_axis_error': float('nan')}).passed
    assert task.dict() == before


@pytest.mark.parametrize('unit', ['rad', 'm/s', 'N', '%', 'percent', 'unknown'])
def test_physical_units_and_percentages_cannot_be_treated_as_dimensionless(unit):
    intent, task, _ = contract()
    axis = next(metric for metric in task.success_metrics if metric.name == 'pitch_axis_error')
    axis.unit = unit
    before = task.dict()
    report = TaskSemanticValidator.assess_task(intent, task)
    assert report.status == 'CONFLICT' and any('单位不兼容' in item for item in report.issues)
    assert not DeterministicEvaluator()._check(axis, {'pitch_axis_error': .1}).passed
    assert task.dict() == before


@pytest.mark.parametrize('operator,value', [('>=', .5), ('<', 1.), ('<=', 50.), ('<', -.1)])
def test_valid_unit_cannot_hide_invalid_axis_constraint(operator, value):
    intent, task, _ = contract()
    axis = next(metric for metric in task.success_metrics if metric.name == 'pitch_axis_error')
    axis.operator, axis.value, axis.unit = operator, value, 'dimensionless'
    report = TaskSemanticValidator.assess_task(intent, task)
    assert report.status == 'CONFLICT' and any('小于 1' in item for item in report.issues)


def test_dimensionless_aliases_do_not_change_physical_metric_units():
    for name in ('front_leg_forward_speed', 'final_pitch_angle'):
        threshold = MetricThreshold(name=name, operator='>=', value=.1, unit='dimensionless')
        assert not DeterministicEvaluator()._check(threshold, {name: 10.}).passed
    score = MetricThreshold(name='landing_stability', operator='>', value=.1, unit='dimensionless')
    assert DeterministicEvaluator()._check(score, {'landing_stability': .8}).passed


def test_actual_comparison_and_unit_pass_gate_review_compile_and_runtime(tmp_path):
    intent, task, plan = contract()
    task.original_instruction = intent.original_instruction = '向前腾空翻跟头'
    task.normalized_description = (
        '控制Go2从足端支撑状态起跳，在四足同时离地的腾空区间内绕机体横向轴正向完成前空翻。'
        '前空翻与绕机体纵向轴的侧空翻、允许机体接触地面的前滚翻具有不同的动作语义。'
        '随后通过足端着地并恢复稳定姿态。')
    for thresholds in (task.success_metrics, plan.success_metrics):
        axis = next(metric for metric in thresholds if metric.name == 'pitch_axis_error')
        axis.operator, axis.value, axis.unit = '<', .5, 'dimensionless'
    plan.terminations = [TerminationSpec(name='forbidden_body_contact',
        condition='非足端刚体接触力超过现有安全策略的1.0 N阈值。')]
    before = task.dict(), plan.dict()
    configuration = settings(tmp_path)
    task_dir = configuration.experiments_path / task.task_id
    task_dir.mkdir(parents=True)
    state = PersistentStateMachine(task_dir / 'state.json')
    orchestrator = TrainingOrchestrator(configuration, CountingProvider())
    assert orchestrator._check_task_semantics(task_dir, state, intent, task) is None
    assert state.record.state == AgentState.RECEIVED
    assert not (task_dir / 'blocking_report.json').exists()
    manifest = EnvironmentInspector(Path(__file__).resolve().parents[2] / 'unitree_rl_gym').inspect('go2')
    review = RewardReviewAgent().review(intent, task, [plan], dict(manifest.dict(), require_objective_links=True), [])
    assert review.approved, review.dict()
    output = task_dir / 'compiled'
    RewardCompiler(manifest).compile(plan, output, task=task)
    config = json.loads((output / 'config.yaml').read_text())
    assert resolve_front_flip_contract(config)['task'] == task.dict()
    axis = next(metric for metric in task.success_metrics if metric.name == 'pitch_axis_error')
    assert DeterministicEvaluator()._check(axis, {'pitch_axis_error': .1}).passed
    assert (task.dict(), plan.dict()) == before
    assert not (task_dir / 'training_execution.json').exists()
