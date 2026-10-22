from types import SimpleNamespace

import pandas as pd
import pytest

from rl_training_agent.rewards.quality_gate import audit_objectives, audit_rollout, compare_pilots
from rl_training_agent.schemas.rewards import RewardPlan, RewardTerm, RewardObjectiveLink, CurriculumStage
from rl_training_agent.schemas.task import TaskSpec, MetricThreshold
from rl_training_agent.training.contact_semantics import collision_count, nonfoot_forces


def case(metric='tracking_error', term='tracking_lin_vel', weight=1.):
    threshold = MetricThreshold(name=metric, operator='<=', value=.25)
    task = TaskSpec(task_id='t', robot='go2', task_name='walk', original_instruction='行走',
        normalized_description='行走', initial_state='standing', success_metrics=[threshold],
        required_behaviors=[], forbidden_behaviors=[], phases=[], required_observations=[],
        required_sensors=[], safety_constraints=[], training_budget={}, visual_evaluation_requirements=[])
    plan = RewardPlan(task_id='t', version=1, design_rationale=['test'],
        terms=[RewardTerm(name=term, implementation='registry:'+term, purpose='target', weight=weight)],
        success_metrics=[threshold.copy()], objective_links=[RewardObjectiveLink(metric=metric,
            reward_terms=[term], rationale='实现对应目标', counterexamples=['检查原地抖动等错误动作'])])
    return task, plan


def test_valid_walk_jump_and_front_leg_contracts():
    for metric, term in [('tracking_error', 'tracking_lin_vel'), ('jump_height', 'jump_height'),
                         ('front_leg_walk_completion', 'front_leg_walk')]:
        assert audit_objectives(*case(metric, term))['passed']


@pytest.mark.parametrize('metric', ['final_roll_angle', 'unknown_target'])
def test_proxy_and_unknown_implementation_cannot_claim_goal_coverage(metric):
    report = audit_objectives(*case(metric, 'roll_rotation'))
    assert not report['passed']
    message = '候选未启用' if metric == 'final_roll_angle' else '缺少已实现'
    assert any(message in e for e in report['errors'])
    assert report['coverage'][0]['active_terms'] == []
    assert not report['coverage'][0]['verified']


def test_missing_links_and_disabled_curriculum_are_rejected():
    task, plan = case()
    plan.objective_links = []
    assert not audit_objectives(task, plan)['passed']
    plan.curriculum = [CurriculumStage(name='off', start_iteration=0, end_iteration=50,
        parameter_changes={'reward_scales': {'tracking_lin_vel': 0}})]
    assert any('候选未启用' in e for e in audit_objectives(task, plan)['errors'])


def test_safety_all_stages_and_immutable_thresholds():
    task, plan = case()
    task.safety_constraints = [MetricThreshold(name='forbidden_collisions', operator='==', value=0)]
    plan.terms.append(RewardTerm(name='collision', implementation='registry:collision', purpose='safety', weight=-1))
    assert audit_objectives(task, plan)['passed']
    plan.curriculum = [CurriculumStage(name='unsafe', start_iteration=0, end_iteration=50,
        parameter_changes={'reward_scales': {'collision': 0}})]
    assert not audit_objectives(task, plan)['passed']
    plan.success_metrics[0].value = 100
    assert any('验收标准' in e for e in audit_objectives(task, plan)['errors'])


def test_collision_covers_base_and_hip_but_excludes_feet():
    torch = pytest.importorskip('torch')
    forces = torch.zeros((2, 4, 3))
    forces[0, 0, 0] = 30
    forces[0, 1, 1] = 2
    forces[:, 3, 2] = 100
    env = SimpleNamespace(contact_forces=forces, feet_indices=torch.tensor([3]), num_bodies=4, device='cpu')
    assert collision_count(env).tolist() == [2., 0.]
    assert nonfoot_forces(env).shape == (2, 3)


def evidence(collision=0.):
    trajectory = pd.DataFrame({'sim_time': [0., .02]})
    rewards = pd.DataFrame({'sim_time': [0., .02], 'raw_tracking_lin_vel': [1., 1.],
        'weighted_tracking_lin_vel': [1., 1.], 'raw_collision': [collision, collision],
        'weighted_collision': [-collision, -collision]})
    return trajectory, rewards


def test_collision_without_penalty_stops_but_zero_safe_penalty_is_valid():
    task, plan = case()
    plan.terms.append(RewardTerm(name='collision', implementation='registry:collision', purpose='safety', weight=-1))
    traj, rew = evidence()
    report = audit_rollout(task, plan, traj, rew, {'tracking_error': 1., 'forbidden_collisions': 2})
    assert not report['passed']
    assert '实际非足端碰撞没有产生碰撞惩罚' in report['errors']
    assert audit_rollout(task, plan, traj, rew, {'tracking_error': 1., 'forbidden_collisions': 0})['passed']
    traj, rew = evidence(1.)
    assert audit_rollout(task, plan, traj, rew, {'tracking_error': 1., 'forbidden_collisions': 2})['passed']


def test_missing_nonfinite_and_misaligned_evidence_stop():
    task, plan = case()
    traj, rew = evidence()
    rew.loc[0, 'sim_time'] = 3
    assert not audit_rollout(task, plan, traj, rew, {'tracking_error': 1})['passed']
    rew = rew.drop(columns=['raw_tracking_lin_vel'])
    rew.loc[1, 'weighted_tracking_lin_vel'] = float('nan')
    assert not audit_rollout(task, plan, traj, rew, {'tracking_error': 1})['passed']


def test_disabled_term_does_not_require_unlogged_column():
    task, plan = case()
    plan.terms.append(RewardTerm(name='landing_recovery', implementation='registry:landing_recovery', purpose='later', weight=0))
    traj, rew = evidence()
    report = audit_rollout(task, plan, traj, rew, {'tracking_error': 1})
    assert report['passed']
    assert any('未启用' in w for w in report['warnings'])


def test_proxy_growth_with_regression_stops_but_plateau_does_not():
    task, plan = case()
    previous = {'metrics': {'tracking_error': .25}, 'contributions': {'signed_tracking_lin_vel': .1}}
    current = {'metrics': {'tracking_error': 1.25}, 'contributions': {'signed_tracking_lin_vel': .3}}
    assert compare_pilots(task, previous, current)['suspected_reward_hacking']
    current['metrics']['tracking_error'] = .25
    assert not compare_pilots(task, previous, current)['suspected_reward_hacking']


def test_pilot_artifact_saves_failure_before_candidate_advances(tmp_path):
    from rl_training_agent.orchestration.orchestrator import TrainingOrchestrator
    from rl_training_agent.utils.io import read_json
    task, plan = case()
    plan.terms.append(RewardTerm(name='collision', implementation='registry:collision', purpose='safety', weight=-1))
    orchestrator = object.__new__(TrainingOrchestrator)
    orchestrator.settings = SimpleNamespace(video_fps=30)
    def rollout(*args, **kwargs):
        target = args[4]
        target.mkdir(parents=True)
        traj, rew = evidence()
        traj['body_collision'] = True
        traj.to_parquet(target / 'trajectory.parquet')
        rew.to_parquet(target / 'rewards.parquet')
        return SimpleNamespace(exit_code=0)
    orchestrator.controller = SimpleNamespace(run_evaluation_rollouts=rollout)
    item = {'id':'c', 'dir':tmp_path, 'manifest':SimpleNamespace(seed=1), 'plan':plan}
    assert not orchestrator._pilot_reward_quality(task, item, tmp_path/'model.pt', 'smoke')
    assert not read_json(tmp_path/'reward_quality/smoke/audit.json')['passed']


def test_body_termination_uses_same_nonfoot_scope():
    torch = pytest.importorskip('torch')
    from rl_training_agent.training.config_runtime import install_runtime_terminations
    class Env:
        def check_termination(self):
            self.reset_buf[:] = False
    env = Env()
    env.device, env.num_bodies = 'cpu', 3
    env.feet_indices = torch.tensor([2])
    env.contact_forces = torch.zeros((2, 3, 3))
    env.contact_forces[0, 0, 0] = 20
    env.contact_forces[1, 2, 2] = 100
    env.reset_buf = torch.tensor([False, False])
    install_runtime_terminations(env, {'terminations': [
        {'name':'forbidden_body_contact', 'condition':'body contact > 1N', 'enabled':True}]})
    env.check_termination()
    assert env.reset_buf.tolist() == [True, False]


def test_behavior_contract_required_and_no_fake_reward_link():
    from rl_training_agent.schemas.task import BehaviorRequirement
    from rl_training_agent.schemas.rewards import RewardBehaviorLink
    task, plan = case()
    task.required_behaviors = [BehaviorRequirement(name='forward', description='前进')]
    assert not audit_objectives(task, plan)['passed']
    plan.behavior_links = [RewardBehaviorLink(behavior='forward', reward_terms=['tracking_lin_vel'],
        rationale='速度追踪', counterexamples=['抖动不算前进'])]
    assert audit_objectives(task, plan)['passed']
    plan.objective_links[0].reward_terms = ['roll_rotation']
    assert not audit_objectives(task, plan)['passed']


def test_missing_safety_evidence_stops_pilot():
    task, plan = case()
    task.safety_constraints = [MetricThreshold(name='forbidden_collisions', operator='==', value=0)]
    traj, rew = evidence()
    assert not audit_rollout(task, plan, traj, rew, {'tracking_error': 1})['passed']


def test_rejected_smoke_candidate_never_starts_screening_or_full_training(tmp_path, monkeypatch):
    from rl_training_agent.orchestration.orchestrator import TrainingOrchestrator
    from rl_training_agent.orchestration.budget import BudgetTracker
    from rl_training_agent.schemas.experiments import ExperimentManifest
    from rl_training_agent.utils.io import read_json
    task, plan = case()
    runner = object.__new__(TrainingOrchestrator)
    runner.settings = SimpleNamespace(smoke_iterations=5, checkpoints_per_run=2)
    transitions = []
    state = SimpleNamespace(transition=lambda *args: transitions.append(args))
    checkpoint = tmp_path / 'checkpoints/model_5.pt'
    checkpoint.parent.mkdir()
    checkpoint.write_text('test')
    manifest = ExperimentManifest(experiment_id='c', task_id='t', config_hash='x',
        reward_version=1, seed=1, robot='go2', training_command=[])
    candidate = {'id':'c', 'dir':tmp_path, 'manifest':manifest, 'plan':plan}
    runner.controller = SimpleNamespace(
        run_smoke_training=lambda *args: SimpleNamespace(exit_code=0, timed_out=False),
        continue_training=lambda *args: pytest.fail('不得继续筛选'),
        run_full_training=lambda *args: pytest.fail('不得开始完整训练'))
    monkeypatch.setattr(runner, '_pilot_reward_quality', lambda *args: False)
    monkeypatch.setattr(runner, '_reserve_training_operation',
        lambda directory, state, budget, operation, amount, **kwargs: budget.consume_iterations(amount), raising=False)
    monkeypatch.setattr(runner, '_complete_training_operation', lambda *args: None, raising=False)
    budget = BudgetTracker(max_iterations=100, max_revisions=1)
    with pytest.raises(RuntimeError, match='奖励完整性审计失败'):
        runner._real_train(task, tmp_path, state, [candidate], budget)
    assert budget.used_iterations == 5
    assert read_json(tmp_path/'blocking_report.json')['state'] == 'HUMAN_REVIEW'


def test_unsorted_curriculum_audit_matches_actual_final_stage():
    task, plan = case()
    plan.curriculum = [CurriculumStage(name='late', start_iteration=100, end_iteration=200,
        parameter_changes={'reward_scales': {'tracking_lin_vel': 1}}),
        CurriculumStage(name='early', start_iteration=0, end_iteration=100,
        parameter_changes={'reward_scales': {'tracking_lin_vel': 0}})]
    traj, rew = evidence()
    report = audit_rollout(task, plan, traj, rew, {'tracking_error': 1})
    assert 'raw_tracking_lin_vel' in report['contributions']


def test_airborne_and_post_landing_metrics_reject_wrong_contact_proxies():
    task, plan = case('feet_air_time', 'feet_air_time')
    assert not audit_objectives(task, plan)['passed']
    task, plan = case('stable_stand_duration', 'stable_support')
    assert audit_objectives(task, plan)['passed']
    task.success_metrics[0].aggregation = 'continuous_after_last_landing'
    plan.success_metrics[0].aggregation = 'continuous_after_last_landing'
    assert not audit_objectives(task, plan)['passed']


def test_stable_support_requires_all_feet_without_requiring_a_jump():
    torch = pytest.importorskip('torch')
    from rl_training_agent.training.action_rewards import _reward_stable_support
    env = SimpleNamespace(contact_forces=torch.zeros((2, 4, 3)),
        feet_indices=torch.arange(4), rpy=torch.zeros((2, 3)),
        root_states=torch.zeros((2, 13)), base_ang_vel=torch.zeros((2, 3)))
    env.contact_forces[:, :, 2] = 10
    env.contact_forces[1, 0, 2] = 0
    assert _reward_stable_support(env).tolist() == [1., 0.]
