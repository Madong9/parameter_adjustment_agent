"""验收单位、倒立坐标系、训练/执行阶段及迁移审计的回归验证。"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rl_training_agent.evaluation.deterministic import DeterministicEvaluator
from rl_training_agent.metrics.trajectory_metrics import TrajectoryMetrics
from rl_training_agent.metrics.velocity import heading_forward_velocity
from rl_training_agent.orchestration.orchestrator import TrainingOrchestrator
from rl_training_agent.providers.mock_provider import MockLLMReasoningProvider
from rl_training_agent.schemas.task import MetricThreshold, TaskPhase, TaskSpec
from rl_training_agent.schemas.rewards import RewardPlan, CurriculumStage
from rl_training_agent.training.config_runtime import install_velocity_tracking, curriculum_segments, prepare_training_env_config
from rl_training_agent.utils.io import read_json, write_json
from scripts.reassess_acceptance import reassess
from rl_training_agent.visual.evidence_builder import SynchronizedEvidenceBuilder


def task():
    raw = MockLLMReasoningProvider().design_task_and_rewards('前腿倒立前进，速度0.5m/s', 'go2', {})
    result = TaskSpec.parse_obj(raw['task_spec'])
    result.phases = [TaskPhase(name='front_stand', description='learn balance'),
                     TaskPhase(name='front_walk', description='learn walking')]
    result.success_metrics = [MetricThreshold(name='front_leg_walk_velocity_tracking',
                                 operator='>=', value=0.45, unit='m/s')]
    return result


def trajectory(speed=0.5, pitch=0.85, roll=0.2):
    n = 20
    return pd.DataFrame({'sim_time':np.arange(n) * 0.04,
        'roll':roll, 'pitch':pitch, 'base_vx':speed * np.cos(pitch),
        'base_vy':speed * np.sin(pitch) * np.sin(roll),
        'base_vz':speed * np.sin(pitch) * np.cos(roll),
        'contact_fl':True, 'contact_fr':True, 'contact_rl':False, 'contact_rr':False,
        'command':[[0.5, 0, 0, 0]] * n})


def test_tilted_velocity_is_physical_speed_not_score():
    data = trajectory()
    assert np.allclose(heading_forward_velocity(data), 0.5)
    values = TrajectoryMetrics().compute(data)
    assert values['front_leg_forward_speed'] == pytest.approx(0.5)
    score = MetricThreshold(name='front_leg_walk_velocity_tracking', operator='>=', value=0.45, unit='m/s')
    assert not DeterministicEvaluator()._check(score, values).passed
    score.unit = '1'
    assert DeterministicEvaluator()._check(score, values).passed
    spec = task()
    TrainingOrchestrator._normalize_task_for_instruction(spec)
    assert spec.success_metrics[0].value == 0.45
    assert DeterministicEvaluator()._check(spec.success_metrics[0], values).passed
    slow = TrajectoryMetrics().compute(trajectory(speed=0.2))
    assert not DeterministicEvaluator()._check(spec.success_metrics[0], slow).passed


def test_missing_velocity_or_posture_fails_closed():
    values = TrajectoryMetrics().compute(trajectory().drop(columns='base_vz'))
    assert 'front_leg_forward_speed' not in values
    threshold = MetricThreshold(name='front_leg_forward_speed', operator='>=', value=0.45, unit='m/s')
    assert not DeterministicEvaluator()._check(threshold, values).passed
    data = trajectory()
    data['contact_rl'] = True
    assert TrajectoryMetrics().compute(data)['front_leg_forward_speed'] == 0


def test_learning_phases_are_separate_from_requested_sequence():
    spec = task()
    TrainingOrchestrator._normalize_task_for_instruction(spec)
    assert all(p.scope == 'training' for p in spec.phases)
    sequential = task()
    sequential.original_instruction = '先前腿倒立站稳，然后前进，速度0.5m/s'
    TrainingOrchestrator._normalize_task_for_instruction(sequential)
    assert all(p.scope == 'execution' for p in sequential.phases)


def test_curriculum_is_contiguous_and_preserves_real_ramp():
    spec = task()
    TrainingOrchestrator._normalize_task_for_instruction(spec)
    raw = MockLLMReasoningProvider().design_task_and_rewards(spec.original_instruction, 'go2', {})
    plan = RewardPlan.parse_obj(raw['reward_plans'][0])
    plan.curriculum = [CurriculumStage(name='front_stand', start_iteration=0, end_iteration=99,
                        parameter_changes={'lin_vel_x':0.0}),
                       CurriculumStage(name='front_walk', start_iteration=100, end_iteration=499,
                        parameter_changes={'lin_vel_x':[0.25, 0.5], 'commands':[['lin_vel_x',0,0.5]]})]
    TrainingOrchestrator._normalize_plan_for_task(spec, plan)
    assert plan.velocity_frame == 'heading'
    assert plan.curriculum[0].parameter_changes['lin_vel_x'] == [0, 0]
    assert plan.curriculum[1].parameter_changes['lin_vel_x'] == [0.25, 0.5]
    assert 'commands' not in plan.curriculum[1].parameter_changes
    assert curriculum_segments({'curriculum':[p.dict() for p in plan.curriculum]}, 500) == [
        (0,100,'front_stand'), (100,400,'front_walk')]
    plan.curriculum = []
    TrainingOrchestrator._normalize_plan_for_task(spec, plan)
    assert [p.name for p in plan.curriculum] == ['front_stand','front_walk']
    spec.original_instruction = '前腿倒立前进'
    spec.normalized_description = spec.original_instruction
    plan.curriculum = []
    TrainingOrchestrator._normalize_plan_for_task(spec, plan)
    assert [p.name for p in plan.curriculum] == ['front_stand','front_walk']
    assert plan.curriculum[1].parameter_changes['lin_vel_x'] == [.3,.3]


def test_runtime_reward_and_metric_use_same_horizontal_velocity():
    torch = pytest.importorskip('torch')
    data = trajectory()
    env = SimpleNamespace(rpy=torch.tensor(data[['roll','pitch']].to_numpy()),
        base_lin_vel=torch.tensor(data[['base_vx','base_vy','base_vz']].to_numpy()),
        commands=torch.tensor(np.tile([0.5,0], (len(data),1))),
        cfg=SimpleNamespace(rewards=SimpleNamespace(tracking_sigma=0.25)))
    install_velocity_tracking(env, {'rewards':{'velocity_frame':'heading'}})
    assert torch.allclose(env._reward_tracking_lin_vel(), torch.ones(len(data), dtype=torch.float64))


def test_front_leg_plan_does_not_inherit_conflicting_default_rewards():
    env_cfg = SimpleNamespace(rewards=SimpleNamespace(scales=SimpleNamespace(
        tracking_lin_vel=1.0, orientation=-1.0, landing_stability=2.0,
        front_leg_stand=0.0, front_leg_walk=0.0)))
    prepare_training_env_config(env_cfg, {'rewards':{'velocity_frame':'heading',
        'scales':{'front_leg_stand':10, 'front_leg_walk':1.5}}})
    assert env_cfg.rewards.scales.tracking_lin_vel == 0
    assert env_cfg.rewards.scales.orientation == 0
    assert env_cfg.rewards.scales.landing_stability == 0
    assert env_cfg.rewards.scales.front_leg_stand == 10


def test_visual_evidence_exposes_acceptance_frame_and_training_scope(tmp_path):
    spec = task()
    TrainingOrchestrator._normalize_task_for_instruction(spec)
    data = trajectory()
    data['base_x'] = np.arange(len(data)) * .02
    data['base_y'] = 0
    data['base_z'] = .42
    data['yaw'] = 0
    data['foot_slip'] = 0
    data['video_frame'] = np.arange(len(data))
    path = SynchronizedEvidenceBuilder().build(spec, data, [], tmp_path/'evidence.json')
    evidence = read_json(path)
    assert evidence['command_tracking']['acceptance_velocity_frame'] == 'heading'
    assert evidence['command_tracking']['front_support_heading_speed_mps']['mean'] == .5
    assert evidence['command_tracking']['measured_base_vx_mps']['mean'] < .45
    assert evidence['phase_context']['training_phases'] == ['front_stand', 'front_walk']
    assert evidence['phase_context']['execution_phases'] == []


def test_audited_migration_preserves_budget_and_rejects_tampering(tmp_path):
    spec = task()
    raw = spec.dict()
    summary = {'selected_experiment':'candidate', 'used_iterations':3000, 'remaining_iterations':0,
               'used_revisions':1, 'max_revisions':1}
    contract = {'run_id':'run', 'robot':spec.robot, 'success_metrics':raw['success_metrics'],
        'safety_constraints':raw['safety_constraints'],
        'task_identity':TrainingOrchestrator._task_acceptance_identity(spec)}
    write_json(tmp_path/'task_spec.json', raw)
    write_json(tmp_path/'acceptance_contract.json', contract)
    write_json(tmp_path/'summary.json', summary)
    write_json(tmp_path/'state.json', {'state':'HUMAN_REVIEW','context':{'run_id':'run'}})
    path = tmp_path/'candidates/candidate/rollouts/round_02/rollout_001/trajectory.parquet'
    path.parent.mkdir(parents=True)
    trajectory().to_parquet(path)
    report = reassess(tmp_path, apply=True)
    assert report['numeric_task_passed'] and report['requires_visual_reassessment']
    assert read_json(tmp_path/'summary.json') == summary
    assert read_json(tmp_path/'acceptance_migrations/horizontal-speed-v2/before.json')['task'] == raw
    assert read_json(tmp_path/'task_spec.json')['success_metrics'][0]['value'] == 0.45
    edited = read_json(tmp_path/'task_spec.json')
    edited['success_metrics'][0]['value'] = 0.1
    write_json(tmp_path/'task_spec.json', edited)
    with pytest.raises(ValueError, match='合同不一致'):
        reassess(tmp_path, apply=True)


def test_resume_recomputes_metrics_from_cached_trajectory(tmp_path):
    directory = tmp_path/'rollout_001'
    directory.mkdir()
    trajectory().to_parquet(directory/'trajectory.parquet')
    for name in ('front','side','overview'):
        (directory/(name+'.mp4')).write_bytes(b'fixture')
    write_json(tmp_path/'rollout_metrics.json', {'representative_rollout':'rollout_001',
        'records':[{'rollout':'rollout_001','metrics':{'front_leg_walk_velocity_tracking':.99}}]})
    orchestrator = TrainingOrchestrator.__new__(TrainingOrchestrator)
    restored = orchestrator._cached_round_rollout(tmp_path, task())
    assert restored[2][0]['metrics']['front_leg_forward_speed'] == pytest.approx(.5)
