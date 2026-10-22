"""使用 CPU 合成证据验证终止前采集与早期前翻审计，不启动模型或仿真。"""
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rl_training_agent.evaluation.deterministic import DeterministicEvaluator
from rl_training_agent.metrics.trajectory_metrics import TrajectoryMetrics
from rl_training_agent.rewards.quality_gate import audit_rollout
from rl_training_agent.training.real_rollout import (
    _install_terminal_state_capture,
    _sample_after_step,
    _sample_current_state,
)

from test_front_flip_backend import contract, quaternions, trajectory


def synchronized_episodes(parts):
    """为合成物理步添加本轮采集器输出的 episode 与终止前快照标记。"""
    frames = []
    offset = 0
    for episode_id, (angles, reason) in enumerate(parts):
        data = trajectory(angles)
        if reason:
            data.loc[data.index[-1], 'termination_reason'] = reason
        data['sim_time'] = (offset + np.arange(len(data))) * .02
        data['episode_id'] = episode_id
        data['episode_step'] = np.arange(1, len(data) + 1)
        data['terminal_state_preserved'] = data['termination_reason'].isin(['reset', 'timeout'])
        data['body_collision'] = data['termination_reason'].eq('reset')
        data['fall'] = data['body_collision']
        data['forbidden_contact_force_max'] = data['body_collision'].astype(float) * 2.
        data['joint_positions'] = [[0.] for _ in range(len(data))]
        data['joint_position_limits'] = [[[-2., 2.]] for _ in range(len(data))]
        data['joint_torques'] = [[0.] for _ in range(len(data))]
        data['joint_torque_limits'] = [[5.] for _ in range(len(data))]
        frames.append(data)
        offset += len(data)
    return pd.concat(frames, ignore_index=True)


def reward_evidence(data, plan):
    """构造逐帧有限奖励，真实碰撞帧含负贡献，其余未触发奖励保持零。"""
    rewards = data[['sim_time', 'episode_id']].copy()
    for term in plan.terms:
        raw = data['body_collision'].astype(float) if term.name == 'collision' else 0.
        rewards['raw_' + term.name] = raw
        rewards['weighted_' + term.name] = raw * term.weight * .02
    return rewards


def test_early_terminal_partial_flip_has_finite_pilot_evidence_but_fails_final_acceptance():
    """已同步采样的早期跌倒可通过奖励完整性审计，绝不能被声明为前翻完成。"""
    angles = np.r_[np.zeros(4), np.linspace(0., 1.2 * math.pi, 21)]
    data = synchronized_episodes([(angles, 'reset'), (angles, 'reset')])
    _, task, plan = contract()
    before = data.copy(deep=True)
    metrics = TrajectoryMetrics().compute_pilot(data)

    assert metrics['pilot_evidence_valid'] == 1.
    assert metrics['pilot_episode_count'] == metrics['pilot_reset_count'] == 2.
    assert metrics['final_pitch_angle'] == pytest.approx(1.2 * math.pi)
    assert metrics['pitch_axis_error'] == pytest.approx(0.)
    assert metrics['landing_stability'] == metrics['stable_stand_duration'] == 0.
    assert metrics['forbidden_collisions'] == 2.
    assert audit_rollout(task, plan, data, reward_evidence(data, plan), metrics)['passed']
    assert not DeterministicEvaluator()._check(task.success_metrics[0], metrics).passed
    strict = TrajectoryMetrics().compute(data)
    assert math.isnan(strict['final_pitch_angle']) and math.isnan(strict['pitch_axis_error'])
    pd.testing.assert_frame_equal(data, before)


def test_timeout_resets_do_not_join_partial_rotations_or_pass_strict_completion():
    """超时同样产生原生重置，两个部分动作不能拼成一圈，严格验收也拒绝跨段。"""
    angles = np.r_[np.zeros(4), np.linspace(0., 1.2 * math.pi, 21)]
    data = synchronized_episodes([(angles, 'timeout'), (angles, 'timeout')])
    metrics = TrajectoryMetrics().compute_pilot(data)
    assert metrics['pilot_evidence_valid'] == 1.
    assert metrics['pilot_episode_count'] == 2. and metrics['pilot_reset_count'] == 0.
    assert metrics['final_pitch_angle'] == pytest.approx(1.2 * math.pi)
    assert math.isnan(TrajectoryMetrics().compute(data)['final_pitch_angle'])


def test_single_successful_episode_preserves_strict_and_pilot_rotation():
    """完整同步的单次前翻两种入口给出相同净旋转，最后超时不会造成跨段积分。"""
    data = synchronized_episodes([(None, 'timeout')])
    metrics = TrajectoryMetrics().compute_pilot(data)
    strict = TrajectoryMetrics().compute(data)
    assert metrics['pilot_evidence_valid'] == 1. and metrics['pilot_episode_count'] == 1.
    assert metrics['final_pitch_angle'] == pytest.approx(2 * math.pi)
    assert metrics['final_pitch_angle'] == pytest.approx(strict['final_pitch_angle'])
    assert metrics['pitch_axis_error'] == pytest.approx(strict['pitch_axis_error'])
    assert metrics['landing_stability'] == pytest.approx(strict['landing_stability'])


def test_repeated_flights_within_episode_use_only_first_flight_progress():
    """同一 episode 重复起跳也不能拼接动作；pilot 只记录首次动作局部进度。"""
    angles = np.r_[np.zeros(4), np.linspace(0., math.pi, 21), np.full(5, math.pi),
                   np.linspace(math.pi, 2 * math.pi, 21), np.full(10, 2 * math.pi)]
    data = synchronized_episodes([(angles, 'timeout')])
    for name in ('contact_fl', 'contact_fr', 'contact_rl', 'contact_rr'):
        data[name] = True
        data.loc[5:24, name] = False
        data.loc[31:50, name] = False
    metrics = TrajectoryMetrics().compute_pilot(data)
    assert metrics['final_pitch_angle'] == pytest.approx(math.pi)
    assert TrajectoryMetrics().compute(data)['final_pitch_angle'] == 0.


def test_single_step_failed_episode_and_standing_do_not_create_missing_or_recovery_evidence():
    """立即终止和原地站立均为有限零进度，初始站立不能算翻转后的稳定恢复。"""
    data = synchronized_episodes([(np.zeros(1), 'reset'), (np.zeros(4), '')])
    metrics = TrajectoryMetrics().compute_pilot(data)
    assert metrics['pilot_evidence_valid'] == 1.
    assert metrics['pilot_episode_count'] == 2.
    for name in ('final_pitch_angle', 'pitch_axis_error', 'landing_stability', 'stable_stand_duration'):
        assert metrics[name] == 0.


def test_pilot_preserves_backward_rotation_direction_and_off_axis_error():
    """局部学习指标仍区分负向前翻与侧轴旋转，不因分段处理放松动作语义。"""
    data = synchronized_episodes([(None, 'timeout')])
    data['quat_y'] *= -1
    assert TrajectoryMetrics().compute_pilot(data)['final_pitch_angle'] == pytest.approx(-2 * math.pi)
    values = quaternions(np.r_[np.zeros(4), np.linspace(0., 2 * math.pi, 41),
                               np.full(10, 2 * math.pi)], axis=0)
    data[['quat_x', 'quat_y', 'quat_z', 'quat_w']] = values
    metrics = TrajectoryMetrics().compute_pilot(data)
    assert metrics['final_pitch_angle'] == pytest.approx(0.)
    assert metrics['pitch_axis_error'] == pytest.approx(1.)


@pytest.mark.parametrize('missing', [
    'quat_y', 'sim_time', 'roll', 'contact_rr', 'termination_reason',
    'episode_id', 'episode_step', 'terminal_state_preserved',
])
def test_missing_sampling_contract_or_physical_evidence_remains_unknown(missing):
    """旧采集器缺少终止前标记、episode 身份或物理字段时保持拒绝，不补造证据。"""
    data = synchronized_episodes([(None, 'timeout')]).drop(columns=[missing])
    metrics = TrajectoryMetrics().compute_pilot(data)
    assert metrics['pilot_evidence_valid'] == 0.
    assert math.isnan(metrics['final_pitch_angle']) and math.isnan(metrics['pitch_axis_error'])


@pytest.mark.parametrize('problem', [
    'nan_quaternion', 'zero_quaternion', 'undersampled_rotation', 'time_gap', 'time_rewind',
    'nan_pose', 'missing_contact', 'unknown_contact', 'unpreserved_terminal',
    'spurious_terminal_marker', 'episode_id_changed', 'episode_step_gap', 'unknown_termination',
])
def test_invalid_episode_evidence_cannot_become_finite_pilot_progress(problem):
    """坏姿态、异常时间、接触缺失与 episode 标记矛盾仍属于采集失效。"""
    data = synchronized_episodes([(None, 'timeout')])
    if problem == 'nan_quaternion':
        data.loc[20, 'quat_y'] = float('nan')
    elif problem == 'zero_quaternion':
        data.loc[20, ['quat_x', 'quat_y', 'quat_z', 'quat_w']] = 0.
    elif problem == 'undersampled_rotation':
        data.loc[10, ['quat_x', 'quat_y', 'quat_z', 'quat_w']] = quaternions([math.pi])[0]
    elif problem == 'time_gap':
        data.loc[20:, 'sim_time'] += 1.
    elif problem == 'time_rewind':
        data.loc[20, 'sim_time'] = data.loc[19, 'sim_time']
    elif problem == 'nan_pose':
        data.loc[20, 'roll'] = float('nan')
    elif problem == 'missing_contact':
        data['contact_fl'] = data['contact_fl'].astype(object)
        data.loc[20, 'contact_fl'] = None
    elif problem == 'unknown_contact':
        data['contact_fl'] = data['contact_fl'].astype(int)
        data.loc[20, 'contact_fl'] = 2
    elif problem == 'unpreserved_terminal':
        data.loc[data.index[-1], 'terminal_state_preserved'] = False
    elif problem == 'spurious_terminal_marker':
        data.loc[20, 'terminal_state_preserved'] = True
    elif problem == 'episode_id_changed':
        data.loc[20:, 'episode_id'] += 1
    elif problem == 'episode_step_gap':
        data.loc[20:, 'episode_step'] += 1
    else:
        data.loc[20, 'termination_reason'] = 'unknown'
    metrics = TrajectoryMetrics().compute_pilot(data)
    _, task, plan = contract()
    assert metrics['pilot_evidence_valid'] == 0.
    assert math.isnan(metrics['final_pitch_angle'])
    assert not audit_rollout(task, plan, data, reward_evidence(data, plan), metrics)['passed']


def capture_environment():
    """模拟原生重置会改写共享物理张量的单环境，记录采集、视频与重置顺序。"""
    torch = pytest.importorskip('torch')
    events = []
    env = SimpleNamespace(num_envs=1, num_bodies=5, device='cpu', dt=.02, sim='fixture')
    env.root_states = torch.zeros(1, 13)
    env.root_states[0, :3] = torch.tensor([1., 2., .5])
    env.root_states[0, 3:7] = torch.tensor(quaternions([.8])[0], dtype=torch.float32)
    env.base_pos, env.base_quat = env.root_states[:, :3], env.root_states[:, 3:7]
    env.rpy = torch.tensor([[0., .8, 0.]])
    env.base_lin_vel, env.base_ang_vel = torch.tensor([[.1, .2, .3]]), torch.tensor([[0., 2., 0.]])
    env.dof_pos, env.dof_vel, env.torques = torch.tensor([[1.2]]), torch.tensor([[3.]]), torch.tensor([[2.]])
    env.actions, env.commands = torch.tensor([[.7]]), torch.tensor([[.4, 0., 0., 0.]])
    env.dof_pos_limits = torch.tensor([[-2., 2.]])
    env.feet_indices = torch.tensor([1, 2, 3, 4])
    env.penalised_contact_indices = env.termination_contact_indices = torch.tensor([0])
    env.contact_forces = torch.zeros(1, 5, 3)
    env.contact_forces[0, 0, 0] = 2.
    env.contact_forces[0, 1:, 2] = 20.
    env.reset_buf, env.time_out_buf = torch.tensor([True]), torch.tensor([False])
    env.episode_length_buf = torch.tensor([11])
    env.rew_buf, env.reward_scales = torch.tensor([-.02]), {'collision': -.02}
    env.last_raw_rewards = {'collision': torch.tensor([1.])}
    rigid = torch.ones(1, 5, 13)
    env.gym = SimpleNamespace(
        fetch_results=lambda *args: events.append('fetch'),
        refresh_rigid_body_state_tensor=lambda *args: events.append('refresh'))

    def native_reset(env_ids):
        """模拟正常执行的原生重置，返回可验证的结果并覆盖新 episode 状态。"""
        events.append('native_reset')
        if len(env_ids):
            env.root_states.zero_()
            env.root_states[0, 6] = 1.
            env.dof_pos.zero_()
            env.dof_vel.zero_()
            env.actions.zero_()
            env.commands.zero_()
            env.episode_length_buf.zero_()
            rigid.zero_()
            env.last_raw_rewards['collision'].fill_(99.)
        return 'native_result'

    env.reset_idx = native_reset

    def collect():
        """执行真实采样函数，测试无需 Isaac Gym 的 CPU 张量路径。"""
        return _sample_current_state(env, rigid, torch.tensor([0]), [[-2., 2.]], [5.])

    return env, collect, events


@pytest.mark.parametrize('timeout', [False, True])
def test_terminal_snapshot_and_video_are_collected_before_native_reset(timeout):
    """终止前物理帧、奖励和视频同步保留，采集之后仍正常执行原生物理重置。"""
    torch = pytest.importorskip('torch')
    env, collect, events = capture_environment()
    env.time_out_buf[0] = timeout
    assert not hasattr(env, '_agent_rollout_terminal_snapshot')
    videos = []

    def record_terminal(snapshot):
        """验证视频回调运行于重置前，并依据同一快照定位终止画面。"""
        events.append('video')
        assert float(env.base_pos[0, 0]) == 1.
        videos.append(snapshot['trajectory']['base_x'])

    _install_terminal_state_capture(env, collect, record_terminal)
    assert env.reset_idx(torch.tensor([0])) == 'native_result'
    sample = _sample_after_step(env, collect, True)
    row, rewards = sample['trajectory'], sample['rewards']
    assert events == ['fetch', 'refresh', 'video', 'native_reset']
    assert videos == [1.]
    assert float(env.base_pos[0, 0]) == 0. and float(env.dof_pos[0, 0]) == 0.
    assert row['base_x'] == 1. and row['quat_y'] == pytest.approx(math.sin(.4))
    assert row['joint_positions'] == pytest.approx([1.2])
    assert row['joint_velocities'] == [3.]
    assert row['actions'] == pytest.approx([.7]) and row['command'][0] == pytest.approx(.4)
    assert row['feet_positions'] == [[1., 1., 1.]] * 4
    assert row['energy'] == pytest.approx(.12)
    assert row['body_collision'] and row['forbidden_contact_force_max'] == 2.
    assert row['episode_step'] == 11 and row['terminal_state_preserved']
    assert row['termination_reason'] == ('timeout' if timeout else 'reset')
    assert row['fall'] is not timeout
    assert rewards['raw_collision'] == 1. and rewards['weighted_collision'] == pytest.approx(-.02)


def test_empty_reset_does_not_collect_or_render_a_snapshot():
    """正常无终止控制步不触发快照复制或视频回调，仍调用原生空重置。"""
    torch = pytest.importorskip('torch')
    env, collect, events = capture_environment()
    _install_terminal_state_capture(env, collect, lambda *args: events.append('video'))
    assert env.reset_idx(torch.tensor([], dtype=torch.long)) == 'native_result'
    assert env._agent_rollout_terminal_snapshot is None
    assert events == ['native_reset']


@pytest.mark.parametrize('failure', ['collection', 'video'])
def test_capture_failure_still_executes_native_reset_and_propagates(failure):
    """采集或视频失败不会阻止原生重置，也不能隐藏异常继续输出伪证据。"""
    torch = pytest.importorskip('torch')
    env, collect, events = capture_environment()

    def fail(*args):
        """用明确异常模拟证据采集或视频输出失败。"""
        raise ValueError('capture fixture failure')

    _install_terminal_state_capture(env, fail if failure == 'collection' else collect,
                                   fail if failure == 'video' else None)
    with pytest.raises(ValueError, match='capture fixture failure'):
        env.reset_idx(torch.tensor([0]))
    assert events[-1] == 'native_reset'
    assert float(env.base_pos[0, 0]) == 0.


def test_terminal_step_without_snapshot_and_contradictory_done_are_rejected():
    """返回的终止状态与快照不一致时立即拒绝，不能退回重置后的混帧采样。"""
    torch = pytest.importorskip('torch')
    env, collect, _ = capture_environment()
    _install_terminal_state_capture(env, collect)
    with pytest.raises(ValueError, match='no synchronized'):
        _sample_after_step(env, collect, True)
    env.reset_idx(torch.tensor([0]))
    with pytest.raises(ValueError, match='disagrees'):
        _sample_after_step(env, collect, False)


def test_capture_hook_rejects_accidental_multi_environment_installation():
    """评估专用 hook 明确拒绝多环境训练，避免常规 PPO 引入终止帧复制。"""
    env, collect, _ = capture_environment()
    env.num_envs = 16
    with pytest.raises(ValueError, match='single rollout'):
        _install_terminal_state_capture(env, collect)
    assert not hasattr(env, '_agent_rollout_terminal_snapshot')
