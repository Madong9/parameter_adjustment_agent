"""前空翻奖励和四元数数值验收的独立 CPU 回归，不启动模型或 PPO。"""
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rl_training_agent.agents.reward_reviewer import RewardReviewAgent
from rl_training_agent.agents.task_semantics import TaskSemanticValidator
from rl_training_agent.environment.inspector import EnvironmentInspector
from rl_training_agent.evaluation.deterministic import DeterministicEvaluator
from rl_training_agent.metrics.trajectory_metrics import TrajectoryMetrics
from rl_training_agent.rewards.compiler import RewardCompiler
from rl_training_agent.rewards.quality_gate import audit_objectives, capability_gaps
from rl_training_agent.rewards.validator import RewardValidationError, validate_plan_execution
from rl_training_agent.schemas.agent_workflow import TaskIntentSpec
from rl_training_agent.schemas.rewards import RewardBehaviorLink, RewardObjectiveLink, RewardPlan, RewardTerm, TerminationSpec
from rl_training_agent.schemas.task import BehaviorRequirement, MetricThreshold, TaskPhase, TaskSpec
from rl_training_agent.training.action_rewards import register_action_rewards
from rl_training_agent.utils.rotation import quaternion_increment_numpy, quaternion_increment_torch


def quaternions(angles, axis=1, initial_yaw=0.):
    """解析生成固定初始航向下绕机体指定轴旋转的 xyzw 姿态。"""
    angles = np.asarray(angles)
    values = np.zeros((len(angles), 4))
    values[:, axis] = np.sin(angles / 2)
    values[:, 3] = np.cos(angles / 2)
    if axis == 1 and initial_yaw:
        values[:, 0] = -math.sin(initial_yaw / 2) * np.sin(angles / 2)
        values[:, 1] = math.cos(initial_yaw / 2) * np.sin(angles / 2)
        values[:, 2] = math.sin(initial_yaw / 2) * np.cos(angles / 2)
        values[:, 3] = math.cos(initial_yaw / 2) * np.cos(angles / 2)
    return values


def trajectory(angles=None, axis=1, initial_yaw=0.):
    """构造准备、单次腾空旋转和稳定落地轨迹，欧拉 pitch 真实回绕。"""
    if angles is None:
        angles = np.r_[np.zeros(4), np.linspace(0, 2 * math.pi, 41), np.full(10, 2 * math.pi)]
    quaternion = quaternions(angles, axis, initial_yaw)
    data = pd.DataFrame(quaternion, columns=['quat_x', 'quat_y', 'quat_z', 'quat_w'])
    data['sim_time'] = np.arange(len(data)) * .02
    data['roll'] = np.where(np.cos(angles) >= 0, 0., math.pi) if axis == 1 else angles if axis == 0 else 0.
    data['pitch'] = np.arcsin(np.sin(angles)) if axis == 1 else 0.
    data['termination_reason'] = ''
    for name in ('contact_fl', 'contact_fr', 'contact_rl', 'contact_rr'):
        data[name] = True
        data.loc[5:43, name] = False
    return data


def contract():
    """构造包含轴向、净旋转、落地与站稳证据的完整前空翻合同。"""
    instruction = '腾空前空翻，足端落地后站稳'
    task = TaskSpec(task_id='front-test', robot='go2', task_name=instruction,
        original_instruction=instruction, normalized_description=instruction, initial_state='全足站立',
        required_behaviors=[BehaviorRequirement(name='forward_rotation', description='完成前向旋转'),
                            BehaviorRequirement(name='stable_recovery', description='落地后站稳')],
        forbidden_behaviors=[], phases=[TaskPhase(name='FLIGHT', description='腾空前翻')],
        required_observations=['base_quat', 'base_ang_vel', 'contact_forces'], required_sensors=[],
        success_metrics=[MetricThreshold(name=name, operator=operator, value=value, unit=unit,
                                        aggregation=aggregation) for name, operator, value, unit, aggregation in (
            ('final_pitch_angle', '>=', 6., 'rad', 'single_action_completion'),
            ('pitch_axis_error', '<=', .25, 'ratio', 'single_action_completion'),
            ('landing_stability', '>=', .8, 'score', 'post_last_airborne_recovery_mean'),
            ('stable_stand_duration', '>=', .15, 's', 'continuous_after_last_landing'))],
        safety_constraints=[MetricThreshold(name=name, operator='==', value=0, unit='count') for name in (
            'forbidden_collisions', 'joint_limit_violations', 'torque_limit_violations')],
        training_budget={}, visual_evaluation_requirements=['核对腾空前翻并足端落地后站稳'])
    parameters = {'net_pitch_progress': {'net_pitch_target': 2 * math.pi},
                  'front_flip_recovery': {'net_pitch_target': 2 * math.pi},
                  'pitch_rotation': {'pitch_rate_target': 6., 'pitch_rate_sigma': 4.}}
    names = ['takeoff_velocity', 'airborne_duration', 'pitch_rotation', 'net_pitch_progress',
             'front_flip_recovery', 'collision', 'dof_pos_limits', 'torque_limits']
    plan = RewardPlan(task_id=task.task_id, version=1, design_rationale=['CPU 合成验证，非真实物理通过'],
        terms=[RewardTerm(name=name, implementation='registry:' + name, purpose=name,
                         weight=-1. if name in names[-3:] else 1., parameters=parameters.get(name, {}),
                         reward_hacking_risks=['错误方向、侧翻和重复起跳必须独立拒绝']) for name in names],
        success_metrics=task.success_metrics,
        objective_links=[RewardObjectiveLink(metric=metric.name,
            reward_terms=[{'final_pitch_angle': 'net_pitch_progress', 'pitch_axis_error': 'pitch_rotation',
                          'landing_stability': 'front_flip_recovery', 'stable_stand_duration': 'front_flip_recovery'}[metric.name]],
            rationale='对应独立实现与验收', counterexamples=['原地站立不算完成']) for metric in task.success_metrics],
        behavior_links=[RewardBehaviorLink(behavior=name, reward_terms=[reward],
            rationale='与原动作一致', counterexamples=['侧翻不能替代前翻']) for name, reward in (
                ('forward_rotation', 'net_pitch_progress'), ('stable_recovery', 'front_flip_recovery'))])
    intent = TaskIntentSpec(original_instruction=instruction, normalized_goal=instruction, robot='go2', action_name=instruction)
    return intent, task, plan


def environment(count=1):
    """构造具有真实四元数、接触和控制步缓存字段的 CPU 环境。"""
    torch = pytest.importorskip('torch')
    class Env:
        """只暴露奖励计算需要的环境状态，不创建 PPO runner。"""
    register_action_rewards(Env)
    env = Env()
    env.dt, env.common_step_counter = .02, 0
    env.episode_length_buf = torch.ones(count, dtype=torch.long)
    env.feet_indices = torch.arange(4)
    env.contact_forces = torch.zeros(count, 4, 3)
    env.contact_forces[:, :, 2] = 2.
    env.base_ang_vel, env.rpy = torch.zeros(count, 3), torch.zeros(count, 3)
    env.root_states = torch.zeros(count, 13)
    env.base_quat = torch.tensor([[0., 0., 0., 1.]]).repeat(count, 1)
    env.cfg = SimpleNamespace(rewards=SimpleNamespace(net_pitch_target=2 * math.pi, pitch_rate_target=6., pitch_rate_sigma=4.))
    return env


def advance(env, quaternion, airborne=True, reset=False):
    """推进一个控制步并设置同步姿态、接触和重置标志。"""
    import torch
    env.common_step_counter += 1
    env.episode_length_buf = torch.ones_like(env.episode_length_buf) if reset else env.episode_length_buf + 1
    env.base_quat = torch.as_tensor(np.asarray(quaternion), dtype=torch.float32).reshape(-1, 4)
    env.contact_forces[:, :, 2] = 0. if airborne else 2.


@pytest.mark.parametrize('initial_yaw', [0., 1.4, -2.5])
def test_quaternion_increment_preserves_body_axis_and_sign(initial_yaw):
    """不同初始航向及四元数正负表示不改变横轴旋转方向，Torch 与 NumPy 一致。"""
    torch = pytest.importorskip('torch')
    values = quaternions([1.2, 1.4], initial_yaw=initial_yaw)
    values[1] *= -1
    actual, valid = quaternion_increment_numpy(values[:1], values[1:])
    assert valid.all() and actual[0] == pytest.approx([0., .2, 0.], abs=1e-8)
    actual_torch, valid_torch = quaternion_increment_torch(torch.tensor(values[:1]), torch.tensor(values[1:]))
    assert valid_torch.all() and actual_torch.numpy() == pytest.approx(actual)


@pytest.mark.parametrize('axis,expected', [(0, 0.), (1, 2 * math.pi), (2, 0.)])
def test_forward_metric_distinguishes_full_pitch_from_roll_and_yaw(axis, expected):
    """完整前翻跨越欧拉奇异区后仍正确计数，完整侧翻或转向不能通过。"""
    data = trajectory(axis=axis)
    metrics = TrajectoryMetrics().compute(data)
    assert metrics['final_pitch_angle'] == pytest.approx(expected)
    assert metrics['pitch_axis_error'] == pytest.approx(0. if axis == 1 else 1.)
    if axis == 1:
        _, task, _ = contract()
        assert all(DeterministicEvaluator()._check(metric, metrics).passed for metric in task.success_metrics)
        assert np.ptp(data.pitch) < math.pi + 1e-6


def test_backward_rotation_and_oscillation_do_not_complete_front_flip():
    """反方向完整旋转保留负值，正反摆动只看净旋转而非路程。"""
    backward = trajectory()
    backward['quat_y'] *= -1
    assert TrajectoryMetrics().compute(backward)['final_pitch_angle'] == pytest.approx(-2 * math.pi)
    angles = np.r_[np.zeros(4), np.linspace(0, math.pi, 21), np.linspace(math.pi, 0, 20), np.zeros(10)]
    assert TrajectoryMetrics().compute(trajectory(angles))['final_pitch_angle'] == pytest.approx(0.)


@pytest.mark.parametrize('problem', ['standing', 'two_flights', 'no_landing', 'reset', 'missing_quaternion', 'zero_quaternion', 'sample_gap'])
def test_invalid_front_flip_evidence_cannot_pass(problem):
    """站立、重复起跳、缺落地、重置和无效/欠采样姿态证据不能变成完成。"""
    data = trajectory()
    if problem == 'standing':
        data[['contact_fl', 'contact_fr', 'contact_rl', 'contact_rr']] = True
    elif problem == 'two_flights':
        data.loc[23, ['contact_fl', 'contact_fr', 'contact_rl', 'contact_rr']] = True
    elif problem == 'no_landing':
        data.loc[44:, ['contact_fl', 'contact_fr', 'contact_rl', 'contact_rr']] = False
    elif problem == 'reset':
        data.loc[20, 'termination_reason'] = 'reset'
    elif problem == 'missing_quaternion':
        data = data.drop(columns=['quat_y'])
    elif problem == 'zero_quaternion':
        data.loc[20, ['quat_x', 'quat_y', 'quat_z', 'quat_w']] = 0.
    else:
        data.loc[20:, 'sim_time'] += 1.
    metrics = TrajectoryMetrics().compute(data)
    _, task, _ = contract()
    assert not DeterministicEvaluator()._check(task.success_metrics[0], metrics).passed


def test_progress_is_cached_capped_and_recovery_requires_complete_landing():
    """同一步多个奖励共享状态，单次正收益封顶，完整旋转后全足落地才奖励恢复。"""
    env = environment()
    assert env._reward_net_pitch_progress().item() == 0.
    assert env._reward_front_flip_recovery().item() == 0.
    total = 0.
    for index, quaternion in enumerate(quaternions(np.linspace(0, 2 * math.pi, 41))[1:], 1):
        advance(env, quaternion, airborne=index < 40)
        reward = env._reward_net_pitch_progress().item()
        total += reward * env.dt
        progress = env._agent_pitch_net.clone()
        recovery = env._reward_front_flip_recovery().item()
        assert np.array_equal(env._agent_pitch_net.numpy(), progress.numpy())
        assert recovery == pytest.approx(1. if index == 40 else 0.)
    assert total == pytest.approx(1., abs=1e-5)
    advance(env, quaternions([2 * math.pi + .1])[0])
    assert env._reward_net_pitch_progress().item() == 0.
    assert env._agent_pitch_flights.item() == 2
    advance(env, [0., 0., 0., 1.], airborne=False, reset=True)
    assert env._reward_front_flip_recovery().item() == 0.
    assert env._agent_pitch_net.item() == 0. and env._agent_pitch_flights.item() == 0


def test_wrong_direction_and_ground_rotation_receive_no_completion_reward():
    """反向翻转与未腾空旋转都没有完成后的站稳奖励。"""
    env = environment(2)
    env._reward_net_pitch_progress()
    for quaternion in quaternions(np.linspace(0, -2 * math.pi, 41))[1:]:
        advance(env, [quaternion, quaternion])
        env.contact_forces[1, :, 2] = 2.
        assert env._reward_net_pitch_progress().tolist() == [0., 0.]
    advance(env, [[0., 0., 0., -1.]] * 2, airborne=False)
    assert env._reward_front_flip_recovery().tolist() == [0., 0.]


def test_pitch_rate_reward_requires_forward_axis_and_airborne_contact():
    """角速度方向、轴向和接触状态都参与旋转代理奖励。"""
    import torch
    env = environment(4)
    env.contact_forces[:, :, 2] = 0.
    env.contact_forces[3, :, 2] = 2.
    env.base_ang_vel = torch.tensor([[0., 6., 0.], [0., -6., 0.], [6., 0., 0.], [0., 6., 0.]])
    rewards = env._reward_pitch_rotation()
    assert rewards[0].item() == 1. and rewards[1].item() < 1e-6
    assert rewards[2].item() < 1e-6 and rewards[3].item() == 0.


@pytest.mark.parametrize('instruction', ['腾空前空翻，足端落地后站稳', '向前腾空翻跟头'])
def test_registered_front_flip_contract_compiles_and_passes_review(tmp_path, instruction):
    """实际项目清单登记新奖励与指标，完整合同通过本地审核并安全编译。"""
    root = Path(__file__).resolve().parents[2] / 'unitree_rl_gym'
    manifest = EnvironmentInspector(root).inspect('go2')
    intent, task, plan = contract()
    intent.original_instruction = task.original_instruction = instruction
    assert TaskSemanticValidator.action_capability_gaps(intent, manifest.dict()) == []
    assert TaskSemanticValidator.assess_task(intent, task).status == 'CONSISTENT'
    assert capability_gaps(task, manifest.dict()) == []
    assert audit_objectives(task, plan)['passed']
    review = RewardReviewAgent().review(intent, task, [plan], dict(manifest.dict(), require_objective_links=True), [])
    assert review.approved, review.conflicts
    RewardCompiler(manifest).compile(plan, tmp_path, task=task)
    assert (tmp_path / 'config.yaml').is_file()
    import json
    from rl_training_agent.training.front_flip_contract import resolve_front_flip_contract
    config = json.loads((tmp_path / 'config.yaml').read_text())
    assert resolve_front_flip_contract(config) is not None


def test_reward_identity_mismatch_is_rejected_by_review_and_compiler(tmp_path):
    """独立入口也不能把旧 G1 任务编号的奖励计划编译成当前 Go2 合同。"""
    root = Path(__file__).resolve().parents[2] / 'unitree_rl_gym'
    manifest = EnvironmentInspector(root).inspect('go2')
    intent, task, plan = contract()
    plan.task_id = 'task-old-g1'
    review = RewardReviewAgent().review(intent, task, [plan], manifest.dict(), [])
    assert not review.approved and any('任务编号' in item for item in review.conflicts)
    with pytest.raises(RewardValidationError, match='编号'):
        RewardCompiler(manifest).compile(plan, tmp_path, task)
    assert not (tmp_path / 'config.yaml').exists()


def test_front_flip_runtime_preserves_contact_timeout_and_restores_pose(tmp_path):
    """完整审核的前翻仅豁免原生姿态终止，身体接触和超时仍导致重置。"""
    import json
    import torch
    from rl_training_agent.training.config_runtime import install_runtime_terminations
    root = Path(__file__).resolve().parents[2] / 'unitree_rl_gym'
    _, task, plan = contract()
    RewardCompiler(EnvironmentInspector(root).inspect('go2')).compile(plan, tmp_path, task=task)
    config = json.loads((tmp_path / 'config.yaml').read_text())

    class NativeEnv:
        """复现 Unitree 的原生接触、普通姿态和超时组合终止。"""

        def __init__(self):
            """构造合法翻转、非法机体接触和超时三种环境。"""
            self.rpy = torch.tensor([[math.pi, 1.2, 0.]] * 3)
            self.device, self.num_bodies = 'cpu', 5
            self.feet_indices = torch.tensor([1, 2, 3, 4])
            self.contact_forces = torch.zeros(3, 5, 3)
            self.contact_forces[1, 0, 2] = 2.
            self.episode_length_buf = torch.tensor([3, 3, 101])
            self.reset_buf = torch.zeros(3, dtype=torch.bool)

        def check_termination(self):
            """执行原生普通姿态、机体触地和超时检查。"""
            self.time_out_buf = self.episode_length_buf > 100
            self.reset_buf = ((self.rpy[:, 0].abs() > .8) | (self.rpy[:, 1].abs() > 1.) |
                              (self.contact_forces[:, 0, 2] > 1.) | self.time_out_buf)

    env = NativeEnv()
    original_pose = env.rpy
    install_runtime_terminations(env, config)
    env.check_termination()
    assert env.reset_buf.tolist() == [False, True, True]
    assert env.rpy is original_pose and env.rpy[0, 1].item() == pytest.approx(1.2)
    config.pop('front_flip_semantics')
    ordinary = NativeEnv()
    install_runtime_terminations(ordinary, config)
    ordinary.check_termination()
    assert ordinary.reset_buf.tolist() == [True, True, True]


@pytest.mark.parametrize('problem', ['axis', 'weight', 'safety'])
def test_runtime_rejects_tampered_front_flip_declaration(tmp_path, problem):
    """修改轴向、实际权重或原始硬约束均不能借前翻声明关闭原生姿态终止。"""
    import json
    from rl_training_agent.training.front_flip_contract import resolve_front_flip_contract
    root = Path(__file__).resolve().parents[2] / 'unitree_rl_gym'
    _, task, plan = contract()
    RewardCompiler(EnvironmentInspector(root).inspect('go2')).compile(plan, tmp_path, task=task)
    config = json.loads((tmp_path / 'config.yaml').read_text())
    if problem == 'axis':
        config['front_flip_semantics']['axis'] = 'body_x'
    elif problem == 'weight':
        config['rewards']['scales']['collision'] = 0.
    else:
        config['safety_constraints'] = []
    with pytest.raises(ValueError):
        resolve_front_flip_contract(config)


@pytest.mark.parametrize('name', ['max_abs_pitch', 'max_abs_roll'])
@pytest.mark.parametrize('operator', ['<=', '=='])
def test_incompatible_user_pose_constraint_cannot_receive_native_exemption(name, operator):
    """用户明确的全程姿态约束不能因前翻例外被删除或放行，包括等于零合同。"""
    intent, task, plan = contract()
    task.safety_constraints.append(MetricThreshold(name=name, operator=operator, value=0., unit='rad'))
    before = task.dict()
    report = TaskSemanticValidator.assess_task(intent, task)
    assert report.status == 'CONFLICT' and any('硬约束冲突' in item for item in report.issues)
    assert task.dict() == before


@pytest.mark.parametrize('problem', ['roll_reward', 'pitch_termination', 'missing_recovery', 'wrong_direction', 'different_targets'])
def test_front_flip_plan_cannot_bypass_axis_or_completion_contract(problem):
    """错误轴、普通俯仰终止、缺恢复、反向速度或目标不一致被本地拒绝。"""
    from rl_training_agent.rewards.action_normalizer import validate_action_semantics
    _, task, plan = contract()
    if problem == 'roll_reward':
        plan.terms.append(RewardTerm(name='roll_rotation', implementation='registered', purpose='wrong axis', weight=1.))
    elif problem == 'pitch_termination':
        plan.terminations = [TerminationSpec(name='fall', condition='abs(pitch)>1.0')]
    elif problem == 'missing_recovery':
        plan.terms = [term for term in plan.terms if term.name != 'front_flip_recovery']
    elif problem == 'wrong_direction':
        next(term for term in plan.terms if term.name == 'pitch_rotation').parameters['pitch_rate_target'] = -6.
    else:
        next(term for term in plan.terms if term.name == 'front_flip_recovery').parameters['net_pitch_target'] = 6.
        assert not audit_objectives(task, plan)['passed']
        return
    with pytest.raises(RewardValidationError):
        validate_plan_execution(plan)
        validate_action_semantics(task, plan)
