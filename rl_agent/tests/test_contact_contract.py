"""接触合同的纯 CPU 回归，不调用模型、Isaac Gym 或 PPO 训练。"""
import copy
import json
from types import ModuleType, SimpleNamespace
import sys

import pandas as pd
import pytest

from rl_training_agent.agents.reward_reviewer import RewardReviewAgent
from rl_training_agent.environment.capability_manifest import RewardRegistryItem
from rl_training_agent.rewards.compiler import RewardCompiler
from rl_training_agent.rewards.quality_gate import audit_rollout
from rl_training_agent.rewards.validator import RewardValidationError, validate_contact_contract
from rl_training_agent.schemas.agent_workflow import TaskIntentSpec
from rl_training_agent.schemas.rewards import CurriculumStage, RewardPlan, RewardTerm, TerminationSpec
from rl_training_agent.schemas.task import MetricThreshold, TaskSpec
from rl_training_agent.training.config_runtime import install_runtime_terminations, prepare_training_env_config
from rl_training_agent.training.contact_semantics import (
    collision_count, contact_audit_metadata, contact_contract, forbidden_contact_flags,
    resolve_contact_contract,
)


def case(force=1., operator='<=', alias='forbidden_contact_force'):
    """构造带真实接触阈值且不涉及动作语义分类的任务。"""
    task = TaskSpec(task_id='contact-test', robot='go2', task_name='safe support',
        original_instruction='安全支撑', normalized_description='安全支撑', initial_state='standing',
        required_behaviors=[], forbidden_behaviors=[], phases=[], required_observations=[],
        required_sensors=[], success_metrics=[], visual_evaluation_requirements=[], training_budget={},
        safety_constraints=[MetricThreshold(name=alias, operator=operator, value=force, unit='N'),
                            MetricThreshold(name='forbidden_collisions', operator='==', value=0, unit='count')])
    plan = RewardPlan(task_id=task.task_id, version=1, design_rationale=['真实接触安全合同'],
        terms=[RewardTerm(name='collision', implementation='registry:collision', purpose='禁止接触', weight=-1,
                          reward_hacking_risks=['不能漏掉 base/hip 接触'])])
    return task, plan


def compiled(task, plan):
    """保留原始硬约束并构建与编译器一致的配置。"""
    constraints = [item.dict() for item in task.safety_constraints]
    terminations = [item.dict() for item in plan.terminations]
    return {'rewards': {'scales': {'collision': -1}, 'terms': [item.dict() for item in plan.terms]},
            'safety_constraints': constraints, 'terminations': terminations,
            'contact_semantics': contact_contract(constraints, terminations)}


def env(values):
    """模拟 base、hip、calf、foot 四个刚体，原生终止故意漏掉 base/hip。"""
    torch = pytest.importorskip('torch')
    class FakeEnv:
        def check_termination(self):
            """只提供原生检查接口；合同必须独立补足禁止部位。"""
            self.reset_buf[:] = False
    result = FakeEnv()
    result.device, result.num_bodies = 'cpu', 4
    result.body_names = ['base', 'hip', 'calf', 'foot']
    result.feet_indices = torch.tensor([3])
    result.contact_forces = torch.zeros((len(values), 4, 3))
    result.contact_forces[:, 0, 0] = torch.tensor(values)
    result.contact_forces[:, 3, 2] = 100
    result.reset_buf = torch.zeros(len(values), dtype=torch.bool)
    return result


def test_compiler_preserves_task_thresholds_and_fixed_safety_baseline(tmp_path):
    task, plan = case(.05)
    before = task.dict()
    registry = RewardRegistryItem(name='collision', implementation='_reward_collision',
        config_key='rewards.scales.collision', expected_raw_range=(0., 20.),
        dependencies=['contact_forces'], sign='negative', default_weight=-1)
    manifest = SimpleNamespace(robot='go2', rewards=[registry])
    RewardCompiler(manifest).compile(plan, tmp_path, task=task)
    config = json.loads((tmp_path / 'config.yaml').read_text())
    assert config['safety_constraints'] == [item.dict() for item in task.safety_constraints]
    assert config['contact_semantics'] == resolve_contact_contract(config)
    assert config['contact_semantics']['collision_force_threshold_n'] == .05
    assert config['contact_semantics']['body_termination_force_threshold_n'] == .05
    assert task.dict() == before
    task.safety_constraints[0].value = 2000
    assert contact_contract(task.safety_constraints)['body_termination_force_threshold_n'] == 1.


@pytest.mark.parametrize('key,value', [
    ('scope', 'thigh_and_calf'), ('collision_force_threshold_n', 1.),
    ('body_termination_force_threshold_n', 100.), ('terminate_on_forbidden_collision', False),
    ('version', 'unknown-version'), ('collision_operator', '>='),
])
def test_runtime_rejects_weakened_or_replaced_compiled_contact_contract(key, value):
    task, plan = case()
    config = compiled(task, plan)
    config['contact_semantics'][key] = value
    with pytest.raises(ValueError, match='接触合同'):
        resolve_contact_contract(config)


def test_original_task_constraint_change_cannot_reuse_old_contract():
    task, plan = case(.05)
    config = compiled(task, plan)
    config['safety_constraints'][0]['value'] = .02
    with pytest.raises(ValueError, match='任务硬约束'):
        resolve_contact_contract(config)


def test_collision_reward_termination_and_scan_share_body_set_and_event_threshold():
    task, plan = case()
    result = env([0., .05, .1, .1001, .9, 2461.])
    install_runtime_terminations(result, compiled(task, plan))
    collision, maximum, terminate = forbidden_contact_flags(result)
    assert collision_count(result).tolist() == [0., 0., 0., 1., 1., 1.]
    assert collision.tolist() == terminate.tolist() == [False, False, False, True, True, True]
    assert maximum.tolist() == pytest.approx([0., .05, .1, .1001, .9, 2461.])
    result.check_termination()
    assert result.reset_buf.tolist() == terminate.tolist()
    metadata = contact_audit_metadata(result)
    assert metadata['forbidden_rigid_body_names'] == ['base', 'hip', 'calf']
    assert metadata['forbidden_rigid_body_indices'] == [0, 1, 2]
    assert metadata['foot_rigid_body_indices'] == [3]


@pytest.mark.parametrize('operator,expected', [('<=', [False, True]), ('<', [True, True])])
def test_stricter_force_contract_never_leaves_zero_collision_reward(operator, expected):
    task, plan = case(.05, operator)
    result = env([.05, .051])
    install_runtime_terminations(result, compiled(task, plan))
    collision, _, terminate = forbidden_contact_flags(result)
    assert collision.tolist() == terminate.tolist() == expected
    assert (collision_count(result) > 0).tolist() == expected


def test_disabled_body_termination_cannot_disable_fixed_hard_safety():
    task, plan = case()
    plan.terminations = [TerminationSpec(name='body_contact', condition='body contact > 1 N', enabled=False)]
    result = env([2.])
    install_runtime_terminations(result, compiled(task, plan))
    result.check_termination()
    assert result.reset_buf.item()


def test_runtime_registers_collision_adapter_without_editing_unitree(monkeypatch):
    class FakeLeggedRobot:
        pass
    module = ModuleType('legged_gym.envs.base.legged_robot')
    module.LeggedRobot = FakeLeggedRobot
    monkeypatch.setitem(sys.modules, module.__name__, module)
    task, plan = case()
    cfg = SimpleNamespace(rewards=SimpleNamespace(scales=SimpleNamespace(collision=-1, orientation=-1)),
        asset=SimpleNamespace(terminate_after_contacts_on=['base'], penalize_contacts_on=['thigh', 'calf']))
    prepare_training_env_config(cfg, compiled(task, plan))
    assert FakeLeggedRobot._reward_collision is collision_count
    assert cfg._agent_contact_contract == compiled(task, plan)['contact_semantics']
    assert cfg.rewards.only_positive_rewards is False


def test_review_rejects_missing_and_disabled_course_collision_penalty():
    task, plan = case()
    intent = TaskIntentSpec(original_instruction='安全支撑', normalized_goal='安全支撑', robot='go2', action_name='safe support')
    capabilities = {'rewards': [{'name': 'collision'}]}
    plan.curriculum = [CurriculumStage(name='unsafe', start_iteration=0, end_iteration=10,
                                     parameter_changes={'reward_scales': {'collision': 0}})]
    review = RewardReviewAgent().review(intent, task, [plan], capabilities, ['禁止奖励投机'])
    assert not review.approved
    assert any('禁止接触硬约束' in item for item in review.conflicts)
    plan.curriculum = []
    plan.terms = []
    with pytest.raises(RewardValidationError, match='负 collision'):
        validate_contact_contract(task, plan)


def test_review_rejects_weaker_body_termination_than_task_contract():
    task, plan = case(.05)
    plan.terminations = [TerminationSpec(name='forbidden_body_contact', condition='body contact > 1 N')]
    with pytest.raises(RewardValidationError, match='终止阈值'):
        validate_contact_contract(task, plan)


@pytest.mark.parametrize('alias', ['forbidden_body_contact', 'body_contact_force'])
def test_force_aliases_preserve_the_same_hard_contract(alias):
    task, plan = case(.05, alias=alias)
    assert validate_contact_contract(task, plan)['body_termination_force_threshold_n'] == .05


def evidence(events, raw):
    """构造严格同帧奖励证据，避免平均值掩盖实际碰撞帧。"""
    trajectory = pd.DataFrame({'sim_time': [0., .02], 'body_collision': events})
    rewards = pd.DataFrame({'sim_time': [0., .02], 'raw_collision': raw,
                            'weighted_collision': [-value * .02 for value in raw]})
    return trajectory, rewards


def test_audit_rejects_average_reward_that_does_not_penalize_actual_collision_frame():
    task, plan = case()
    traj, rewards = evidence([True, False], [0., 1.])
    report = audit_rollout(task, plan, traj, rewards,
        {'forbidden_contact_force': 20., 'forbidden_collisions': 1})
    assert not report['passed']
    assert any('同帧' in item for item in report['errors'])


def test_small_safe_positive_contact_force_does_not_imply_missing_penalty():
    task, plan = case()
    traj, rewards = evidence([False, False], [0., 0.])
    assert audit_rollout(task, plan, traj, rewards,
        {'forbidden_contact_force': .05, 'forbidden_collisions': 0})['passed']


def test_production_audit_requires_resolved_contact_metadata_and_checks_body_completeness():
    task, plan = case()
    traj, rewards = evidence([False, False], [0., 0.])
    metrics = {'forbidden_contact_force': 0., 'forbidden_collisions': 0}
    assert not audit_rollout(task, plan, traj, rewards, metrics, require_contact_metadata=True)['passed']
    result = env([0.])
    install_runtime_terminations(result, compiled(task, plan))
    metadata = contact_audit_metadata(result)
    metadata['body_collision_scope'] = result._agent_contact_contract['scope']
    metadata['body_collision_force_threshold_n'] = result._agent_contact_contract['collision_force_threshold_n']
    assert audit_rollout(task, plan, traj, rewards, metrics, metadata, True)['passed']
    incomplete = copy.deepcopy(metadata)
    incomplete['forbidden_rigid_body_indices'] = [2]
    report = audit_rollout(task, plan, traj, rewards, metrics, incomplete, True)
    assert not report['passed']
    assert any('完整的非足端' in item for item in report['errors'])


def test_ground_roll_does_not_create_implicit_body_contact_exemption():
    task, plan = case()
    task.original_instruction = '地面侧向翻滚，身体低力接触地面'
    assert validate_contact_contract(task, plan)['scope'] == 'all_non_foot_rigid_bodies'
    config = compiled(task, plan)
    config['contact_semantics']['allowed_contact_phases'] = ['rolling']
    with pytest.raises(ValueError, match='任务硬约束'):
        resolve_contact_contract(config)


@pytest.mark.parametrize('condition', ['body contact < 1 N', 'body contact > 1 kN',
                                     'body contact > NaN', 'body contact whenever unsafe'])
def test_unsupported_body_termination_text_is_rejected_before_environment_creation(condition):
    task, plan = case()
    plan.terminations = [TerminationSpec(name='body_contact', condition=condition)]
    with pytest.raises(RewardValidationError, match='禁止接触合同无效'):
        validate_contact_contract(task, plan)


def test_legacy_contract_is_explicitly_upgraded_without_weakening_body_scope():
    original = {'contact_semantics': {'version': 'all-non-foot-v1',
        'collision_force_threshold_n': .1, 'body_termination_force_threshold_n': 1.}}
    resolved = resolve_contact_contract(original)
    assert resolved['scope'] == 'all_non_foot_rigid_bodies'
    assert resolved['terminate_on_forbidden_collision']
    original['contact_semantics']['body_termination_force_threshold_n'] = 100.
    with pytest.raises(ValueError, match='旧接触合同'):
        resolve_contact_contract(original)


def test_environment_missing_contact_tensors_cannot_execute_explicit_contract():
    task, plan = case()
    result = SimpleNamespace(check_termination=lambda: None)
    with pytest.raises(ValueError, match='缺少接触张量'):
        install_runtime_terminations(result, compiled(task, plan))


@pytest.mark.parametrize('metadata', [[], 'bad-json-shape', 4, True])
def test_malformed_metadata_fails_audit_without_crashing(metadata):
    task, plan = case()
    traj, rewards = evidence([False, False], [0., 0.])
    report = audit_rollout(task, plan, traj, rewards,
        {'forbidden_contact_force': 0., 'forbidden_collisions': 0}, metadata, True)
    assert not report['passed']


def test_fabricated_huge_body_count_is_rejected_without_allocating_a_range():
    task, plan = case()
    traj, rewards = evidence([False, False], [0., 0.])
    result = env([0.])
    install_runtime_terminations(result, compiled(task, plan))
    metadata = contact_audit_metadata(result)
    metadata.update(body_collision_scope=result._agent_contact_contract['scope'],
                    body_collision_force_threshold_n=.1, rigid_body_count=10 ** 12)
    report = audit_rollout(task, plan, traj, rewards,
        {'forbidden_contact_force': 0., 'forbidden_collisions': 0}, metadata, True)
    assert not report['passed']
