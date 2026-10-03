from types import SimpleNamespace, ModuleType
import sys

import pytest

from rl_training_agent.environment.inspector import EnvironmentInspector
from rl_training_agent.providers.mock_provider import MockLLMReasoningProvider
from rl_training_agent.orchestration.orchestrator import TrainingOrchestrator
from rl_training_agent.rewards.compiler import RewardCompiler
from rl_training_agent.rewards.validator import RewardValidationError, validate_plan_execution
from rl_training_agent.schemas.rewards import RewardPlan, RewardTerm, CurriculumStage, TerminationSpec
from rl_training_agent.schemas.task import TaskSpec
from rl_training_agent.settings import load_settings
from rl_training_agent.training.action_rewards import register_action_rewards
from rl_training_agent.training.config_runtime import prepare_training_env_config, reward_scales_for_stage, install_runtime_terminations, apply_runtime_stage


def old_plan():
    return RewardPlan(task_id='task-flip', version=1, design_rationale=['test'], terms=[
        RewardTerm(name=name, implementation='native', purpose='test', weight=weight, active_phases=phases)
        for name, weight, phases in [('feet_air_time', 2., ['all']), ('jump_height', 2., ['takeoff']),
            ('tracking_ang_vel', 2., ['flight']), ('landing_stability', 2., ['landing']), ('orientation', -.2, ['all'])]],
        curriculum=[CurriculumStage(name='stability_curriculum', start_iteration=0, end_iteration=12000,
            parameter_changes={'landing_stability': {'start_value': .5, 'end_value': 1.5}})],
        terminations=[TerminationSpec(name='fall', condition='abs(pitch) > 1.0 or abs(roll) > 0.8')])


def test_failed_flip_plan_gets_executable_rewards_and_termination(tmp_path):
    manifest = EnvironmentInspector(load_settings().training_root).inspect('go2')
    task = TaskSpec.parse_obj(MockLLMReasoningProvider().design_task_and_rewards('侧向翻跟头', 'go2', {})['task_spec'])
    task.original_instruction = '侧向翻跟头'
    plan = old_plan()
    with pytest.raises(RewardValidationError):
        validate_plan_execution(plan)
    audit = TrainingOrchestrator._normalize_plan_for_task(task, plan)
    assert audit
    RewardCompiler(manifest).compile(plan, tmp_path)
    config = __import__('json').loads((tmp_path / 'config.yaml').read_text())
    scales = reward_scales_for_stage(config, 'side_flip_learning')
    assert all(scales[name] > 0 for name in ('takeoff_velocity', 'airborne_duration', 'roll_rotation', 'landing_recovery'))
    assert 'orientation' not in scales and 'tracking_ang_vel' not in scales
    assert plan.terminations[0].condition == 'abs(pitch) > 1.0'
    assert plan.success_metrics == task.success_metrics
    assert __import__('json').loads((tmp_path / 'effective_reward_scales.json').read_text())['side_flip_learning'] == scales


@pytest.mark.parametrize('problem', ['phase', 'parameter', 'override_sign', 'condition', 'command_nan'])
def test_generic_invalid_execution_is_rejected(problem):
    plan = RewardPlan(task_id='task-x', version=1, design_rationale=['test'], terms=[
        RewardTerm(name='tracking_lin_vel', implementation='native', purpose='test', weight=1.)],
        curriculum=[CurriculumStage(name='walk', start_iteration=0, end_iteration=100)])
    if problem == 'phase':
        plan.terms[0].active_phases = ['fly']
    elif problem == 'parameter':
        plan.curriculum[0].parameter_changes['ignored_reward'] = {'schedule': 'linear'}
    elif problem == 'override_sign':
        plan.curriculum[0].parameter_changes['reward_scales'] = {'tracking_lin_vel': -1.}
    elif problem == 'condition':
        plan.terms[0].activation_condition = 'when airborne'
    else:
        plan.curriculum[0].parameter_changes['lin_vel_x'] = [0., float('nan')]
    with pytest.raises(RewardValidationError):
        validate_plan_execution(plan)


def test_action_rewards_distinguish_yaw_and_roll_and_require_real_landing(monkeypatch):
    torch = pytest.importorskip('torch')
    class Env:
        pass
    register_action_rewards(Env)
    env = Env()
    env.feet_indices = torch.arange(4)
    env.contact_forces = torch.zeros(3, 4, 3)
    env.base_ang_vel = torch.tensor([[0., 0., 6.], [6., 0., 0.], [6., 0., 0.]])
    env.root_states = torch.zeros(3, 13)
    env.rpy = torch.zeros(3, 3)
    env.episode_length_buf = torch.full((3,), 5)
    env.cfg = SimpleNamespace(rewards=SimpleNamespace(roll_rate_target=6., roll_rate_sigma=4., takeoff_velocity_target=.8))
    env.contact_forces[2, :, 2] = 2.
    assert env._reward_roll_rotation()[1] == 1.
    assert env._reward_roll_rotation()[0] < .001
    assert env._reward_roll_rotation()[2] == 0.
    assert env._reward_airborne_duration().tolist() == [1., 1., 0.]
    assert env._reward_landing_recovery().tolist() == [0., 0., 0.]
    env.base_ang_vel[:] = 0.
    env.contact_forces[:, :, 2] = 2.
    assert env._reward_landing_recovery().tolist() == [1., 1., 0.]
    env.episode_length_buf[:] = 0
    assert env._reward_landing_recovery().tolist() == [0., 0., 0.]
    env.root_states[:, 9] = torch.tensor([.8, -.8, 0.])
    assert env._reward_takeoff_velocity().tolist() == [1., 0., 0.]


def test_registration_precedes_env_construction_and_removes_inherited_rewards(monkeypatch):
    class Env:
        pass
    module = ModuleType('legged_gym.envs.base.legged_robot')
    module.LeggedRobot = Env
    monkeypatch.setitem(sys.modules, module.__name__, module)
    cfg = SimpleNamespace(rewards=SimpleNamespace(scales=SimpleNamespace(
        tracking_lin_vel=1., orientation=-1., torques=-.1)),
        asset=SimpleNamespace(terminate_after_contacts_on=[], penalize_contacts_on=[]))
    config = {'rewards': {'scales': {'roll_rotation': 1.}, 'terms': []}}
    prepare_training_env_config(cfg, config)
    assert hasattr(Env, '_reward_roll_rotation')
    assert cfg.rewards.scales.roll_rotation == 1.
    assert cfg.rewards.scales.tracking_lin_vel == cfg.rewards.scales.orientation == cfg.rewards.scales.torques == 0.


def test_combined_termination_uses_individual_axis_limits():
    torch = pytest.importorskip('torch')
    env = SimpleNamespace(rpy=torch.tensor([[.9, 0.], [0., .9], [0., 1.1]]),
        reset_buf=torch.zeros(3, dtype=torch.bool), check_termination=lambda: None)
    install_runtime_terminations(env, {'terminations': [
        {'name': 'fall', 'condition': 'abs(pitch) > 1.0 or abs(roll) > 0.8'}]})
    env.check_termination()
    assert env.reset_buf.tolist() == [True, False, True]


def test_zero_initial_weight_is_registered_for_later_curriculum():
    cfg = SimpleNamespace(rewards=SimpleNamespace(scales=SimpleNamespace(tracking_lin_vel=1.)),
        asset=SimpleNamespace(terminate_after_contacts_on=[], penalize_contacts_on=[]))
    config = {'rewards': {'scales': {'tracking_lin_vel': 0.}, 'terms': []}, 'curriculum': [
        {'name': 'stand', 'parameter_changes': {'reward_scales': {'tracking_lin_vel': 0.}}},
        {'name': 'walk', 'parameter_changes': {'reward_scales': {'tracking_lin_vel': 1.}}}]}
    prepare_training_env_config(cfg, config)
    assert cfg.rewards.scales.tracking_lin_vel == 1.
    env = SimpleNamespace(cfg=cfg, dt=.02, reward_scales={'tracking_lin_vel': .02}, command_ranges={})
    apply_runtime_stage(env, config, 'stand', {})
    assert env.reward_scales['tracking_lin_vel'] == 0.
    apply_runtime_stage(env, config, 'walk', {})
    assert env.reward_scales['tracking_lin_vel'] == .02


def test_flip_conflicting_hard_constraints_are_retained_and_rejected():
    from rl_training_agent.schemas.task import MetricThreshold
    from rl_training_agent.rewards.action_normalizer import validate_action_semantics
    task = TaskSpec.parse_obj(MockLLMReasoningProvider().design_task_and_rewards('侧向翻跟头', 'go2', {})['task_spec'])
    task.original_instruction = '侧向翻跟头'
    task.safety_constraints = [MetricThreshold(name='roll_limit', operator='<=', value=.8, unit='rad')]
    plan = old_plan()
    TrainingOrchestrator._normalize_plan_for_task(task, plan)
    assert 'abs(roll)' in plan.terminations[0].condition
    with pytest.raises(RewardValidationError, match='硬约束冲突'):
        validate_action_semantics(task, plan)
