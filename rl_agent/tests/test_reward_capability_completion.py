from types import SimpleNamespace
import math

import pytest

from rl_training_agent.training.action_rewards import _reward_net_roll_progress
from rl_training_agent.training.config_runtime import prepare_training_env_config
from rl_training_agent.environment.inspector import EnvironmentInspector
from rl_training_agent.settings import load_settings
from rl_training_agent.rewards.quality_gate import capability_gaps


def env():
    torch = pytest.importorskip('torch')
    return SimpleNamespace(rpy=torch.zeros((1, 3)), contact_forces=torch.zeros((1, 4, 3)),
        feet_indices=torch.arange(4), episode_length_buf=torch.tensor([1]),
        cfg=SimpleNamespace(rewards=SimpleNamespace(net_roll_target=2*math.pi)), dt=.02)


def step(e, angle, length):
    e.rpy[0, 0] = math.atan2(math.sin(angle), math.cos(angle))
    e.episode_length_buf[0] = length
    return _reward_net_roll_progress(e).item() * e.dt


def test_unwrapped_net_rotation_matches_full_turn_and_caps_second_turn():
    e = env()
    assert step(e, 0, 1) == 0
    values = [step(e, i*math.pi/10, i+1) for i in range(1, 41)]
    assert sum(values[:20]) == pytest.approx(1, abs=1e-6)
    assert sum(values[20:]) == pytest.approx(0, abs=1e-6)


def test_oscillation_and_repeated_jumps_cannot_repeat_progress_bonus():
    e = env()
    step(e, 0, 1)
    assert step(e, math.pi/2, 2) == pytest.approx(.25)
    assert step(e, 0, 3) == pytest.approx(-.25)
    assert step(e, math.pi/2, 4) == pytest.approx(0)
    assert step(e, math.pi, 5) == pytest.approx(.25)
    e.contact_forces[:, :, 2] = 10
    assert step(e, math.pi, 6) == 0
    e.contact_forces[:] = 0
    assert step(e, math.pi, 7) == 0


def test_ground_motion_is_not_rewarded_and_episode_reset_clears_state():
    e = env()
    step(e, 0, 1)
    e.contact_forces[:, :, 2] = 10
    assert step(e, math.pi/2, 2) == 0
    e.contact_forces[:] = 0
    assert step(e, math.pi/2, 3) == 0
    assert step(e, 0, 1) == 0
    assert step(e, -math.pi/2, 2) == pytest.approx(.25)


def test_registry_only_exposes_implemented_safety_and_net_progress():
    manifest = EnvironmentInspector(load_settings().training_root).inspect('go2')
    rewards = {r.name:r for r in manifest.rewards}
    assert rewards['torque_limits'].sign == 'negative'
    assert rewards['net_roll_progress'].parameters['net_roll_target'] == pytest.approx(2*math.pi)
    assert rewards['net_roll_progress'].expected_raw_range[0] < 0
    task = SimpleNamespace(success_metrics=[SimpleNamespace(name='final_roll_angle', required=True, aggregation='final')],
        safety_constraints=[SimpleNamespace(name='torque_limit_violations', required=True)])
    assert capability_gaps(task, manifest.dict()) == []
    incomplete = {'rewards':[{'name':'roll_rotation'}, {'name':'torques'}]}
    assert capability_gaps(task, incomplete) == ['reward_signal:final_roll_angle', 'safety_reward:torque_limits']


def test_runtime_installs_zero_default_safety_scale():
    cfg = SimpleNamespace(rewards=SimpleNamespace(scales=SimpleNamespace(torques=-.001)),
        asset=SimpleNamespace(terminate_after_contacts_on=[], penalize_contacts_on=[]))
    prepare_training_env_config(cfg, {'rewards': {'scales': {'torque_limits':-1}, 'terms':[]}})
    assert cfg.rewards.scales.torque_limits == -1
    assert cfg.rewards.scales.torques == 0


@pytest.mark.parametrize('target', [0, -1])
def test_zero_or_negative_net_target_rejected_before_training(target):
    from rl_training_agent.schemas.rewards import RewardPlan, RewardTerm
    from rl_training_agent.rewards.validator import validate_plan_execution, RewardValidationError
    plan = RewardPlan(task_id='t', version=1, design_rationale=['test'], terms=[
        RewardTerm(name='net_roll_progress', implementation='registered', purpose='net', weight=1,
            parameters={'net_roll_target':target})])
    with pytest.raises(RewardValidationError):
        validate_plan_execution(plan)


def test_one_environment_reset_does_not_reset_another_progress():
    torch = pytest.importorskip('torch')
    e = env()
    e.rpy = torch.zeros((2, 3))
    e.contact_forces = torch.zeros((2, 4, 3))
    e.episode_length_buf = torch.tensor([1, 1])
    _reward_net_roll_progress(e)
    e.episode_length_buf[:] = 2
    e.rpy[:, 0] = math.pi/2
    assert (_reward_net_roll_progress(e)*e.dt).tolist() == pytest.approx([.25, .25])
    e.episode_length_buf[:] = torch.tensor([1, 3])
    e.rpy[:, 0] = torch.tensor([0., math.pi])
    assert (_reward_net_roll_progress(e)*e.dt).tolist() == pytest.approx([0., .25])


def test_safety_registry_does_not_advertise_missing_native_function(tmp_path):
    from rl_training_agent.environment.reward_registry import RewardRegistry
    registry = RewardRegistry(tmp_path)
    registry._methods = lambda: {}
    registry._weights = lambda robot: {}
    assert 'torque_limits' not in {item.name for item in registry.inspect('go2')}


def test_implementation_gap_stops_redesign_before_calling_gpt(tmp_path):
    from test_failure_reward_redesign import setup_redesign
    from rl_training_agent.schemas.task import MetricThreshold
    from rl_training_agent.rewards.validator import RewardValidationError
    from rl_training_agent.utils.io import read_json
    runner, task, selected, diagnosis, manifest, calls = setup_redesign(tmp_path, ['valid'])
    task.success_metrics.append(MetricThreshold(name='unimplemented_action_completion', operator='>=', value=1))
    with pytest.raises(RewardValidationError, match='补齐本地奖励实现'):
        runner._redesign_failed_rewards(task, tmp_path, selected, diagnosis, 1, manifest)
    assert calls == []
    assert read_json(tmp_path/'reward_redesign/round_01/capability_gaps.json')['gaps'] == [
        'reward_signal:unimplemented_action_completion']
