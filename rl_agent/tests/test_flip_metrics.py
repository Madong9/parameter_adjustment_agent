import numpy as np
import pandas as pd
import pytest

from rl_training_agent.environment.inspector import EnvironmentInspector
from rl_training_agent.environment.metric_registry import normalize_task_metrics
from rl_training_agent.metrics.trajectory_metrics import TrajectoryMetrics
from rl_training_agent.evaluation.deterministic import DeterministicEvaluator
from rl_training_agent.providers.mock_provider import MockLLMReasoningProvider
from rl_training_agent.schemas.task import TaskSpec, MetricThreshold
from rl_training_agent.settings import load_settings


def flip_trajectory():
    angles = np.r_[np.zeros(10), np.linspace(0, 2 * np.pi, 21), np.full(20, 2 * np.pi)]
    data = pd.DataFrame({'sim_time': np.arange(len(angles)) * 0.1,
        'roll': (angles + np.pi) % (2 * np.pi) - np.pi, 'pitch': 0.,
        'base_wx': np.r_[np.zeros(10), np.full(21, np.pi), np.zeros(20)]})
    for name in ('contact_fl', 'contact_fr', 'contact_rl', 'contact_rr'):
        data[name] = True
        data.loc[11:29, name] = False
    return data


def test_flip_acceptance_preserves_quantity_and_threshold(tmp_path):
    manifest = EnvironmentInspector(load_settings().training_root).inspect('go2')
    spec = TaskSpec.parse_obj(MockLLMReasoningProvider().design_task_and_rewards('侧向翻跟头', 'go2', {})['task_spec'])
    spec.required_observations = ['base_quat', 'base_ang_vel', 'feet_air_time']
    spec.safety_constraints = []
    spec.success_metrics = [MetricThreshold(name=name, operator='>=', unit=unit, value=value)
        for name, unit, value in [('final_roll_angle', 'rad', 6.), ('max_roll_velocity', 'rad/s', 3.),
            ('landing_stability', 'score', .8), ('stable_stand_duration', 's', 1.), ('feet_air_time', 's', .5)]]
    assert normalize_task_metrics(spec, manifest.evaluation_metrics) == []
    assert EnvironmentInspector(tmp_path).validate_task(spec, manifest)[0] == []
    metrics = TrajectoryMetrics().compute(flip_trajectory())
    assert metrics['final_roll_angle'] == pytest.approx(2 * np.pi)
    assert metrics['max_roll_velocity'] == pytest.approx(np.pi)
    assert metrics['feet_air_time'] == pytest.approx(1.9)
    assert metrics['landing_stability'] == 1.
    assert metrics['stable_stand_duration'] == pytest.approx(2.1)
    assert all(DeterministicEvaluator()._check(item, metrics).passed for item in spec.success_metrics)


def test_standing_and_oscillation_cannot_fake_flip_or_landing():
    data = flip_trajectory()
    data['roll'] = np.sin(np.arange(len(data)) * .2)
    for name in ('contact_fl', 'contact_fr', 'contact_rl', 'contact_rr'):
        data[name] = True
    metrics = TrajectoryMetrics().compute(data)
    assert metrics['final_roll_angle'] < 2
    assert metrics['feet_air_time'] == 0
    assert metrics['landing_stability'] == 0
    airborne = flip_trajectory()
    for name in ('contact_fl', 'contact_fr', 'contact_rl', 'contact_rr'):
        airborne.loc[11:, name] = False
    metrics = TrajectoryMetrics().compute(airborne)
    assert metrics['landing_stability'] == 0
    assert metrics['stable_stand_duration'] == 0


def test_missing_axis_rate_and_reset_do_not_invent_rotation():
    data = flip_trajectory().drop(columns=['base_wx'])
    data['angular_velocity'] = 20.
    data['termination_reason'] = ''
    data.loc[20, 'termination_reason'] = 'reset'
    metrics = TrajectoryMetrics().compute(data)
    assert 'max_roll_velocity' not in metrics
    assert 'final_roll_angle' not in metrics
    threshold = MetricThreshold(name='max_roll_velocity', operator='>=', value=5., unit='rad/s')
    assert not DeterministicEvaluator()._check(threshold, metrics).passed
    threshold.unit = 'rad'
    assert not DeterministicEvaluator()._check(threshold, {'max_roll_velocity': 20.}).passed
