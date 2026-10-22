"""奖励局部修复权限来自合同谓词，合法字段和固定任务验收不能被重写。"""
import copy

import pytest

from rl_training_agent.rewards.repair_scope import build_repair_scope
from rl_training_agent.schemas.rewards import CurriculumStage, RewardBehaviorLink, RewardTerm
from rl_training_agent.schemas.task import BehaviorRequirement, MetricThreshold

from test_reward_quality_gate import case
from test_front_flip_backend import contract


def capabilities(*names):
    return {'require_objective_links': True, 'rewards': [
        {'name': name, 'sign': 'negative' if name in ('collision', 'dof_pos_limits', 'torque_limits') else 'positive'}
        for name in names]}


def permissions(scope, path):
    return [item for item in scope['allowed_patches'] if item['path'] == path]


def test_valid_plan_has_no_patch_permissions_and_inputs_are_unchanged():
    task, plan = case()
    before = copy.deepcopy((task.dict(), plan.dict()))
    report = build_repair_scope(task, plan, capabilities('tracking_lin_vel'))
    assert report == {'issues': [], 'allowed_patches': [], 'blocked': []}
    assert (task.dict(), plan.dict()) == before


def test_missing_links_only_allow_append_with_exact_identity():
    task, plan = case()
    plan.objective_links = []
    task.required_behaviors = [BehaviorRequirement(name='forward', description='前进')]
    report = build_repair_scope(task, plan, capabilities('tracking_lin_vel'))
    assert report['allowed_patches'] == [
        {'op': 'add', 'path': ['objective_links', '-'], 'identity': {'metric': 'tracking_error'}},
        {'op': 'add', 'path': ['behavior_links', '-'], 'identity': {'behavior': 'forward'}},
    ]


def test_invalid_link_leaves_do_not_allow_valid_references_or_explanations():
    task, plan = case()
    link = plan.objective_links[0]
    link.reward_terms.append('unknown')
    link.rationale = ' '
    link.counterexamples = ['', ' ']
    report = build_repair_scope(task, plan, capabilities('tracking_lin_vel'))
    paths = [item['path'] for item in report['allowed_patches']]
    assert ['objective_links', 0, 'reward_terms', 0] not in paths
    assert ['objective_links', 0, 'reward_terms'] not in paths
    assert permissions(report, ['objective_links', 0, 'reward_terms', 1]) == [
        {'op': 'replace', 'path': ['objective_links', 0, 'reward_terms', 1], 'required_value': 'tracking_lin_vel'}]
    assert ['objective_links', 0, 'rationale'] in paths
    assert ['objective_links', 0, 'counterexamples', 0] in paths
    assert ['objective_links', 0, 'counterexamples', 1] in paths


def test_empty_link_lists_only_allow_append():
    task, plan = case()
    plan.objective_links[0].reward_terms = []
    plan.objective_links[0].counterexamples = []
    report = build_repair_scope(task, plan, capabilities('tracking_lin_vel'))
    assert permissions(report, ['objective_links', 0, 'counterexamples', '-'])[0]['value_kind'] == 'nonempty_string'
    assert permissions(report, ['objective_links', 0, 'reward_terms', '-'])[0]['required_value'] == 'tracking_lin_vel'
    assert not permissions(report, ['objective_links', 0, 'rationale'])


def test_partial_metric_change_can_only_restore_changed_task_values():
    task, plan = case()
    plan.success_metrics[0].value = 999
    plan.success_metrics[0].required = False
    report = build_repair_scope(task, plan, capabilities('tracking_lin_vel'))
    assert report['allowed_patches'] == [
        {'op': 'replace', 'path': ['success_metrics', 0, 'value'], 'required_value': task.success_metrics[0].value},
        {'op': 'replace', 'path': ['success_metrics', 0, 'required'], 'required_value': True},
    ]
    assert all(item['path'][0] not in ('task_spec', 'task_id', 'version', 'velocity_frame') for item in report['allowed_patches'])


def test_missing_metric_append_requires_exact_task_object():
    task, plan = case()
    plan.success_metrics = []
    report = build_repair_scope(task, plan, capabilities('tracking_lin_vel'))
    assert report['allowed_patches'] == [
        {'op': 'add', 'path': ['success_metrics', '-'], 'required_value': task.success_metrics[0].dict()}]


def test_unsupported_parameters_and_invalid_execution_fields_are_local():
    task, plan = case()
    term = plan.terms[0]
    term.parameters = {'tracking_sigma': -1, 'unknown_parameter': {'nested': True}, 'base_height_target': .3}
    term.weight = -1
    term.activation_condition = 'if airborne'
    term.active_phases = ['wrong']
    plan.curriculum = [CurriculumStage(name='learning', start_iteration=0, end_iteration=10)]
    report = build_repair_scope(task, plan, capabilities('tracking_lin_vel'))
    assert permissions(report, ['terms', 0, 'parameters', 'tracking_sigma'])
    assert permissions(report, ['terms', 0, 'parameters', 'unknown_parameter'])[0]['op'] == 'remove'
    assert not permissions(report, ['terms', 0, 'parameters', 'base_height_target'])
    assert permissions(report, ['terms', 0, 'activation_condition'])[0]['required_value'] is None
    assert permissions(report, ['terms', 0, 'active_phases', 0])[0]['required_value'] == 'all'
    assert permissions(report, ['terms', 0, 'weight'])
    assert not permissions(report, ['terms', 0])


def test_curriculum_sorted_checks_keep_original_array_indices():
    task, plan = case()
    plan.curriculum = [
        CurriculumStage(name='later', start_iteration=100, end_iteration=200),
        CurriculumStage(name='earlier', start_iteration=50, end_iteration=150),
    ]
    report = build_repair_scope(task, plan, capabilities('tracking_lin_vel'))
    assert permissions(report, ['curriculum', 1, 'start_iteration'])[0]['required_value'] == 0
    assert permissions(report, ['curriculum', 0, 'start_iteration'])
    assert not permissions(report, ['curriculum', 1, 'end_iteration'])


def test_curriculum_invalid_parameters_are_repaired_without_replacing_stage():
    task, plan = case()
    plan.curriculum = [CurriculumStage(name='learning', start_iteration=0, end_iteration=10,
        parameter_changes={'reward_scales': {'tracking_lin_vel': -2, 'missing_reward': 1},
            'command_scale': -1, 'lin_vel_x': [2, 1], 'unsupported': 'value', 'base_height_target': .3})]
    report = build_repair_scope(task, plan, capabilities('tracking_lin_vel'))
    prefix = ['curriculum', 0, 'parameter_changes']
    assert permissions(report, prefix + ['reward_scales', 'tracking_lin_vel'])
    assert permissions(report, prefix + ['reward_scales', 'missing_reward'])[0]['op'] == 'remove'
    assert permissions(report, prefix + ['unsupported'])[0]['op'] == 'remove'
    assert permissions(report, prefix + ['command_scale'])
    assert permissions(report, prefix + ['lin_vel_x', 0]) and permissions(report, prefix + ['lin_vel_x', 1])
    assert not permissions(report, prefix + ['base_height_target'])
    assert not permissions(report, prefix)


def test_disabled_safety_override_does_not_allow_valid_base_or_other_stage():
    task, plan = case()
    task.safety_constraints = [MetricThreshold(name='forbidden_collisions', operator='==', value=0)]
    plan.terms.append(RewardTerm(name='collision', implementation='registry:collision', purpose='安全', weight=-1))
    plan.curriculum = [
        CurriculumStage(name='valid', start_iteration=0, end_iteration=10),
        CurriculumStage(name='invalid', start_iteration=10, end_iteration=20,
                        parameter_changes={'reward_scales': {'collision': 0}}),
    ]
    report = build_repair_scope(task, plan, capabilities('tracking_lin_vel', 'collision'))
    assert report['allowed_patches'] == [
        {'op': 'replace', 'path': ['curriculum', 1, 'parameter_changes', 'reward_scales', 'collision']}]


def test_missing_registered_signal_can_only_append_designated_name():
    task, plan = case('final_roll_angle', 'roll_rotation')
    report = build_repair_scope(task, plan, capabilities('roll_rotation', 'net_roll_progress'))
    assert permissions(report, ['terms', '-']) == [
        {'op': 'add', 'path': ['terms', '-'], 'identity': {'name': 'net_roll_progress'}}]
    assert not permissions(report, ['terms', 0, 'name'])


def test_unavailable_backend_and_identity_drift_do_not_authorize_substitutions():
    task, plan = case('unknown_goal')
    plan.task_id = 'wrong'
    report = build_repair_scope(task, plan, capabilities('tracking_lin_vel'))
    codes = {item['code'] for item in report['blocked']}
    assert {'missing_objective_backend', 'task_identity_mismatch'} <= codes
    assert not permissions(report, ['terms', '-'])
    assert not permissions(report, ['task_id'])
    assert not permissions(report, ['success_metrics'])


def test_front_flip_missing_reward_only_adds_registered_required_name():
    _, task, plan = contract()
    manifest = capabilities(*(term.name for term in plan.terms))
    plan.terms = [term for term in plan.terms if term.name != 'airborne_duration']
    report = build_repair_scope(task, plan, manifest)
    assert permissions(report, ['terms', '-']) == [
        {'op': 'add', 'path': ['terms', '-'], 'identity': {'name': 'airborne_duration'}}]
    assert all(item['path'][0] != 'task_spec' for item in report['allowed_patches'])


def test_rotation_target_and_recovery_alignment_do_not_expose_other_parameters():
    _, task, plan = contract()
    manifest = capabilities(*(term.name for term in plan.terms))
    progress = next((index, term) for index, term in enumerate(plan.terms) if term.name == 'net_pitch_progress')
    recovery = next((index, term) for index, term in enumerate(plan.terms) if term.name == 'front_flip_recovery')
    progress[1].parameters['net_pitch_target'] = 8.
    report = build_repair_scope(task, plan, manifest)
    assert permissions(report, ['terms', recovery[0], 'parameters', 'net_pitch_target']) == [
        {'op': 'replace', 'path': ['terms', recovery[0], 'parameters', 'net_pitch_target'], 'required_value': 8.}]
    assert not permissions(report, ['terms', progress[0], 'parameters', 'net_pitch_target'])
    progress[1].parameters['net_pitch_target'] = .5
    report = build_repair_scope(task, plan, manifest)
    assert permissions(report, ['terms', progress[0], 'parameters', 'net_pitch_target'])
    assert not permissions(report, ['terms', recovery[0], 'parameters', 'net_pitch_target'])


@pytest.mark.parametrize('invalid', [{'unexpected': 1}, [1, 2]])
def test_invalid_scalar_container_is_blocked_without_whole_container_permission(invalid):
    task, plan = case()
    plan.terms[0].parameters = {'tracking_sigma': invalid}
    report = build_repair_scope(task, plan, capabilities('tracking_lin_vel'))
    assert report['blocked']
    assert not permissions(report, ['terms', 0, 'parameters', 'tracking_sigma'])


def test_duplicate_links_are_blocked_without_list_rewrite_permissions():
    task, plan = case()
    plan.objective_links.append(plan.objective_links[0].copy(deep=True))
    report = build_repair_scope(task, plan, capabilities('tracking_lin_vel'))
    assert any(item['code'] == 'duplicate_objective_links' for item in report['blocked'])
    assert report['allowed_patches'] == []


def test_unknown_reward_only_allows_exact_removal_and_registered_signal_addition():
    task, plan = case('final_roll_angle', 'invented_rotation')
    plan.terms[0].parameters['invented'] = 'invalid'
    report = build_repair_scope(task, plan, capabilities('net_roll_progress'))
    term_grants = [item for item in report['allowed_patches'] if item['path'][0] == 'terms']
    assert term_grants == [
        {'op': 'remove', 'path': ['terms', 0]},
        {'op': 'add', 'path': ['terms', '-'], 'identity': {'name': 'net_roll_progress'}},
    ]
    assert not report['blocked']
    assert permissions(report, ['objective_links', 0, 'reward_terms', 0])[0]['required_value'] == 'net_roll_progress'


def test_unused_registry_alternatives_are_not_authorized_as_link_targets():
    task, plan = case('landing_stability', 'landing_stability')
    plan.objective_links[0].reward_terms = ['wrong']
    report = build_repair_scope(task, plan, capabilities('landing_stability', 'landing_recovery', 'front_flip_recovery'))
    assert permissions(report, ['objective_links', 0, 'reward_terms', 0]) == [
        {'op': 'replace', 'path': ['objective_links', 0, 'reward_terms', 0], 'required_value': 'landing_stability'}]
    assert not permissions(report, ['terms', '-'])


def test_front_flip_mandatory_additions_are_available_to_behavior_links():
    _, task, plan = contract()
    manifest = capabilities(*(term.name for term in plan.terms))
    task.required_behaviors = [BehaviorRequirement(name='takeoff', description='起跳')]
    plan.behavior_links = [RewardBehaviorLink(behavior='takeoff', reward_terms=['wrong'],
        rationale='起跳对应信号', counterexamples=['抖动不算起跳'])]
    plan.terms = [term for term in plan.terms if term.name != 'takeoff_velocity']
    report = build_repair_scope(task, plan, manifest)
    assert any(item.get('required_value') == 'takeoff_velocity'
               for item in permissions(report, ['behavior_links', 0, 'reward_terms', 0]))


def test_front_flip_zero_base_weight_is_repairable_even_with_positive_overrides():
    _, task, plan = contract()
    manifest = capabilities(*(term.name for term in plan.terms))
    index = next(index for index, term in enumerate(plan.terms) if term.name == 'takeoff_velocity')
    plan.terms[index].weight = 0
    plan.curriculum = [CurriculumStage(name='learning', start_iteration=0, end_iteration=100,
        parameter_changes={'reward_scales': {'takeoff_velocity': 1}})]
    report = build_repair_scope(task, plan, manifest)
    assert permissions(report, ['terms', index, 'weight'])
    assert not permissions(report, ['curriculum', 0, 'parameter_changes', 'reward_scales', 'takeoff_velocity'])


def test_disabled_objective_link_gate_does_not_authorize_link_or_signal_edits():
    task, plan = case('unknown_goal')
    plan.objective_links = []
    task.required_behaviors = [BehaviorRequirement(name='forward', description='前进')]
    manifest = capabilities('tracking_lin_vel')
    manifest['require_objective_links'] = False
    report = build_repair_scope(task, plan, manifest)
    assert report == {'issues': [], 'allowed_patches': [], 'blocked': []}
