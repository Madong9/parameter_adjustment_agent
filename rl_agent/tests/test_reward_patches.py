"""局部补丁不能越权、串候选、按删除后的索引改错字段或覆盖已训练方案。"""
import copy

import pytest
from pydantic import ValidationError

from rl_training_agent.rewards.patches import apply_reward_repairs, reward_plan_sha256
from test_front_flip_backend import contract


def target(plan, patches, index=1):
    return {'candidate_index': index, 'base_sha256': reward_plan_sha256(plan), 'allowed_patches': patches}


def response(plan, patches, index=1):
    return {'candidate_index': index, 'base_sha256': reward_plan_sha256(plan), 'patches': patches}


def test_local_weight_repair_preserves_all_other_content_and_source():
    _, _, plan = contract()
    before = plan.dict()
    edit = {'op': 'replace', 'path': ['terms', 0, 'weight'], 'value': 2.}
    revised, audit = apply_reward_repairs([plan], [target(plan, [dict(op='replace', path=edit['path'])])],
                                        {'repairs': [response(plan, [edit])]})
    expected = copy.deepcopy(before)
    expected['terms'][0]['weight'] = 2.
    assert revised[0].dict() == expected and plan.dict() == before
    assert len(audit['accepted']) == 1 and not audit['rejected']


def test_one_candidates_unauthorized_change_is_atomic_and_other_candidate_can_succeed():
    _, _, first = contract()
    second = first.copy(deep=True)
    second.version = 2
    allowed = [{'op': 'replace', 'path': ['terms', 0, 'weight']}]
    bad = response(first, [{'op': 'replace', 'path': ['terms', 0, 'weight'], 'value': 2.},
                           {'op': 'replace', 'path': ['design_rationale', 0], 'value': '越权重写'}])
    good = response(second, [{'op': 'replace', 'path': ['terms', 0, 'weight'], 'value': 3.}], 2)
    revised, audit = apply_reward_repairs([first, second],
        [target(first, allowed), target(second, allowed, 2)], {'repairs': [bad, good]})
    assert revised[0] == first and revised[1].terms[0].weight == 3.
    assert audit['rejected'][0]['candidate_index'] == 1
    assert audit['unchanged_candidate_indexes'] == [1]


@pytest.mark.parametrize('change', ['stale_hash', 'unrequested_candidate', 'duplicate_candidate', 'protected_version'])
def test_stale_duplicate_and_unrequested_changes_never_touch_source(change):
    _, _, plan = contract()
    edit = {'op': 'replace', 'path': ['terms', 0, 'weight'], 'value': 2.}
    repairs = [response(plan, [edit])]
    allowed = [{'op': 'replace', 'path': edit['path']}]
    if change == 'stale_hash':
        repairs[0]['base_sha256'] = '0' * 64
    elif change == 'unrequested_candidate':
        repairs[0]['candidate_index'] = 2
    elif change == 'duplicate_candidate':
        repairs *= 2
    else:
        repairs[0]['patches'] = [{'op': 'replace', 'path': ['version'], 'value': 99}]
        allowed = [{'op': 'replace', 'path': ['version']}]
    revised, audit = apply_reward_repairs([plan], [target(plan, allowed)], {'repairs': repairs})
    assert revised == [plan] and audit['rejected'] and not audit['accepted']


def test_whole_plan_response_is_rejected_by_patch_schema():
    _, _, plan = contract()
    with pytest.raises(ValidationError):
        apply_reward_repairs([plan], [], plan.dict())


@pytest.mark.parametrize('path', [['terms', -1, 'weight'], ['terms', True, 'weight'], ['terms', '0', 'weight']])
def test_patch_indices_are_strict_and_must_resolve_against_source(path):
    _, _, plan = contract()
    payload = {'repairs': [response(plan, [{'op': 'replace', 'path': path, 'value': 2.}])]}
    try:
        revised, audit = apply_reward_repairs([plan], [target(plan, [{'op': 'replace', 'path': path}])], payload)
    except ValidationError:
        return
    assert revised == [plan] and audit['rejected']


def test_removals_do_not_shift_other_patch_indices():
    _, _, plan = contract()
    grants = [{'op': 'remove', 'path': ['terms', 0]}, {'op': 'remove', 'path': ['terms', 1]},
              {'op': 'replace', 'path': ['terms', 2, 'weight']}]
    edits = [dict(grants[0]), dict(grants[1]), dict(grants[2], value=2.)]
    revised, audit = apply_reward_repairs([plan], [target(plan, grants)], {'repairs': [response(plan, edits)]})
    assert audit['accepted']
    assert revised[0].terms[0].name == plan.terms[2].name and revised[0].terms[0].weight == 2.
    assert revised[0].terms[1:] == plan.terms[3:]


def test_removal_before_nested_add_does_not_change_source_address():
    _, _, plan = contract()
    plan.terms[1].parameters = {}
    original_later_parameters = dict(plan.terms[2].parameters)
    edits = [{'op': 'remove', 'path': ['terms', 0]},
             {'op': 'add', 'path': ['terms', 1, 'parameters', 'min_air_time_threshold'], 'value': .1}]
    grants = [{'op': edit['op'], 'path': edit['path']} for edit in edits]
    revised, audit = apply_reward_repairs([plan], [target(plan, grants)], {'repairs': [response(plan, edits)]})
    assert audit['accepted']
    assert revised[0].terms[0].name == plan.terms[1].name
    assert revised[0].terms[0].parameters == {'min_air_time_threshold': .1}
    assert revised[0].terms[1].parameters == original_later_parameters


def test_overlapping_paths_are_rejected_atomically():
    _, _, plan = contract()
    grants = [{'op': 'remove', 'path': ['terms', 0]}, {'op': 'replace', 'path': ['terms', 0, 'weight']}]
    revised, audit = apply_reward_repairs([plan], [target(plan, grants)],
        {'repairs': [response(plan, [dict(grants[0]), dict(grants[1], value=2.)])]})
    assert revised == [plan] and not audit['accepted']


def test_task_metric_patch_requires_exact_local_value():
    _, task, plan = contract()
    path = ['success_metrics', 0, 'value']
    exact = task.success_metrics[0].value
    for grant, value in [({'op': 'replace', 'path': path}, .1),
                         ({'op': 'replace', 'path': path, 'required_value': exact}, .1)]:
        revised, audit = apply_reward_repairs([plan], [target(plan, [grant])],
            {'repairs': [response(plan, [{'op': 'replace', 'path': path, 'value': value}])]})
        assert revised == [plan] and audit['rejected']


def test_new_entries_require_authorized_identity_and_schema_valid_contents():
    _, _, plan = contract()
    plan.objective_links = []
    grant = {'op': 'add', 'path': ['objective_links', '-'], 'identity': {'metric': 'final_pitch_angle'}}
    invalid = {'metric': 'pitch_axis_error', 'reward_terms': ['pitch_rotation'],
               'rationale': '错误目标', 'counterexamples': ['反例']}
    revised, audit = apply_reward_repairs([plan], [target(plan, [grant])],
        {'repairs': [response(plan, [dict(op='add', path=grant['path'], value=invalid)])]})
    assert revised == [plan] and audit['rejected']
    correct = dict(invalid, metric='final_pitch_angle', reward_terms=['net_pitch_progress'])
    revised, audit = apply_reward_repairs([plan], [target(plan, [grant])],
        {'repairs': [response(plan, [dict(op='add', path=grant['path'], value=correct)])]})
    assert audit['accepted'] and revised[0].objective_links[0].metric == 'final_pitch_angle'
    assert not plan.objective_links


def test_free_text_append_is_limited_to_explicit_nonempty_string_grant():
    _, _, plan = contract()
    plan.objective_links[0].counterexamples = []
    path = ['objective_links', 0, 'counterexamples', '-']
    for value, accepted in [('侧翻不能代替前翻', True), ('  ', False), ({'arbitrary': 'object'}, False)]:
        revised, audit = apply_reward_repairs([plan], [target(plan, [dict(
            op='add', path=path, value_kind='nonempty_string')])],
            {'repairs': [response(plan, [dict(op='add', path=path, value=value)])]})
        assert bool(audit['accepted']) is accepted


def test_candidate_schema_failure_does_not_commit_any_other_edit():
    _, _, plan = contract()
    paths = [['terms', 0, 'weight'], ['terms', 1, 'weight']]
    revised, audit = apply_reward_repairs([plan], [target(plan, [dict(op='replace', path=p) for p in paths])],
        {'repairs': [response(plan, [dict(op='replace', path=paths[0], value=2.),
                                    dict(op='replace', path=paths[1], value='not a number')])]})
    assert revised == [plan] and audit['rejected']
