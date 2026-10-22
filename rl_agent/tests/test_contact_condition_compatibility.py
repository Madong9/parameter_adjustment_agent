"""明确的中文接触力条件与英文表达使用同一执行合同，不推测安全阈值。"""
import copy

import pytest

from rl_training_agent.rewards.validator import RewardValidationError, validate_contact_contract
from rl_training_agent.schemas.rewards import TerminationSpec
from rl_training_agent.training.contact_semantics import (
    body_termination_bound, contact_contract, is_body_termination, resolve_contact_contract,
)
from test_front_flip_backend import contract


@pytest.mark.parametrize('condition,operator,value', [
    ('非足端刚体接触力超过现有安全策略的1.0 N阈值。', '>', 1.),
    ('非足端刚体接触力大于 0.25 N', '>', .25),
    ('非足端接触力大于等于当前安全合同的0.1 N阈值', '>=', .1),
    ('非足端身体接触力达到或超过 0.5 N。', '>=', .5),
    ('非足端刚体接触力不小于 0 N', '>=', 0.),
])
def test_explicit_chinese_condition_preserves_exact_bound(condition, operator, value):
    item = {'name': 'contact_guard', 'condition': condition}
    before = copy.deepcopy(item)
    assert is_body_termination(item)
    assert body_termination_bound(item) == (operator, value)
    equivalent = {'name': 'body_contact', 'condition': 'body contact %s %s N' % (operator, value)}
    assert contact_contract(terminations=[item]) == contact_contract(terminations=[equivalent])
    config = {'terminations': [item], 'contact_semantics': contact_contract(terminations=[item])}
    assert resolve_contact_contract(config) == config['contact_semantics']
    assert item == before


@pytest.mark.parametrize('condition', [
    '非足端刚体接触力超过安全阈值',
    '非足端刚体接触力超过 1.0',
    '非足端刚体接触力超过 1 kg',
    '非足端刚体接触力小于 1 N',
    '非足端刚体接触力未超过 1 N',
    '非足端刚体接触力超过 1 N 或者关节超限',
    '非足端刚体接触力超过 1 N 但落地时允许',
    '非足端刚体接触力超过 1 N 且持续 2 秒',
    '非足端刚体接触力超过 NaN N',
    '非足端刚体接触力超过 -1 N',
    '非足端刚体接触力超过 1% N',
    '足端接触力超过 1 N',
])
def test_ambiguous_different_scope_or_compound_condition_is_rejected(condition):
    item = {'name': 'body_contact', 'condition': condition}
    with pytest.raises(ValueError):
        body_termination_bound(item)
    with pytest.raises(ValueError):
        contact_contract(terminations=[item])


def test_chinese_condition_cannot_raise_fixed_safety_limit():
    _, task, plan = contract()
    plan.terminations = [TerminationSpec(name='forbidden_body_contact',
        condition='非足端刚体接触力超过 2 N阈值。')]
    with pytest.raises(RewardValidationError, match='不得高于'):
        validate_contact_contract(task, plan)
