"""明确的动作对照说明不能成为额外训练目标，独立动作要求仍须拒绝。"""
import pytest

from rl_training_agent.agents.task_semantics import TaskSemanticValidator
from rl_training_agent.schemas.agent_workflow import TaskIntentSpec
from rl_training_agent.utils.task_semantics import (
    ForwardFlipKind, RollMotionKind, classify_forward_flip, classify_roll_motion,
)


@pytest.mark.parametrize('description', [
    '前空翻与绕机体纵向轴的侧空翻、允许机体接触地面的前滚翻具有不同的动作语义。',
    '前空翻与侧空翻不同。',
    '前空翻和侧翻是不同的动作。',
    '前空翻区别于侧滚。',
    '前空翻不同于允许机体接触地面的前滚翻。',
    '前空翻有别于侧空翻或前滚翻。',
    '前空翻不能与侧翻混同。',
    '前空翻不得和前滚翻等同。',
    '前空翻不可与侧翻混为一谈。',
    '前空翻不能混同为侧空翻。',
    '前空翻不应等同于地面前滚翻。',
    'front flip differs from a side flip',
    'front flip is different from a side flip and a forward roll',
    'front flip is distinct from a side flip',
    'front flip is not equivalent to a forward roll',
    'front flip is not the same as a side flip',
])
def test_explicit_contrast_keeps_front_flip_subject(description):
    """保留前空翻主目标，去除明确被区分开的侧翻及地面前滚翻对照项。"""
    assert classify_forward_flip(description) == ForwardFlipKind.AERIAL
    assert classify_roll_motion(description) == RollMotionKind.NONE
    intent = TaskIntentSpec(original_instruction='向前腾空翻跟头', robot='go2',
        action_name='前空翻', normalized_goal=description)
    assert TaskSemanticValidator.assess_intent(intent).status == 'CONSISTENT'


@pytest.mark.parametrize('description', [
    '侧空翻与前空翻不同。',
    '侧空翻区别于前滚翻。',
    '侧空翻不能与前空翻混同。',
    '侧空翻不能混同为前空翻。',
    'side flip is different from a front flip',
])
def test_reverse_contrast_keeps_actual_side_flip_goal(description):
    """比较方向决定被保留的主语，不能把真实侧空翻洗成前空翻。"""
    assert classify_roll_motion(description) == RollMotionKind.AERIAL_SIDE_FLIP
    assert classify_forward_flip(description) == ForwardFlipKind.NONE
    intent = TaskIntentSpec(original_instruction='向前腾空翻跟头', robot='go2',
        action_name='motion', normalized_goal=description)
    assert TaskSemanticValidator.assess_intent(intent).status == 'CONFLICT'


@pytest.mark.parametrize('description', [
    '前空翻与侧翻不同，但必须完成起跳腾空侧翻。',
    '前空翻区别于侧翻，然后完成侧空翻。',
    '前空翻不能与侧空翻混同，并完成一次侧空翻。',
    '前空翻不能混同为侧翻并完成腾空侧翻。',
    '前空翻与侧空翻不同，但两者都必须完成。',
    '前空翻与侧空翻不同，两个动作均需要执行。',
    '前空翻与侧空翻不同，两者均执行。',
    '前空翻区别于侧翻，两者都完成。',
    '前空翻和侧空翻不同，但都必须做。',
    '前空翻与必须完成的侧空翻具有不同的动作语义。',
    '前空翻和侧空翻都要完成。',
    '腾空前空翻，并完成侧空翻。',
    '不是前空翻，而是侧空翻。',
    'front flip differs from a side flip; perform a side flip',
    'front flip differs from a side flip; perform both',
    'front flip is different from a side flip, both are required',
])
def test_comparisons_cannot_hide_independent_side_flip_requirement(description):
    """只处理有明确句法的对照项，后续动作和真正的双动作要求仍被检查。"""
    assert classify_roll_motion(description) != RollMotionKind.NONE
    intent = TaskIntentSpec(original_instruction='向前腾空翻跟头', robot='go2',
        action_name='motion', normalized_goal=description)
    assert TaskSemanticValidator.assess_intent(intent).status == 'CONFLICT'


def test_actual_task_description_excludes_comparison_but_keeps_flight():
    """离线复现 10 月 21 日任务的比较句及前空翻动作，不读取旧实验或改动合同。"""
    description = (
        '控制Go2从足端支撑状态起跳，在四足同时离地的腾空区间内绕机体横向轴（body pitch轴）'
        '正向完成向前腾空前空翻，随后通过足端着地并尝试恢复稳定姿态。'
        '前空翻与绕机体纵向轴的侧空翻、允许机体接触地面的前滚翻具有不同的动作语义。'
        '不要求持续向前行走，不设置未指定的水平速度目标。')
    assert classify_forward_flip(description) == ForwardFlipKind.AERIAL
    assert classify_roll_motion(description) == RollMotionKind.NONE
    assert TaskSemanticValidator._requires_flight(description)
