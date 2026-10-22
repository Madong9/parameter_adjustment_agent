"""验证监督前翻训练时的否定比较语义与候选代理，避免重跑昂贵训练。"""
import pytest

from rl_training_agent.agents.task_semantics import TaskSemanticValidator
from rl_training_agent.orchestration.orchestrator import TrainingOrchestrator
from rl_training_agent.providers.mock_provider import MockLLMReasoningProvider
from rl_training_agent.schemas.agent_workflow import TaskIntentSpec
from rl_training_agent.schemas.task import TaskSpec
from rl_training_agent.utils.task_semantics import RollMotionKind, classify_roll_motion


@pytest.mark.parametrize('comparison', [
    '空中旋转方向必须为前翻，不得将横向轴前翻误判为纵向轴侧翻',
    '起跳前空翻，不要把前空翻误认为侧空翻',
    '腾空前空翻，禁止把前空翻当作侧翻',
    'front flip, do not mistake a front flip for a side flip',
    'front flip, must not confuse a front flip with a side flip',
])
def test_forbidden_comparison_does_not_replace_actual_forward_goal(comparison):
    """真实模型的错误动作排除说明不能被当成新增侧翻要求。"""
    intent = TaskIntentSpec(original_instruction='向前腾空翻跟头', robot='go2',
        action_name='单次前空翻', normalized_goal='腾空前空翻后足端落地站稳',
        required_behaviors=[comparison])
    assert classify_roll_motion(comparison) == RollMotionKind.NONE
    assert TaskSemanticValidator.assess_intent(intent).status == 'CONSISTENT'


@pytest.mark.parametrize('requirement', [
    '不得将前空翻误判为侧翻，但必须做腾空侧翻',
    '不得将前翻误判为纵向轴侧翻并完成一次腾空侧翻',
    '不要把前空翻误认为侧翻，然后起跳腾空侧空翻',
    'front flip, do not mistake a front flip for a side flip; perform a side flip',
    '起跳腾空侧翻',
])
def test_independent_side_flip_requirement_is_still_rejected(requirement):
    """否定比较仅截止于目标动作词，后续真实侧翻要求继续被冲突检查拒绝。"""
    intent = TaskIntentSpec(original_instruction='向前腾空翻跟头', robot='go2',
        action_name='单次前空翻', normalized_goal=requirement)
    assert classify_roll_motion(requirement) != RollMotionKind.NONE
    assert TaskSemanticValidator.assess_intent(intent).status == 'CONFLICT'


@pytest.mark.parametrize('instruction', [
    '向前腾空翻跟头', '起跳完成单次前空翻，落地站稳', 'front flip',
    'forward somersault with takeoff and stable landing',
])
def test_front_flip_ranking_uses_pitch_rewards_instead_of_ordinary_velocity(instruction):
    """巨大的普通速度奖励不能影响前翻代理，起跳措辞也不能将其路由成跳跃。"""
    task = TaskSpec.parse_obj(MockLLMReasoningProvider(1).design_task_and_rewards(
        '稳定站立', 'go2', {})['task_spec'])
    task.original_instruction = instruction
    task.task_name = 'aerial_front_flip'
    names = ['takeoff_velocity', 'airborne_duration', 'pitch_rotation',
             'net_pitch_progress', 'front_flip_recovery']
    latest = {'Episode/rew_raw_mean_' + name: (index + 1) / 10.
              for index, name in enumerate(names)}
    latest['Episode/rew_raw_mean_tracking_lin_vel'] = 999.
    latest['Episode/rew_raw_mean_jump_height'] = 999.
    score, terms = TrainingOrchestrator._task_reward_proxy(task, latest)
    assert score == pytest.approx(.3) and terms == names
