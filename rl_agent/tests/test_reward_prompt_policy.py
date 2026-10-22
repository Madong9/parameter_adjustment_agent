import json
from pathlib import Path

from rl_training_agent.agents.prompt_compiler import RewardPromptCompiler
from rl_training_agent.agents.reward_prompt_policy import render_reward_policy, REWARD_PROMPT_VERSION
from rl_training_agent.providers.opencli_chatgpt import OpenCLIChatGPTWebProvider
from rl_training_agent.rewards.quality_gate import METRIC_SIGNALS, SAFETY_SIGNALS, METRIC_SIGNAL_OVERRIDES
from rl_training_agent.schemas.agent_workflow import TaskIntentSpec, TaskRewardBundle
from rl_training_agent.schemas.rewards import RewardPlan
from rl_training_agent.training.config_runtime import REWARD_PARAMETER_NAMES


def section(prompt, name):
    return json.loads(next(line[len(name)+2:] for line in prompt.splitlines() if line.startswith(name+': ')))


def test_policy_uses_live_implementation_and_safety_tables():
    prompt = render_reward_policy()
    assert section(prompt, 'LOCAL_METRIC_REWARD_SIGNALS') == {k: sorted(v) for k, v in METRIC_SIGNALS.items()}
    policy = section(prompt, 'RUNTIME_REWARD_POLICY')
    assert policy['safety_metric_rewards'] == SAFETY_SIGNALS
    assert policy['safety_metric_rewards']['torque_limit_violations'] == 'torque_limits'
    assert set(policy['runtime_parameter_names']) == REWARD_PARAMETER_NAMES
    assert len(policy['metric_signal_overrides']) == len(METRIC_SIGNAL_OVERRIDES)
    assert policy['version'] == REWARD_PROMPT_VERSION
    assert section(prompt, 'LOCAL_METRIC_REWARD_SIGNALS')['final_roll_angle'] == ['net_roll_progress']
    assert policy['capability_gap_output']['objective_reward_terms'] == []
    assert policy['capability_gap_output']['executable'] is False


def test_initial_design_is_deterministic_and_has_generic_shared_policy():
    template = Path(__file__).parents[1]/'rl_training_agent/prompts/reward_design_agent.md'
    compiler = RewardPromptCompiler(template)
    intent = TaskIntentSpec(original_instruction='平地行走', robot='go2', action_name='walk', normalized_goal='平地行走')
    first = compiler.compile(intent, {'environment_manifest': {'rewards': []}}, 3)
    second = compiler.compile(intent, {'environment_manifest': {'rewards': []}}, 3)
    assert first == second
    assert first['version'] == 'reward-design-v12'
    for text in ('objective_links', 'behavior_links', '[CAPABILITY_GAP]', 'CANDIDATE_COUNT: 3',
                 '训练迭代课程', '反例', 'torque_limits', '固定动作语义与验收合同'):
        assert text in first['prompt']


def test_front_flip_policy_requires_complete_executable_candidate():
    """前翻明确对应关系、全程学习门控和参数位置，不把验收指标误认成缺奖励实现。"""
    from rl_training_agent.rewards.front_flip_baseline import FRONT_FLIP_OBJECTIVES, FRONT_FLIP_WEIGHTS
    contract = section(render_reward_policy(), 'RUNTIME_REWARD_POLICY')['front_flip_contract']
    assert contract['objective_reward_terms'] == FRONT_FLIP_OBJECTIVES
    assert contract['baseline_weights'] == FRONT_FLIP_WEIGHTS
    assert contract['active_phases'] == ['all']
    assert 'terms.parameters' in contract['parameter_location']
    assert '不要放在 curriculum.parameter_changes' in contract['parameter_location']
    assert '每个候选' in contract['execution_gating']


def test_legacy_and_failure_redesign_use_same_policy_without_browser(monkeypatch):
    provider = OpenCLIChatGPTWebProvider()
    calls = []
    class Reply:
        def dict(self):
            return {}
    def request(prompt, schema, title):
        calls.append((prompt, schema, title))
        return Reply()
    monkeypatch.setattr(provider, '_request_model', request)
    provider.design_task_and_rewards('行走', 'go2', {'rewards': []})
    provider.redesign_rewards({'immutable_task': {'task_id': 't'}, 'validation_feedback': '缺少 torque_limits'})
    assert [c[1] for c in calls] == [TaskRewardBundle, RewardPlan]
    for prompt, _, _ in calls:
        assert render_reward_policy() in prompt
    assert 'immutable_task' in calls[1][0] and 'validation_feedback' in calls[1][0]
    assert '完整继承' in calls[1][0] and 'RewardPlan' in calls[1][0]


def test_gap_is_schema_valid_but_cannot_pass_local_gate():
    from rl_training_agent.schemas.task import TaskSpec, MetricThreshold
    from rl_training_agent.schemas.rewards import RewardTerm, RewardObjectiveLink
    from rl_training_agent.rewards.quality_gate import audit_objectives
    target = MetricThreshold(name='final_roll_angle', operator='>=', value=6., unit='rad')
    task = TaskSpec(task_id='t', robot='go2', task_name='roll', original_instruction='侧翻',
        normalized_description='侧翻', initial_state='standing', required_behaviors=[], forbidden_behaviors=[],
        phases=[], required_observations=[], required_sensors=[], safety_constraints=[],
        training_budget={}, visual_evaluation_requirements=[], success_metrics=[target])
    plan = RewardPlan(task_id='t', version=1, design_rationale=['[CAPABILITY_GAP] 缺少净滚转奖励'],
        terms=[RewardTerm(name='roll_rotation', implementation='registry:roll_rotation', purpose='角速度代理', weight=1.)],
        success_metrics=[target], objective_links=[RewardObjectiveLink(metric='final_roll_angle', reward_terms=[],
            rationale='代理不能证明完整翻滚', counterexamples=['高速往复转动净角度为零'])])
    assert not audit_objectives(task, plan)['passed']
