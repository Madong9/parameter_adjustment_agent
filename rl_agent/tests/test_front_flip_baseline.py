"""验证模型候选遗漏后的独立前翻基线仍经过完整审查，不请求模型或启动 PPO。"""
import json
from pathlib import Path

import pytest

from rl_training_agent.agents.reward_reviewer import RewardReviewAgent
from rl_training_agent.environment.inspector import EnvironmentInspector
from rl_training_agent.orchestration.orchestrator import TrainingOrchestrator
from rl_training_agent.providers.mock_provider import MockLLMReasoningProvider
from rl_training_agent.rewards.compiler import RewardCompiler
from rl_training_agent.rewards.front_flip_baseline import (BASELINE_VERSION, FRONT_FLIP_WEIGHTS,
                                                         build_front_flip_baseline)
from rl_training_agent.rewards.quality_gate import audit_objectives, effective_scales
from rl_training_agent.schemas.task import BehaviorRequirement, MetricThreshold
from rl_training_agent.settings import Settings, load_settings
from test_front_flip_backend import contract


def manifest():
    """读取项目实际注册表，不访问真实任务、凭据或 GPU。"""
    return EnvironmentInspector(Path(__file__).parents[2] / 'unitree_rl_gym').inspect('go2')


def recorded_shape_task():
    """复现本轮中文动作要求、执行阶段和原始验收阈值，不读取真实实验。"""
    intent, task, _ = contract()
    intent.original_instruction = task.original_instruction = '向前腾空翻跟头'
    task.required_behaviors = [BehaviorRequirement(name=name, description=description) for name, description in (
        ('机器人进行向前方向的腾空翻转动作', '机器人完成包含起跳、空中旋转和落地恢复的向前腾空翻转动作'),
        ('旋转轴应为横向轴（pitch轴）', '翻转主要绕机体横向轴进行正向旋转'),
        ('完成起跳、空中旋转和落地恢复过程', '动作链包含起跳、腾空旋转、落地以及恢复稳定支撑'))]
    return intent, task


def test_baseline_preserves_task_passes_actual_review_and_compiler(tmp_path):
    """本地完整基线对本轮合同形状通过原有审查、编译与运行时合同复核。"""
    from rl_training_agent.training.front_flip_contract import resolve_front_flip_contract
    intent, task = recorded_shape_task()
    before = task.dict()
    actual_manifest = manifest()
    plan = build_front_flip_baseline(task, actual_manifest.dict(), version=4)
    assert plan is not None and task.dict() == before
    assert plan.version == 4 and plan.parent_version is None
    assert plan.success_metrics == task.success_metrics
    assert {term.name: term.weight for term in plan.terms} == FRONT_FLIP_WEIGHTS
    assert all(term.active_phases == ['all'] for term in plan.terms)
    assert {link.behavior for link in plan.behavior_links} == {item.name for item in task.required_behaviors}
    assert audit_objectives(task, plan)['passed']
    assert all(set(FRONT_FLIP_WEIGHTS) <= set(scales) for scales in effective_scales(plan).values())
    review = RewardReviewAgent().review(intent, task, [plan],
        dict(actual_manifest.dict(), require_objective_links=True), [])
    assert review.approved, review.conflicts
    RewardCompiler(actual_manifest).compile(plan, tmp_path, task)
    config = json.loads((tmp_path / 'config.yaml').read_text())
    assert resolve_front_flip_contract(config) is not None
    assert config['safety_constraints'] == before['safety_constraints']
    assert task.dict() == before


@pytest.mark.parametrize('missing', ['net_pitch_progress', 'airborne_duration', 'collision', 'pitch_axis_error'])
def test_missing_backend_never_gets_a_baseline(missing):
    """缺失真实奖励、独立验收或安全实现时不以本地模板冒充完整能力。"""
    _, task = recorded_shape_task()
    capabilities = manifest().dict()
    capabilities['rewards'] = [item for item in capabilities['rewards'] if item['name'] != missing]
    capabilities['evaluation_metrics'] = [name for name in capabilities['evaluation_metrics'] if name != missing]
    assert build_front_flip_baseline(task, capabilities) is None


@pytest.mark.parametrize('instruction', ['向前翻跟头', '贴地前滚翻后站起', '侧空翻', '稳定站立', '连续腾空前空翻'])
def test_baseline_does_not_choose_unrelated_or_repeated_motion(instruction):
    """不为含糊、地面、侧向、普通动作或连续动作选择单次前翻配方。"""
    _, task = recorded_shape_task()
    task.original_instruction = instruction
    assert build_front_flip_baseline(task, manifest().dict()) is None


@pytest.mark.parametrize('extra', ['后腿行走', '单腿落地', '保持前向速度1m/s'])
def test_baseline_does_not_claim_unsupported_behavior(extra):
    """额外支撑与速度目标不能被一个前翻模板的文字对应表声称已经覆盖。"""
    _, task = recorded_shape_task()
    task.required_behaviors.append(BehaviorRequirement(name='extra', description=extra))
    assert build_front_flip_baseline(task, manifest().dict()) is None


def test_baseline_does_not_remove_conflicting_hard_constraint():
    """与前翻冲突的原始俯仰硬约束保留，模板拒绝生成候选。"""
    _, task = recorded_shape_task()
    task.safety_constraints.append(MetricThreshold(name='max_abs_pitch', operator='<=', value=.5, unit='rad'))
    before = task.dict()
    assert build_front_flip_baseline(task, manifest().dict()) is None
    assert task.dict() == before


def test_extra_required_metric_has_no_invented_baseline_mapping():
    """本模板范围外的关键验收指标保持缺口，不自行降低或删除目标。"""
    _, task = recorded_shape_task()
    task.success_metrics.append(MetricThreshold(name='tracking_error', operator='<=', value=.1))
    assert build_front_flip_baseline(task, manifest().dict()) is None


def test_baseline_requires_matching_robot_capabilities():
    """不能为其他机型借用 Go2 模板或不匹配的机型能力清单。"""
    _, task = recorded_shape_task()
    capabilities = manifest().dict()
    capabilities['robot'] = 'g1'
    assert build_front_flip_baseline(task, capabilities) is None
    task.robot = 'g1'
    assert build_front_flip_baseline(task, capabilities) is None


def test_new_run_clears_stale_baseline_display_but_preserves_candidate(tmp_path):
    """新运行不会继承旧基线采用记录，旧候选奖励和 checkpoint 仍保留。"""
    names = ['local_reward_baseline.json', 'reward_review_before_baseline.json', 'reward_review_with_baseline.json']
    for name in names:
        (tmp_path / name).write_text('{}')
    candidate = tmp_path / 'candidates/old'
    candidate.mkdir(parents=True)
    (candidate / 'reward_plan.json').write_text('{}')
    (candidate / 'model_0.pt').write_bytes(b'old')
    TrainingOrchestrator._clear_stale_run_outputs(tmp_path)
    assert all(not (tmp_path / name).exists() for name in names)
    assert (candidate / 'reward_plan.json').is_file() and (candidate / 'model_0.pt').read_bytes() == b'old'


def test_candidate_rejection_adds_reviewed_baseline_before_training(tmp_path, monkeypatch):
    """完整编排先保存模型拒绝记录，再审核并编译本地候选，训练入口桩阻止创建 PPO。"""
    intent, task = recorded_shape_task()
    _, _, invalid = contract()
    invalid.terms = [term for term in invalid.terms if term.name not in ('airborne_duration', 'pitch_rotation')]
    class IncompleteProvider(MockLLMReasoningProvider):
        """每次奖励请求都返回相同遗漏，模拟模型修订没有补齐执行信号。"""
        calls = 0

        def understand_task(self, instruction, robot):
            """返回明确前空翻理解。"""
            return intent.copy(deep=True)

        def design_task_bundle(self, compiled_prompt):
            """输出可解析但无法通过审查的候选，不运行真实浏览器。"""
            self.calls += 1
            return {'task_spec': task.dict(), 'reward_plans': [invalid.dict()],
                    'reward_hacking_risks': ['错误旋转方向不能通过验收']}

    settings = Settings(**{**load_settings().dict(), 'experiment_root': str(tmp_path / 'experiments'),
        'artifact_root': str(tmp_path / 'artifacts'), 'memory_enabled': False, 'num_reward_candidates': 1,
        'reward_design_review_retries': 1})
    provider = IncompleteProvider(1)
    orchestrator = TrainingOrchestrator(settings, provider)
    from rl_training_agent.feasibility.agent import FeasibilityPipeline
    from rl_training_agent.feasibility.schema import CompleteFeasibilityReport, FeasibilityStatus
    monkeypatch.setattr(FeasibilityPipeline, 'assess', lambda *args: CompleteFeasibilityReport(
        status=FeasibilityStatus.CONDITIONAL, task='离线编排测试', motion_type='ACROBATIC',
        validation_level='MOCK_VALIDATED', evidence_mode='MOCK'))

    def reached_training(*args, **kwargs):
        """在已审查且已编译候选之后停止，绝不创建真实或 Mock PPO 进程。"""
        raise RuntimeError('已到达训练入口')

    monkeypatch.setattr(orchestrator.controller.adapter, 'training_command', reached_training)
    with pytest.raises(RuntimeError, match='已到达训练入口'):
        orchestrator.train(task.original_instruction, 'go2', dry_run=True)
    root = settings.experiments_path / orchestrator._task_id(task.original_instruction, 'go2')
    assert provider.calls == 1  # 不支持补丁的 Provider 不会退回完整重生成。
    assert json.loads((root / 'reward_review_before_baseline.json').read_text())['approved'] is False
    review = json.loads((root / 'reward_review.json').read_text())
    assert review['approved'] is True and review['passed_candidate_indexes'] == [2]
    baseline = json.loads((root / 'local_reward_baseline.json').read_text())
    assert baseline['source'] == BASELINE_VERSION
    original = json.loads((root / 'design_response_attempt_00.json').read_text())
    assert 'airborne_duration' not in {term['name'] for term in original['reward_plans'][0]['terms']}
    assert list(root.glob('candidates/*/config.yaml'))
    assert not list(root.rglob('*.pt'))
    frozen = json.loads((root / 'acceptance_contract.json').read_text())
    assert frozen['success_metrics'] == [metric.dict() for metric in task.success_metrics]
    assert frozen['safety_constraints'] == [metric.dict() for metric in task.safety_constraints]
