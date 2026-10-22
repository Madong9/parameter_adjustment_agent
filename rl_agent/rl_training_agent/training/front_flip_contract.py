"""只对经过本地完整合同审核的前空翻启用运行时旋转规则。"""
from __future__ import annotations


def compile_front_flip_contract(task, plan):
    """审核任务、正向旋转和安全奖励后记录原始合同，不替换侧翻或普通任务。"""
    from ..utils.task_semantics import ForwardFlipKind, classify_forward_flip
    if task is None or classify_forward_flip(task.original_instruction) != ForwardFlipKind.AERIAL:
        return None
    from ..agents.task_semantics import TaskSemanticValidator
    from ..schemas.agent_workflow import TaskIntentSpec
    from ..rewards.action_normalizer import validate_action_semantics
    from ..rewards.quality_gate import require_objectives
    from ..rewards.validator import RewardValidationError
    intent = TaskIntentSpec(original_instruction=task.original_instruction, robot=task.robot,
                          action_name=task.task_name, normalized_goal=task.normalized_description)
    report = TaskSemanticValidator.assess_task(intent, task)
    if report.status != 'CONSISTENT':
        raise RewardValidationError('前空翻合同无效：' + '；'.join(report.issues))
    validate_action_semantics(task, plan)
    require_objectives(task, plan)
    return {'version': 'front-flip-v1', 'axis': 'body_y', 'direction': 'positive',
            'quaternion_order': 'xyzw', 'task': task.dict()}


def resolve_front_flip_contract(config):
    """启动时复核编译声明、原始任务与全部奖励，防止任意配置关闭原生姿态终止。"""
    declared = config.get('front_flip_semantics')
    if declared is None:
        return None
    if not isinstance(declared, dict):
        raise ValueError('front_flip_semantics 必须为完整编译声明对象')
    from ..schemas.task import TaskSpec
    from ..schemas.rewards import RewardPlan
    task = TaskSpec.parse_obj(declared.get('task', {}))
    if (task.task_id != config.get('task_id') or task.robot != config.get('robot') or
            [item.dict() for item in task.safety_constraints] != config.get('safety_constraints')):
        raise ValueError('前空翻声明与编译任务身份或安全合同不一致')
    plan = RewardPlan(task_id=config['task_id'], version=config['reward_version'],
        design_rationale=['已编译的前空翻合同'], terms=config['rewards']['terms'],
        terminations=config.get('terminations', []), curriculum=config.get('curriculum', []),
        success_metrics=config.get('success_metrics', []), objective_links=config.get('objective_links', []),
        behavior_links=config.get('behavior_links', []))
    scales = config['rewards']['scales']
    if scales != {term.name: term.weight for term in plan.terms}:
        raise ValueError('前空翻声明的奖励与实际权重不一致')
    expected = compile_front_flip_contract(task, plan)
    if expected is None or expected != declared:
        raise ValueError('前空翻编译声明与本地旋转规则不一致')
    return expected
