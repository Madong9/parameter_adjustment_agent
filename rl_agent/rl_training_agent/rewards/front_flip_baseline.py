"""为已明确且实现完整的 Go2 单次前空翻构建可审查的本地奖励基线。"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Set

from ..schemas.agent_workflow import TaskIntentSpec
from ..schemas.rewards import (CurriculumStage, RewardBehaviorLink, RewardObjectiveLink,
                               RewardPlan, RewardTerm)
from ..schemas.task import BehaviorRequirement, TaskSpec
from ..utils.task_semantics import ForwardFlipKind, _positive_motion_text, classify_forward_flip


BASELINE_VERSION = 'front-flip-baseline-v1'
FRONT_FLIP_WEIGHTS = {'takeoff_velocity': 1., 'airborne_duration': .5,
                      'pitch_rotation': 1., 'net_pitch_progress': 3.,
                      'front_flip_recovery': 2., 'collision': -2.,
                      'dof_pos_limits': -1., 'torque_limits': -1.}
FRONT_FLIP_OBJECTIVES = {'final_pitch_angle': ['net_pitch_progress'],
                         'pitch_axis_error': ['pitch_rotation'],
                         'landing_stability': ['front_flip_recovery'],
                         'stable_stand_duration': ['front_flip_recovery']}
COUNTEREXAMPLES = ['初始站立没有起跳或完整旋转，不计为前翻完成',
                   '侧翻、后翻或往复转动不能代替正向横轴净旋转',
                   '重复起跳、机体碰撞或未站稳不能代替单次腾空前翻后足端稳定落地']


def _behavior_signals(behavior: BehaviorRequirement) -> Set[str]:
    """只识别本基线实际覆盖的前翻动作要求，不为速度或其他支撑动作虚构对应。"""
    text = _positive_motion_text(behavior.name + ' ' + behavior.description)
    if any(word in text for word in ('侧翻', '侧滚', '后翻', '后空翻', '单腿', '后腿', '前腿',
                                      '倒立', '速度', '行走', '连续', '重复', '多次',
                                      'side flip', 'backflip', 'handstand', 'speed', 'walk')):
        return set()
    signals = set()
    if any(word in text for word in ('前翻', '前空翻', '向前', '前向', '旋转', '横向轴', '横轴',
                                      'pitch', 'front flip', 'forward', 'rotation')):
        signals.update(('pitch_rotation', 'net_pitch_progress'))
    if any(word in text for word in ('起跳', 'takeoff')):
        signals.update(('takeoff_velocity', 'airborne_duration'))
    if any(word in text for word in ('腾空', '空中', '离地', 'airborne', 'flight')):
        signals.add('airborne_duration')
    if any(word in text for word in ('落地', '站稳', '恢复', '稳定', '支撑',
                                      'landing', 'recovery', 'stable')):
        signals.add('front_flip_recovery')
    return signals


def build_front_flip_baseline(task: TaskSpec, capabilities: Dict[str, Any],
                              version: int = 1) -> Optional[RewardPlan]:
    """保留原任务和预算，构造接触门控基线；超出模板范围或缺实现时不提供候选。"""
    if (task.robot != 'go2' or capabilities.get('robot') != task.robot or
            classify_forward_flip(task.original_instruction) != ForwardFlipKind.AERIAL):
        return None
    if any(word in task.original_instruction.lower() for word in (
            '连续', '重复', '多次', 'repeated', 'consecutive')):
        return None
    from ..agents.task_semantics import TaskSemanticValidator
    intent = TaskIntentSpec(original_instruction=task.original_instruction, robot=task.robot,
                           action_name=task.task_name, normalized_goal=task.normalized_description)
    if TaskSemanticValidator.assess_task(intent, task).status != 'CONSISTENT':
        return None
    registered = {item['name']: item for item in capabilities.get('rewards', [])}
    if not set(FRONT_FLIP_WEIGHTS) <= set(registered):
        return None
    if not set(FRONT_FLIP_OBJECTIVES) <= set(capabilities.get('evaluation_metrics', [])):
        return None
    required = {metric.name for metric in task.success_metrics if metric.required}
    if required != set(FRONT_FLIP_OBJECTIVES):
        return None
    from .quality_gate import SAFETY_SIGNALS
    if any(metric.required and metric.name not in SAFETY_SIGNALS for metric in task.safety_constraints):
        return None
    links = []
    for behavior in task.required_behaviors:
        signals = _behavior_signals(behavior)
        if not signals:
            return None
        links.append(RewardBehaviorLink(behavior=behavior.name, reward_terms=sorted(signals),
            rationale='本地前翻基线使用已实现的接触门控、正向横轴旋转和完成后恢复信号',
            counterexamples=list(COUNTEREXAMPLES)))
    target = max(2 * math.pi, next(metric.value for metric in task.success_metrics
                                  if metric.name == 'final_pitch_angle' and metric.required))
    terms = []
    for name, weight in FRONT_FLIP_WEIGHTS.items():
        parameters = dict(registered[name].get('parameters', {}))
        if name in ('net_pitch_progress', 'front_flip_recovery'):
            parameters['net_pitch_target'] = target
        terms.append(RewardTerm(name=name, weight=weight,
            implementation=registered[name]['implementation'], parameters=parameters,
            dependencies=registered[name].get('dependencies', []), active_phases=['all'],
            purpose='本地前翻基线的执行信号：' + name, reward_hacking_risks=list(COUNTEREXAMPLES)))
    return RewardPlan(task_id=task.task_id, version=version, velocity_frame=task.velocity_frame,
        design_rationale=[BASELINE_VERSION + '：模型候选全部被拒绝后的独立本地基线',
            '动作内起跳、腾空和落地由运行时状态门控，全部信号在学习课程有效',
            '权重为本地起始配置，必须通过原有审查和编译；不证明动作已学会'],
        terms=terms, curriculum=[CurriculumStage(name='front_flip_learning', start_iteration=0,
            end_iteration=max(0, task.training_budget.max_total_iterations - 1),
            parameter_changes={name: [0., 0.] for name in ('lin_vel_x', 'lin_vel_y', 'ang_vel_yaw', 'heading')})],
        success_metrics=[metric.copy(deep=True) for metric in task.success_metrics],
        objective_links=[RewardObjectiveLink(metric=metric.name,
            reward_terms=FRONT_FLIP_OBJECTIVES[metric.name],
            rationale='已实现训练信号支持该目标，动作完成仍使用独立轨迹与视觉验收',
            counterexamples=list(COUNTEREXAMPLES)) for metric in task.success_metrics
                         if metric.required], behavior_links=links)
