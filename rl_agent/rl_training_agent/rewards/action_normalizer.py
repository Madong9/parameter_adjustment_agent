"""修正已知动作与现有奖励实现的语义冲突，记录每一次配置调整。"""
import re
import math

from ..schemas.rewards import CurriculumStage, RewardTerm
from ..utils.task_semantics import is_side_flip_text
from ..training.action_rewards import ACTION_REWARDS


def validate_action_semantics(task, plan):
    if not is_side_flip_text((task.original_instruction, task.task_name)):
        return
    from .validator import RewardValidationError
    for metric in task.safety_constraints:
        if metric.required and metric.name in ('roll_limit', 'max_abs_roll') and metric.operator in ('<', '<=') and metric.value < math.pi:
            raise RewardValidationError('完整侧翻与任务的 roll 硬约束冲突，不能通过删除硬约束放行')
    if not {'takeoff_velocity', 'airborne_duration', 'roll_rotation', 'landing_recovery'} <= {t.name for t in plan.terms if t.weight > 0}:
        raise RewardValidationError('侧翻缺少真实接触门控的起跳、旋转和恢复信号')


def normalize_action_plan(task, plan):
    if not is_side_flip_text((task.original_instruction, task.task_name)):
        return []
    audit = []
    replacements = {'feet_air_time': 'airborne_duration', 'tracking_ang_vel': 'roll_rotation',
                    'jump_height': 'takeoff_velocity', 'landing_stability': 'landing_recovery'}
    incompatible = {'orientation', 'ang_vel_xy', 'lin_vel_z', 'base_height', 'stand_still', 'tracking_lin_vel'}
    terms = {}
    for term in plan.terms:
        if term.name in incompatible:
            audit.append('移除阻碍侧翻的全程奖励：' + term.name)
            continue
        new_name = replacements.get(term.name, term.name)
        if new_name != term.name:
            audit.append('侧翻奖励实现替换：%s -> %s' % (term.name, new_name))
            term.name = new_name
            term.parameters = {}
        if new_name in {'takeoff_velocity', 'airborne_duration', 'roll_rotation', 'landing_recovery'}:
            term.implementation = 'rl_training_agent/training/action_rewards.py:_reward_' + new_name
            term.dependencies = ACTION_REWARDS[new_name]
            term.expected_raw_range = (0., 1.)
            term.active_phases = ['all']
            term.activation_condition = None
        terms[new_name] = term
    for name in ('takeoff_velocity', 'airborne_duration', 'roll_rotation', 'landing_recovery'):
        if name not in terms or terms[name].weight <= 0:
            terms[name] = RewardTerm(name=name,
                implementation='rl_training_agent/training/action_rewards.py:_reward_' + name,
                purpose='侧翻接触状态门控的物理动作奖励', weight=1.,
                dependencies=ACTION_REWARDS[name], expected_raw_range=(0., 1.),
                reward_hacking_risks=['必须独立验证单次净旋转、腾空、落地及非足端碰撞'])
            audit.append('补充侧翻物理奖励：' + name)
    plan.terms = list(terms.values())
    # 学习课程是迭代阶段；动作内起跳/飞行/落地由运行时接触状态判断。
    plan.curriculum = [CurriculumStage(name='side_flip_learning', start_iteration=0,
        end_iteration=max(0, task.training_budget.max_total_iterations - 1),
        parameter_changes={'lin_vel_x': [0., 0.], 'lin_vel_y': [0., 0.],
                           'ang_vel_yaw': [0., 0.], 'heading': [0., 0.]})]
    for term in plan.terms:
        term.active_phases = ['all']
    audit.append('侧翻使用全程学习课程与动作接触门控，平移和 yaw 命令为零')
    # 保留独立硬约束；仅移除不适用于完整翻转的普通 roll 跌倒条件。
    roll_safety = any(m.required and m.name in ('roll_limit', 'max_abs_roll') for m in task.safety_constraints)
    if not roll_safety:
        for termination in plan.terminations:
            if not termination.enabled or termination.is_timeout:
                continue
            parts = re.split(r'\s+or\s+', termination.condition)
            kept = [p for p in parts if not re.fullmatch(r'\s*abs\(roll\)\s*>\s*[0-9.]+\s*', p)]
            if len(kept) != len(parts):
                termination.condition = ' or '.join(kept)
                termination.enabled = bool(kept)
                audit.append('移除侧翻会必经的普通 roll 跌倒终止；其余终止保留')
    return audit
