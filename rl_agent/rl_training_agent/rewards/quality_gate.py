"""目标覆盖与奖励投机审计；代理奖励不能被声明为完整动作信号。"""
import math

from .validator import RewardValidationError
from ..training.config_runtime import reward_scales_for_stage

# 此表是本地审核的实现能力，不采信模型自行宣称的奖励语义。
METRIC_SIGNALS = {
    'tracking_error': {'tracking_lin_vel'},
    'walking_speed_tracking': {'tracking_lin_vel'},
    'tracking_lin_vel': {'tracking_lin_vel'},
    'fall_rate': {'orientation', 'termination'},
    'yaw_tracking_error': {'tracking_ang_vel'},
    'jump_height': {'jump_height'},
    'takeoff_time_spread': {'feet_synchrony'},
    'landing_pitch_abs': {'landing_stability', 'orientation'},
    'feet_air_time': {'airborne_duration'},  # 验收要求全足同时腾空，步态单足时长不能替代。
    'max_roll_velocity': {'roll_rotation'},
    'final_roll_angle': set(),  # 角速度代理不等于净滚转；须实现后显式注册。
    'landing_stability': {'landing_stability', 'landing_recovery'},
    'stable_stand_duration': {'stable_support', 'landing_recovery'},
    'front_leg_stand_duration': {'front_leg_stand'},
    'rear_leg_stand_duration': {'rear_leg_stand'},
    'rear_stand_duration': {'rear_leg_stand'},
    'front_leg_walk_velocity_tracking': {'front_leg_walk'},
    'rear_leg_walk_velocity_tracking': {'rear_leg_walk'},
    'front_leg_walk_completion': {'front_leg_walk'},
    'rear_leg_walk_completion': {'rear_leg_walk'},
}
SAFETY_SIGNALS = {
    'forbidden_collisions': 'collision', 'forbidden_contact_force': 'collision',
    'joint_limit_violations': 'dof_pos_limits', 'torque_limit_violations': 'torque_limits',
}


def effective_scales(plan):
    config = {'rewards': {'scales': {t.name: t.weight for t in plan.terms},
                           'terms': [t.dict() for t in plan.terms]},
              'curriculum': [s.dict() for s in plan.curriculum]}
    return {s.name: reward_scales_for_stage(config, s.name) for s in sorted(plan.curriculum, key=lambda s: s.start_iteration)} or {
        'all': reward_scales_for_stage(config, None)}


def audit_objectives(task, plan):
    scales = effective_scales(plan)
    active = {n for values in scales.values() for n, w in values.items() if w != 0}
    errors, links = [], {}
    for link in plan.objective_links:
        if link.metric in links:
            errors.append('重复目标对应关系：' + link.metric)
        links[link.metric] = link
    coverage = []
    behavior_links = {link.behavior: link for link in plan.behavior_links}
    if len(behavior_links) != len(plan.behavior_links):
        errors.append('动作要求对应关系重复')
    for behavior in task.required_behaviors:
        link = behavior_links.get(behavior.name)
        if (link is None or not link.reward_terms or not set(link.reward_terms) <= active or
                not link.rationale.strip() or not any(c.strip() for c in link.counterexamples)):
            errors.append('缺少有效的动作要求—奖励—反例说明：' + behavior.name)
    for metric in task.success_metrics:
        if not metric.required:
            continue
        supported = METRIC_SIGNALS.get(metric.name, set())
        if (metric.name in {'stable_stand_duration', 'landing_stability'} and
                metric.aggregation in {'continuous_after_last_landing', 'post_last_airborne_recovery_mean'}):
            supported = {'landing_recovery'}
        if not any(m.dict() == metric.dict() for m in plan.success_metrics):
            errors.append('不得修改或遗漏任务验收标准：' + metric.name)
        usable = supported & active
        link = links.get(metric.name)
        if not usable:
            errors.append('关键目标缺少已实现且有效的训练信号：' + metric.name)
        if link is None or not link.rationale.strip() or not any(c.strip() for c in link.counterexamples):
            errors.append('缺少目标—奖励—反例对应表：' + metric.name)
        elif not link.reward_terms or not set(link.reward_terms) <= usable:
            errors.append('目标对应表引用无效或语义不匹配的奖励：' + metric.name)
        coverage.append({'metric': metric.name, 'supported_terms': sorted(supported),
                         'active_terms': sorted(usable), 'verified': bool(usable and link and
                          link.reward_terms and set(link.reward_terms) <= usable and
                          link.rationale.strip() and any(c.strip() for c in link.counterexamples))})
    for metric in task.safety_constraints:
        term = SAFETY_SIGNALS.get(metric.name)
        if metric.required and term and not all(v.get(term, 0) < 0 for v in scales.values()):
            errors.append('安全约束在部分课程缺少负奖励：%s -> %s' % (metric.name, term))
    return {'passed': not errors, 'errors': errors, 'coverage': coverage,
            'effective_scales': scales, 'scope': '实现覆盖审核，非动作可行性或学习成功证明'}


def require_objectives(task, plan):
    report = audit_objectives(task, plan)
    if not report['passed']:
        raise RewardValidationError('；'.join(report['errors']))
    return report


def audit_rollout(task, plan, trajectory, rewards, metrics):
    """同一 pilot rollout 的反例与触发检查；零值不等于失败。"""
    errors, warnings = [], []
    if trajectory.empty or rewards.empty or len(trajectory) != len(rewards):
        errors.append('轨迹与奖励证据缺失或长度不一致')
    elif ('sim_time' not in trajectory or 'sim_time' not in rewards or
          not (trajectory.sim_time.to_numpy() == rewards.sim_time.to_numpy()).all()):
        errors.append('轨迹与奖励时间未对齐')
    contributions = {}
    current_scales = list(effective_scales(plan).values())[-1]
    for term in plan.terms:
        if current_scales.get(term.name, 0) == 0:
            warnings.append('评估课程未启用该奖励：' + term.name)
            continue
        for prefix in ('raw_', 'weighted_'):
            name = prefix + term.name
            if name not in rewards or not all(math.isfinite(float(v)) for v in rewards[name]):
                errors.append('缺失或非有限奖励证据：' + name)
            else:
                contributions[name] = float(rewards[name].abs().mean())
                if prefix == 'weighted_':
                    contributions['signed_' + term.name] = float(rewards[name].mean())
        if contributions.get('raw_' + term.name) == 0:
            warnings.append('奖励未触发，须检查阶段和门控：' + term.name)
    if (metrics.get('forbidden_collisions', 0) > 0 and
            contributions.get('raw_collision', 0) == 0):
        errors.append('实际非足端碰撞没有产生碰撞惩罚')
    for metric, term in SAFETY_SIGNALS.items():
        if (metrics.get(metric, 0) > 0 and term in current_scales and
                current_scales[term] < 0 and contributions.get('signed_' + term, 0) >= 0):
            errors.append('安全违规未产生有效负贡献：%s -> %s' % (metric, term))
    for metric in list(task.success_metrics) + list(task.safety_constraints):
        value = metrics.get(metric.name)
        if metric.required and (not isinstance(value, (int, float)) or not math.isfinite(value)):
            errors.append('关键验收证据缺失：' + metric.name)
    # 错误动作反例：代理表现良好却没有目标进度，交给 GPT 诊断，不宣称学习不可能。
    if metrics.get('max_roll_velocity', 0) >= 4 and metrics.get('final_roll_angle', 0) < .5:
        warnings.append('高角速度、低净滚转：不能把代理收益作为动作进步')
    if metrics.get('feet_air_time', 0) > 0 and metrics.get('landing_stability') == 0:
        warnings.append('已有腾空但没有稳定恢复：检查重复起跳投机')
    return {'passed': not errors, 'errors': errors, 'warnings': warnings,
            'contributions': contributions, 'metrics': metrics,
            'scope': '奖励完整性检查；短训练不要求完成最终动作'}


def compare_pilots(task, previous, current):
    """保守检查重复 pilot 的代理增长与目标倒退，静止平台期仅告警。"""
    def progress(report):
        values = []
        for metric in task.success_metrics:
            value = report['metrics'].get(metric.name)
            if not metric.required or not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            scale = max(abs(metric.value), .001)
            if metric.operator in ('>=', '>'):
                values.append(min(1., max(0., value / scale)))
            elif metric.operator in ('<=', '<'):
                values.append(1. / (1. + max(0., value - metric.value) / scale))
        return sum(values) / len(values) if values else None
    def positive_return(report):
        return sum(v for k, v in report['contributions'].items()
                   if k.startswith('signed_') and v > 0)
    before, after = progress(previous), progress(current)
    proxy_before, proxy_after = positive_return(previous), positive_return(current)
    suspicious = (before is not None and after is not None and before - after > .1 and
                  proxy_after > max(1e-6, proxy_before * 1.2))
    return {'suspected_reward_hacking': suspicious, 'previous_progress': before,
            'current_progress': after, 'previous_proxy': proxy_before, 'current_proxy': proxy_after,
            'note': '只阻止代理收益增加且目标明显倒退；不把早期尚未学会当作训练失败'}
