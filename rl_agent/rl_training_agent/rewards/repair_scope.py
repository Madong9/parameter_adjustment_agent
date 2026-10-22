"""从确定性合同检查生成奖励补丁权限；诊断文本不参与授权。"""
from __future__ import annotations

import math
from typing import Any, Dict

from .quality_gate import METRIC_SIGNALS, METRIC_SIGNAL_OVERRIDES, SAFETY_SIGNALS
from ..training.config_runtime import COMMAND_RANGE_NAMES, REWARD_PARAMETER_NAMES, _phase_matches
from ..training.contact_semantics import body_termination_bound, contact_contract, is_body_termination
from ..utils.task_semantics import (ForwardFlipKind, classify_forward_flip, is_front_leg_support_text,
                                    is_side_flip_text)


def build_repair_scope(task, plan, capabilities) -> Dict[str, Any]:
    """仅授权当前候选中可定位的问题；路径相对于 RewardPlan，既有合法字段冻结。"""
    issues, permissions, blocked = [], [], []
    registry = {item['name']: item for item in capabilities.get('rewards', []) if item.get('name')}
    terms = {term.name: (index, term) for index, term in enumerate(plan.terms)}
    stages = list(enumerate(plan.curriculum))
    weight_limit = capabilities.get('max_abs_reward_weight', 100.)

    def issue(code, message, path=None, blocking=False):
        item = {'code': code, 'message': message}
        if path is not None:
            item['path'] = list(path)
        issues.append(item)
        if blocking:
            blocked.append(dict(item))

    def allow(op, path, **constraints):
        item = dict(op=op, path=list(path), **constraints)
        if item not in permissions:
            permissions.append(item)

    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    def parameter(path, value, code, message):
        """运行时标量参数若变成容器则不授权整容器替换。"""
        issue(code, message, path, blocking=isinstance(value, (dict, list, tuple)))
        if not isinstance(value, (dict, list, tuple)):
            allow('replace', path)

    def add_term(name, code):
        if name not in registry:
            issue('missing_backend', '缺少已注册的必要训练信号：' + name, blocking=True)
            return
        issue(code, '缺少必要奖励：' + name, ['terms'])
        allow('add', ['terms', '-'], identity={'name': name})

    def enable_term(name, sign, every_stage=False, require_base=False):
        """按运行时覆盖优先级定位关停来源，不能开放无关课程或奖励。"""
        if name not in terms:
            add_term(name, 'missing_required_reward')
            return
        index, term = terms[name]
        if require_base and (not finite(term.weight) or term.weight * sign <= 0):
            parameter(['terms', index, 'weight'], term.weight,
                      'disabled_required_reward', '必要奖励的基础权重或符号无效：' + name)
        entries = stages or [(None, None)]
        valid = []
        failures = []
        for stage_index, stage in entries:
            changes = stage.parameter_changes if stage is not None else {}
            overrides = changes.get('reward_scales', {})
            if isinstance(overrides, dict) and name in overrides:
                value = overrides[name]
                path = ['curriculum', stage_index, 'parameter_changes', 'reward_scales', name]
                active = True
            else:
                value, path = term.weight, ['terms', index, 'weight']
                active = _phase_matches(term.active_phases, stage.name if stage is not None else None)
            okay = active and finite(value) and value * sign > 0
            valid.append(okay)
            if not okay:
                failures.append((active, value, path))
        if (all(valid) if every_stage else any(valid)):
            return
        for active, value, path in failures:
            if not active:
                issue('inactive_required_reward', '必要奖励未覆盖学习课程：' + name,
                      ['terms', index, 'active_phases'])
                allow('add', ['terms', index, 'active_phases', '-'], required_value='all')
            if not finite(value) or value * sign <= 0:
                parameter(path, value, 'disabled_required_reward', '必要奖励权重或符号无效：' + name)

    if plan.task_id != task.task_id:
        issue('task_identity_mismatch', '任务身份不一致，不能由奖励补丁改变身份', ['task_id'], True)
    try:
        contact = contact_contract(task.safety_constraints, plan.terminations)
        for index, termination in enumerate(plan.terminations):
            item = termination.dict()
            if termination.enabled and not termination.is_timeout and is_body_termination(item):
                _, bound = body_termination_bound(item)
                if bound > contact['body_termination_force_threshold_n']:
                    issue('unsafe_contact_termination', '接触终止阈值超过任务或固定安全底线',
                          ['terminations', index], True)
    except (TypeError, ValueError):
        issue('invalid_contact_contract', '接触合同无法安全执行，不能通过修改硬约束放行', blocking=True)

    # 执行参数和注册表错误直接定位到原始列表下标。
    for index, term in enumerate(plan.terms):
        path = ['terms', index]
        registered = registry.get(term.name)
        if registered is None:
            issue('unregistered_reward', '奖励未注册，只能移除该不可执行条目：' + term.name, path)
            allow('remove', path)
            continue
        expected_sign = (registered or {}).get('sign')
        if (not finite(term.weight) or abs(term.weight) > weight_limit or
                (expected_sign == 'positive' and term.weight < 0) or
                (expected_sign == 'negative' and term.weight > 0)):
            parameter(path + ['weight'], term.weight, 'invalid_reward_weight', '奖励权重或符号不合法')
        if term.activation_condition:
            issue('unexecutable_activation_condition', '文本激活条件不会在运行时执行', path + ['activation_condition'])
            allow('replace', path + ['activation_condition'], required_value=None)
        if term.weight and stages and not any(_phase_matches(term.active_phases, stage.name) for _, stage in stages):
            issue('inactive_reward', '奖励在所有课程均未激活', path + ['active_phases'])
            for phase_index, phase in enumerate(term.active_phases):
                if not any(_phase_matches([phase], stage.name) for _, stage in stages):
                    allow('replace', path + ['active_phases', phase_index], required_value='all')
        for name, value in term.parameters.items():
            key_path = path + ['parameters', name]
            if name not in REWARD_PARAMETER_NAMES:
                issue('unsupported_reward_parameter', '参数没有运行时实现：' + name, key_path)
                allow('remove', key_path)
            elif (not finite(value) or
                  (('sigma' in name or name in {'takeoff_velocity_target', 'net_roll_target', 'net_pitch_target', 'pitch_rate_target'}) and value <= 0) or
                  (name == 'min_air_time_threshold' and value < 0)):
                parameter(key_path, value, 'invalid_reward_parameter', '奖励参数数值无效：' + name)

    sorted_stages = sorted(stages, key=lambda item: item[1].start_iteration)
    if len({stage.name for _, stage in stages}) != len(stages):
        issue('duplicate_curriculum_names', '课程名称重复，无法唯一对应活动阶段', ['curriculum'], True)
    for position, (index, stage) in enumerate(sorted_stages):
        path = ['curriculum', index]
        if position == 0 and stage.start_iteration != 0:
            issue('curriculum_start', '第一课程必须从零开始', path + ['start_iteration'])
            allow('replace', path + ['start_iteration'], required_value=0)
        if position and stage.start_iteration < sorted_stages[position - 1][1].end_iteration:
            issue('curriculum_overlap', '课程边界重叠', path + ['start_iteration'])
            allow('replace', path + ['start_iteration'])
        for name, value in stage.parameter_changes.items():
            key_path = path + ['parameter_changes', name]
            if name not in COMMAND_RANGE_NAMES | {'command_scale', 'base_height_target', 'reward_scales', 'commands'}:
                issue('unsupported_curriculum_parameter', '课程参数没有运行时实现：' + name, key_path)
                allow('remove', key_path)
            elif name in ('command_scale', 'base_height_target') and (not finite(value) or value < 0):
                parameter(key_path, value, 'invalid_curriculum_parameter', '课程参数必须为非负有限数值')
            elif name == 'commands' and (not isinstance(value, str) or not any(token in value.lower() for token in ('low', 'increase', 'full'))):
                parameter(key_path, value, 'invalid_commands', '课程命令没有对应运行时实现')
            elif name == 'reward_scales':
                if not isinstance(value, dict):
                    issue('invalid_reward_scales_shape', '课程权重必须为对象，不能整体重写课程', key_path, True)
                    continue
                for reward, weight in value.items():
                    weight_path = key_path + [reward]
                    if reward not in terms:
                        issue('unknown_override_reward', '课程引用不存在的奖励：' + reward, weight_path)
                        allow('remove', weight_path)
                        continue
                    base_weight = terms[reward][1].weight
                    sign = registry.get(reward, {}).get('sign')
                    if (not finite(weight) or abs(weight) > weight_limit or weight * base_weight < 0 or
                            (sign == 'positive' and weight < 0) or (sign == 'negative' and weight > 0)):
                        parameter(weight_path, weight, 'invalid_curriculum_weight', '课程奖励权重或符号非法')
            elif name in COMMAND_RANGE_NAMES:
                if isinstance(value, (list, tuple)):
                    if len(value) != 2:
                        issue('invalid_command_range_shape', '命令范围必须是两个数值', key_path, True)
                    else:
                        inverted = all(finite(item) for item in value) and value[0] > value[1]
                        for offset, number in enumerate(value):
                            if not finite(number) or inverted:
                                parameter(key_path + [offset], number, 'invalid_command_range', '命令范围数值或顺序无效')
                elif not finite(value):
                    parameter(key_path, value, 'invalid_command_range', '命令范围数值无效')

    # 依据 TaskSpec 恢复候选验收，绝不允许模型修改 TaskSpec 或放宽阈值。
    for metric in task.success_metrics:
        if not metric.required or any(item.dict() == metric.dict() for item in plan.success_metrics):
            continue
        matches = [(index, item) for index, item in enumerate(plan.success_metrics) if item.name == metric.name]
        issue('task_metric_mismatch', '候选必须保留任务验收：' + metric.name, ['success_metrics'])
        if not matches:
            allow('add', ['success_metrics', '-'], required_value=metric.dict())
        elif len(matches) == 1:
            index, current = matches[0]
            for key, value in metric.dict().items():
                if current.dict()[key] != value:
                    allow('replace', ['success_metrics', index, key], required_value=value)
        else:
            issue('ambiguous_metric_identity', '重复验收名称，无法确定应修复的条目', ['success_metrics'], True)

    def repair_links(field, key, requirements):
        links = getattr(plan, field)
        names = [getattr(link, key) for link in links]
        if len(set(names)) != len(names):
            issue('duplicate_' + field, '对应关系名称重复，不能重新排序或删除整表', [field], True)
        for name, usable in requirements:
            matches = [(index, link) for index, link in enumerate(links) if getattr(link, key) == name]
            if not matches:
                issue('missing_' + field, '缺少对应关系：' + name, [field])
                allow('add', [field, '-'], identity={key: name})
                continue
            if len(matches) != 1:
                continue
            index, link = matches[0]
            path = [field, index]
            if not link.rationale.strip():
                issue('empty_link_rationale', '对应关系缺少有效说明', path + ['rationale'])
                allow('replace', path + ['rationale'], value_kind='nonempty_string')
            if not any(item.strip() for item in link.counterexamples):
                issue('empty_link_counterexamples', '对应关系缺少有效反例', path + ['counterexamples'])
                if not link.counterexamples:
                    allow('add', path + ['counterexamples', '-'], value_kind='nonempty_string')
                else:
                    for offset in range(len(link.counterexamples)):
                        allow('replace', path + ['counterexamples', offset], value_kind='nonempty_string')
            if not link.reward_terms:
                issue('empty_link_rewards', '对应关系缺少奖励引用', path + ['reward_terms'])
                for reward in sorted(usable):
                    allow('add', path + ['reward_terms', '-'], required_value=reward)
            for offset, reward in enumerate(link.reward_terms):
                if reward not in usable:
                    issue('invalid_link_reward', '对应关系奖励引用无效：' + reward, path + ['reward_terms', offset])
                    for replacement in sorted(usable):
                        allow('replace', path + ['reward_terms', offset], required_value=replacement)

    # 链接可引用当前有效奖励，以及本轮已被允许补充或启用的目标信号。
    active = set()
    for name, (_, term) in terms.items():
        if name not in registry:
            continue
        for _, stage in stages or [(None, None)]:
            overrides = stage.parameter_changes.get('reward_scales', {}) if stage is not None else {}
            weight = overrides.get(name) if isinstance(overrides, dict) and name in overrides else (
                term.weight if _phase_matches(term.active_phases, stage.name if stage is not None else None) else 0)
            if finite(weight) and weight != 0:
                active.add(name)
    objective_requirements = []
    for metric in task.success_metrics:
        if not metric.required or not capabilities.get('require_objective_links', True):
            continue
        supported = METRIC_SIGNAL_OVERRIDES.get((metric.name, metric.aggregation), METRIC_SIGNALS.get(metric.name, set()))
        available = supported & set(registry)
        if not available:
            issue('missing_objective_backend', '目标缺少已实现的训练信号：' + metric.name, blocking=True)
        if not supported & active:
            existing = available & set(terms)
            for name in sorted(existing or available):
                sign = -1 if registry[name].get('sign') == 'negative' else 1
                enable_term(name, sign)
        objective_requirements.append((metric.name, supported))
    for metric in task.safety_constraints:
        if metric.required and metric.name in SAFETY_SIGNALS:
            enable_term(SAFETY_SIGNALS[metric.name], -1, every_stage=True)

    forward = classify_forward_flip(task.original_instruction) == ForwardFlipKind.AERIAL
    side = is_side_flip_text((task.original_instruction, task.task_name))
    required = ({'takeoff_velocity', 'airborne_duration', 'pitch_rotation', 'net_pitch_progress', 'front_flip_recovery'}
                if forward else {'takeoff_velocity', 'airborne_duration', 'roll_rotation', 'landing_recovery'} if side else set())
    for name in sorted(required):
        enable_term(name, 1, every_stage=forward, require_base=True)
    leg_rewards = set()
    if is_front_leg_support_text((task.original_instruction,)):
        leg_rewards.update(('front_leg_stand', 'front_leg_walk'))
    if '后腿' in task.original_instruction or any('rear_leg' in item.name for item in task.required_behaviors):
        leg_rewards.update(('rear_leg_stand', 'rear_leg_walk'))
        if 'orientation' in terms:
            issue('rear_leg_posture_conflict', '水平姿态奖励与后腿目标冲突，不能推断安全替代项',
                  ['terms', terms['orientation'][0]], True)
    for name in sorted(leg_rewards - set(terms)):
        add_term(name, 'missing_leg_reward')
    if side and any(metric.required and metric.name in ('roll_limit', 'max_abs_roll') and
                    metric.operator in ('<', '<=') and metric.value < math.pi for metric in task.safety_constraints):
        issue('task_roll_safety_conflict', '完整侧翻与任务 roll 硬约束冲突，不能修改任务底线', blocking=True)
    if forward:
        for index, term in enumerate(plan.terms):
            if term.name in ('net_roll_progress', 'roll_rotation') and term.weight > 0:
                issue('wrong_rotation_axis', '前空翻不能使用侧翻奖励，无法授权整体替换', ['terms', index], True)
        for index, termination in enumerate(plan.terminations):
            if termination.enabled and not termination.is_timeout and any(
                    text in termination.condition for text in ('abs(pitch)', 'abs(roll)')):
                issue('conflicting_termination', '姿态终止与前空翻冲突，不能整体禁用混合安全条件', ['terminations', index], True)
    if capabilities.get('require_objective_links', True):
        repaired_names = set()
        for entry in permissions:
            path = entry['path']
            if path == ['terms', '-'] and 'identity' in entry:
                repaired_names.add(entry['identity']['name'])
            elif len(path) >= 3 and path[0] == 'terms' and isinstance(path[1], int) and path[2] in ('weight', 'active_phases'):
                repaired_names.add(plan.terms[path[1]].name)
            elif len(path) == 5 and path[0] == 'curriculum' and path[2:4] == ['parameter_changes', 'reward_scales'] and entry['op'] != 'remove':
                repaired_names.add(path[4])
        usable = active | repaired_names
        repair_links('objective_links', 'metric', [(name, supported & usable) for name, supported in objective_requirements])
        repair_links('behavior_links', 'behavior', [(item.name, usable) for item in task.required_behaviors])
    for term_name, metric_name, parameter_name in (
            ('net_roll_progress', 'final_roll_angle', 'net_roll_target'),
            ('net_pitch_progress', 'final_pitch_angle', 'net_pitch_target')):
        target_metric = next((item for item in task.success_metrics if item.name == metric_name and item.required), None)
        if term_name not in terms or target_metric is None:
            continue
        index, term = terms[term_name]
        target = term.parameters.get(parameter_name, 2 * math.pi)
        valid_target = finite(target) and target > 0 and target >= target_metric.value
        if not valid_target:
            path = ['terms', index, 'parameters', parameter_name]
            issue('insufficient_rotation_target', '净旋转奖励目标不能低于任务验收', path)
            if not isinstance(target, (dict, list, tuple)):
                allow('replace' if parameter_name in term.parameters else 'add', path)
        if term_name == 'net_pitch_progress' and 'front_flip_recovery' in terms and valid_target:
            recovery_index, recovery = terms['front_flip_recovery']
            if recovery.parameters.get(parameter_name, 2 * math.pi) != target:
                path = ['terms', recovery_index, 'parameters', parameter_name]
                invalid_shape = isinstance(recovery.parameters.get(parameter_name), (dict, list, tuple))
                issue('recovery_rotation_target_mismatch', '恢复奖励必须使用相同净旋转目标', path, invalid_shape)
                if not invalid_shape:
                    allow('replace' if parameter_name in recovery.parameters else 'add', path, required_value=target)
    return {'issues': issues, 'allowed_patches': permissions, 'blocked': blocked}
