"""仅修复明确的 JSON 表示方式，并保护格式重试中已经有效的结构与数值。"""
from __future__ import annotations

import copy
import json
import math
import re
from typing import Any, Dict, List, Type

from pydantic import BaseModel

from ..schemas.agent_workflow import TaskRewardBundle
from ..schemas.rewards import CurriculumStage, RewardPlan
from ..schemas.reward_repairs import RewardSchemaFieldRepair


_METRIC_COMPARISON = re.compile(
    r'\s*([A-Za-z][A-Za-z0-9_]*)\s*(<=|>=|==|<|>)\s*'
    r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
    r'\s*([A-Za-z][A-Za-z0-9_/%^·-]*|1)?\s*')
_CANDIDATE_VERSION = re.compile(r'candidate-([0-9]{1,9})')


def schema_patch_permissions(data, schema, validation_errors):
    """直接以 Pydantic 报错位置授权 add/replace，不开放有效字段或整表重写。"""
    if schema not in (TaskRewardBundle, RewardPlan) or not isinstance(data, dict):
        return []
    permissions = []
    for error in validation_errors:
        path = list(error.get('loc', ()))
        if not path:
            continue
        if '__root__' in path:
            if (path[-1] == '__root__' and 'curriculum' in path and
                    error.get('msg') == 'curriculum start_iteration must not exceed end_iteration'):
                paths = [path[:-1] + [key] for key in ('start_iteration', 'end_iteration')]
            else:
                continue
        else:
            paths = [path]
        for location in paths:
            parent = data
            try:
                for key in location[:-1]:
                    parent = parent[key]
                key = location[-1]
                if isinstance(parent, dict) and isinstance(key, str):
                    op = 'replace' if key in parent else 'add'
                    value = parent.get(key)
                elif isinstance(parent, list) and type(key) is int and 0 <= key < len(parent):
                    op, value = 'replace', parent[key]
                else:
                    continue
                if isinstance(value, (dict, list)) and value and not error.get('type', '').startswith('type_error.'):
                    continue
                grant = {'op': op, 'path': location}
                if grant not in permissions:
                    permissions.append(grant)
            except (KeyError, IndexError, TypeError):
                continue
    return permissions


def render_schema_patch_request(prompt, data, schema, validation_errors, rejected=()):
    from ..rewards.patches import reward_plan_sha256
    request = {'base_sha256': reward_plan_sha256(data),
               'allowed_patches': schema_patch_permissions(data, schema, validation_errors)}
    repair = (prompt + '\n\n只返回 RewardSchemaFieldRepair 字段补丁，不重新输出整份奖励。'
              '\nop/path 必须匹配 allowed_patches；只修复 validation_errors，所有其他字段沿用原始来源。'
              '\nVALIDATION_ERRORS: ' + json.dumps(validation_errors, ensure_ascii=False, default=str) +
              '\nREJECTED_REPAIR_REGRESSIONS: ' + json.dumps(list(rejected), ensure_ascii=False) +
              '\nSCHEMA_PATCH_REQUEST: ' + json.dumps(request, ensure_ascii=False) +
              '\nSOURCE_JSON_SCHEMA: ' + schema.schema_json() +
              '\nJSON Schema: ' + RewardSchemaFieldRepair.schema_json() +
              '\nInvalid JSON response to repair:\n' + json.dumps(data, ensure_ascii=False))
    return repair, request


def apply_schema_field_patches(original, raw, schema, validation_errors):
    from ..rewards.patches import _parent, reward_plan_sha256
    response = RewardSchemaFieldRepair.parse_obj(raw)
    if response.base_sha256 != reward_plan_sha256(original):
        raise ValueError('格式补丁来源哈希不一致')
    permissions = schema_patch_permissions(original, schema, validation_errors)
    merged, paths = copy.deepcopy(original), []
    for patch in response.patches:
        if {'op': patch.op, 'path': patch.path} not in permissions:
            raise ValueError('格式补丁越过 validation_errors 指定位置：%s' % patch.path)
        if any(patch.path[:len(other)] == other or other[:len(patch.path)] == patch.path for other in paths):
            raise ValueError('格式补丁路径重复或重叠')
        parent, key = _parent(merged, patch.path)
        parent[key] = copy.deepcopy(patch.value)
        paths.append(patch.path)
    regressions = repair_preservation_errors(original, merged, schema, validation_errors)
    if regressions:
        raise ValueError('格式补丁改变了有效字段：' + '；'.join(regressions))
    return merged, {'scope': 'schema_error_field_patches_v1', 'base_sha256': response.base_sha256,
                    'patches': [patch.dict() for patch in response.patches], 'effective_response': merged}


def project_legacy_schema_reply(original, repaired, schema, validation_errors):
    """模型仍返回整份对象时，也只取可定位且身份一致的错误字段，禁止整表覆盖。"""
    normalized, conversions = normalize_reward_response(repaired, schema)
    merged, adjustments = merge_schema_repairs(original, normalized, schema, validation_errors)
    allowed = schema_patch_permissions(original, schema, validation_errors)
    usable = [item for item in adjustments if any(item['path'] == grant['path'] for grant in allowed)]
    if len(usable) != len(adjustments):
        # 兼容合并器的授权不能大于新协议的精确错误位置。
        merged = copy.deepcopy(original)
        for item in usable:
            parent = merged
            for key in item['path'][:-1]:
                parent = parent[key]
            parent[item['path'][-1]] = copy.deepcopy(item['normalized'])
    regressions = repair_preservation_errors(original, normalized, schema,
        [{'loc': tuple(grant['path'])} for grant in allowed])
    paths = [tuple(item['path']) for item in usable]
    used_conversions = [item for item in conversions if any(
        tuple(item['path'])[:len(path)] == path or path[:len(item['path'])] == tuple(item['path']) for path in paths)]
    return merged, {'scope': 'validated_reward_field_repairs_v1', 'adjustments': usable,
                    'repair_normalizations': used_conversions, 'ignored_repair_changes': regressions,
                    'effective_response': merged}


def normalize_reward_response(payload: Any, schema: Type[BaseModel]):
    """规范明确的指标比较和说明表示，保留执行内容及可还原的原始信息。"""
    normalized = copy.deepcopy(payload)
    adjustments = []
    if not isinstance(normalized, dict) or schema not in (TaskRewardBundle, RewardPlan):
        return normalized, adjustments
    plans = normalized.get('reward_plans') if schema is TaskRewardBundle else [normalized]
    if isinstance(plans, list):
        for index, plan in enumerate(plans):
            if not isinstance(plan, dict):
                continue
            prefix = ['reward_plans', index] if schema is TaskRewardBundle else []
            for field in ('version', 'parent_version'):
                value = plan.get(field)
                match = _CANDIDATE_VERSION.fullmatch(value) if isinstance(value, str) else None
                if match is not None and int(match.group(1)) > 0:
                    plan[field] = int(match.group(1))
                    adjustments.append({'path': prefix + [field], 'original': value,
                                        'normalized': plan[field]})
            rationale = plan.get('design_rationale')
            if isinstance(rationale, str) and rationale.strip():
                plan['design_rationale'] = [rationale]
                adjustments.append({'path': prefix + ['design_rationale'], 'original': rationale,
                                    'normalized': [rationale]})
    if schema is not TaskRewardBundle:
        return normalized, adjustments
    for field in ('termination_suggestions', 'reward_hacking_risks'):
        suggestions = normalized.get(field)
        if not isinstance(suggestions, list):
            continue
        for index, item in enumerate(suggestions):
            if not isinstance(item, dict):
                continue
            try:
                text = json.dumps(item, ensure_ascii=False, allow_nan=False,
                                  separators=(',', ':'))
                if json.loads(text) != item:
                    continue
            except (TypeError, ValueError, OverflowError, RecursionError):
                continue
            suggestions[index] = text
            adjustments.append({'path': [field, index], 'original': item, 'normalized': text})
    task = normalized.get('task_spec')
    if not isinstance(task, dict):
        return normalized, adjustments
    for field in ('success_metrics', 'safety_constraints'):
        metrics = task.get(field)
        if not isinstance(metrics, list):
            continue
        for index, item in enumerate(metrics):
            if not isinstance(item, str):
                continue
            match = _METRIC_COMPARISON.fullmatch(item)
            if match is None:
                continue
            name, operator, number, unit = match.groups()
            value = float(number)
            if not math.isfinite(value):
                continue
            replacement = {'name': name, 'operator': operator, 'value': value, 'unit': unit or ''}
            metrics[index] = replacement
            adjustments.append({'path': ['task_spec', field, index],
                                'original': item, 'normalized': replacement})
    return normalized, adjustments


def normalize_metric_comparisons(payload: Any, schema: Type[BaseModel]):
    """兼容已有无损转换调用，继续使用统一的回复类型规范。"""
    return normalize_reward_response(payload, schema)


def merge_schema_repairs(original: Any, repaired: Any, schema: Type[BaseModel],
                         validation_errors: List[Dict[str, Any]]):
    """只取奖励回复中可定位的错误字段，其他内容始终沿用完整来源。

    不合并已有容器的整体替换或根校验错误。数组长度及元素身份必须一致，
    避免把重排/删减候选后的同一索引误认为原字段。调用方仍需完整校验结果。
    """
    merged = copy.deepcopy(original)
    adjustments = []
    if schema not in (TaskRewardBundle, RewardPlan) or not isinstance(original, dict):
        return merged, adjustments
    missing = object()
    for error in validation_errors:
        path = tuple(error.get('loc', ()))
        if not path or '__root__' in path:
            continue
        before, after, target = original, repaired, merged
        try:
            for key in path[:-1]:
                if isinstance(before, list):
                    if (not isinstance(key, int) or key < 0 or not isinstance(after, list)
                            or len(before) != len(after)):
                        raise ValueError('changed list')
                    item, replacement = before[key], after[key]
                    if isinstance(item, dict):
                        identity = {name: item[name] for name in
                                    ('name', 'task_id', 'version', 'metric', 'behavior') if name in item}

                        def matches_identity(value):
                            return isinstance(value, dict) and all(
                                value.get(name, missing) == expected for name, expected in identity.items())

                        if (not identity or not matches_identity(replacement)
                                or sum(matches_identity(value) for value in before) != 1
                                or sum(matches_identity(value) for value in after) != 1):
                            raise ValueError('changed item identity')
                before, after, target = before[key], after[key], target[key]
            key = path[-1]
            if isinstance(before, dict) and isinstance(after, dict):
                previous = before.get(key, missing)
                replacement = after[key]
            elif (isinstance(before, list) and isinstance(after, list)
                  and len(before) == len(after) and isinstance(key, int) and key >= 0):
                previous, replacement = before[key], after[key]
            else:
                continue
            if isinstance(previous, (dict, list)) or previous == replacement:
                continue
            target[key] = copy.deepcopy(replacement)
            adjustment = {'path': list(path), 'normalized': copy.deepcopy(replacement),
                          'original_present': previous is not missing}
            if previous is not missing:
                adjustment['original'] = copy.deepcopy(previous)
            adjustments.append(adjustment)
        except (KeyError, IndexError, TypeError, ValueError):
            # 缺少父对象、数组形状改变或无法确定身份时，继续保留来源。
            continue
    if repair_preservation_errors(original, merged, schema, validation_errors):
        return copy.deepcopy(original), []
    return merged, adjustments


def repair_preservation_errors(original: Any, repaired: Any, schema: Type[BaseModel],
                               validation_errors: List[Dict[str, Any]]) -> List[str]:
    """仅允许修改校验失败的位置；已有合法字段、候选和课程不能被重写丢失。"""
    invalid = {tuple(error.get('loc', ())) for error in validation_errors}
    errors = []

    def walk(before, after, descriptor, path):
        """按 Schema 认识的字段递归保护来源，忽略模型的额外未知键。"""
        if path in invalid:
            return
        if isinstance(before, dict):
            model = descriptor if isinstance(descriptor, type) else getattr(descriptor, 'type_', None)
            fields = getattr(model, '__fields__', None)
            if model is CurriculumStage and any(
                    tuple(error.get('loc', ())) == path + ('__root__',) and
                    error.get('msg') == 'curriculum start_iteration must not exceed end_iteration'
                    for error in validation_errors):
                invalid.update((path + ('start_iteration',), path + ('end_iteration',)))
            if fields is not None and isinstance(after, dict):
                before, after = dict(before), dict(after)
                for name, field in fields.items():
                    if not field.required:
                        default = field.get_default()
                        if name not in before:
                            before[name] = copy.deepcopy(default)
                        if name not in after:
                            after[name] = copy.deepcopy(default)
            keys = set(before) & set(fields) if fields is not None else set(before)
            if not keys:
                return
            if not isinstance(after, dict):
                errors.append('.'.join(map(str, path)) + '：有效对象被替换')
                return
            for key in sorted(keys):
                if key not in after:
                    errors.append('.'.join(map(str, path + (key,))) + '：已有字段被删除')
                    continue
                walk(before[key], after[key], fields.get(key) if fields is not None else None, path + (key,))
        elif isinstance(before, list):
            if not isinstance(after, list) or len(after) != len(before):
                errors.append('.'.join(map(str, path)) + '：已有列表数量被改变')
                return
            subfields = getattr(descriptor, 'sub_fields', None)
            item_descriptor = subfields[0] if subfields else None
            for index, value in enumerate(before):
                walk(value, after[index], item_descriptor, path + (index,))
        else:
            expected, actual = before, after
            if descriptor is not None and hasattr(descriptor, 'validate'):
                expected, first_error = descriptor.validate(before, {}, loc=path)
                actual, second_error = descriptor.validate(after, {}, loc=path)
                if first_error is not None or second_error is not None:
                    expected, actual = before, after
            if expected != actual:
                errors.append('.'.join(map(str, path)) + '：有效值被改变')

    if isinstance(original, dict):
        walk(original, repaired, schema, ())
    return errors
