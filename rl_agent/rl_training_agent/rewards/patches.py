"""以本地问题白名单合并 GPT 补丁，所有路径都绑定同一候选快照。"""
import copy
import hashlib
import json

from ..schemas.reward_repairs import RewardRepairBundle
from ..schemas.rewards import RewardPlan


def reward_plan_sha256(plan):
    data = plan.dict() if hasattr(plan, 'dict') else plan
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _parent(data, path):
    value = data
    for key in path[:-1]:
        if isinstance(value, list):
            if type(key) is not int or key < 0:
                raise ValueError('数组路径必须使用非负整数')
        elif not isinstance(value, dict) or not isinstance(key, str):
            raise ValueError('字段路径与来源类型不一致')
        value = value[key]
    return value, path[-1]


def _apply_candidate(plan, target, repair):
    source = plan.dict()
    digest = reward_plan_sha256(source)
    if repair.base_sha256 != digest or target['base_sha256'] != digest:
        raise ValueError('补丁来源哈希不一致，不能应用过期修复')
    operations = []
    for patch in repair.patches:
        path = patch.path
        if path[0] in ('task_id', 'version', 'parent_version', 'velocity_frame'):
            raise ValueError('任务身份、版本与坐标系禁止由补丁修改')
        if len(path) == 1 and path[0] in (
                'terms', 'curriculum', 'objective_links', 'behavior_links', 'success_metrics', 'terminations'):
            raise ValueError('补丁不能整体替换已有奖励、课程或合同列表')
        grants = [grant for grant in target['allowed_patches']
                  if grant['op'] == patch.op and grant['path'] == path]
        authorized = False
        for grant in grants:
            if 'required_value' in grant and patch.value != grant['required_value']:
                continue
            identity = grant.get('identity')
            if identity and (not isinstance(patch.value, dict) or any(
                    patch.value.get(key) != value for key, value in identity.items())):
                continue
            if grant.get('value_kind') == 'nonempty_string' and (
                    not isinstance(patch.value, str) or not patch.value.strip()):
                continue
            if path[0] == 'success_metrics' and 'required_value' not in grant:
                continue
            if (patch.op == 'add' and path[-1] == '-' and not identity and 'required_value' not in grant
                    and grant.get('value_kind') != 'nonempty_string'):
                continue
            authorized = True
            break
        if not authorized:
            raise ValueError('补丁越过本地报错字段白名单：%s %s' % (patch.op, path))
        for previous in operations:
            other = previous.path
            same_append = path == other and patch.op == previous.op == 'add' and path[-1] == '-'
            if same_append:
                if patch.value == previous.value:
                    raise ValueError('重复追加同一修复项')
                continue
            if path[:len(other)] == other or other[:len(path)] == path:
                raise ValueError('同一请求中补丁路径重叠，拒绝不明确的执行顺序')
        parent, key = _parent(source, path)
        if isinstance(parent, list):
            if patch.op == 'add':
                if key != '-':
                    raise ValueError('列表仅支持追加，不支持插入或重新排序')
                if identity and any(isinstance(item, dict) and all(item.get(name) == value
                        for name, value in identity.items()) for item in parent):
                    raise ValueError('不能追加来源中已有身份的条目')
                if identity and any(other.op == 'add' and other.path == path and
                        isinstance(other.value, dict) and all(other.value.get(name) == value
                        for name, value in identity.items()) for other in operations):
                    raise ValueError('同一请求不能重复追加相同身份')
            elif type(key) is not int or not 0 <= key < len(parent):
                raise ValueError('补丁下标不属于来源列表')
        elif isinstance(parent, dict) and isinstance(key, str):
            if (patch.op == 'add') == (key in parent):
                raise ValueError('补丁操作与来源字段存在状态不一致')
        else:
            raise ValueError('补丁父路径不是可修改的对象或数组')
        operations.append(patch)

    merged = copy.deepcopy(source)
    # 所有下标都相对于来源：先修改和追加，再从后向前删除。
    # 追加不会移动来源项；先删除会让后续嵌套 add 写到错误奖励。
    ordered = ([patch for patch in operations if patch.op == 'replace'] +
               [patch for patch in operations if patch.op == 'add'] +
               sorted((patch for patch in operations if patch.op == 'remove'),
                      key=lambda patch: tuple((1, key) if type(key) is int else (0, key)
                                              for key in patch.path), reverse=True))
    for patch in ordered:
        parent, key = _parent(merged, patch.path)
        if patch.op == 'remove':
            del parent[key]
        elif isinstance(parent, list):
            if patch.op == 'add':
                parent.append(copy.deepcopy(patch.value))
            else:
                parent[key] = copy.deepcopy(patch.value)
        else:
            parent[key] = copy.deepcopy(patch.value)
    # 单个候选内原子合并。无效字段补丁不能先污染下一次重试的来源。
    return RewardPlan.parse_obj(merged)


def apply_reward_repairs(plans, targets, raw):
    """独立处理各候选：越权或过期补丁不覆盖来源，合法候选互不影响。"""
    response = RewardRepairBundle.parse_obj(raw.dict() if hasattr(raw, 'dict') else raw)
    allowed = {target['candidate_index']: target for target in targets}
    result = [plan.copy(deep=True) for plan in plans]
    audit = {'accepted': [], 'rejected': [], 'unchanged_candidate_indexes': []}
    seen = set()
    duplicates = {repair.candidate_index for repair in response.repairs
                  if sum(other.candidate_index == repair.candidate_index for other in response.repairs) > 1}
    for repair in response.repairs:
        index = repair.candidate_index
        if index in seen:
            continue
        seen.add(index)
        try:
            if index in duplicates:
                raise ValueError('同一候选重复返回补丁')
            if index not in allowed or index > len(plans):
                raise ValueError('未授权修改该候选')
            revised = _apply_candidate(plans[index - 1], allowed[index], repair)
            if revised.dict() == plans[index - 1].dict():
                continue
            result[index - 1] = revised
            audit['accepted'].append({'candidate_index': index, 'base_sha256': repair.base_sha256,
                                      'patches': [patch.dict() for patch in repair.patches],
                                      'effective_sha256': reward_plan_sha256(revised)})
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            audit['rejected'].append({'candidate_index': index, 'reason': str(exc)})
    changed = {item['candidate_index'] for item in audit['accepted']}
    audit['unchanged_candidate_indexes'] = [index for index in range(1, len(plans) + 1) if index not in changed]
    return result, audit
