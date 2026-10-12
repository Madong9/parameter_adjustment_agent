"""编译、奖励、终止和采集共用的禁止接触合同，不修改 Unitree 源码。"""
import math
import re

COLLISION_FORCE_THRESHOLD = 0.1
BODY_TERMINATION_FORCE_THRESHOLD = 1.0
CONTACT_SEMANTICS_VERSION = 'all-non-foot-v2'
CONTACT_SCOPE = 'all_non_foot_rigid_bodies'
CONTACT_FORCE_METRICS = {'forbidden_contact_force', 'forbidden_body_contact', 'body_contact_force'}
CONTACT_METRICS = CONTACT_FORCE_METRICS | {'forbidden_collisions'}


def _dictionary(item):
    """把 Pydantic 或 JSON 合同项转成普通字典。"""
    return item.dict() if hasattr(item, 'dict') else dict(item)


def contact_contract(safety_constraints=(), terminations=()):
    """保留任务原阈值，并与固定非足端禁触底线取更严格规则。"""
    force_limit = BODY_TERMINATION_FORCE_THRESHOLD
    force_operator = '>'
    constraints = []
    for item in safety_constraints:
        metric = _dictionary(item)
        if metric.get('name') not in CONTACT_METRICS or not metric.get('required', True):
            continue
        value = metric.get('value')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('禁止接触合同阈值必须是有限非负数：' + metric['name'])
        operator = metric.get('operator')
        if metric['name'] == 'forbidden_collisions':
            if operator not in ('==', '<=') or value != 0:
                raise ValueError('禁止碰撞合同必须保留固定零碰撞底线')
            if metric.get('unit', '') not in ('', 'count', 'frames'):
                raise ValueError('禁止碰撞计数单位必须为 count 或 frames')
        else:
            if operator not in ('<', '<=', '==') or (operator == '==' and value != 0):
                raise ValueError('禁止接触力只支持上界或等于零的硬约束')
            if operator == '<' and value == 0:
                raise ValueError('接触力范数不能小于零，合同不可执行')
            if metric.get('unit', '') not in ('', 'N'):
                raise ValueError('禁止接触力合同必须使用 N 单位')
            if value < force_limit or (value == force_limit and operator == '<'):
                force_limit = float(value)
                force_operator = '>=' if operator == '<' else '>'
        constraints.append(metric)
    for item in terminations:
        item = _dictionary(item)
        if not item.get('enabled', True) or item.get('is_timeout', False):
            continue
        if not is_body_termination(item):
            continue
        operator, value = body_termination_bound(item)
        if value < force_limit or (value == force_limit and operator == '>='):
            force_limit, force_operator = value, operator
    event_limit = min(COLLISION_FORCE_THRESHOLD, force_limit)
    event_operator = force_operator if force_limit <= COLLISION_FORCE_THRESHOLD else '>'
    return {
        'version': CONTACT_SEMANTICS_VERSION,
        'scope': CONTACT_SCOPE,
        # 碰撞事件阈值不是最大接触力上界，两项规则分别执行。
        'collision_force_threshold_n': event_limit,
        'collision_operator': event_operator,
        'body_termination_force_threshold_n': force_limit,
        'body_termination_operator': force_operator,
        'terminate_on_forbidden_collision': True,
        'safety_constraints': constraints,
    }


def is_body_termination(item):
    """识别已有计划中的身体接触终止项。"""
    name = str(item.get('name', '')).lower()
    condition = str(item.get('condition', '')).lower()
    return (any(token in name for token in ('body_contact', 'body_collision', 'forbidden_contact')) or
            any(token in condition for token in ('body contact', 'body_contact', 'forbidden_contact_force')))


def body_termination_bound(item):
    """只接受有明确 N 阈值的身体接触上界条件，不执行含糊自然语言。"""
    condition = str(item.get('condition', '')).strip().lower()
    subject = r'(?:(?:forbidden[ _]+)?body[ _]+contact(?:[ _]+force)?|forbidden[ _]+contact[ _]+force)'
    match = re.fullmatch(subject + r'\s*(>=|>)\s*([0-9]+(?:\.[0-9]+)?)\s*n?', condition)
    if match is None:
        raise ValueError('身体接触终止条件必须为 body contact > 数值 N（可使用 >=）')
    value = float(match.group(2))
    if not math.isfinite(value):
        raise ValueError('身体接触终止阈值必须有限')
    return match.group(1), value


def resolve_contact_contract(config):
    """校验编译声明，旧配置显式升级；不能用手改配置放宽安全底线。"""
    expected = contact_contract(config.get('safety_constraints', ()), config.get('terminations', ()))
    declared = config.get('contact_semantics')
    if declared is None:
        return expected
    if not isinstance(declared, dict):
        raise ValueError('contact_semantics 必须为对象')
    if declared.get('version') == 'all-non-foot-v1':
        # 旧合同没有 scope/任务阈值，仍必须验证已有声明，禁止静默忽略被放宽的阈值。
        for key in ('collision_force_threshold_n', 'body_termination_force_threshold_n'):
            if declared.get(key) != expected[key]:
                raise ValueError('旧接触合同与当前硬约束不一致：' + key)
        if declared.get('scope', CONTACT_SCOPE) != CONTACT_SCOPE:
            raise ValueError('旧接触合同不得缩小禁止身体部位')
        return expected
    if declared != expected:
        raise ValueError('编译接触合同与任务硬约束不一致，不得放宽或替换禁止接触定义')
    return expected


def env_contact_contract(env):
    """取得环境已安装的合同，独立奖励测试使用相同固定安全底线。"""
    return getattr(env, '_agent_contact_contract', None) or getattr(
        getattr(env, 'cfg', None), '_agent_contact_contract', None) or contact_contract()


def nonfoot_indices(env):
    """按真实刚体索引排除足端，不采信 thigh/calf 等不完整字符串列表。"""
    import torch
    if hasattr(env, '_agent_nonfoot_indices'):
        return env._agent_nonfoot_indices
    feet = set(int(i) for i in env.feet_indices.detach().cpu().tolist())
    env._agent_nonfoot_indices = torch.tensor([i for i in range(env.num_bodies) if i not in feet],
                                              dtype=torch.long, device=env.device)
    return env._agent_nonfoot_indices


def nonfoot_forces(env):
    """返回合同禁止部位的接触力范数。"""
    import torch
    return torch.norm(env.contact_forces[:, nonfoot_indices(env), :], dim=-1)


def collision_count(env):
    """按共同事件阈值计数接触刚体；评估按此事件统计碰撞帧数。"""
    return _collision_mask(nonfoot_forces(env), env_contact_contract(env)).sum(dim=1).float()


def _collision_mask(forces, contract):
    """严格接触力上界也使用同源事件门控，防止小力违规不产生惩罚。"""
    limit = contract['collision_force_threshold_n']
    return forces >= limit if contract['collision_operator'] == '>=' else forces > limit


def forbidden_contact_flags(env):
    """从奖励同源接触数据产生评估碰撞事件和硬终止标记。"""
    import torch
    forces = nonfoot_forces(env)
    contract = env_contact_contract(env)
    collision = _collision_mask(forces, contract).any(dim=1)
    maximum = forces.max(dim=1).values if forces.shape[1] else torch.zeros(
        forces.shape[0], dtype=forces.dtype, device=forces.device)
    limit = contract['body_termination_force_threshold_n']
    exceeded = maximum >= limit if contract['body_termination_operator'] == '>=' else maximum > limit
    terminated = exceeded | collision if contract['terminate_on_forbidden_collision'] else exceeded
    return collision, maximum, terminated


def contact_audit_metadata(env):
    """记录生效合同与解析部位，供训练和 rollout 证据离线复核。"""
    indices = [int(index) for index in nonfoot_indices(env).detach().cpu().tolist()]
    names = getattr(env, 'body_names', None)
    if names is None and hasattr(env, 'gym'):
        names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
    return {'contact_semantics': dict(env_contact_contract(env)),
            'rigid_body_count': int(env.num_bodies),
            'forbidden_rigid_body_indices': indices,
            'forbidden_rigid_body_names': [names[index] for index in indices] if names is not None else [],
            'foot_rigid_body_indices': [int(index) for index in env.feet_indices.detach().cpu().tolist()]}
