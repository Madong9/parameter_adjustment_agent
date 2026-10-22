from __future__ import annotations

import ast
import math
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Set

from ..environment.capability_manifest import RewardRegistryItem
from ..schemas.rewards import RewardPlan


class RewardValidationError(ValueError):
    """奖励计划或隔离实现违反了安全约束。"""


def validate_contact_contract(task, plan: RewardPlan) -> Dict:
    """在训练前核对固定禁触底线、任务阈值和所有课程的碰撞惩罚。"""
    from ..training.contact_semantics import contact_contract, CONTACT_METRICS, is_body_termination, body_termination_bound
    from ..training.config_runtime import reward_scales_for_stage
    try:
        contract = contact_contract(task.safety_constraints, plan.terminations)
    except (ValueError, TypeError) as exc:
        raise RewardValidationError('禁止接触合同无效：' + str(exc)) from exc
    required = any(metric.required and metric.name in CONTACT_METRICS for metric in task.safety_constraints)
    if required:
        config = {'rewards': {'scales': {term.name: term.weight for term in plan.terms},
                              'terms': [term.dict() for term in plan.terms]},
                  'curriculum': [stage.dict() for stage in plan.curriculum]}
        names = [stage.name for stage in plan.curriculum] or [None]
        if not all(reward_scales_for_stage(config, name).get('collision', 0) < 0 for name in names):
            raise RewardValidationError('禁止接触硬约束在所有课程都必须有有效负 collision 奖励')
    for termination in plan.terminations:
        item = termination.dict()
        if not termination.enabled or termination.is_timeout or not is_body_termination(item):
            continue
        _, limit = body_termination_bound(item)
        if limit > contract['body_termination_force_threshold_n']:
            raise RewardValidationError('身体接触终止阈值不得高于任务合同或固定安全底线')
    return contract


def validate_plan_execution(plan: RewardPlan, max_abs_weight: float = 100.0) -> None:
    """拒绝声明了但运行时不会执行的阶段、条件和课程参数。"""
    from ..training.config_runtime import _phase_matches, REWARD_PARAMETER_NAMES, COMMAND_RANGE_NAMES
    stages = sorted(plan.curriculum, key=lambda s: s.start_iteration)
    if len({s.name for s in stages}) != len(stages):
        raise RewardValidationError('课程阶段名称重复')
    for index, stage in enumerate(stages):
        if index == 0 and stage.start_iteration != 0:
            raise RewardValidationError('课程必须从第零次迭代开始')
        if index and stage.start_iteration < stages[index - 1].end_iteration:
            raise RewardValidationError('课程阶段重叠：' + stage.name)
        allowed = COMMAND_RANGE_NAMES | {'command_scale', 'base_height_target', 'reward_scales', 'commands'}
        unknown = set(stage.parameter_changes) - allowed
        if unknown:
            raise RewardValidationError('课程参数没有运行时实现：' + ', '.join(sorted(unknown)))
        overrides = stage.parameter_changes.get('reward_scales', {})
        if not isinstance(overrides, dict):
            raise RewardValidationError('课程 reward_scales 必须是对象')
        for key in ('command_scale', 'base_height_target'):
            if key in stage.parameter_changes:
                value = stage.parameter_changes[key]
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise RewardValidationError('课程参数不是有效非负数值：' + key)
        if 'commands' in stage.parameter_changes:
            value = stage.parameter_changes['commands']
            if not isinstance(value, str) or not any(x in value.lower() for x in ('low', 'increase', 'full')):
                raise RewardValidationError('课程 commands 没有对应运行时实现，请使用明确命令范围')
        for name, value in overrides.items():
            term = next((t for t in plan.terms if t.name == name), None)
            if term is None or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > max_abs_weight:
                raise RewardValidationError('课程奖励权重非法：' + name)
            if term.weight * value < 0:
                raise RewardValidationError('课程奖励权重改变符号：' + name)
        for name in COMMAND_RANGE_NAMES & set(stage.parameter_changes):
            value = stage.parameter_changes[name]
            values = value if isinstance(value, (list, tuple)) else [value, value]
            if len(values) != 2 or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in values) or values[0] > values[1]:
                raise RewardValidationError('课程命令范围非法：' + name)
    for term in plan.terms:
        if term.activation_condition:
            raise RewardValidationError('activation_condition 文本不会执行；须使用已注册门控奖励：' + term.name)
        if term.weight and stages and not any(_phase_matches(term.active_phases, stage.name) for stage in stages):
            raise RewardValidationError('奖励在所有训练阶段均未激活：' + term.name)
        if set(term.parameters) - REWARD_PARAMETER_NAMES:
            raise RewardValidationError('奖励参数没有运行时实现：%s；不支持的参数：%s' % (
                term.name, ', '.join(sorted(set(term.parameters) - REWARD_PARAMETER_NAMES))))
        for name, value in term.parameters.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RewardValidationError('奖励参数不是有限数值：' + name)
            if ('sigma' in name or name in {'takeoff_velocity_target', 'net_roll_target', 'net_pitch_target', 'pitch_rate_target'}) and value <= 0:
                raise RewardValidationError('奖励尺度必须大于零：' + name)
            if name == 'min_air_time_threshold' and value < 0:
                raise RewardValidationError('最小腾空时间不能为负数')


def validate_registered_weights(plan, registry, max_abs_weight=100., ignore_unknown=False):
    """审核和编译共用注册表符号与权重边界，避免问题延迟到编译才发现。"""
    for term in plan.terms:
        if term.name not in registry:
            if ignore_unknown:
                continue
            raise RewardValidationError("reward is not present in registry: %s" % term.name)
        if not isinstance(term.parameters, dict):
            raise RewardValidationError("reward parameters must be an object: %s" % term.name)
        if not math.isfinite(term.weight):
            raise RewardValidationError("reward weight is not finite: %s" % term.name)
        if abs(term.weight) > max_abs_weight:
            raise RewardValidationError("reward weight exceeds safety limit: %s" % term.name)
        item = registry[term.name]
        expected_sign = item.get('sign') if isinstance(item, dict) else item.sign
        if expected_sign == "negative" and term.weight > 0:
            raise RewardValidationError("penalty reward has unsafe positive sign: %s" % term.name)
        if expected_sign == "positive" and term.weight < 0:
            raise RewardValidationError("positive reward has unsafe negative sign: %s" % term.name)
        for stage in plan.curriculum:
            override = stage.parameter_changes.get('reward_scales', {}).get(term.name, term.weight)
            if (expected_sign == 'negative' and override > 0) or (expected_sign == 'positive' and override < 0):
                raise RewardValidationError('课程权重违反奖励符号：' + term.name)


class RewardPlanValidator:
    def __init__(self, registry: Iterable[RewardRegistryItem], max_abs_weight: float = 100.0):
        """初始化 RewardPlanValidator 实例及其运行依赖。"""
        self.registry = {item.name: item for item in registry}
        self.max_abs_weight = max_abs_weight

    def validate(self, plan: RewardPlan) -> None:
        """校验奖励计划只使用安全且已注册的奖励项。"""
        validate_plan_execution(plan, self.max_abs_weight)
        validate_registered_weights(plan, self.registry, self.max_abs_weight)


class RewardCodeValidator:
    """用于隔离候选奖励函数的 AST 与张量冒烟验证器。"""

    FORBIDDEN_NODES = (ast.Import, ast.ImportFrom, ast.With, ast.AsyncWith, ast.Lambda,
                       ast.Global, ast.Nonlocal, ast.ClassDef, ast.Try, ast.Raise)
    FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "os", "sys", "subprocess",
                       "socket", "requests", "pathlib", "shutil", "pickle", "input"}
    ALLOWED_TORCH_CALLS = {"abs", "sum", "mean", "square", "sqrt", "exp", "clip", "clamp", "norm",
                           "where", "maximum", "minimum", "isfinite", "zeros_like", "ones_like"}

    def __init__(self, allowed_tensors: Iterable[str]):
        """初始化 RewardCodeValidator 实例及其运行依赖。"""
        self.allowed_tensors: Set[str] = set(allowed_tensors)

    def validate_ast(self, source: str) -> ast.Module:
        """对生成奖励源码执行 AST 白名单检查。"""
        tree = ast.parse(source)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if len(tree.body) != 1 or len(functions) != 1 or functions[0].name != "reward":
            raise RewardValidationError("generated code must define exactly one reward(tensors) function")
        function = functions[0]
        args = function.args
        if (function.decorator_list or function.returns or args.defaults or args.kw_defaults or
                args.vararg or args.kwarg or args.kwonlyargs or getattr(args, 'posonlyargs', []) or
                len(args.args) != 1 or args.args[0].arg != 'tensors' or args.args[0].annotation):
            raise RewardValidationError("generated reward must be a plain reward(tensors) function")
        for node in ast.walk(tree):
            if isinstance(node, self.FORBIDDEN_NODES):
                raise RewardValidationError("forbidden AST node: %s" % type(node).__name__)
            if isinstance(node, ast.Name) and node.id in self.FORBIDDEN_NAMES:
                raise RewardValidationError("forbidden name: %s" % node.id)
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise RewardValidationError("dunder attribute access is forbidden")
            if isinstance(node, (ast.For, ast.While, ast.AsyncFor, ast.ListComp, ast.SetComp,
                                 ast.DictComp, ast.GeneratorExp, ast.NamedExpr, ast.Delete,
                                 ast.AugAssign, ast.AsyncFunctionDef)):
                raise RewardValidationError("generated reward must be a bounded pure tensor expression")
            if isinstance(node, ast.FunctionDef) and node is not function:
                raise RewardValidationError("nested reward functions are forbidden")
            if isinstance(node, ast.Assign) and any(
                    not isinstance(target, ast.Name) or target.id in ('torch', 'tensors')
                    for target in node.targets):
                raise RewardValidationError("only local variable assignment is allowed")
            if isinstance(node, ast.AnnAssign):
                raise RewardValidationError("annotated assignment is forbidden")
            if isinstance(node, ast.Attribute) and (
                    not isinstance(node.value, ast.Name) or node.value.id != 'torch' or
                    node.attr not in self.ALLOWED_TORCH_CALLS):
                raise RewardValidationError("only whitelisted torch attributes may be used")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, (ast.Name, ast.Attribute)):
                    raise RewardValidationError("indirect function calls are forbidden")
                if isinstance(node.func, ast.Name) and node.func.id not in {"float", "len"}:
                    raise RewardValidationError("function call is not allowed: %s" % node.func.id)
                if isinstance(node.func, ast.Attribute):
                    if not isinstance(node.func.value, ast.Name) or node.func.value.id != "torch" or node.func.attr not in self.ALLOWED_TORCH_CALLS:
                        raise RewardValidationError("only whitelisted torch functions may be called")
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "tensors":
                key = None
                slice_node = node.slice.value if isinstance(node.slice, ast.Index) else node.slice
                if isinstance(slice_node, ast.Constant):
                    key = slice_node.value
                if key not in self.allowed_tensors:
                    raise RewardValidationError("tensor is not in capability whitelist: %s" % key)
        return tree

    def tensor_smoke_test(self, source: str, batch_size: int = 8) -> Dict[str, float]:
        """在限时、限内存、禁止写文件的独立进程中验证生成奖励。"""
        self.validate_ast(source)
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 1024:
            raise RewardValidationError("batch_size must be within 1..1024")
        root = str(Path(__file__).resolve().parents[2])
        script = ("import sys; sys.path.insert(0, %r); "
                  "from rl_training_agent.rewards.validator import _tensor_smoke_worker; "
                  "_tensor_smoke_worker()") % root
        environment = {'PATH': str(Path(sys.executable).parent), 'PYTHONDONTWRITEBYTECODE': '1',
                       'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
                       'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        try:
            with tempfile.TemporaryDirectory(prefix='reward-smoke-') as directory:
                result = subprocess.run(
                    [sys.executable, '-I', '-c', script],
                    input=json.dumps({'source': source, 'batch_size': batch_size,
                                      'allowed_tensors': sorted(self.allowed_tensors)}),
                    capture_output=True, text=True, cwd=directory, env=environment,
                    timeout=20, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RewardValidationError("isolated reward smoke test failed or timed out") from exc
        if result.returncode != 0:
            raise RewardValidationError("isolated reward smoke test exceeded resource limits or failed")
        try:
            payload = json.loads(result.stdout)
        except (ValueError, TypeError) as exc:
            raise RewardValidationError("isolated reward worker returned invalid evidence") from exc
        if 'error' in payload:
            raise RewardValidationError(str(payload['error']))
        return payload['stats']

    def _tensor_smoke_in_process(self, source: str, batch_size: int) -> Dict[str, float]:
        """仅供独立 worker 在资源限制生效后执行受检张量函数。"""
        import numpy as np
        import torch
        # 在进入受限全局命名空间前初始化 Torch 的可选 NumPy 桥接。
        torch.from_numpy(np.zeros(1, dtype=np.float32))
        tree = self.validate_ast(source)
        namespace = {"torch": torch, "__builtins__": {"float": float, "len": len}}
        exec(compile(tree, "<generated_reward>", "exec"), namespace)
        tensors = {name: torch.randn(batch_size, 3) for name in self.allowed_tensors}
        result = namespace["reward"](tensors)
        if not isinstance(result, torch.Tensor) or result.shape != (batch_size,):
            raise RewardValidationError("reward output shape must be [batch_size]")
        if not torch.isfinite(result).all():
            raise RewardValidationError("reward output contains NaN or Inf")
        return {"min": float(result.min()), "max": float(result.max()), "mean": float(result.mean())}


def _tensor_smoke_worker() -> None:
    """初始化 CPU Torch 后限制当前 worker 资源，不继承宿主凭据环境。"""
    import resource
    import torch
    import numpy as np
    request = json.loads(sys.stdin.read())
    torch.set_num_threads(1)
    torch.from_numpy(np.zeros(1, dtype=np.float32))
    with open('/proc/self/statm', encoding='ascii') as handle:
        virtual_bytes = int(handle.read().split()[0]) * os.sysconf('SC_PAGE_SIZE')
    resource.setrlimit(resource.RLIMIT_AS, (virtual_bytes + 256 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    try:
        stats = RewardCodeValidator(request['allowed_tensors'])._tensor_smoke_in_process(
            request['source'], request['batch_size'])
        print(json.dumps({'stats': stats}, allow_nan=False))
    except Exception as exc:
        print(json.dumps({'error': str(exc)[:500]}))
