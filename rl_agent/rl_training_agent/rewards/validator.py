from __future__ import annotations

import ast
import math
from typing import Dict, Iterable, Set

from ..environment.capability_manifest import RewardRegistryItem
from ..schemas.rewards import RewardPlan


class RewardValidationError(ValueError):
    """奖励计划或隔离实现违反了安全约束。"""


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
            raise RewardValidationError('奖励参数没有运行时实现：' + term.name)
        for name, value in term.parameters.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RewardValidationError('奖励参数不是有限数值：' + name)
            if ('sigma' in name or name == 'takeoff_velocity_target') and value <= 0:
                raise RewardValidationError('奖励尺度必须大于零：' + name)


class RewardPlanValidator:
    def __init__(self, registry: Iterable[RewardRegistryItem], max_abs_weight: float = 100.0):
        """初始化 RewardPlanValidator 实例及其运行依赖。"""
        self.registry = {item.name: item for item in registry}
        self.max_abs_weight = max_abs_weight

    def validate(self, plan: RewardPlan) -> None:
        """校验奖励计划只使用安全且已注册的奖励项。"""
        validate_plan_execution(plan, self.max_abs_weight)
        for term in plan.terms:
            if term.name not in self.registry:
                raise RewardValidationError("reward is not present in registry: %s" % term.name)
            if not isinstance(term.parameters, dict):
                raise RewardValidationError("reward parameters must be an object: %s" % term.name)
            if not math.isfinite(term.weight):
                raise RewardValidationError("reward weight is not finite: %s" % term.name)
            if abs(term.weight) > self.max_abs_weight:
                raise RewardValidationError("reward weight exceeds safety limit: %s" % term.name)
            expected_sign = self.registry[term.name].sign
            if expected_sign == "negative" and term.weight > 0:
                raise RewardValidationError("penalty reward has unsafe positive sign: %s" % term.name)
            if expected_sign == "positive" and term.weight < 0:
                raise RewardValidationError("positive reward has unsafe negative sign: %s" % term.name)
            for stage in plan.curriculum:
                override = stage.parameter_changes.get('reward_scales', {}).get(term.name, term.weight)
                if (expected_sign == 'negative' and override > 0) or (expected_sign == 'positive' and override < 0):
                    raise RewardValidationError('课程权重违反奖励符号：' + term.name)


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
        if len(functions) != 1 or functions[0].name != "reward":
            raise RewardValidationError("generated code must define exactly one reward(tensors) function")
        for node in ast.walk(tree):
            if isinstance(node, self.FORBIDDEN_NODES):
                raise RewardValidationError("forbidden AST node: %s" % type(node).__name__)
            if isinstance(node, ast.Name) and node.id in self.FORBIDDEN_NAMES:
                raise RewardValidationError("forbidden name: %s" % node.id)
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise RewardValidationError("dunder attribute access is forbidden")
            if isinstance(node, ast.Call):
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
        """执行生成奖励的形状和有限值张量冒烟测试。"""
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
