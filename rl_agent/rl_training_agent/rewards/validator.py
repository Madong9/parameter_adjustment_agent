from __future__ import annotations

import ast
import math
from typing import Dict, Iterable, Set

from ..environment.capability_manifest import RewardRegistryItem
from ..schemas.rewards import RewardPlan


class RewardValidationError(ValueError):
    """奖励计划或隔离实现违反了安全约束。"""


class RewardPlanValidator:
    def __init__(self, registry: Iterable[RewardRegistryItem], max_abs_weight: float = 100.0):
        """初始化 RewardPlanValidator 实例及其运行依赖。"""
        self.registry = {item.name: item for item in registry}
        self.max_abs_weight = max_abs_weight

    def validate(self, plan: RewardPlan) -> None:
        """校验奖励计划只使用安全且已注册的奖励项。"""
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
