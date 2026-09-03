from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List

from .capability_manifest import RewardRegistryItem
from ..utils.paths import AGENT_ROOT, relative_display


DEPENDENCIES = {
    "lin_vel_z": ["base_lin_vel"], "ang_vel_xy": ["base_ang_vel"],
    "orientation": ["projected_gravity"], "base_height": ["base_pos"],
    "torques": ["torques"], "dof_vel": ["dof_vel"],
    "dof_acc": ["dof_vel", "last_dof_vel"], "action_rate": ["actions", "last_actions"],
    "collision": ["contact_forces"], "termination": ["reset_buf", "time_out_buf"],
    "dof_pos_limits": ["dof_pos", "dof_pos_limits"], "dof_vel_limits": ["dof_vel"],
    "torque_limits": ["torques", "torque_limits"],
    "tracking_lin_vel": ["base_lin_vel", "commands"],
    "tracking_ang_vel": ["base_ang_vel", "commands"],
    "feet_air_time": ["contact_forces", "feet_air_time"],
    "stumble": ["contact_forces"], "stand_still": ["dof_pos", "commands"],
    "feet_contact_forces": ["contact_forces"],
    "jump_height": ["base_pos"], "feet_synchrony": ["contact_forces"],
    "landing_stability": ["contact_forces", "rpy"],
    "horizontal_drift": ["base_pos", "env_origins"],
    "rear_leg_stand": ["contact_forces", "base_pos", "rpy"],
    "rear_leg_walk": ["contact_forces", "base_pos", "rpy", "base_lin_vel", "commands"],
    "front_leg_stand": ["contact_forces", "base_pos", "rpy"],
    "front_leg_walk": ["contact_forces", "base_pos", "rpy", "base_lin_vel", "commands"],
}

# 这些环境奖励函数返回的是非负“代价”或违规量，权重必须为负数。
# 不能仅依赖默认权重推断方向，因为 Python 会把 -0.0 视为大于等于 0，
# 而 Unitree 默认配置恰好会用 -0.0 关闭 orientation、base_height 等惩罚项。
PENALTY_REWARDS = {
    "lin_vel_z", "ang_vel_xy", "orientation", "base_height", "torques", "dof_vel",
    "dof_acc", "action_rate", "collision", "termination", "dof_pos_limits",
    "dof_vel_limits", "torque_limits", "stumble", "feet_stumble", "stand_still",
    "feet_contact_forces", "horizontal_drift",
}


def _literal_assignments(class_node: ast.ClassDef) -> Dict[str, float]:
    """提取类定义中的数值常量赋值。"""
    values: Dict[str, float] = {}
    for node in class_node.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            if isinstance(value, (int, float)):
                values[node.targets[0].id] = float(value)
    return values


class RewardRegistry:
    def __init__(self, training_root: Path):
        """初始化 RewardRegistry 实例及其运行依赖。"""
        self.training_root = training_root

    def _methods(self) -> Dict[str, str]:
        """扫描并收集实际环境中的奖励实现符号。"""
        files = list((self.training_root / "legged_gym" / "envs").glob("**/*.py"))
        result: Dict[str, str] = {}
        for path in files:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("_reward_"):
                    result[node.name[8:]] = "%s:%s" % (relative_display(path), node.name)
        return result

    def _weights(self, robot: str) -> Dict[str, float]:
        """合并基础配置与机器人配置中的奖励权重。"""
        base = self.training_root / "legged_gym" / "envs" / "base" / "legged_robot_config.py"
        tree = ast.parse(base.read_text(encoding="utf-8"))
        values: Dict[str, float] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "scales":
                values.update(_literal_assignments(node))
                break
        robot_cfg = self.training_root / "legged_gym" / "envs" / robot / (robot + "_config.py")
        if robot_cfg.exists():
            tree = ast.parse(robot_cfg.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == "scales":
                    values.update(_literal_assignments(node))
        return values

    def inspect(self, robot: str = "go2") -> List[RewardRegistryItem]:
        """检查实际项目并返回机器可读的能力清单。"""
        methods = self._methods()
        weights = self._weights(robot)
        items: List[RewardRegistryItem] = []
        for name, implementation in sorted(methods.items()):
            if (name.startswith("rear_leg_") or name.startswith("front_leg_")) and robot != "go2":
                continue
            weight = weights.get(name, 0.0)
            unit_interval = (name.startswith("tracking_") or name.startswith("rear_leg_") or
                             name.startswith("front_leg_") or name == "termination")
            expected = [0.0, 1.0] if unit_interval else [0.0, 100.0]
            sign = "negative" if name in PENALTY_REWARDS else ("positive" if weight >= 0 else "negative")
            items.append(RewardRegistryItem(
                name=name, implementation=implementation,
                config_key="rewards.scales.%s" % name, expected_raw_range=expected,
                default_weight=weight, sign=sign,
                dependencies=DEPENDENCIES.get(name, [])))
        return items
