"""把 Agent 编译配置安全地应用到 Unitree 训练、播放和评估环境。"""
from __future__ import annotations

import copy
import math
import re
from types import MethodType
from typing import Any, Dict, Iterable, List, Optional, Tuple


REWARD_PARAMETER_NAMES = {
    "base_height_target", "tracking_sigma", "jump_height_target", "max_contact_force",
    "rear_stand_height_target", "rear_stand_pitch_target", "rear_stand_height_sigma",
    "rear_stand_pitch_sigma",
    "front_stand_height_target", "front_stand_pitch_target", "front_stand_height_sigma",
    "front_stand_pitch_sigma",
}
COMMAND_RANGE_NAMES = {"lin_vel_x", "lin_vel_y", "ang_vel_yaw", "heading"}


def _enabled_terminations(config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """迭代配置中所有启用且非超时的终止条件。"""
    return (item for item in config.get("terminations", [])
            if item.get("enabled", True) and not item.get("is_timeout", False))


def _number_from_condition(condition: str, default: float) -> float:
    """从受限条件文本中提取第一个有限阈值。"""
    match = re.search(r"(?:>|>=|<|<=)\s*([0-9]+(?:\.[0-9]+)?)", condition or "")
    value = float(match.group(1)) if match else float(default)
    return value if math.isfinite(value) and value > 0 else float(default)


def _phase_token(value: str) -> str:
    """把自然语言阶段名称归一化为便于匹配的英文标记。"""
    token = re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")
    for suffix in ("_learning", "_phase", "_stage"):
        if token.endswith(suffix):
            token = token[:-len(suffix)]
    return token.replace("walking", "walk").replace("standing", "stand").replace("balancing", "balance")


def _phase_matches(active_phases: Iterable[str], stage_name: Optional[str]) -> bool:
    """判断奖励声明的活动阶段是否覆盖当前课程阶段。"""
    phases = {_phase_token(item) for item in active_phases}
    if not phases or "all" in phases or stage_name is None:
        return True
    stage = _phase_token(stage_name)
    return stage in phases or any(stage.startswith(item) or item.startswith(stage) for item in phases if item)


def reward_scales_for_stage(config: Dict[str, Any], stage_name: Optional[str]) -> Dict[str, float]:
    """计算指定课程阶段真正生效的奖励权重。"""
    scales = {name: float(value) for name, value in config["rewards"]["scales"].items()}
    terms = config.get("rewards", {}).get("terms", [])
    if terms:
        for term in terms:
            name = str(term["name"])
            if name in scales and not _phase_matches(term.get("active_phases", ["all"]), stage_name):
                scales[name] = 0.0
    stage = next((item for item in config.get("curriculum", []) if item.get("name") == stage_name), None)
    overrides = (stage or {}).get("parameter_changes", {}).get("reward_scales", {})
    if isinstance(overrides, dict):
        for name, value in overrides.items():
            if name in scales:
                scales[name] = float(value)
    return scales


def curriculum_segments(config: Dict[str, Any], total_iterations: int) -> List[Tuple[int, int, Optional[str]]]:
    """把计划中的绝对课程边界按本次训练长度缩放为连续分段。"""
    if total_iterations <= 0:
        raise ValueError("训练迭代次数必须为正数")
    stages = sorted(config.get("curriculum", []), key=lambda item: int(item.get("start_iteration", 0)))
    if not stages:
        return [(0, total_iterations, None)]
    source_horizon = max(int(item.get("end_iteration", 0)) for item in stages) + 1
    if source_horizon <= 0:
        return [(0, total_iterations, None)]
    starts = [min(total_iterations - 1, max(0, round(
        int(item.get("start_iteration", 0)) * total_iterations / source_horizon))) for item in stages]
    starts[0] = 0
    segments: List[Tuple[int, int, Optional[str]]] = []
    for index, stage in enumerate(stages):
        start = starts[index]
        end = starts[index + 1] if index + 1 < len(starts) else total_iterations
        if end > start:
            segments.append((start, end - start, str(stage.get("name", ""))))
    return segments or [(0, total_iterations, None)]


def _apply_reward_parameters(env_cfg: Any, config: Dict[str, Any], stage_name: Optional[str] = None) -> None:
    """将白名单奖励参数写入环境配置。"""
    for term in config.get("rewards", {}).get("terms", []):
        for name, value in term.get("parameters", {}).items():
            if name in REWARD_PARAMETER_NAMES:
                setattr(env_cfg.rewards, name, float(value))
    if stage_name is not None:
        stage = next((item for item in config.get("curriculum", []) if item.get("name") == stage_name), None)
        changes = (stage or {}).get("parameter_changes", {})
        if "base_height_target" in changes:
            env_cfg.rewards.base_height_target = float(changes["base_height_target"])


def _apply_command_changes(target: Any, changes: Dict[str, Any], originals: Dict[str, Any]) -> None:
    """从原始范围出发应用一个课程阶段的数值化命令变化。"""
    for name, value in originals.items():
        setattr(target, name, copy.deepcopy(value)) if not isinstance(target, dict) else target.__setitem__(name, copy.deepcopy(value))
    scale = changes.get("command_scale")
    legacy = str(changes.get("commands", "")).lower()
    if scale is None and "low" in legacy:
        scale = 0.25
    elif scale is None and ("increase" in legacy or "full" in legacy):
        scale = 1.0
    if scale is not None:
        factor = max(0.0, float(scale))
        for name in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw"):
            original = originals.get(name)
            if isinstance(original, (list, tuple)) and len(original) == 2:
                value = [float(original[0]) * factor, float(original[1]) * factor]
                setattr(target, name, value) if not isinstance(target, dict) else target.__setitem__(name, value)
    for name in COMMAND_RANGE_NAMES:
        if name not in changes:
            continue
        value = changes[name]
        if isinstance(value, (int, float)):
            requested = float(value)
            # Unitree 会把平面速度模长 <=0.2 m/s 的命令清零。标量表示明确目标而非
            # 对称采样范围；前向目标过小时提升到刚超过死区，避免“行走课程”实际静止。
            if name == "lin_vel_x" and 0.0 < abs(requested) <= 0.2:
                requested = math.copysign(0.21, requested)
            value = [requested, requested]
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("课程命令范围 %s 必须是两个数值" % name)
        result = [float(value[0]), float(value[1])]
        setattr(target, name, result) if not isinstance(target, dict) else target.__setitem__(name, result)


def prepare_training_env_config(env_cfg: Any, config: Dict[str, Any]) -> None:
    """在创建训练环境前注册全部奖励、参数和身体接触终止链接。"""
    for name, weight in config["rewards"]["scales"].items():
        if not hasattr(env_cfg.rewards.scales, name):
            raise ValueError("compiled reward is not registered: %s" % name)
        setattr(env_cfg.rewards.scales, name, float(weight))
    _apply_reward_parameters(env_cfg, config)
    if any("body" in str(item.get("name", "")).lower() or
           "body contact" in str(item.get("condition", "")).lower()
           for item in _enabled_terminations(config)):
        patterns = list(dict.fromkeys(list(env_cfg.asset.terminate_after_contacts_on) +
                                      list(env_cfg.asset.penalize_contacts_on)))
        env_cfg.asset.terminate_after_contacts_on = patterns


def prepare_evaluation_env_config(env_cfg: Any, config: Dict[str, Any]) -> None:
    """在创建播放或评估环境前应用最终课程阶段配置。"""
    prepare_training_env_config(env_cfg, config)
    stages = sorted(config.get("curriculum", []), key=lambda item: int(item.get("start_iteration", 0)))
    stage_name = str(stages[-1].get("name", "")) if stages else None
    for name, weight in reward_scales_for_stage(config, stage_name).items():
        setattr(env_cfg.rewards.scales, name, float(weight))
    _apply_reward_parameters(env_cfg, config, stage_name)
    if stages:
        changes = stages[-1].get("parameter_changes", {})
        ranges = env_cfg.commands.ranges
        originals = {name: copy.deepcopy(getattr(ranges, name)) for name in COMMAND_RANGE_NAMES
                     if hasattr(ranges, name)}
        _apply_command_changes(ranges, changes, originals)


def install_runtime_terminations(env: Any, config: Dict[str, Any]) -> None:
    """给环境安装配置声明的 roll、pitch 终止检查。"""
    pitch_limit = roll_limit = None
    for item in _enabled_terminations(config):
        name = str(item.get("name", "")).lower()
        condition = str(item.get("condition", "")).lower()
        if "pitch" in name or "pitch" in condition:
            pitch_limit = _number_from_condition(condition, 1.0)
        if "roll" in name or "roll" in condition:
            roll_limit = _number_from_condition(condition, 0.8)
    if pitch_limit is None and roll_limit is None:
        return
    original = env.check_termination

    def check_termination_with_orientation(self: Any) -> None:
        """运行项目原生终止检查，并追加配置声明的姿态限制。"""
        original()
        if pitch_limit is not None:
            self.reset_buf |= self.rpy[:, 1].abs() > pitch_limit
        if roll_limit is not None:
            self.reset_buf |= self.rpy[:, 0].abs() > roll_limit

    env.check_termination = MethodType(check_termination_with_orientation, env)


def apply_runtime_stage(env: Any, config: Dict[str, Any], stage_name: Optional[str],
                        original_command_ranges: Dict[str, Any]) -> None:
    """在训练环境内切换当前阶段的奖励权重、目标参数和命令范围。"""
    for name, weight in reward_scales_for_stage(config, stage_name).items():
        if name in env.reward_scales:
            env.reward_scales[name] = float(weight) * env.dt
    _apply_reward_parameters(env.cfg, config, stage_name)
    stage = next((item for item in config.get("curriculum", []) if item.get("name") == stage_name), None)
    _apply_command_changes(env.command_ranges, (stage or {}).get("parameter_changes", {}),
                           original_command_ranges)
