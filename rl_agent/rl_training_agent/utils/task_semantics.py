"""提供跨任务理解、奖励归一化和审查共享的确定性动作语义规则。"""
from __future__ import annotations

import re
from enum import Enum
from typing import Iterable


class RollMotionKind(str, Enum):
    """区分侧滚与腾空侧翻，含糊文本不能成为自动动作配方。"""

    NONE = "NONE"
    GROUND_ROLL = "GROUND_ROLL"
    AERIAL_SIDE_FLIP = "AERIAL_SIDE_FLIP"
    AMBIGUOUS = "AMBIGUOUS"


def _positive_motion_text(text: str) -> str:
    """排除明确否定的动作短语，避免“不是侧空翻”反而触发空翻路由。"""
    value = str(text).lower()
    tokens = ("侧向翻跟头", "侧空翻", "侧向空翻", "侧向翻滚", "侧向翻转", "侧滚", "侧翻",
              "起跳", "腾空", "跳跃", "贴地", "地面翻滚", "地面侧滚",
              "aerial side flip", "side flip", "side_flip", "lateral flip", "ground roll",
              "lateral roll", "side roll", "airborne", "takeoff")
    pattern = "|".join(re.escape(token) for token in sorted(tokens, key=len, reverse=True))
    return re.sub(r"(?:不是|不做|不要|不得|禁止|无需|不允许|不|非|not\s+(?:an?\s+)?|no\s+|without\s+)"
                  r"\s*(?:进行|执行|做|发生)?\s*(?:" + pattern + r")", "", value)


def classify_roll_motion(text: str) -> RollMotionKind:
    """只根据用户原文的肯定目标及明确不腾空要求识别侧滚类别。"""
    raw = str(text).lower()
    positive = _positive_motion_text(raw)
    lateral = any(token in positive for token in (
        "侧向翻", "侧翻", "侧空翻", "侧滚", "side roll", "lateral roll",
        "side flip", "side_flip", "lateral flip"))
    if not lateral:
        return RollMotionKind.NONE
    aerial = any(token in positive for token in (
        "侧空翻", "侧向空翻", "侧向翻跟头", "side flip", "side_flip", "lateral flip",
        "起跳", "腾空", "airborne", "takeoff"))
    ground = any(token in positive for token in (
        "贴地", "地面侧滚", "地面翻滚", "地面滚动", "侧滚", "ground roll", "side roll", "lateral roll"))
    ground = ground or bool(re.search(
        r"(?:不|不要|无需|禁止)(?:进行)?\s*(?:起跳|腾空|跳跃)|without\s+(?:takeoff|airborne)", raw))
    if ground and aerial:
        return RollMotionKind.AMBIGUOUS
    if aerial:
        return RollMotionKind.AERIAL_SIDE_FLIP
    if ground:
        return RollMotionKind.GROUND_ROLL
    return RollMotionKind.AMBIGUOUS


def is_side_flip_text(parts: Iterable[str]) -> bool:
    """仅显式腾空侧翻匹配旧接口，地面侧滚和未澄清动作不自动套空翻配方。"""
    # 第一项应是用户原文；后续模型命名不允许覆盖已经明确的动作类别。
    values = [str(part) for part in parts if part]
    return bool(values and classify_roll_motion(values[0]) == RollMotionKind.AERIAL_SIDE_FLIP)


def is_front_leg_support_text(parts: Iterable[str]) -> bool:
    """判断文本是否要求以前腿支撑，而不是要求把前腿抬离地面。"""
    text = " ".join(str(part) for part in parts if part).lower()
    mentions_front = any(token in text for token in (
        "前腿", "前脚", "前足", "front leg", "front-leg", "foreleg"))
    support_action = any(token in text for token in (
        "站立", "走路", "行走", "倒立", "前进", "移动", "迈步",
        "stand", "walk", "locomotion", "handstand", "inverted"))
    explicitly_lifted = any(token in text for token in (
        "抬起前腿", "前腿离地", "前足离地", "lift front leg", "front legs off ground"))
    return mentions_front and support_action and not explicitly_lifted
