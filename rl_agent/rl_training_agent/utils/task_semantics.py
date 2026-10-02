"""提供跨任务理解、奖励归一化和审查共享的确定性动作语义规则。"""
from __future__ import annotations

from typing import Iterable


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
