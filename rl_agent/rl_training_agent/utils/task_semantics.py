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


class ForwardFlipKind(str, Enum):
    """区分前向地面翻滚与腾空前翻，不将前翻映射为侧翻。"""

    NONE = "NONE"
    GROUND = "GROUND_FORWARD_ROLL"
    AERIAL = "AERIAL_FRONT_FLIP"
    AMBIGUOUS = "AMBIGUOUS_FORWARD_FLIP"


# 方向与翻转之间允许明确的动作修饰词，不能跨标点连接两个独立目标。
_FORWARD_MOTION_PATTERN = (
    r"(?:向前|朝前|往前|前向)"
    r"(?:[ \t]|主动|起跳|腾空|在空中|进行|完成|一个|一次|一圈|的)*"
    r"(?:翻跟头|翻筋斗|翻滚|翻转|空翻)"
)


def _remove_motion_contrasts(text: str) -> str:
    """保留比较句的主语动作，只排除明确作为不同动作对照的短语。"""
    motions = (
        "侧向翻跟头", "侧空翻", "侧向空翻", "侧向翻滚", "侧向翻转", "侧滚", "侧翻",
        "向前翻跟头", "前翻跟头", "向前翻滚", "前空翻", "前滚翻", "前翻", "前滚",
        "front flip", "frontflip", "forward flip", "forward roll", "forward somersault",
        "front somersault", "aerial side flip", "side flip", "side_flip", "lateral flip",
        "ground roll", "lateral roll", "side roll",
    )
    motion = "(?:" + _FORWARD_MOTION_PATTERN + "|" + "|".join(
        re.escape(token) for token in sorted(motions, key=len, reverse=True)) + ")"
    # 只允许轴向、接触方式等已知修饰词；不能吞掉“必须/完成/执行”等独立要求。
    modifiers = (r"(?:(?:绕|沿|机体|身体|机器人|纵向|横向|纵轴|横轴|roll|pitch|body|轴|的|"
                 r"允许|持续|接触|触地|地面|贴地|腾空|起跳|在空中|aerial|ground)[ \t]*)*")
    target = r"(?:an?[ \t]+)?" + modifiers + motion
    targets = target + r"(?:[ \t]*(?:、|或|和|及|以及|[ \t]+and[ \t]+)[ \t]*" + target + r")*"
    subject = r"(?P<motion>" + motion + r")[ \t]*"

    def keep_subject(match: re.Match) -> str:
        """对照后的独立肯定要求继续保留，包括明确要求同时执行两者。"""
        remainder = re.split(r"[。.!！?？\n]", match.string[match.end():], maxsplit=1)[0]
        if (re.search(r"(?:两者|二者|(?:这)?两[个种]动作)[^，,。.!！?？\n]{0,24}"
                      r"(?:完成|执行|进行|做|训练)", remainder)
                or re.search(r"(?:都|均)[ \t]*(?:必须|需要|要|需)?[ \t]*(?:完成|执行|进行|做)", remainder)
                or re.search(r"\bboth\b", remainder)):
            return match.group(0)
        return match.group('motion')

    comparisons = (
        subject + r"(?:与|和|跟|同)[ \t]*" + targets +
        r"[ \t]*(?:(?:具有|存在|有|是)[ \t]*)?(?:不同|区别|差异)",
        subject + r"(?:区别于|不同于|有别于)[ \t]*" + targets,
        subject + r"(?:不能|不得|不可|不应)[ \t]*(?:与|和|同)[ \t]*" + targets +
        r"[ \t]*(?:混同|混淆|等同|混为一谈)",
        subject + r"(?:不能|不得|不可|不应)[ \t]*(?:混同|混淆|等同)(?:为|于)[ \t]*" + targets,
        subject + r"(?:differs[ \t]+from|is[ \t]+(?:different|distinct)[ \t]+from|"
        r"is[ \t]+not[ \t]+(?:equivalent[ \t]+to|the[ \t]+same[ \t]+as))[ \t]+" + targets,
    )
    for comparison in comparisons:
        text = re.sub(comparison, keep_subject, text)
    return text


def _positive_motion_text(text: str) -> str:
    """排除明确否定的动作短语，避免“不是侧空翻”反而触发空翻路由。"""
    value = str(text).lower()
    tokens = ("侧向翻跟头", "侧空翻", "侧向空翻", "侧向翻滚", "侧向翻转", "侧滚", "侧翻",
              "向前翻跟头", "前翻跟头", "向前翻滚", "前空翻", "前滚翻", "前翻", "前滚",
              "front flip", "frontflip", "forward flip", "forward roll", "forward somersault",
              "front somersault",
              "起跳", "腾空", "在空中", "跳跃", "贴地", "地面翻滚", "地面侧滚",
              "aerial side flip", "side flip", "side_flip", "lateral flip", "ground roll",
              "lateral roll", "side roll", "airborne", "takeoff")
    pattern = _FORWARD_MOTION_PATTERN + "|" + "|".join(
        re.escape(token) for token in sorted(tokens, key=len, reverse=True))
    # 仅移除禁止的比较表达，截止到其目标动作词，保留随后独立的肯定要求。
    value = re.sub(
        r"(?:不得|不要|禁止|不能|不允许)[ \t]*(?:将|把)"
        r"[^，,。.;；!！?？\n]{0,80}?(?:误判为|误认为|混同为|当作|替换为)[ \t]*"
        r"(?:(?:纵向轴|横向轴|纵轴|横轴|机体|身体|的)[ \t]*)*(?:" + pattern + r")",
        "", value)
    value = re.sub(
        r"(?:do\s+not|don't|must\s+not)\s+(?:mistake|confuse)\s+"
        r"[^,.;!?\n]{0,80}?\s+(?:for|with)\s+(?:an?\s+)?(?:" + pattern + r")",
        "", value)
    value = _remove_motion_contrasts(value)
    return re.sub(r"(?:不是|不做|不要|不得|禁止|无需|不允许|不|非|not\s+(?:an?\s+)?|no\s+|without\s+)"
                  r"[ \t]*(?:(?:进行|执行|做|发生|机体|身体|以|沿|在|接触|滚动|地面|的)[ \t]*)*"
                  r"(?:" + pattern + r")", "", value)


def classify_forward_flip(text: str) -> ForwardFlipKind:
    """从肯定的用户目标识别前翻接触方式，含糊翻跟头先澄清。"""
    raw = str(text).lower()
    positive = _positive_motion_text(raw)
    mentions_forward = re.search(_FORWARD_MOTION_PATTERN, positive) is not None
    if not mentions_forward and not any(token in positive for token in (
            "向前翻跟头", "前翻跟头", "向前翻滚", "前空翻", "前滚翻", "前翻", "前滚",
            "front flip", "frontflip", "forward flip", "forward roll", "forward somersault",
            "front somersault")):
        return ForwardFlipKind.NONE
    aerial = any(token in positive for token in (
        "前空翻", "起跳", "腾空", "在空中", "front flip", "frontflip", "airborne", "takeoff", "aerial"))
    ground = any(token in positive for token in (
        "贴地", "前滚翻", "地面翻滚", "地面滚动", "forward roll", "ground roll"))
    ground = ground or bool(re.search(
        r"(?:不|不要|无需|禁止)(?:进行)?\s*(?:起跳|腾空|跳跃|在空中)|without\s+(?:takeoff|airborne)", raw))
    if ground and aerial:
        return ForwardFlipKind.AMBIGUOUS
    if aerial:
        return ForwardFlipKind.AERIAL
    if ground:
        return ForwardFlipKind.GROUND
    return ForwardFlipKind.AMBIGUOUS


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
