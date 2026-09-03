from __future__ import annotations

from pathlib import Path
from typing import Union


AGENT_ROOT = Path(__file__).resolve().parents[2]


def resolve_relative(path: Union[str, Path], base: Path = AGENT_ROOT) -> Path:
    """解析配置中的相对路径，避免持久化主机专属路径。"""
    value = Path(path)
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def ensure_within(path: Union[str, Path], root: Union[str, Path]) -> Path:
    """校验并返回位于允许根目录内的安全路径。"""
    candidate = Path(path).resolve()
    allowed = Path(root).resolve()
    try:
        candidate.relative_to(allowed)
    except ValueError as exc:
        raise ValueError("path escapes allowed root: %s" % path) from exc
    return candidate


def relative_display(path: Union[str, Path], base: Path = AGENT_ROOT) -> str:
    """把运行时路径转换为适合持久化的相对显示路径。"""
    candidate = Path(path).resolve()
    try:
        return candidate.relative_to(base.resolve()).as_posix()
    except ValueError:
        try:
            return "../" + candidate.relative_to(base.parent.resolve()).as_posix()
        except ValueError:
            return candidate.name
