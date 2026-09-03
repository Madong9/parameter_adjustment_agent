from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    """返回带时区的当前 UTC 时间字符串。"""
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    """通过临时文件和原子替换安全写入文本。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, data: Any) -> None:
    """以稳定格式原子写入 JSON 数据。"""
    if hasattr(data, "dict"):
        data = data.dict()
    atomic_write_text(path, json.dumps(
        json_safe(data), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def json_safe(data: Any) -> Any:
    """递归把 NaN/Inf 转换为 null，确保产物和模型证据都是严格 JSON。"""
    if isinstance(data, float):
        return data if math.isfinite(data) else None
    if isinstance(data, dict):
        return {key: json_safe(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [json_safe(value) for value in data]
    return data


def read_json(path: Path) -> Any:
    """读取并解析指定 JSON 文件。"""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_text(text: str) -> str:
    """计算文本内容的 SHA-256 摘要。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
