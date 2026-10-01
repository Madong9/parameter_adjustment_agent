"""实现适合流式读取和离线分析的 JSONL 结构化事件记录器。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from ..utils.io import json_safe, utc_now


class JsonlEventRecorder:
    """以单行 JSON 追加状态、Provider、训练和资源事件。"""

    def __init__(self, path: Path):
        """保存事件文件路径。"""
        self.path = path

    def emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        """追加一条带 UTC 时间的严格 JSON 事件并立即刷盘。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"at": utc_now(), "event_type": event_type, **json_safe(payload)}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
