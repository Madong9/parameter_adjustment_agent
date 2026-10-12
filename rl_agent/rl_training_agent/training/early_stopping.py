from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path
from typing import Dict, Optional


class EarlyStoppingMonitor:
    FAILURE_PATTERNS = {
        "nan_or_inf": re.compile(r"\b(?:nan|inf)\b", re.IGNORECASE),
        "gpu_oom": re.compile(r"out of memory|cuda oom", re.IGNORECASE),
        "process_error": re.compile(r"traceback|segmentation fault", re.IGNORECASE),
    }

    def __init__(self, stale_seconds: int = 600, max_abs_reward: float = 1e6):
        """初始化 EarlyStoppingMonitor 实例及其运行依赖。"""
        self.stale_seconds = stale_seconds
        self.max_abs_reward = max_abs_reward

    def inspect_log(self, path: Path) -> Optional[str]:
        """限量读取日志尾部，检查停滞、数值异常、OOM 和进程错误。"""
        try:
            with path.open("rb") as handle:
                stat = os.fstat(handle.fileno())
                if time.time() - stat.st_mtime > self.stale_seconds:
                    return "log_stale"
                # UTF-8 每个字符最多四字节，保留原来的十万字符检查窗口。
                handle.seek(max(0, stat.st_size - 400000))
                text = handle.read(400000).decode("utf-8", errors="replace")[-100000:]
        except FileNotFoundError:
            return "missing_log"
        for reason, pattern in self.FAILURE_PATTERNS.items():
            if pattern.search(text):
                return reason
        rewards = re.findall(
            r"Mean reward:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
            r"(?=\s|$)", text)
        if rewards:
            value = float(rewards[-1])
            if not math.isfinite(value) or abs(value) > self.max_abs_reward:
                return "reward_exploded"
        return None

    def inspect_metrics(self, metrics: Dict[str, float]) -> Optional[str]:
        """依据结构化训练指标判断提前停止原因。"""
        if any(not math.isfinite(float(value)) for value in metrics.values()):
            return "nan_or_inf"
        if metrics.get("all_rewards_zero", 0.0) > 0.5:
            return "all_rewards_zero"
        if metrics.get("immediate_termination_rate", 0.0) > 0.9:
            return "episodes_immediately_terminate"
        if metrics.get("fall_rate", 0.0) > 0.95:
            return "persistent_falls"
        if metrics.get("kl", 0.0) > 0.2:
            return "kl_limit"
        return None
