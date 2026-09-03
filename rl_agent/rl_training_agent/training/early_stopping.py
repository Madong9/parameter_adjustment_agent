from __future__ import annotations

import math
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
        """检查日志停滞、数值异常、OOM 和进程错误。"""
        if not path.exists():
            return "missing_log"
        if time.time() - path.stat().st_mtime > self.stale_seconds:
            return "log_stale"
        text = path.read_text(encoding="utf-8", errors="replace")[-100000:]
        for reason, pattern in self.FAILURE_PATTERNS.items():
            if pattern.search(text):
                return reason
        rewards = re.findall(r"Mean reward:\s*([-+0-9.eE]+)", text)
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

