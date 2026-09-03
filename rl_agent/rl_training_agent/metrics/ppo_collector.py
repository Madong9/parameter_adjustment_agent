from __future__ import annotations

from pathlib import Path

from ..schemas.metrics import PPOStatistics
from .tensorboard_reader import TensorBoardReader


class PPOCollector:
    TAGS = {
        "mean_reward": "Train/mean_reward", "value_loss": "Loss/value_function",
        "surrogate_loss": "Loss/surrogate", "learning_rate": "Loss/learning_rate",
        "mean_noise_std": "Policy/mean_noise_std", "fps": "Perf/total_fps",
    }

    def collect(self, log_dir: Path) -> PPOStatistics:
        """从 TensorBoard 日志汇总最新 PPO 指标。"""
        latest = TensorBoardReader().latest(log_dir)
        kwargs = {field: latest.get(tag, 0.0) for field, tag in self.TAGS.items()}
        iterations = [item["step"] for series in TensorBoardReader().read(log_dir).values() for item in series[-1:]]
        kwargs["iteration"] = max(iterations) if iterations else 0
        return PPOStatistics(**kwargs)
