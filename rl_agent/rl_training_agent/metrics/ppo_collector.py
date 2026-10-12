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
        series_by_tag = TensorBoardReader().read(log_dir)
        latest = {tag: series[-1]["value"] for tag, series in series_by_tag.items() if series}
        kwargs = {field: latest.get(tag, 0.0) for field, tag in self.TAGS.items()}
        iterations = [series[-1]["step"] for series in series_by_tag.values() if series]
        kwargs["iteration"] = max(iterations) if iterations else 0
        return PPOStatistics(**kwargs)
