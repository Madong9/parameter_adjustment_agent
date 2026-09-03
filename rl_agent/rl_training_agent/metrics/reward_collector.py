from __future__ import annotations

from typing import Iterable, List

import numpy as np

from ..schemas.metrics import RewardStatistics


class RewardCollector:
    def summarize(self, name: str, raw_values: Iterable[float], weight: float,
                  total_weighted_values: Iterable[float], saturation_limit: float = 1.0,
                  phase: str = "all") -> RewardStatistics:
        """汇总单项奖励的原始值、加权值和贡献统计。"""
        raw = np.asarray(list(raw_values), dtype=float)
        total = np.asarray(list(total_weighted_values), dtype=float)
        if raw.size == 0:
            raw = np.zeros(1)
        weighted = raw * weight
        denominator = float(np.mean(np.abs(total))) if total.size else 0.0
        contribution = float(np.mean(np.abs(weighted)) / denominator) if denominator > 1e-12 else 0.0
        return RewardStatistics(
            name=name, raw_mean=float(raw.mean()), raw_std=float(raw.std()),
            raw_min=float(raw.min()), raw_max=float(raw.max()), weight=weight,
            weighted_mean=float(weighted.mean()), weighted_std=float(weighted.std()),
            contribution_ratio=contribution, nonzero_ratio=float(np.count_nonzero(raw) / raw.size),
            saturation_ratio=float(np.mean(np.abs(raw) >= saturation_limit)),
            per_phase={phase: float(raw.mean())}, episode_sum=float(raw.sum()))

