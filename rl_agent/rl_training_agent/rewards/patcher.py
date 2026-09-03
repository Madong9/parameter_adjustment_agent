from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..utils.io import atomic_write_text, sha256_text
from .validator import RewardCodeValidator


class IsolatedRewardPatcher:
    def __init__(self, generated_root: Path, allowed_tensors):
        """初始化 IsolatedRewardPatcher 实例及其运行依赖。"""
        self.generated_root = generated_root
        self.validator = RewardCodeValidator(allowed_tensors)

    def create(self, name: str, source: str) -> Dict[str, object]:
        """校验并保存隔离的新奖励实现。"""
        if not name.replace("_", "").isalnum():
            raise ValueError("reward name must be alphanumeric with underscores")
        stats = self.validator.tensor_smoke_test(source)
        path = self.generated_root / (name + ".py")
        atomic_write_text(path, source.rstrip() + "\n")
        return {"name": name, "sha256": sha256_text(source), "smoke": stats, "path": path.name}

