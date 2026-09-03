from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional


class CheckpointManager:
    PATTERN = re.compile(r"model_(\d+)\.pt$")

    @classmethod
    def list_checkpoints(cls, directory: Path) -> List[Path]:
        """按训练迭代排序查找所有 checkpoint。"""
        paths = []
        for path in directory.glob("**/model_*.pt"):
            match = cls.PATTERN.search(path.name)
            if match:
                paths.append(path)
        return sorted(paths, key=lambda path: int(cls.PATTERN.search(path.name).group(1)))

    @classmethod
    def latest(cls, directory: Path) -> Optional[Path]:
        """返回每个 TensorBoard 标量标签的最新值。"""
        paths = cls.list_checkpoints(directory)
        return paths[-1] if paths else None

