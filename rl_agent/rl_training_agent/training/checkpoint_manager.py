from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional


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

    @classmethod
    def prune(cls, directory: Path, keep_per_run: int = 1,
              protected: Iterable[Path] = ()) -> List[Path]:
        """按运行目录保留末次 checkpoint，并返回已删除的中间 checkpoint。"""
        if keep_per_run <= 0:
            raise ValueError("keep_per_run must be positive")
        protected_paths = {path.resolve() for path in protected}
        grouped = {}
        for path in cls.list_checkpoints(directory):
            grouped.setdefault(path.parent, []).append(path)
        removed: List[Path] = []
        for paths in grouped.values():
            ordered = sorted(
                paths, key=lambda path: int(cls.PATTERN.search(path.name).group(1)))
            retained = {path.resolve() for path in ordered[-keep_per_run:]}
            retained.update(protected_paths)
            for path in ordered:
                if path.resolve() not in retained:
                    path.unlink()
                    removed.append(path)
        return removed
