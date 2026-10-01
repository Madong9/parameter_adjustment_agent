"""实现实验、记忆和索引 JSON 产物的显式格式版本迁移。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..utils.io import read_json, write_json


CURRENT_FORMAT_VERSION = 2


class ArtifactMigrator:
    """对已知 JSON 产物执行幂等、非破坏性的字段补全迁移。"""

    def __init__(self, experiments_root: Path, memory_root: Path):
        """保存允许迁移的实验和记忆根目录。"""
        self.experiments_root = experiments_root.resolve()
        self.memory_root = memory_root.resolve()

    def candidates(self) -> List[Path]:
        """枚举具备稳定 Schema 的可迁移 JSON 文件。"""
        paths: List[Path] = []
        if self.experiments_root.is_dir():
            for pattern in ("task-*/state.json", "task-*/summary.json", "task-*/loop_status.json"):
                paths.extend(self.experiments_root.glob(pattern))
        if self.memory_root.is_dir():
            for pattern in ("records/*.json", "semantic/*.json", "procedural/*.json"):
                paths.extend(self.memory_root.glob(pattern))
        return sorted(set(path.resolve() for path in paths))

    def migrate(self, dry_run: bool = True) -> Dict[str, Any]:
        """为旧产物增加格式版本；演练模式只报告而不写盘。"""
        changed = []
        skipped = []
        for path in self.candidates():
            value = read_json(path)
            if not isinstance(value, dict) or int(value.get("format_version", 0)) >= CURRENT_FORMAT_VERSION:
                skipped.append(str(path))
                continue
            value["format_version"] = CURRENT_FORMAT_VERSION
            if not dry_run:
                write_json(path, value)
            changed.append(str(path))
        return {"format_version": CURRENT_FORMAT_VERSION, "dry_run": dry_run,
                "changed": changed, "skipped": skipped}
