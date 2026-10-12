"""实现实验、记忆和索引 JSON 产物的显式格式版本迁移。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..utils.io import read_json, write_json
from ..utils.paths import ensure_within


CURRENT_FORMAT_VERSION = 2


class ArtifactMigrator:
    """对已知 JSON 产物执行幂等、非破坏性的字段补全迁移。"""

    def __init__(self, experiments_root: Path, memory_root: Path):
        """保存允许迁移的实验和记忆根目录。"""
        self.experiments_root = experiments_root.resolve()
        self.memory_root = memory_root.resolve()

    def candidates(self) -> List[Path]:
        """枚举具备稳定 Schema 的可迁移 JSON 文件。"""
        paths = []
        if self.experiments_root.is_dir():
            for pattern in ("task-*/state.json", "task-*/summary.json", "task-*/loop_status.json"):
                paths.extend((path, self.experiments_root) for path in self.experiments_root.glob(pattern))
        if self.memory_root.is_dir():
            for pattern in ("records/*.json", "semantic/*.json", "procedural/*.json"):
                paths.extend((path, self.memory_root) for path in self.memory_root.glob(pattern))
        safe: List[Path] = []
        for path, root in paths:
            if self._safe_candidate(path, root):
                safe.append(path)
        return sorted(set(safe))

    @staticmethod
    def _safe_candidate(path: Path, root: Path) -> bool:
        """拒绝文件或任务子目录链接，只允许当前根目录中的普通文件。"""
        try:
            path.relative_to(root)
            current = path
            while current != root:
                if current.is_symlink():
                    return False
                current = current.parent
            ensure_within(path, root)
            return path.is_file()
        except (OSError, ValueError):
            return False

    def migrate(self, dry_run: bool = True) -> Dict[str, Any]:
        """为旧产物增加格式版本；演练模式只报告而不写盘。"""
        changed = []
        skipped = []
        for path in self.candidates():
            root = next((root for root in (self.experiments_root, self.memory_root)
                         if path == root or root in path.parents), None)
            if root is None or not self._safe_candidate(path, root):
                skipped.append(str(path))
                continue
            value = read_json(path)
            if not isinstance(value, dict) or int(value.get("format_version", 0)) >= CURRENT_FORMAT_VERSION:
                skipped.append(str(path))
                continue
            value["format_version"] = CURRENT_FORMAT_VERSION
            if not dry_run:
                if not self._safe_candidate(path, root):
                    skipped.append(str(path))
                    continue
                write_json(path, value)
            changed.append(str(path))
        return {"format_version": CURRENT_FORMAT_VERSION, "dry_run": dry_run,
                "changed": changed, "skipped": skipped}
