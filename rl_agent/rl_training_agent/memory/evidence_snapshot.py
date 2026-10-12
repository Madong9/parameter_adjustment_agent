"""保存可移动且不可覆盖的长期经验证据，避免新任务运行改写旧真值。"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from ..storage.locking import FileLock
from ..utils.io import read_json, write_json
from ..utils.paths import ensure_within


SNAPSHOT_MANIFEST = "snapshot_manifest.json"


def file_sha256(path: Path) -> str:
    """流式计算产物摘要，不把大型 checkpoint 一次读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_evidence_path(root: Path, source: str) -> Path:
    """拒绝绝对路径、路径跳转和任意父目录链接，并保留原始文件名。"""
    relative = Path(source)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts or "\\" in source:
        raise ValueError("证据路径必须是安全相对路径")
    path = root / relative
    current = path
    while current != root:
        if current.is_symlink():
            raise ValueError("证据路径不能包含符号链接")
        current = current.parent
    ensure_within(path, root)
    return path


def create_evidence_snapshot(memory_root: Path, memory_id: str, task_dir: Path,
                             paths: Iterable[Path],
                             checkpoint_provenance: Optional[Dict[str, Any]] = None,
                             expected_hashes: Optional[Dict[str, str]] = None
                             ) -> Tuple[str, Dict[str, str]]:
    """复制验证过的少量证据并原子发布；同一记忆的既有快照只能复用，不能重写。"""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", memory_id):
        raise ValueError("不安全的记忆标识")
    memory_root = memory_root.resolve()
    task_dir = task_dir.resolve()
    snapshots = safe_evidence_path(memory_root, "evidence")
    snapshots.mkdir(parents=True, exist_ok=True)
    target = safe_evidence_path(memory_root, "evidence/" + memory_id)
    with FileLock(memory_root / ".evidence.lock"):
        staging = Path(tempfile.mkdtemp(prefix=memory_id + "-", dir=str(snapshots)))
        try:
            hashes: Dict[str, str] = {}
            for original in sorted(set(paths)):
                source = original.relative_to(task_dir).as_posix()
                original = safe_evidence_path(task_dir, source)
                if not original.is_file() or original.suffix.lower() in (".pt", ".mp4", ".webm", ".avi"):
                    raise ValueError("快照来源缺失或属于不应复制的大型策略/视频")
                destination = safe_evidence_path(staging, source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, destination)
                digest = file_sha256(destination)
                if expected_hashes and source in expected_hashes and digest != expected_hashes[source]:
                    raise ValueError("复制时经验证据已发生变化：" + source)
                if digest != file_sha256(original):
                    raise ValueError("快照复制期间来源发生变化：" + source)
                hashes[source] = digest
            manifest = {"format_version": 1, "memory_id": memory_id,
                        "hashes": hashes, "checkpoint_provenance": checkpoint_provenance or {}}
            write_json(staging / SNAPSHOT_MANIFEST, manifest)
            hashes = {**hashes, SNAPSHOT_MANIFEST: file_sha256(staging / SNAPSHOT_MANIFEST)}
            if target.exists():
                existing = read_json(safe_evidence_path(target, SNAPSHOT_MANIFEST))
                if existing != manifest or any(
                        file_sha256(safe_evidence_path(target, source)) != digest
                        for source, digest in hashes.items()):
                    raise ValueError("不可变证据快照已存在且内容不一致")
            else:
                os.rename(str(staging), str(target))
            return target.relative_to(memory_root).as_posix(), hashes
        finally:
            if staging.exists():
                shutil.rmtree(staging)
