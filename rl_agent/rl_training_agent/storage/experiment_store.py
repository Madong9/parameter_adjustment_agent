from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..utils.io import read_json, write_json
from ..utils.paths import ensure_within
from .locking import FileLock


class ExperimentStore:
    def __init__(self, root: Path):
        """初始化 ExperimentStore 实例及其运行依赖。"""
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def task_dir(self, task_id: str) -> Path:
        """校验任务标识并返回安全任务目录。"""
        if not task_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in task_id):
            raise ValueError("unsafe task id")
        return ensure_within(self.root / task_id, self.root)

    def initialize_task(self, task_id: str) -> Path:
        """加锁创建任务、候选和最终产物目录。"""
        path = self.task_dir(task_id)
        with FileLock(self.root / ".store.lock"):
            path.mkdir(parents=True, exist_ok=True)
            (path / "candidates").mkdir(exist_ok=True)
            (path / "final").mkdir(exist_ok=True)
        return path

    def candidate_dir(self, task_id: str, experiment_id: str) -> Path:
        """校验实验标识并返回安全候选目录。"""
        if not experiment_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError("unsafe experiment id")
        path = self.task_dir(task_id) / "candidates" / experiment_id
        return ensure_within(path, self.root)

    def write(self, task_id: str, relative: str, value: Any) -> Path:
        """将检查结果写入指定的机器可读文件。"""
        path = ensure_within(self.task_dir(task_id) / relative, self.task_dir(task_id))
        with FileLock(self.task_dir(task_id) / ".lock"):
            write_json(path, value)
        return path

    def read(self, task_id: str, relative: str) -> Any:
        """读取并合并目录下的 TensorBoard 标量事件。"""
        return read_json(ensure_within(self.task_dir(task_id) / relative, self.task_dir(task_id)))

