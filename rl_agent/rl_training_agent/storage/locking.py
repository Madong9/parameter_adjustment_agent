from __future__ import annotations

import fcntl
from pathlib import Path
from typing import IO, Optional


class FileLock:
    def __init__(self, path: Path):
        """初始化 FileLock 实例及其运行依赖。"""
        self.path = path
        self.handle: Optional[IO[str]] = None

    def __enter__(self):
        """获取文件锁并进入受保护的写入上下文。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """释放文件锁并关闭锁文件句柄。"""
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
        self.handle = None

