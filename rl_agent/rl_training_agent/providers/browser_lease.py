"""按浏览器 profile 串行化完整网页请求，保护绑定标签页及主备会话。"""
from contextlib import contextmanager
from functools import wraps
import hashlib
import threading

from ..storage.locking import FileLock
from ..utils.paths import AGENT_ROOT


class BrowserLease:
    """支持同一 Provider 嵌套调用的线程锁与跨进程文件锁。"""

    def __init__(self):
        """初始化线程内重入锁与嵌套计数。"""
        self.lock = threading.RLock()
        self.depth = 0

    @contextmanager
    def acquire(self, profile):
        """完整请求期间持锁；同 profile 的不同别名也不能同时绑定同一标签页。"""
        with self.lock:
            key = hashlib.sha256((profile or 'default').encode()).hexdigest()[:20]
            file_lock = FileLock(AGENT_ROOT / 'artifacts' / 'browser_locks' / (key + '.lock'))
            outer = self.depth == 0
            if outer:
                file_lock.__enter__()
            self.depth += 1
            try:
                yield
            finally:
                self.depth -= 1
                if outer:
                    file_lock.__exit__(None, None, None)


def serialize_browser_operation(method):
    """将页面导航、附件、发送、等待及关闭操作纳入同一个可重入租约。"""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        """使用 Provider 的 profile 保护本次浏览器操作。"""
        with self._browser_lease.acquire(self.settings.profile):
            return method(self, *args, **kwargs)
    return wrapped
