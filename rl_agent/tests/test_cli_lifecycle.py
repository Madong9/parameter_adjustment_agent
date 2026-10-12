"""验证 CLI 终止信号与真实子进程清理入口，不发送系统信号或启动训练。"""
import signal

import pytest

from rl_training_agent.cli import _managed_training_signals
from rl_training_agent.training.process_manager import ProcessManager


def test_training_signal_cleanup_and_handler_restore(monkeypatch):
    """训练信号先回收子组，再退出；上下文退出恢复调用者信号处理器。"""
    calls = []
    previous = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(ProcessManager, "stop_all_active", lambda grace_seconds: calls.append(grace_seconds))
    with pytest.raises(SystemExit) as caught:
        with _managed_training_signals("train"):
            handler = signal.getsignal(signal.SIGTERM)
            assert callable(handler)
            handler(signal.SIGTERM, None)
    assert caught.value.code == 128 + signal.SIGTERM
    assert calls and all(value == 5.0 for value in calls)
    assert signal.getsignal(signal.SIGTERM) == previous


def test_regular_worker_exit_also_cleans_children(monkeypatch):
    """正常完成恢复命令也清理登记资源，而不是只处理停止按钮。"""
    calls = []
    monkeypatch.setattr(ProcessManager, "stop_all_active", lambda grace_seconds: calls.append(grace_seconds))
    with _managed_training_signals("resume"):
        pass
    assert calls == [5.0]


def test_ui_does_not_install_worker_signal_handler(monkeypatch):
    """桌面服务由自身 shutdown 管理，不能套用训练 CLI 的退出逻辑。"""
    calls = []
    previous = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(ProcessManager, "stop_all_active", lambda grace_seconds: calls.append(grace_seconds))
    with _managed_training_signals("desktop"):
        assert signal.getsignal(signal.SIGTERM) == previous
    assert not calls
