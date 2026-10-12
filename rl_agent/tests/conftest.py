"""测试默认不读取工作区私密 .env，只允许显式的临时配置夹具。"""
from pathlib import Path
import tempfile

import pytest

import rl_training_agent.settings as settings_module


@pytest.fixture(autouse=True)
def isolate_private_environment_files(monkeypatch):
    """阻止 load_settings 隐式读取真实密钥，保留临时 .env 解析器测试。"""
    original = settings_module._load_local_env_file
    temp_root = Path(tempfile.gettempdir()).resolve()

    def load_test_file(path=None):
        """仅向原加载器传递明确指定、位于系统临时根目录的文件。"""
        if path is None:
            return None
        candidate = Path(path).resolve()
        if temp_root not in candidate.parents:
            raise AssertionError("测试不能读取工作区私密环境文件")
        return original(candidate)

    monkeypatch.setattr(settings_module, "_load_local_env_file", load_test_file)
