import json
import math
import subprocess
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_training_agent.environment.project_adapter import UnitreeProjectAdapter
from rl_training_agent.settings import load_settings
from rl_training_agent.training.checkpoint_manager import CheckpointManager
from rl_training_agent.training.early_stopping import EarlyStoppingMonitor
from rl_training_agent.training.process_manager import ProcessManager
from rl_training_agent.training.real_rollout import _tracked_camera_pose
from rl_training_agent.training.real_train import _ensure_environment_tools_on_path, _install_numpy_compatibility_aliases
from rl_training_agent.training.config_runtime import (
    curriculum_segments,
    install_runtime_terminations,
    prepare_training_env_config,
    reward_scales_for_stage,
)


def adapter(tmp_path):
    """构造测试所需的adapter数据。"""
    settings = load_settings()
    return UnitreeProjectAdapter(settings.training_root, settings.agent_root, tmp_path)


def test_command_builder_argument_array_and_injection(tmp_path):
    """验证“command builder argument array and injection”场景的预期行为。"""
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    command = adapter(tmp_path).training_command("go2", config, 50, 1, "run;touch PWNED")
    assert isinstance(command, list) and command[0] == sys.executable
    assert "run;touch PWNED" in command


def test_command_rejects_escaped_config(tmp_path):
    """验证“command rejects escaped config”场景的预期行为。"""
    outside = tmp_path.parent / "outside.yaml"
    outside.write_text("{}")
    with pytest.raises(ValueError):
        adapter(tmp_path).training_command("go2", outside, 1, 1, "run")


def test_play_command_loads_exact_checkpoint_without_headless(tmp_path):
    """验证实时播放命令直接加载受控 checkpoint 并启用图形窗口。"""
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model_3000.pt"
    config.write_text("{}")
    checkpoint.write_text("model")
    command = adapter(tmp_path).play_command("go2", config, checkpoint, seed=2, num_envs=1)
    assert command[0] == sys.executable
    assert "real_play.py" in " ".join(command)
    assert "--checkpoint" in command and "--headless" not in command


def test_tracked_camera_offset_rotates_with_robot_yaw():
    """验证机体前视相机在机器人转向后仍保持相对前方位置。"""
    position, target = _tracked_camera_pose([1.0, 2.0, 0.4], math.pi / 2, [1.4, 0.0, 0.38])
    assert position == pytest.approx((1.0, 3.4, 0.78))
    assert target == pytest.approx((1.0, 2.0, 0.4))


def test_process_failure_and_logs(tmp_path):
    """验证“process failure and logs”场景的预期行为。"""
    result = ProcessManager().run(["python", "-c", "import sys;print('x');sys.exit(3)"],
                                  tmp_path, tmp_path / "run", 10)
    assert result.exit_code == 3
    assert "x" in (tmp_path / "run" / "stdout.log").read_text()


def test_process_timeout(tmp_path):
    """验证“process timeout”场景的预期行为。"""
    result = ProcessManager().run(["python", "-c", "import time;time.sleep(2)"],
                                  tmp_path, tmp_path / "run", 1)
    assert result.timed_out


def test_process_reports_training_iteration_progress(tmp_path, capsys):
    """验证子训练进程的迭代信息会写入快照并转发给上位机日志。"""
    command = ["python", "-u", "-c", "print('Learning iteration 5/10')",
               "--run-name", "candidate-01-v01", "--max-iterations", "10"]
    result = ProcessManager().run(command, tmp_path, tmp_path / "run", 10)
    progress = json.loads((tmp_path / "run" / "training_progress.json").read_text())
    assert result.exit_code == 0
    assert progress["status"] == "completed" and progress["iteration"] == 10
    assert "迭代 5/10" in capsys.readouterr().out


def test_checkpoint_discovery_and_resume(tmp_path):
    """验证“checkpoint discovery and resume”场景的预期行为。"""
    for value in (50, 5, 100):
        (tmp_path / ("model_%d.pt" % value)).write_text("x")
    assert CheckpointManager.latest(tmp_path).name == "model_100.pt"


def test_early_stop_patterns_and_metrics(tmp_path):
    """验证“early stop patterns and metrics”场景的预期行为。"""
    log = tmp_path / "stdout.log"
    log.write_text("CUDA out of memory")
    assert EarlyStoppingMonitor().inspect_log(log) == "gpu_oom"
    assert EarlyStoppingMonitor().inspect_metrics({"kl": 0.3}) == "kl_limit"
    assert EarlyStoppingMonitor().inspect_metrics({"fall_rate": 1.0}) == "persistent_falls"


def test_stale_log_and_missing_checkpoint(tmp_path):
    """验证“stale log and missing checkpoint”场景的预期行为。"""
    log = tmp_path / "stdout.log"
    log.write_text("healthy")
    old = time.time() - 100
    os.utime(log, (old, old))
    assert EarlyStoppingMonitor(stale_seconds=1).inspect_log(log) == "log_stale"
    assert CheckpointManager.latest(tmp_path / "missing") is None


def test_real_training_restores_removed_numpy_aliases(monkeypatch):
    """验证真实训练入口会为旧版 Isaac Gym 恢复 NumPy 标量别名。"""
    import numpy as np
    monkeypatch.delitem(np.__dict__, "float", raising=False)
    _install_numpy_compatibility_aliases()
    assert np.float is float


def test_real_training_adds_current_environment_tools_to_path(monkeypatch):
    """验证真实训练入口会把当前 Python 环境工具目录加入 PATH。"""
    import sys
    monkeypatch.setenv("PATH", "/usr/bin")
    _ensure_environment_tools_on_path()
    assert os.environ["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).resolve().parent)


def runtime_config():
    """构造包含两阶段课程和姿态终止条件的编译配置。"""
    return {
        "rewards": {
            "scales": {"base_height": -1.0, "orientation": -1.0, "tracking_lin_vel": 1.5},
            "terms": [
                {"name": "base_height", "weight": -1.0, "active_phases": ["stand_learning"],
                 "parameters": {"base_height_target": 0.45}},
                {"name": "orientation", "weight": -1.0, "active_phases": ["all"], "parameters": {}},
                {"name": "tracking_lin_vel", "weight": 1.5,
                 "active_phases": ["walking_learning"], "parameters": {}},
            ],
        },
        "curriculum": [
            {"name": "stand", "start_iteration": 0, "end_iteration": 4000,
             "parameter_changes": {"command_scale": 0.1}},
            {"name": "walk", "start_iteration": 4001, "end_iteration": 12000,
             "parameter_changes": {"command_scale": 1.0}},
        ],
        "terminations": [
            {"name": "body_contact", "condition": "forbidden body contact > 1 N", "enabled": True},
            {"name": "fall_pitch", "condition": "abs(pitch)>1.0 rad", "enabled": True},
            {"name": "fall_roll", "condition": "abs(roll)>0.8 rad", "enabled": True},
        ],
    }


def test_runtime_config_applies_significant_training_fields():
    """验证奖励参数、身体终止链接和阶段奖励会真正应用。"""
    scales = SimpleNamespace(base_height=0.0, orientation=0.0, tracking_lin_vel=0.0)
    cfg = SimpleNamespace(
        rewards=SimpleNamespace(scales=scales, base_height_target=0.25),
        asset=SimpleNamespace(terminate_after_contacts_on=["base"], penalize_contacts_on=["thigh", "calf"]),
    )
    config = runtime_config()
    prepare_training_env_config(cfg, config)
    assert cfg.rewards.base_height_target == 0.45
    assert cfg.asset.terminate_after_contacts_on == ["base", "thigh", "calf"]
    assert curriculum_segments(config, 1500) == [(0, 500, "stand"), (500, 1000, "walk")]
    assert reward_scales_for_stage(config, "stand")["tracking_lin_vel"] == 0.0
    assert reward_scales_for_stage(config, "walk")["base_height"] == 0.0


def test_runtime_orientation_termination_is_enforced():
    """验证配置中的 roll 与 pitch 限制会追加到环境终止逻辑。"""
    import torch

    class FakeEnv:
        """提供终止包装器所需的最小环境接口。"""

        def __init__(self):
            """初始化两条并行环境状态。"""
            self.reset_buf = torch.tensor([False, False])
            self.rpy = torch.tensor([[0.0, 1.1, 0.0], [0.9, 0.0, 0.0]])

        def check_termination(self):
            """模拟项目原生接触终止检查。"""
            self.reset_buf[:] = False

    env = FakeEnv()
    install_runtime_terminations(env, runtime_config())
    env.check_termination()
    assert env.reset_buf.tolist() == [True, True]
