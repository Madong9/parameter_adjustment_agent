from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Optional

from ..environment.project_adapter import UnitreeProjectAdapter
from ..utils.io import write_json
from .checkpoint_manager import CheckpointManager
from .process_manager import ProcessManager, ProcessResult


class TrainingController:
    """业务级训练 API；特意不提供任意 shell 执行接口。"""

    def __init__(self, adapter: UnitreeProjectAdapter, process_manager: Optional[ProcessManager] = None,
                 timeout_seconds: int = 86400):
        """初始化 TrainingController 实例及其运行依赖。"""
        self.adapter = adapter
        self.process_manager = process_manager or ProcessManager()
        self.timeout_seconds = timeout_seconds
        self._status: Dict[str, str] = {}

    def create_experiment(self, experiment_dir: Path, manifest: dict) -> None:
        """创建候选实验目录及其标准子目录。"""
        experiment_dir.mkdir(parents=True, exist_ok=False)
        for name in ("metrics", "checkpoints", "rollouts", "prompts", "responses"):
            (experiment_dir / name).mkdir()
        write_json(experiment_dir / "manifest.json", manifest)
        self._status[manifest["experiment_id"]] = "created"

    def _run(self, experiment_id: str, robot: str, config: Path, iterations: int, seed: int,
             experiment_dir: Path, resume_checkpoint: Optional[Path] = None,
             num_envs: Optional[int] = None) -> ProcessResult:
        """执行受限子流程并返回结构化结果。"""
        command = self.adapter.training_command(robot, config, iterations, seed, experiment_id,
                                                resume_checkpoint, num_envs)
        self._status[experiment_id] = "running"
        result = self.process_manager.run(command, self.adapter.training_root, experiment_dir,
                                          self.timeout_seconds)
        self._status[experiment_id] = "completed" if result.exit_code == 0 else "failed"
        return result

    def run_smoke_training(self, experiment_id: str, robot: str, config: Path, iterations: int,
                           seed: int, experiment_dir: Path, num_envs: int = 64) -> ProcessResult:
        """启动短迭代候选冒烟训练。"""
        return self._run(experiment_id, robot, config, iterations, seed, experiment_dir, num_envs=num_envs)

    def continue_training(self, experiment_id: str, robot: str, config: Path, iterations: int,
                          seed: int, experiment_dir: Path, checkpoint: Path) -> ProcessResult:
        """从指定 checkpoint 继续训练。"""
        return self._run(experiment_id, robot, config, iterations, seed, experiment_dir, checkpoint)

    def run_full_training(self, experiment_id: str, robot: str, config: Path, iterations: int,
                          seed: int, experiment_dir: Path) -> ProcessResult:
        """从随机初始化启动完整训练。"""
        return self._run(experiment_id, robot, config, iterations, seed, experiment_dir)

    def stop_training(self) -> None:
        """停止当前训练进程组。"""
        self.process_manager.stop()

    def get_training_status(self, experiment_id: str) -> str:
        """返回指定实验的内存运行状态。"""
        return self._status.get(experiment_id, "unknown")

    def run_evaluation_rollouts(self, experiment_id: str, robot: str, config: Path, checkpoint: Path,
                                output_dir: Path, seed: int, fps: int, seconds: float = 10.0) -> ProcessResult:
        """运行确定性仿真并保存同步评估材料。"""
        latest = self.load_checkpoint(checkpoint)
        output_dir.mkdir(parents=True, exist_ok=True)
        command = self.adapter.rollout_command(robot, config, latest, output_dir, seed, fps, seconds)
        return self.process_manager.run(command, self.adapter.training_root, output_dir,
                                        min(self.timeout_seconds, 3600))

    def load_checkpoint(self, path: Path) -> Path:
        """校验并返回受支持的 checkpoint 文件。"""
        if not path.is_file() or not CheckpointManager.PATTERN.match(path.name):
            raise ValueError("checkpoint must be an existing model_<iteration>.pt file")
        return path
