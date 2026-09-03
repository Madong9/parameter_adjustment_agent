from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

from ..utils.paths import ensure_within


class UnitreeProjectAdapter:
    """为已检查的 unitree_rl_gym 入口构造受限命令。"""

    def __init__(self, training_root: Path, agent_root: Path, experiment_root: Path):
        """初始化 UnitreeProjectAdapter 实例及其运行依赖。"""
        self.training_root = training_root.resolve()
        self.agent_root = agent_root.resolve()
        self.experiment_root = experiment_root.resolve()

    def _relative_to_training(self, path: Path) -> str:
        """把路径转换为相对于训练项目的参数值。"""
        return os.path.relpath(str(path.resolve()), str(self.training_root))

    def training_command(self, robot: str, compiled_config: Path, iterations: int, seed: int,
                         experiment_id: str, resume_checkpoint: Optional[Path] = None,
                         num_envs: Optional[int] = None, headless: bool = True) -> List[str]:
        """构造安全的真实训练参数数组。"""
        config = ensure_within(compiled_config, self.experiment_root)
        script = self.agent_root / "rl_training_agent" / "training" / "real_train.py"
        command = [sys.executable, "-u", self._relative_to_training(script), "--task", robot,
                   "--config", self._relative_to_training(config), "--max-iterations", str(iterations),
                   "--seed", str(seed), "--run-name", experiment_id]
        if headless:
            command.append("--headless")
        if num_envs is not None:
            command.extend(["--num-envs", str(num_envs)])
        if resume_checkpoint is not None:
            checkpoint = ensure_within(resume_checkpoint, self.experiment_root)
            command.extend(["--resume-checkpoint", self._relative_to_training(checkpoint)])
        return command

    def evaluation_command(self, robot: str, checkpoint: Path, seed: int) -> List[str]:
        """构造项目原生评估入口的参数数组。"""
        checkpoint = ensure_within(checkpoint, self.experiment_root)
        return [sys.executable, "-u", "legged_gym/scripts/play.py", "--task", robot, "--seed", str(seed),
                "--load_run", self._relative_to_training(checkpoint.parent),
                "--checkpoint", checkpoint.stem.replace("model_", "")]

    def rollout_command(self, robot: str, config: Path, checkpoint: Path, output_dir: Path,
                        seed: int, fps: int, seconds: float = 10.0) -> List[str]:
        """构造同步录制真实仿真 rollout 的参数数组。"""
        config = ensure_within(config, self.experiment_root)
        checkpoint = ensure_within(checkpoint, self.experiment_root)
        output_dir = ensure_within(output_dir, self.experiment_root)
        script = self.agent_root / "rl_training_agent" / "training" / "real_rollout.py"
        return [sys.executable, "-u", self._relative_to_training(script), "--task", robot,
                "--config", self._relative_to_training(config),
                "--checkpoint", self._relative_to_training(checkpoint),
                "--output", self._relative_to_training(output_dir), "--seed", str(seed),
                "--fps", str(fps), "--seconds", str(seconds)]

    def play_command(self, robot: str, config: Path, checkpoint: Path, seed: int = 1,
                     num_envs: int = 1) -> List[str]:
        """构造直接加载 checkpoint 的实时 Viewer 播放命令。"""
        config = ensure_within(config, self.experiment_root)
        checkpoint = ensure_within(checkpoint, self.experiment_root)
        script = self.agent_root / "rl_training_agent" / "training" / "real_play.py"
        return [sys.executable, "-u", self._relative_to_training(script), "--task", robot,
                "--config", self._relative_to_training(config),
                "--checkpoint", self._relative_to_training(checkpoint),
                "--seed", str(seed), "--num-envs", str(num_envs)]
