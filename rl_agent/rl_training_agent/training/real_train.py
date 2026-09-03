"""在内存中应用 Agent 奖励配置的受限真实训练包装器。"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

from rl_training_agent.training.config_runtime import (
    apply_runtime_stage,
    curriculum_segments,
    install_runtime_terminations,
    prepare_training_env_config,
)


def _install_numpy_compatibility_aliases() -> None:
    """为旧版 Isaac Gym 恢复被新版 NumPy 删除的标量类型别名。"""
    aliases = {"float": float, "int": int, "bool": bool}
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def _ensure_environment_tools_on_path() -> None:
    """确保当前 Python 环境中的 Ninja 等工具可被 PyTorch 扩展构建器发现。"""
    environment_bin = str(Path(sys.executable).resolve().parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if environment_bin not in path_entries:
        os.environ["PATH"] = environment_bin + os.pathsep + os.environ.get("PATH", "")


def parse_args() -> argparse.Namespace:
    """解析并返回受限命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["go2", "h1", "h1_2", "g1"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-iterations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--resume-checkpoint")
    return parser.parse_args()


def main() -> int:
    """解析命令行参数并执行对应的 Agent 工作流。"""
    args = parse_args()
    _install_numpy_compatibility_aliases()
    _ensure_environment_tools_on_path()
    # 此进程必须先导入 Isaac Gym，再间接导入 torch。
    import isaacgym  # noqa: F401
    from legged_gym.envs import task_registry
    from legged_gym.utils.helpers import get_args

    config_path = Path(args.config).resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    prepare_training_env_config(env_cfg, data)
    train_cfg.seed = args.seed
    train_cfg.runner.max_iterations = args.max_iterations
    train_cfg.runner.run_name = args.run_name
    # 显式 checkpoint 会在下方加载，因此禁用项目内置的“最新运行”隐式查找。
    train_cfg.runner.resume = False

    # 复用真实项目的 Isaac Gym 参数解析器，但不转发包装器专用参数。
    original = sys.argv
    forwarded = [original[0], "--task", args.task, "--seed", str(args.seed),
                 "--max_iterations", str(args.max_iterations), "--run_name", args.run_name]
    if args.headless:
        forwarded.append("--headless")
    if args.num_envs is not None:
        forwarded.extend(["--num_envs", str(args.num_envs)])
    sys.argv = forwarded
    try:
        gym_args = get_args()
    finally:
        sys.argv = original
    env, _ = task_registry.make_env(name=args.task, args=gym_args, env_cfg=env_cfg)
    install_runtime_terminations(env, data)
    original_command_ranges = copy.deepcopy(env.command_ranges)
    log_root = os.path.dirname(config_path)
    runner, _ = task_registry.make_alg_runner(env=env, args=gym_args, train_cfg=train_cfg, log_root=log_root)
    if args.resume_checkpoint:
        runner.load(args.resume_checkpoint)
    segments = curriculum_segments(data, args.max_iterations)
    for index, (_, segment_iterations, stage_name) in enumerate(segments):
        apply_runtime_stage(env, data, stage_name, original_command_ranges)
        print("[课程] 阶段：%s；迭代：%d" % (stage_name or "all", segment_iterations), flush=True)
        runner.learn(num_learning_iterations=segment_iterations, init_at_random_ep_len=index == 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
