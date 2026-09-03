"""加载 Agent checkpoint 并在 Isaac Gym Viewer 中实时播放策略。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rl_training_agent.training.real_train import (
    _ensure_environment_tools_on_path,
    _install_numpy_compatibility_aliases,
)
from rl_training_agent.training.config_runtime import (
    install_runtime_terminations,
    prepare_evaluation_env_config,
)


def parse_args() -> argparse.Namespace:
    """解析实时策略播放所需的受限参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["go2", "h1", "h1_2", "g1"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    return parser.parse_args()


def _gym_args(task: str, seed: int, num_envs: int):
    """构造启用图形 Viewer 的 Isaac Gym 参数。"""
    from legged_gym.utils.helpers import get_args
    original = sys.argv
    sys.argv = [original[0], "--task", task, "--seed", str(seed), "--num_envs", str(num_envs)]
    try:
        return get_args()
    finally:
        sys.argv = original


def main() -> int:
    """加载真实策略并持续执行，直到 Viewer 关闭或达到可选步数限制。"""
    args = parse_args()
    if args.num_envs <= 0 or args.num_envs > 16:
        raise ValueError("play 的并行环境数量必须位于 1 到 16 之间")
    if args.max_steps < 0:
        raise ValueError("max-steps 不能为负数")
    _install_numpy_compatibility_aliases()
    _ensure_environment_tools_on_path()

    import isaacgym  # noqa: F401
    import torch
    from legged_gym.envs import task_registry

    config = json.loads(Path(args.config).resolve().read_text(encoding="utf-8"))
    checkpoint = Path(args.checkpoint).resolve()
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = args.num_envs
    env_cfg.env.test = True
    env_cfg.terrain.num_rows = min(5, env_cfg.terrain.num_rows)
    env_cfg.terrain.num_cols = min(5, env_cfg.terrain.num_cols)
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    prepare_evaluation_env_config(env_cfg, config)

    gym_args = _gym_args(args.task, args.seed, args.num_envs)
    env, _ = task_registry.make_env(name=args.task, args=gym_args, env_cfg=env_cfg)
    install_runtime_terminations(env, config)
    train_cfg.seed = args.seed
    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(
        env=env, args=gym_args, train_cfg=train_cfg, log_root=None)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    observations = env.get_observations()
    step = 0
    print("[播放] 已加载策略：%s" % checkpoint.name, flush=True)
    print("[播放] 关闭 Viewer 窗口或按 Ctrl+C 退出。", flush=True)
    try:
        while args.max_steps == 0 or step < args.max_steps:
            with torch.inference_mode():
                actions = policy(observations.detach())
            observations, _, _, _, _ = env.step(actions.detach())
            step += 1
    except KeyboardInterrupt:
        print("\n[播放] 用户已停止策略播放。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
