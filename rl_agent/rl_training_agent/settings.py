from __future__ import annotations

import os
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field, validator

from .utils.paths import AGENT_ROOT, resolve_relative


class Settings(BaseModel):
    training_project: str = "../unitree_rl_gym"
    experiment_root: str = "experiments"
    artifact_root: str = "artifacts"
    default_robot: str = "go2"
    provider: str = "opencli"
    num_reward_candidates: int = 3
    smoke_iterations: int = 50
    screening_iterations: int = 300
    trend_iterations: int = 500
    mid_iterations: int = 1000
    full_iterations: int = 3000
    max_total_iterations: int = 12000
    max_reward_revisions: int = 3
    evaluation_seeds: List[int] = Field(default_factory=lambda: [1, 2, 3])
    rollouts_per_seed: int = 20
    video_fps: int = 30
    training_timeout_seconds: int = 86400
    log_stale_seconds: int = 600
    max_abs_reward_weight: float = 100.0
    allowed_robots: List[str] = Field(default_factory=lambda: ["go2", "h1", "h1_2", "g1"])

    @validator(
        "num_reward_candidates", "smoke_iterations", "screening_iterations", "trend_iterations",
        "mid_iterations", "full_iterations", "max_total_iterations", "rollouts_per_seed",
        "video_fps", "training_timeout_seconds", "log_stale_seconds",
    )
    def positive(cls, value: int) -> int:
        """校验数值必须为正数。"""
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @validator("max_reward_revisions")
    def nonnegative_revisions(cls, value: int) -> int:
        """允许关闭自动修订，但不允许负数修订预算。"""
        if value < 0:
            raise ValueError("max_reward_revisions must be nonnegative")
        return value

    @validator("evaluation_seeds")
    def usable_evaluation_seeds(cls, values: List[int]) -> List[int]:
        """校验评估随机种子非空且不重复。"""
        if not values:
            raise ValueError("evaluation_seeds must not be empty")
        if len(values) != len(set(values)):
            raise ValueError("evaluation_seeds must be unique")
        return values

    @property
    def agent_root(self) -> Path:
        """返回 Agent 目录的运行时绝对路径。"""
        return AGENT_ROOT

    @property
    def training_root(self) -> Path:
        """解析并返回训练项目目录。"""
        return resolve_relative(self.training_project)

    @property
    def experiments_path(self) -> Path:
        """解析并返回实验存储目录。"""
        return resolve_relative(self.experiment_root)

    @property
    def artifacts_path(self) -> Path:
        """解析并返回公共产物目录。"""
        return resolve_relative(self.artifact_root)


class OpenCLISettings(BaseModel):
    session: str = "rl-training-agent"
    profile: str = ""
    bind_existing_tab: bool = True
    chatgpt_url: str = "https://chatgpt.com/"
    connect_timeout: int = 10
    command_timeout: int = 30
    submit_timeout: int = 20
    response_timeout: int = 300
    prompt_attachment_threshold: int = 4000
    max_retries: int = 2
    owned_session: bool = False


def _env_value(name: str, current: object) -> object:
    """按当前字段类型解析环境变量覆盖值。"""
    raw = os.getenv(name)
    if raw is None:
        return current
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, list):
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    return raw


def load_settings(path: Path = AGENT_ROOT / "config" / "agent.yaml") -> Settings:
    """加载 Agent YAML 配置并应用环境变量覆盖。"""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    mapping = {
        "EXPERIMENT_ROOT": "experiment_root", "MAX_REWARD_REVISIONS": "max_reward_revisions",
        "MAX_TOTAL_ITERATIONS": "max_total_iterations", "NUM_REWARD_CANDIDATES": "num_reward_candidates",
        "SMOKE_ITERATIONS": "smoke_iterations", "SCREENING_ITERATIONS": "screening_iterations",
        "FULL_ITERATIONS": "full_iterations", "EVALUATION_SEEDS": "evaluation_seeds",
        "ROLLOUTS_PER_SEED": "rollouts_per_seed", "VIDEO_FPS": "video_fps",
    }
    for env_name, key in mapping.items():
        data[key] = _env_value(env_name, data.get(key, Settings.__fields__[key].default))
    return Settings(**data)


def load_opencli_settings(path: Path = AGENT_ROOT / "config" / "opencli.yaml") -> OpenCLISettings:
    """加载 OpenCLI YAML 配置并应用环境变量覆盖。"""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    mapping = {
        "OPENCLI_PROFILE": "profile", "OPENCLI_SESSION": "session",
        "OPENCLI_BIND_EXISTING_TAB": "bind_existing_tab", "CHATGPT_URL": "chatgpt_url",
        "OPENCLI_CONNECT_TIMEOUT": "connect_timeout", "OPENCLI_COMMAND_TIMEOUT": "command_timeout",
        "OPENCLI_SUBMIT_TIMEOUT": "submit_timeout",
        "OPENCLI_RESPONSE_TIMEOUT": "response_timeout", "OPENCLI_MAX_RETRIES": "max_retries",
        "OPENCLI_PROMPT_ATTACHMENT_THRESHOLD": "prompt_attachment_threshold",
    }
    for env_name, key in mapping.items():
        data[key] = _env_value(env_name, data.get(key, OpenCLISettings.__fields__[key].default))
    return OpenCLISettings(**data)
