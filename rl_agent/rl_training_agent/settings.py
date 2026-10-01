from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Literal, Union

import yaml
from pydantic import BaseModel, Field, validator

from .utils.paths import AGENT_ROOT, resolve_relative


def _load_local_env_file(path: Path = AGENT_ROOT / ".env") -> None:
    """读取被 Git 忽略的本地环境文件，且不覆盖已有进程环境变量。"""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not name or not name.replace("_", "").isalnum() or name[0].isdigit():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if value:
            os.environ.setdefault(name, value)


class Settings(BaseModel):
    training_project: str = "../unitree_rl_gym"
    experiment_root: str = "experiments"
    artifact_root: str = "artifacts"
    default_robot: str = "go2"
    provider: str = "multi-agent"
    task_planner_provider: str = "opencli"
    reward_designer_provider: str = "bailian"
    visual_critic_provider: str = "opencli"
    diagnosis_provider: str = "bailian"
    num_reward_candidates: int = 3
    smoke_iterations: int = 50
    screening_iterations: int = 300
    trend_iterations: int = 500
    mid_iterations: int = 1000
    full_iterations: int = 3000
    max_total_iterations: int = 12000
    max_reward_revisions: int = 3
    feasibility_admission_mode: Literal["strict", "budgeted_exploration"] = "budgeted_exploration"
    exploration_max_iterations: int = 3000
    exploration_max_revisions: int = 1
    evaluation_seeds: List[int] = Field(default_factory=lambda: [1, 2, 3])
    rollouts_per_seed: int = 20
    checkpoints_per_run: int = 1
    video_fps: int = 30
    counterfactual_enabled: bool = True
    counterfactual_zero_command_speed_max: float = 0.15
    training_timeout_seconds: int = 86400
    log_stale_seconds: int = 600
    max_abs_reward_weight: float = 100.0
    allowed_robots: List[str] = Field(default_factory=lambda: ["go2", "h1", "h1_2", "g1"])
    gpu_ids: List[int] = Field(default_factory=lambda: [0])
    rag_enabled: bool = True
    rag_index_path: str = "rag/index.json"
    rag_document_roots: List[str] = Field(default_factory=lambda: [
        "docs/agent/REWARD_DESIGN.md",
        "docs/agent/VISUAL_CRITIC.md",
        "docs/agent/SAFETY.md",
    ])
    rag_top_k: int = 6
    rag_chunk_chars: int = 1200
    rag_chunk_overlap: int = 120
    rag_max_context_chars: int = 6000
    rag_bm25_weight: float = 0.65
    rag_vector_weight: float = 0.35
    memory_enabled: bool = True
    memory_root: str = "memory"
    memory_top_k: int = 4
    memory_max_context_chars: int = 4000
    memory_require_multi_seed: bool = True
    memory_min_semantic_support: int = 2
    memory_max_records: int = 500
    memory_max_age_days: int = 365
    memory_min_confidence: float = 0.55

    @validator(
        "num_reward_candidates", "smoke_iterations", "screening_iterations", "trend_iterations",
        "mid_iterations", "full_iterations", "max_total_iterations", "rollouts_per_seed",
        "checkpoints_per_run",
        "video_fps", "training_timeout_seconds", "log_stale_seconds", "rag_top_k",
        "rag_chunk_chars", "rag_max_context_chars", "memory_top_k",
        "memory_max_context_chars", "memory_min_semantic_support",
        "memory_max_records", "memory_max_age_days",
        "exploration_max_iterations",
    )
    def positive(cls, value: int) -> int:
        """校验数值必须为正数。"""
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @validator("max_reward_revisions", "exploration_max_revisions")
    def nonnegative_revisions(cls, value: int) -> int:
        """允许关闭自动修订，但不允许负数修订预算。"""
        if value < 0:
            raise ValueError("max_reward_revisions must be nonnegative")
        return value

    @validator("evaluation_seeds", "gpu_ids")
    def unique_nonempty_integer_list(cls, values: List[int], field: Any) -> List[int]:
        """校验随机种子或 GPU 编号非空且不重复。"""
        if not values:
            raise ValueError("%s must not be empty" % field.name)
        if len(values) != len(set(values)):
            raise ValueError("%s must be unique" % field.name)
        return values

    @validator("rag_chunk_overlap")
    def nonnegative_rag_overlap(cls, value: int) -> int:
        """保证 RAG 文本窗口重叠不为负数。"""
        if value < 0:
            raise ValueError("rag_chunk_overlap must be nonnegative")
        return value

    @validator("memory_min_confidence")
    def memory_confidence_interval(cls, value: float) -> float:
        """保证长期记忆最低可信度位于零到一之间。"""
        if not 0.0 <= value <= 1.0:
            raise ValueError("memory_min_confidence must be within [0, 1]")
        return value

    @validator("rag_bm25_weight", "rag_vector_weight")
    def rag_weight_interval(cls, value: float) -> float:
        """保证混合检索权重位于零到一之间。"""
        if not 0.0 <= value <= 1.0:
            raise ValueError("RAG weights must be within [0, 1]")
        return value

    @validator("counterfactual_zero_command_speed_max")
    def positive_counterfactual_threshold(cls, value: float) -> float:
        """保证零命令漂移阈值为正数。"""
        if value <= 0:
            raise ValueError("counterfactual threshold must be positive")
        return value

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

    @property
    def rag_index_file(self) -> Path:
        """相对于公共产物目录解析 RAG 持久化索引文件。"""
        path = Path(self.rag_index_path).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (self.artifacts_path / path).resolve()

    @property
    def memory_path(self) -> Path:
        """相对于公共产物目录解析长期记忆目录。"""
        path = Path(self.memory_root).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (self.artifacts_path / path).resolve()

    def resolve_agent_path(self, path: Union[str, Path]) -> Path:
        """相对于 Agent 根目录解析 RAG 文档来源等可移植路径。"""
        return resolve_relative(path)


class OpenCLISettings(BaseModel):
    session: str = "rl-training-agent"
    profile: str = ""
    bind_existing_tab: bool = True
    chatgpt_url: str = "https://chatgpt.com/"
    connect_timeout: int = 20
    auto_launch_browser: bool = True
    bridge_browser_executable: str = ""
    command_timeout: int = 30
    submit_timeout: int = 45
    response_timeout: int = 300
    prompt_attachment_threshold: int = 4000
    visual_image_attachment_limit: int = 2
    max_retries: int = 2
    owned_session: bool = False
    force_chat_mode: bool = True
    # 旧版仅上传文件的豆包兼容开关；生产推理由 opencli-doubao 主备 Provider 负责。
    use_doubao_on_quota: bool = False
    doubao_url: str = "https://www.doubao.com/"  # 本机 OpenCLI 打开的豆包对话地址。
    doubao_upload_input_selector: str = "input[type=file]"
    doubao_result_selector: str = "input.share-link, a.share-link"  # 旧版上传结果定位器。
    doubao_max_wait: int = 60
    doubao_session: str = "rl-training-agent-doubao"

    @validator("visual_image_attachment_limit")
    def positive_visual_attachment_limit(cls, value: int) -> int:
        """保证视觉评价至少能够上传一张关键证据图。"""
        if value <= 0:
            raise ValueError("visual_image_attachment_limit must be positive")
        return value


class BailianSettings(BaseModel):
    """定义百炼 OpenAI 兼容接口的无密钥持久化配置。"""

    model: str = "glm-4.7"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env: str = "DASHSCOPE_API_KEY"
    timeout_seconds: int = 300
    max_retries: int = 2
    enable_thinking: bool = False
    response_format_json: bool = True

    @validator("timeout_seconds")
    def positive_timeout(cls, value: int) -> int:
        """保证百炼请求超时时间为正数。"""
        if value <= 0:
            raise ValueError("timeout_seconds must be positive")
        return value

    @validator("max_retries")
    def nonnegative_retries(cls, value: int) -> int:
        """保证百炼重试次数不为负数。"""
        if value < 0:
            raise ValueError("max_retries must be nonnegative")
        return value


def _env_value(name: str, current: object) -> object:
    """按当前字段类型解析环境变量覆盖值。"""
    raw = os.getenv(name)
    if raw is None:
        return current
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    return raw


def load_settings(path: Path = AGENT_ROOT / "config" / "agent.yaml") -> Settings:
    """加载 Agent YAML 配置并应用环境变量覆盖。"""
    _load_local_env_file()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    mapping = {
        "EXPERIMENT_ROOT": "experiment_root", "MAX_REWARD_REVISIONS": "max_reward_revisions",
        "MAX_TOTAL_ITERATIONS": "max_total_iterations", "NUM_REWARD_CANDIDATES": "num_reward_candidates",
        "SMOKE_ITERATIONS": "smoke_iterations", "SCREENING_ITERATIONS": "screening_iterations",
        "FULL_ITERATIONS": "full_iterations", "EVALUATION_SEEDS": "evaluation_seeds",
        "ROLLOUTS_PER_SEED": "rollouts_per_seed", "VIDEO_FPS": "video_fps",
        "CHECKPOINTS_PER_RUN": "checkpoints_per_run",
        "GPU_IDS": "gpu_ids",
        "RAG_TOP_K": "rag_top_k", "RAG_MAX_CONTEXT_CHARS": "rag_max_context_chars",
        "MEMORY_TOP_K": "memory_top_k", "MEMORY_MAX_CONTEXT_CHARS": "memory_max_context_chars",
        "MEMORY_REQUIRE_MULTI_SEED": "memory_require_multi_seed",
        "MEMORY_MIN_SEMANTIC_SUPPORT": "memory_min_semantic_support",
        "MEMORY_MAX_RECORDS": "memory_max_records",
        "MEMORY_MAX_AGE_DAYS": "memory_max_age_days",
        "MEMORY_MIN_CONFIDENCE": "memory_min_confidence",
    }
    for env_name, key in mapping.items():
        data[key] = _env_value(env_name, data.get(key, Settings.__fields__[key].default))
    return Settings(**data)


def load_opencli_settings(path: Path = AGENT_ROOT / "config" / "opencli.yaml") -> OpenCLISettings:
    """加载 OpenCLI YAML 配置并应用环境变量覆盖。"""
    _load_local_env_file()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    mapping = {
        "OPENCLI_PROFILE": "profile", "OPENCLI_SESSION": "session",
        "OPENCLI_BIND_EXISTING_TAB": "bind_existing_tab", "CHATGPT_URL": "chatgpt_url",
        "OPENCLI_CONNECT_TIMEOUT": "connect_timeout", "OPENCLI_COMMAND_TIMEOUT": "command_timeout",
        "OPENCLI_AUTO_LAUNCH_BROWSER": "auto_launch_browser",
        "OPENCLI_BRIDGE_BROWSER_EXECUTABLE": "bridge_browser_executable",
        "OPENCLI_SUBMIT_TIMEOUT": "submit_timeout",
        "OPENCLI_RESPONSE_TIMEOUT": "response_timeout", "OPENCLI_MAX_RETRIES": "max_retries",
        "OPENCLI_PROMPT_ATTACHMENT_THRESHOLD": "prompt_attachment_threshold",
        "OPENCLI_VISUAL_IMAGE_ATTACHMENT_LIMIT": "visual_image_attachment_limit",
        "OPENCLI_FORCE_CHAT_MODE": "force_chat_mode",
        "OPENCLI_USE_DOUBAO_ON_QUOTA": "use_doubao_on_quota",
        "OPENCLI_DOUBAO_URL": "doubao_url",
        "OPENCLI_DOUBAO_UPLOAD_INPUT_SELECTOR": "doubao_upload_input_selector",
        "OPENCLI_DOUBAO_RESULT_SELECTOR": "doubao_result_selector",
        "OPENCLI_DOUBAO_MAX_WAIT": "doubao_max_wait",
        "OPENCLI_DOUBAO_SESSION": "doubao_session",
    }
    for env_name, key in mapping.items():
        data[key] = _env_value(env_name, data.get(key, OpenCLISettings.__fields__[key].default))
    return OpenCLISettings(**data)


def load_bailian_settings(path: Path = AGENT_ROOT / "config" / "bailian.yaml") -> BailianSettings:
    """加载百炼配置，并允许通过环境变量覆盖模型和专属域名。"""
    _load_local_env_file()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    data = data or {}
    mapping = {
        "BAILIAN_MODEL": "model",
        "DASHSCOPE_BASE_URL": "base_url",
        "BAILIAN_TIMEOUT_SECONDS": "timeout_seconds",
        "BAILIAN_MAX_RETRIES": "max_retries",
    }
    for env_name, key in mapping.items():
        data[key] = _env_value(env_name, data.get(key, BailianSettings.__fields__[key].default))
    return BailianSettings(**data)
