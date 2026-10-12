"""生成提示词、Schema、校验和闭环策略的程序记忆快照。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from ..agents.prompt_compiler import RewardPromptCompiler
from ..schemas.agent_workflow import ProceduralMemoryRecord, TaskIntentSpec, TaskRewardBundle
from ..schemas.decisions import TrainingDiagnosis
from ..schemas.metrics import EvaluationResult
from ..utils.io import utc_now
from .reward_experience.prompts import REWARD_EXPERIENCE_PROMPT


class ProceduralMemoryAgent:
    """以确定性哈希记录当前 Agent 执行规则，而不把对话当作程序知识。"""

    VERSION = "memory-procedure-v2"
    WORKFLOW_STATES = [
        "RECEIVED", "ENVIRONMENT_INSPECTED", "TASK_UNDERSTANDING", "TASK_FEASIBILITY_CHECK",
        "MOTION_PROTOTYPE_GENERATING", "STATIC_MOTION_VALIDATION", "DYNAMIC_MOTION_VALIDATION",
        "RAG_RETRIEVING",
        "CONTEXT_BUILDING", "PROMPT_COMPILING", "REWARD_DESIGNING", "REWARD_REVIEWING",
        "TASK_DESIGNED", "REWARD_CANDIDATES_CREATED", "CONFIGS_COMPILED", "VALIDATED",
        "SMOKE_TRAINING", "CANDIDATE_SCREENING", "FULL_TRAINING", "ROLLOUT_COLLECTING",
        "VISUAL_EVALUATING", "NUMERIC_EVALUATING", "DIAGNOSING", "MEMORY_CURATING",
        "CONTINUE_TRAINING", "REVISE_REWARD", "REVISE_CURRICULUM", "ROLLBACK", "RESTART",
        "HUMAN_REVIEW", "COMPLETED", "FAILED",
    ]
    VALIDATION_RULES = [
        "奖励名称必须来自环境能力清单",
        "确定性指标、视觉验收和硬约束共同决定完成",
        "HUMAN_REVIEW、Provider 故障和格式错误不得晋升",
        "长期情景经验必须包含多个不同随机种子和不可变证据",
    ]
    DIAGNOSIS_POLICY = {
        "complete_requires_all_gates": True,
        "model_cannot_override_deterministic_failure": True,
        "budget_exhaustion": "HUMAN_REVIEW",
    }
    POLICY_FILES = (
        "agents/reward_reviewer.py", "rewards/validator.py", "rewards/compiler.py",
        "evaluation/deterministic.py", "evaluation/completion.py", "evaluation/fusion.py",
        "orchestration/state_machine.py", "memory/curator.py", "memory/consolidator.py",
        "memory/reward_experience/validator.py", "memory/reward_experience/eligibility.py",
    )

    def __init__(self, prompt_root: Optional[Path] = None):
        """允许离线测试使用隔离模板，默认读取仓库实际提示词。"""
        self.package_root = Path(__file__).resolve().parents[1]
        self.prompt_root = prompt_root or self.package_root / "prompts"

    @staticmethod
    def _schema_hash(model: object) -> str:
        """计算 Pydantic Schema 的稳定 SHA-256 摘要。"""
        schema = getattr(model, "schema")()
        encoded = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def snapshot(self) -> ProceduralMemoryRecord:
        """生成当前提示词、结构协议、验证边界和诊断策略快照。"""
        hashes = {
            model.__name__: self._schema_hash(model)
            for model in (TaskIntentSpec, TaskRewardBundle, TrainingDiagnosis, EvaluationResult)
        }
        prompt_hashes = {
            path.relative_to(self.prompt_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(self.prompt_root.rglob("*.md")) if path.is_file()
        }
        prompt_hashes["reward_experience_inline"] = hashlib.sha256(
            REWARD_EXPERIENCE_PROMPT.encode("utf-8")).hexdigest()
        policy_hashes = {
            name: hashlib.sha256((self.package_root / name).read_bytes()).hexdigest()
            for name in self.POLICY_FILES if (self.package_root / name).is_file()
        }
        payload = {
            "version": self.VERSION,
            "prompt_versions": {"reward_design": RewardPromptCompiler.VERSION},
            "prompt_hashes": prompt_hashes,
            "policy_hashes": policy_hashes,
            "schema_hashes": hashes,
            "validation_rules": list(self.VALIDATION_RULES),
            "workflow_states": list(self.WORKFLOW_STATES),
            "diagnosis_policy": dict(self.DIAGNOSIS_POLICY),
        }
        digest = hashlib.sha256(json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()[:24]
        now = utc_now()
        return ProceduralMemoryRecord(
            procedure_id="procedure-" + digest,
            **payload,
            created_at=now,
            updated_at=now,
        )
