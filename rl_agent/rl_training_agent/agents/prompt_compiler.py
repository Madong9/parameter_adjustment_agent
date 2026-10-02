"""把结构化任务和检索上下文编译为版本固定的奖励设计提示词。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..schemas.agent_workflow import TaskIntentSpec, TaskRewardBundle
from ..utils.io import sha256_text


class RewardPromptCompiler:
    """使用仓库内固定模板生成可哈希、可审计的模型请求。"""

    VERSION = "reward-design-v3"

    def __init__(self, template_path: Path):
        """保存提示词模板路径。"""
        self.template_path = template_path

    def compile(self, intent: TaskIntentSpec, context: Dict[str, Any],
                candidate_count: int) -> Dict[str, str]:
        """编译完整提示词并返回版本与内容哈希。"""
        template = self.template_path.read_text(encoding="utf-8")
        prompt = "\n".join([
            template,
            "PROMPT_VERSION: " + self.VERSION,
            "CANDIDATE_COUNT: %d" % candidate_count,
            "TASK_INTENT_SPEC: " + json.dumps(
                intent.dict(), ensure_ascii=False, separators=(",", ":")),
            "CONTEXT_SNAPSHOT: " + json.dumps(
                context, ensure_ascii=False, separators=(",", ":"), default=str),
            "OUTPUT_JSON_SCHEMA: " + TaskRewardBundle.schema_json(),
        ])
        return {"version": self.VERSION, "sha256": sha256_text(prompt), "prompt": prompt}
