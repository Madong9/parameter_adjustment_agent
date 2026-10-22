"""把结构化任务和检索上下文编译为版本固定的奖励设计提示词。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from ..schemas.agent_workflow import TaskIntentSpec, TaskRewardBundle
from ..utils.io import sha256_text
from .reward_prompt_policy import REWARD_PROMPT_VERSION, render_reward_policy
from .schema_contract import render_schema_contract


class RewardPromptCompiler:
    """使用仓库内固定模板生成可哈希、可审计的模型请求。"""

    VERSION = REWARD_PROMPT_VERSION

    def __init__(self, template_path: Path):
        """保存提示词模板路径。"""
        self.template_path = template_path

    def compile(self, intent: TaskIntentSpec, context: Dict[str, Any],
                candidate_count: int, task_id: str | None = None) -> Dict[str, str]:
        """编译完整提示词并返回版本与内容哈希。"""
        template = self.template_path.read_text(encoding="utf-8")
        current_task_id = task_id if task_id is not None else "task-" + hashlib.sha1(
            (intent.robot + intent.original_instruction).encode("utf-8")).hexdigest()[:10]
        if not current_task_id.strip():
            raise ValueError("current task_id must not be empty")
        identity = {
            "task_id": current_task_id,
            "robot": intent.robot,
            "original_instruction": intent.original_instruction,
        }
        schema = TaskRewardBundle.schema()
        required_fields = {"TaskRewardBundle": schema.get("required", [])}
        for name, definition in schema.get("definitions", {}).items():
            if definition.get("required"):
                required_fields[name] = definition["required"]
        prompt = "\n".join([
            template,
            render_reward_policy(),
            "PROMPT_VERSION: " + self.VERSION,
            "CANDIDATE_COUNT: %d" % candidate_count,
            "CURRENT_TASK_IDENTITY: " + json.dumps(
                identity, ensure_ascii=False, separators=(",", ":")),
            "CURRENT_TASK_IDENTITY 是当前请求的权威身份：task_spec 的 task_id、robot、"
            "original_instruction 必须逐字一致，所有 reward_plans[*].task_id 也必须一致。"
            "CONTEXT_SNAPSHOT 中 RAG、长期记忆、历史实验的 task_id 和 robot 仅是历史证据，"
            "不得复用为当前身份，不得用历史 G1 等机器人替换当前机器人。",
            "SCHEMA_REQUIRED_FIELDS: " + json.dumps(
                required_fields, ensure_ascii=False, separators=(",", ":")),
            "SCHEMA_REQUIRED_FIELDS 从当前 Schema 自动提取，所有列出的字段必须显式给出。"
            "尤其不得漏掉 task_spec.phases、required_observations、required_sensors、"
            "safety_constraints、visual_evaluation_requirements，"
            "每个奖励项必须包含说明其作用的 purpose。"
            "reward_plans[*].version 必须是正整数（例如 1、2、3），不能用 candidate-01 等标签。"
            "parent_version 为整数或 null；design_rationale 必须是字符串数组，例如 [\"设计说明\"]，不能是一段字符串。"
            "若提供 curriculum，每个课程项必须包含 name、start_iteration、end_iteration；"
            "边界是非负训练迭代整数且起点不晚于终点，不是动作阶段名。",
            "SAFETY_CONSTRAINT_FORMAT_EXAMPLE: " + json.dumps(
                {"name": "forbidden_collisions", "operator": "<=", "value": 0,
                 "unit": "count", "aggregation": "max", "required": True},
                ensure_ascii=False, separators=(",", ":")),
            "上面的安全约束仅示范 MetricThreshold 对象格式，适用于已明确禁止接触的合同；"
            "safety_constraints 必须是对象列表，不能是字符串或只写描述。"
            "name 必须为已有指标，operator 使用 >、>=、<、<=、==，value 必须为数值，"
            "unit 必须匹配实际指标。不要为了补齐结构补造阈值或直接复制示例为当前合同；"
            "继承已有任务的硬约束和阈值，初次设计的假设必须有当前意图或环境证据支持并明确标注。"
            "缺证据时如实报告能力缺口，不得清空安全约束来掩盖问题。",
            "TASK_INTENT_SPEC: " + json.dumps(
                intent.dict(), ensure_ascii=False, separators=(",", ":")),
            "CONTEXT_SNAPSHOT: " + json.dumps(
                context, ensure_ascii=False, separators=(",", ":"), default=str),
            "OUTPUT_JSON_SCHEMA: " + json.dumps(schema, ensure_ascii=False),
            render_schema_contract(TaskRewardBundle),
        ])
        return {"version": self.VERSION, "sha256": sha256_text(prompt), "prompt": prompt}
