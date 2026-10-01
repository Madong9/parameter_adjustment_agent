"""按角色把 OpenCLI/GPT 与百炼 GLM 组合成一个兼容 Provider。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from ..schemas.agent_workflow import TaskIntentSpec
from ..feasibility.motion_prototype.schema import MotionPrototype
from ..schemas.decisions import TrainingDiagnosis
from ..schemas.experiments import ConversationHandle, ProviderHealth
from ..schemas.task import TaskSpec
from ..schemas.visual import VisualBehaviorReport
from ..settings import Settings
from .registry import ProviderRegistry


class MultiAgentProvider:
    """为每个推理职责选择独立 Provider，同时兼容现有编排器接口。"""

    def __init__(self, settings: Settings, record_dir: Optional[Path] = None):
        """创建任务/视觉 GPT Agent 与奖励/诊断 GLM Agent。"""
        self.settings = settings
        self.registry = ProviderRegistry()
        role_names = {
            "task_planner": settings.task_planner_provider,
            "motion_prototype": settings.task_planner_provider,
            "reward_designer": settings.reward_designer_provider,
            "visual_critic": settings.visual_critic_provider,
            "diagnosis": settings.diagnosis_provider,
            "reward_experience": settings.diagnosis_provider,
        }
        for role, name in role_names.items():
            self.registry.validate_role(role, name)
        instances: Dict[str, Any] = {}
        for name in sorted(set(role_names.values())):
            path = record_dir / name if record_dir else None
            instances[name] = self.registry.create(name, settings, path)
        self.task_agent = instances[role_names["task_planner"]]
        self.motion_agent = instances[role_names["motion_prototype"]]
        self.reward_agent = instances[role_names["reward_designer"]]
        self.visual_agent = instances[role_names["visual_critic"]]
        self.diagnosis_agent = instances[role_names["diagnosis"]]
        self._instances = list(instances.values())

    def doctor(self) -> ProviderHealth:
        """合并 OpenCLI 和百炼配置健康状态。"""
        reports = []
        for provider in self._instances:
            reports.append(provider.doctor() if hasattr(provider, "doctor") else provider.health())
        available = all(bool(item.available if hasattr(item, "available") else item.get("available"))
                        for item in reports)
        opencli = next((item for item in reports if hasattr(item, "opencli_available")), None)
        details = []
        for item in reports:
            details.extend(list(item.details) if hasattr(item, "details") else item.get("details", []))
        return ProviderHealth(
            available=available,
            opencli_available=bool(opencli and opencli.opencli_available),
            extension_connected=bool(opencli and opencli.extension_connected),
            chatgpt_logged_in=bool(opencli and opencli.chatgpt_logged_in),
            image_upload_supported=bool(opencli and opencli.image_upload_supported),
            recoverable=all(bool(getattr(item, "recoverable", True)) for item in reports),
            details=details,
        )

    def understand_task(self, instruction: str, robot: str) -> TaskIntentSpec:
        """委托 GPT 任务理解 Agent 生成 TaskIntentSpec。"""
        return self.task_agent.understand_task(instruction, robot)

    def generate_motion_prototype(self, intent: TaskIntentSpec) -> MotionPrototype:
        """委托任务理解 Provider 描述语义动作阶段。"""
        return self.motion_agent.generate_motion_prototype(intent)

    def design_task_bundle(self, compiled_prompt: str) -> Dict[str, Any]:
        """委托百炼 GLM 奖励设计 Agent 生成候选。"""
        return self.reward_agent.design_task_bundle(compiled_prompt)

    def design_task_and_rewards(self, instruction: str, robot: str,
                                capabilities: Dict[str, Any]) -> Dict[str, Any]:
        """保留旧接口，并在未使用新编译器时交由 GPT 完成兼容设计。"""
        return self.task_agent.design_task_and_rewards(instruction, robot, capabilities)

    def diagnose_training(self, payload: Dict[str, Any]) -> TrainingDiagnosis:
        """委托百炼 GLM 诊断 Agent 生成闭环决策。"""
        return self.diagnosis_agent.diagnose_training(payload)

    def summarize_reward_experience(self, payload: Dict[str, Any]) -> Any:
        """由诊断 Provider 承担经验文字归纳，保留单一配置来源。"""
        return self.diagnosis_agent.summarize_reward_experience(payload)

    def design_visual_evaluation(self, task: TaskSpec) -> Dict[str, Any]:
        """委托 GPT 视觉 Agent 设计评估事件。"""
        return self.visual_agent.design_visual_evaluation(task)

    def critique_visual_behavior(self, task: TaskSpec, files: List[Path]) -> VisualBehaviorReport:
        """委托 GPT 多模态视觉 Agent 独立评价动作。"""
        return self.visual_agent.critique_visual_behavior(task, files)

    def open_or_bind(self) -> None:
        """打开或绑定 GPT 所用的 OpenCLI 浏览器会话。"""
        self.task_agent.open_or_bind()

    def new_conversation(self, title_hint: str) -> ConversationHandle:
        """通过 GPT Agent 创建兼容会话。"""
        return self.task_agent.new_conversation(title_hint)

    def send_text(self, prompt: str, conversation: ConversationHandle) -> str:
        """通过 GPT Agent 发送兼容文本请求。"""
        return self.task_agent.send_text(prompt, conversation)

    def send_with_files(self, prompt: str, files: List[Path],
                        conversation: ConversationHandle) -> str:
        """通过 GPT Agent 发送兼容多模态请求。"""
        return self.task_agent.send_with_files(prompt, files, conversation)

    def parse_json_response(self, raw_response: str, schema: Type[BaseModel]) -> BaseModel:
        """复用 GPT Agent 的严格 JSON 解析。"""
        return self.task_agent.parse_json_response(raw_response, schema)

    def close(self) -> None:
        """释放多 Agent 团队持有的全部外部资源。"""
        for provider in self._instances:
            provider.close()
