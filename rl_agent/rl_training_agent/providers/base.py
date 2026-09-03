from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Protocol, Type, TypeVar

from pydantic import BaseModel

from ..schemas.decisions import TrainingDiagnosis
from ..schemas.experiments import ConversationHandle, ProviderHealth
from ..schemas.rewards import RewardPlan
from ..schemas.task import TaskSpec
from ..schemas.visual import VisualBehaviorReport

ModelT = TypeVar("ModelT", bound=BaseModel)


class LLMReasoningProvider(Protocol):
    def doctor(self) -> ProviderHealth:
        """检查运行环境、外部依赖和服务健康状态。"""
        ...
    def open_or_bind(self) -> None:
        """打开或绑定配置指定的 ChatGPT 浏览器会话。"""
        ...
    def new_conversation(self, title_hint: str) -> ConversationHandle:
        """创建新会话并返回可持久化的会话句柄。"""
        ...
    def send_text(self, prompt: str, conversation: ConversationHandle) -> str:
        """向指定网页会话发送文本并返回最新助手回复。"""
        ...
    def send_with_files(self, prompt: str, files: List[Path], conversation: ConversationHandle) -> str:
        """上传本地文件、发送提示词并返回最新助手回复。"""
        ...
    def parse_json_response(self, raw_response: str, schema: Type[ModelT]) -> ModelT:
        """提取并校验网页回复中的结构化 JSON。"""
        ...
    def design_task_and_rewards(self, instruction: str, robot: str, capabilities: Dict[str, Any]) -> Dict[str, Any]:
        """依据任务描述和环境能力生成任务规格与奖励候选。"""
        ...
    def design_visual_evaluation(self, task: TaskSpec) -> Dict[str, Any]:
        """为任务生成视觉评估输入与事件设计。"""
        ...
    def critique_visual_behavior(self, task: TaskSpec, files: List[Path]) -> VisualBehaviorReport:
        """基于视觉材料生成不受奖励数值锚定的行为评论。"""
        ...
    def diagnose_training(self, payload: Dict[str, Any]) -> TrainingDiagnosis:
        """融合视觉、物理和 PPO 证据生成训练诊断。"""
        ...
    def close(self) -> None:
        """释放 Provider 持有或绑定的浏览器资源。"""
        ...

