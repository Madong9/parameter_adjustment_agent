"""实现可配置 Provider 注册表、能力声明和角色启动前校验。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Optional

from ..settings import Settings, load_bailian_settings, load_opencli_settings
from .bailian_glm import BailianGLMProvider
from .fallback import FallbackProvider
from .mock_provider import MockLLMReasoningProvider
from .opencli_chatgpt import OpenCLIChatGPTWebProvider
from .opencli_doubao import OpenCLIDoubaoWebProvider


@dataclass(frozen=True)
class ProviderDescriptor:
    """描述 Provider 工厂及其明确支持的推理能力。"""

    name: str
    capabilities: FrozenSet[str]
    factory: Callable[[Settings, Optional[Path]], Any]


class ProviderRegistry:
    """集中注册 Provider，并在产生外部请求前验证角色能力。"""

    ROLE_CAPABILITIES = {
        "task_planner": "task_understanding",
        "motion_prototype": "motion_prototype_generation",
        "reward_designer": "reward_design",
        "visual_critic": "visual_critique",
        "diagnosis": "training_diagnosis",
        "reward_experience": "reward_experience_summarization",
    }

    def __init__(self) -> None:
        """注册内置 ChatGPT、豆包主备组合、百炼和 Mock Provider。"""
        self._providers: Dict[str, ProviderDescriptor] = {}
        self.register(ProviderDescriptor(
            "opencli", frozenset({
                "task_understanding", "motion_prototype_generation", "reward_design",
                "visual_critique", "training_diagnosis", "reward_experience_summarization"}),
            lambda settings, path: OpenCLIChatGPTWebProvider(record_dir=path)))
        self.register(ProviderDescriptor(
            "doubao", frozenset({
                "task_understanding", "motion_prototype_generation", "reward_design",
                "visual_critique", "training_diagnosis", "reward_experience_summarization"}),
            lambda settings, path: OpenCLIDoubaoWebProvider(record_dir=path)))
        self.register(ProviderDescriptor(
            "opencli-doubao", frozenset({
                "task_understanding", "motion_prototype_generation", "reward_design",
                "visual_critique", "training_diagnosis", "reward_experience_summarization"}),
            self._create_opencli_doubao))
        self.register(ProviderDescriptor(
            "bailian", frozenset({"reward_design", "training_diagnosis", "reward_experience_summarization"}),
            lambda settings, path: BailianGLMProvider(load_bailian_settings(), record_dir=path)))
        self.register(ProviderDescriptor(
            "mock", frozenset({
                "task_understanding", "motion_prototype_generation", "reward_design",
                "visual_critique", "training_diagnosis", "reward_experience_summarization"}),
            lambda settings, path: MockLLMReasoningProvider(settings.num_reward_candidates)))

    @staticmethod
    def _create_opencli_doubao(settings: Settings, path: Optional[Path]) -> FallbackProvider:
        """创建强制聊天模式的 ChatGPT 主服务与独立豆包备用服务。"""
        opencli = load_opencli_settings()
        primary_path = path / "chatgpt" if path else None
        fallback_path = path / "doubao" if path else None
        return FallbackProvider(
            OpenCLIChatGPTWebProvider(opencli, record_dir=primary_path),
            OpenCLIDoubaoWebProvider(opencli, record_dir=fallback_path),
            record_dir=path,
        )

    def register(self, descriptor: ProviderDescriptor) -> None:
        """注册或显式替换同名 Provider 描述。"""
        self._providers[descriptor.name] = descriptor

    def descriptor(self, name: str) -> ProviderDescriptor:
        """读取 Provider 描述，不存在时给出可操作错误。"""
        if name not in self._providers:
            raise ValueError("未注册 Provider：%s；可用项：%s" % (
                name, ", ".join(sorted(self._providers))))
        return self._providers[name]

    def validate_role(self, role: str, provider_name: str) -> None:
        """验证指定 Provider 具备该 Agent 角色所需能力。"""
        if role not in self.ROLE_CAPABILITIES:
            raise ValueError("未知 Agent 角色：%s" % role)
        required = self.ROLE_CAPABILITIES[role]
        descriptor = self.descriptor(provider_name)
        if required not in descriptor.capabilities:
            raise ValueError("Provider %s 不支持角色 %s 所需能力 %s" % (
                provider_name, role, required))

    def create(self, name: str, settings: Settings, record_dir: Optional[Path]) -> Any:
        """使用注册工厂创建 Provider 实例。"""
        return self.descriptor(name).factory(settings, record_dir)

    def inventory(self) -> Dict[str, Any]:
        """返回上位机和诊断命令可展示的能力清单。"""
        return {name: sorted(item.capabilities) for name, item in sorted(self._providers.items())}
