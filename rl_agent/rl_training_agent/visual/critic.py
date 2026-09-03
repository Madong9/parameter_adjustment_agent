from __future__ import annotations

from pathlib import Path
from typing import List

from ..providers.base import LLMReasoningProvider
from ..schemas.task import TaskSpec
from ..schemas.visual import VisualBehaviorReport


class VisualCritic:
    def __init__(self, provider: LLMReasoningProvider):
        """初始化 VisualCritic 实例及其运行依赖。"""
        self.provider = provider

    def evaluate(self, task: TaskSpec, clean_and_annotated_files: List[Path]) -> VisualBehaviorReport:
        """执行视觉与确定性证据的联合评估。"""
        return self.provider.critique_visual_behavior(task, clean_and_annotated_files)
