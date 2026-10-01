"""导出结构化事件记录能力。"""

from .events import JsonlEventRecorder
from .gpu import sample_gpu

__all__ = ["JsonlEventRecorder", "sample_gpu"]
