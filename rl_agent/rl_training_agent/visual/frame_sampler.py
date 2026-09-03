from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


class FrameSampler:
    @staticmethod
    def uniform_indices(frame_count: int, count: int) -> List[int]:
        """生成覆盖完整视频的均匀帧索引。"""
        if frame_count <= 0 or count <= 0:
            return []
        return sorted(set(np.linspace(0, frame_count - 1, min(count, frame_count), dtype=int).tolist()))

    @staticmethod
    def event_indices(frame_count: int, event_frames: Iterable[int], radius: int = 2) -> List[int]:
        """生成事件附近的密集帧索引。"""
        values = set()
        for frame in event_frames:
            values.update(range(max(0, int(frame) - radius), min(frame_count, int(frame) + radius + 1)))
        return sorted(values)

    @staticmethod
    def decode(path: Path, indices: Sequence[int]) -> List[Tuple[int, np.ndarray]]:
        """按索引顺序解码视频帧并忽略坏帧。"""
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError("bad video: %s" % path.name)
        result: List[Tuple[int, np.ndarray]] = []
        for index in sorted(set(indices)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if ok and frame is not None and frame.size:
                result.append((index, frame))
        capture.release()
        return result

