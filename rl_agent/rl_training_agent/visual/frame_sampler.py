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
        """连续索引顺序解码，仅在跳帧时定位，并始终释放视频句柄。"""
        capture = cv2.VideoCapture(str(path))
        result: List[Tuple[int, np.ndarray]] = []
        try:
            if not capture.isOpened():
                raise ValueError("bad video: %s" % path.name)
            next_frame = 0
            for index in sorted(set(index for index in indices if index >= 0)):
                if index != next_frame and not capture.set(cv2.CAP_PROP_POS_FRAMES, int(index)):
                    next_frame = None
                    continue
                ok, frame = capture.read()
                next_frame = index + 1 if ok else None
                if ok and frame is not None and frame.size:
                    result.append((index, frame))
        finally:
            capture.release()
        return result
