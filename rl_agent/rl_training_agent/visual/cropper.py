from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def crop_frame(frame: np.ndarray, box: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    """按边界框裁剪帧并校验输入范围。"""
    if frame is None or frame.size == 0:
        raise ValueError("cannot crop an empty frame")
    if box is None:
        return frame.copy()
    x, y, width, height = box
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > frame.shape[1] or y + height > frame.shape[0]:
        raise ValueError("crop is outside frame bounds")
    return frame[y:y + height, x:x + width].copy()


def center_crop(frame: np.ndarray, aspect_ratio: float = 4.0 / 3.0) -> np.ndarray:
    """按目标宽高比进行居中裁剪。"""
    height, width = frame.shape[:2]
    current = width / height
    if current > aspect_ratio:
        new_width = int(height * aspect_ratio)
        return crop_frame(frame, ((width - new_width) // 2, 0, new_width, height))
    new_height = int(width / aspect_ratio)
    return crop_frame(frame, (0, (height - new_height) // 2, width, new_height))

