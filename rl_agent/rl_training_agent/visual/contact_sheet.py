from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .overlay_renderer import render_overlay


class ContactSheetBuilder:
    def __init__(self, tile_size: Tuple[int, int] = (320, 240), columns: int = 4, max_dimension: int = 4096):
        """初始化 ContactSheetBuilder 实例及其运行依赖。"""
        self.tile_size = tile_size
        self.columns = columns
        self.max_dimension = max_dimension

    def build(self, frames: Sequence[Tuple[int, np.ndarray]], output: Path,
              annotations: Optional[Dict[int, dict]] = None) -> Path:
        """构建并保存人类可读及机器可读的训练报告。"""
        if not frames:
            raise ValueError("contact sheet requires at least one decoded frame")
        columns = min(self.columns, len(frames))
        rows = int(math.ceil(len(frames) / columns))
        tile_width, tile_height = self.tile_size
        canvas = np.full((rows * tile_height, columns * tile_width, 3), 245, dtype=np.uint8)
        for position, (frame_index, frame) in enumerate(frames):
            tile = cv2.resize(frame, self.tile_size, interpolation=cv2.INTER_AREA)
            if annotations is not None:
                values = dict(annotations.get(frame_index, {}))
                values.setdefault("frame", frame_index)
                tile = render_overlay(tile, values)
            row, column = divmod(position, columns)
            canvas[row * tile_height:(row + 1) * tile_height,
                   column * tile_width:(column + 1) * tile_width] = tile
        scale = min(1.0, self.max_dimension / max(canvas.shape[:2]))
        if scale < 1.0:
            canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), canvas):
            raise IOError("failed to write contact sheet: %s" % output.name)
        return output

    def build_multi_camera(self, camera_frames: Dict[str, Sequence[Tuple[int, np.ndarray]]], output: Path) -> Path:
        """合并多相机帧并生成接触图。"""
        merged: List[Tuple[int, np.ndarray]] = []
        for camera, frames in camera_frames.items():
            for index, frame in frames:
                labeled = frame.copy()
                cv2.putText(labeled, camera, (10, labeled.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2, cv2.LINE_AA)
                merged.append((index, labeled))
        return self.build(merged, output)

