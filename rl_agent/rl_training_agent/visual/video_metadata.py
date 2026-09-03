from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
from pydantic import BaseModel


class VideoMetadata(BaseModel):
    video_fps: float
    frame_count: int
    width: int
    height: int
    duration: float
    simulation_dt: float
    control_decimation: int
    video_start_sim_time: float
    camera: str
    seed: int
    checkpoint: str


def read_video_metadata(path: Path, simulation_dt: float = 0.005, control_decimation: int = 4,
                        camera: str = "side", seed: int = 0, checkpoint: str = "") -> VideoMetadata:
    """读取视频属性并构造同步元数据。"""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("video cannot be opened: %s" % path.name)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0:
        raise ValueError("video FPS must be greater than zero")
    return VideoMetadata(video_fps=fps, frame_count=frames, width=width, height=height,
                         duration=frames / fps, simulation_dt=simulation_dt,
                         control_decimation=control_decimation, video_start_sim_time=0.0,
                         camera=camera, seed=seed, checkpoint=checkpoint)


def time_to_frame(sim_time: float, metadata: VideoMetadata) -> int:
    """把仿真时间转换为受边界约束的视频帧号。"""
    frame = round((sim_time - metadata.video_start_sim_time) * metadata.video_fps)
    return max(0, min(metadata.frame_count - 1, int(frame)))

