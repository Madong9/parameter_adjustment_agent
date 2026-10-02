"""把机体速度转换到只随航向旋转的地面水平坐标系，供验收和视觉证据共用。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def heading_forward_velocity(trajectory: pd.DataFrame) -> np.ndarray:
    """还原水平前向速度；缺少姿态或完整速度时不假定机器人水平。"""
    required = {"base_vx", "base_vy", "base_vz", "roll", "pitch"}
    if not required.issubset(trajectory.columns):
        return np.full(len(trajectory), np.nan)
    roll = trajectory["roll"].to_numpy(dtype=float)
    pitch = trajectory["pitch"].to_numpy(dtype=float)
    return (np.cos(pitch) * trajectory["base_vx"].to_numpy(dtype=float) +
            np.sin(pitch) * (np.sin(roll) * trajectory["base_vy"].to_numpy(dtype=float) +
                             np.cos(roll) * trajectory["base_vz"].to_numpy(dtype=float)))
