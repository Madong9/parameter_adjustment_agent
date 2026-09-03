from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pandas as pd

from ..utils.io import write_json
from .video_metadata import read_video_metadata


class DryRunRolloutRecorder:
    """在无需 Isaac Gym 或 GPU 的情况下生成同步确定性媒体。"""

    def record(self, output_dir: Path, fps: int = 30, seconds: float = 2.0, seed: int = 1,
               checkpoint: str = "mock_model_300.pt") -> Dict[str, Path]:
        """生成无需 GPU 的同步模拟视频与轨迹。"""
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_count = int(fps * seconds)
        width, height = 640, 480
        rows = []
        videos: Dict[str, Path] = {}
        for camera in ("front", "side", "overview"):
            path = output_dir / (camera + ".mp4")
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError("OpenCV MP4 writer is unavailable")
            for frame_index in range(frame_count):
                t = frame_index / fps
                phase = min(1.0, t / seconds)
                jump = max(0.0, math.sin(math.pi * phase)) * 90
                frame = np.full((height, width, 3), (235, 240, 245), dtype=np.uint8)
                cv2.line(frame, (0, 380), (width, 380), (90, 90, 90), 3)
                center_x = 320 + (frame_index if camera == "side" else 0)
                center_y = int(330 - jump)
                cv2.rectangle(frame, (center_x - 60, center_y - 28), (center_x + 60, center_y + 28), (50, 100, 210), -1)
                for offset in (-45, -15, 15, 45):
                    cv2.line(frame, (center_x + offset, center_y + 25),
                             (center_x + offset, center_y + 60), (20, 20, 20), 6)
                cv2.putText(frame, "%s  t=%.2f" % (camera, t), (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (20, 20, 20), 2)
                writer.write(frame)
            writer.release()
            videos[camera] = path
        for frame_index in range(frame_count):
            t = frame_index / fps
            phase = min(1.0, t / seconds)
            airborne = 0.22 < phase < 0.78
            z = 0.42 + max(0.0, math.sin(math.pi * phase)) * 0.18
            rows.append({
                "sim_time": t, "video_frame": frame_index, "base_x": 0.002 * frame_index,
                "base_y": 0.0, "base_z": z, "quat_x": 0.0, "quat_y": 0.0, "quat_z": 0.0, "quat_w": 1.0,
                "roll": 0.02 * math.sin(t), "pitch": 0.03 * math.sin(2 * t), "yaw": 0.0,
                "base_vx": 0.06, "base_vy": 0.0, "base_vz": 0.18 * math.pi / seconds * math.cos(math.pi * phase),
                "angular_velocity": 0.03, "action_rate": 0.1, "joint_positions": [0.0] * 12,
                "joint_velocities": [0.0] * 12, "joint_torques": [0.1] * 12, "actions": [0.0] * 12,
                "feet_positions": [[0.0, 0.0, 0.0]] * 4, "feet_velocities": [[0.0, 0.0, 0.0]] * 4,
                "contact_fl": not airborne, "contact_fr": not airborne, "contact_rl": not airborne,
                "contact_rr": not airborne, "contact_forces": [0.0 if airborne else 20.0] * 4,
                "command": [0.0, 0.0, 0.0, 0.0], "termination_reason": "timeout" if frame_index == frame_count - 1 else "",
                "penalized_contact_force_max": 0.0, "termination_contact_force_max": 0.0,
                "nonfoot_contact_force_max": 0.0, "forbidden_contact_force_max": 0.0,
                "body_collision": False,
                "fall": False, "foot_slip": 0.0, "energy": 0.01,
                "base_speed": 0.06, "reward_total": 1.0,
            })
        trajectory = pd.DataFrame(rows)
        trajectory_path = output_dir / "trajectory.parquet"
        trajectory.to_parquet(trajectory_path, index=False)
        reward_path = output_dir / "rewards.parquet"
        trajectory[["sim_time", "video_frame", "reward_total"]].assign(
            raw_tracking=1.0, weighted_tracking=1.0).to_parquet(reward_path, index=False)
        metadata = read_video_metadata(videos["side"], camera="side", seed=seed, checkpoint=checkpoint)
        write_json(output_dir / "metadata.json", metadata)
        return {**videos, "trajectory": trajectory_path, "rewards": reward_path,
                "metadata": output_dir / "metadata.json"}
