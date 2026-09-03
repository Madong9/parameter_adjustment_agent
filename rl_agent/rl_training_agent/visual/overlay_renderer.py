from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np


def render_overlay(frame: np.ndarray, values: Dict[str, Any]) -> np.ndarray:
    """在帧上绘制同步事件和物理指标。"""
    result = frame.copy()
    compact_keys = ("frame_time", "event_phase", "command_target", "measured_velocity",
                    "pose", "feet_contacts", "foot_forces", "safety_flags")
    if any(key in values for key in compact_keys):
        lines = [str(values[key]) for key in compact_keys if key in values]
    else:
        lines = ["%s: %s" % (key, values.get(key, "")) for key in
                 ("frame", "sim_time", "phase", "event", "base_height", "roll_pitch",
                  "contact_count", "task_metric")]
    overlay = result.copy()
    line_height = max(15, min(21, result.shape[0] // max(10, len(lines) + 2)))
    cv2.rectangle(overlay, (5, 5), (result.shape[1] - 5, min(result.shape[0] - 5, line_height * len(lines) + 10)),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, result, 0.45, 0, result)
    for index, line in enumerate(lines):
        cv2.putText(result, line, (10, 18 + line_height * index), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return result
