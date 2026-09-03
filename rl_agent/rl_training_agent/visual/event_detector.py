from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


class EventDetector:
    """以轨迹为准的通用、跳跃和步态事件检测器。"""

    def _event(self, frame: pd.Series, name: str) -> Dict[str, object]:
        """把轨迹行转换为标准事件记录。"""
        return {"name": name, "frame": int(frame["video_frame"]), "sim_time": float(frame["sim_time"])}

    def detect(self, data: pd.DataFrame) -> List[Dict[str, object]]:
        """从同步轨迹检测通用、跳跃和步态事件。"""
        if data.empty or "video_frame" not in data or "sim_time" not in data:
            return []
        events: List[Dict[str, object]] = [self._event(data.iloc[0], "stable_start")]
        for column, name in (("roll", "max_roll"), ("pitch", "max_pitch"),
                             ("angular_velocity", "max_angular_velocity"), ("action_rate", "max_action_rate"),
                             ("foot_slip", "max_foot_slip")):
            if column in data:
                events.append(self._event(data.loc[data[column].abs().idxmax()], name))
        if "base_speed" in data:
            moving = data.index[data["base_speed"] > 0.05]
            if len(moving):
                events.append(self._event(data.loc[moving[0]], "motion_start"))
        contact_cols = [column for column in ("contact_fl", "contact_fr", "contact_rl", "contact_rr")
                        if column in data]
        if contact_cols:
            contacts = data[contact_cols].astype(bool).sum(axis=1)
            all_air = contacts == 0
            transitions = all_air.astype(int).diff().fillna(0)
            takeoff = data.index[transitions == 1]
            landing = data.index[transitions == -1]
            if len(takeoff):
                events.append(self._event(data.loc[takeoff[0]], "all_feet_takeoff"))
                before = max(0, int(takeoff[0]) - 3)
                events.append(self._event(data.iloc[before], "crouch_start"))
                events.append(self._event(data.iloc[max(0, int(takeoff[0]) - 1)], "crouch_bottom"))
                events.append(self._event(data.loc[takeoff[0]], "first_foot_takeoff"))
            if len(landing):
                events.append(self._event(data.loc[landing[0]], "first_foot_landing"))
                events.append(self._event(data.loc[landing[0]], "all_feet_landing"))
                recovery_index = min(len(data) - 1, int(landing[0]) + 5)
                events.append(self._event(data.iloc[recovery_index], "recovery"))
            for column, foot in zip(contact_cols, ("left_front", "right_front", "left_hind", "right_hind")):
                changes = data[column].astype(int).diff().fillna(0)
                touchdowns = data.index[changes == 1]
                if len(touchdowns):
                    events.append(self._event(data.loc[touchdowns[0]], foot + "_touchdown"))
        if "base_vz" in data:
            events.append(self._event(data.loc[data["base_vz"].idxmax()], "peak_vertical_velocity"))
        if "base_z" in data:
            events.append(self._event(data.loc[data["base_z"].idxmax()], "jump_peak"))
        if "body_collision" in data and data["body_collision"].any():
            events.append(self._event(data.loc[data.index[data["body_collision"].astype(bool)][0]], "body_collision"))
        if "fall" in data and data["fall"].any():
            events.append(self._event(data.loc[data.index[data["fall"].astype(bool)][0]], "fall"))
        events.append(self._event(data.iloc[-1], "episode_end"))
        return sorted(events, key=lambda item: (item["frame"], item["name"]))
