from __future__ import annotations

from pathlib import Path
from typing import Dict, List


class TensorBoardReader:
    def read(self, log_dir: Path) -> Dict[str, List[dict]]:
        """读取并合并目录下的 TensorBoard 标量事件。"""
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        result: Dict[str, List[dict]] = {}
        files = sorted(log_dir.glob("**/events.out.tfevents.*"))
        for file in files:
            accumulator = EventAccumulator(str(file), size_guidance={"scalars": 0})
            accumulator.Reload()
            for tag in accumulator.Tags().get("scalars", []):
                result.setdefault(tag, []).extend(
                    {"step": item.step, "value": item.value, "wall_time": item.wall_time}
                    for item in accumulator.Scalars(tag))
        for values in result.values():
            values.sort(key=lambda item: (item["step"], item["wall_time"]))
        return result

    def latest(self, log_dir: Path) -> Dict[str, float]:
        """返回每个 TensorBoard 标量标签的最新值。"""
        return {tag: values[-1]["value"] for tag, values in self.read(log_dir).items() if values}

