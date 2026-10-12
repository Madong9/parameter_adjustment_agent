from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import pandas as pd

from ..schemas.task import TaskSpec
from ..utils.io import atomic_write_text, read_json, write_json
from .contact_sheet import ContactSheetBuilder
from .event_detector import EventDetector
from .evidence_builder import SynchronizedEvidenceBuilder
from .frame_sampler import FrameSampler


@dataclass
class VisualEvidenceArtifacts:
    """保存一次视觉评估所需的同步轨迹和全部附件。"""

    trajectory: pd.DataFrame
    visual_files: List[Path]
    clean_sheet: Path
    annotated_sheet: Path
    multiview_sheet: Path
    behavior_evidence: Path
    attachment_manifest: Path


class VisualEvaluationPipeline:
    """从现有三视角 rollout 构建可重复使用的增强视觉评估附件。"""

    @staticmethod
    def _media_paths(rollout_dir: Path) -> Dict[str, Path]:
        """返回 rollout 目录内约定的媒体和轨迹路径。"""
        return {
            "front": rollout_dir / "front.mp4",
            "side": rollout_dir / "side.mp4",
            "overview": rollout_dir / "overview.mp4",
            "trajectory": rollout_dir / "trajectory.parquet",
            "metadata": rollout_dir / "metadata.json",
        }

    def build(self, task: TaskSpec, rollout_dir: Path,
              media: Optional[Dict[str, Path]] = None) -> VisualEvidenceArtifacts:
        """生成完整轨迹证据、连续帧标注和同帧三视角接触图。"""
        paths = dict(media or self._media_paths(rollout_dir))
        required = [paths[name] for name in ("front", "side", "overview", "trajectory")]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("视觉评估缺少 rollout 文件：%s" % ", ".join(missing))
        trajectory = pd.read_parquet(paths["trajectory"])
        events = EventDetector().detect(trajectory)
        write_json(rollout_dir / "events.json", events)
        evidence_builder = SynchronizedEvidenceBuilder()
        uniform_indices = FrameSampler.uniform_indices(len(trajectory), 12)
        critical_indices = evidence_builder.critical_indices(len(trajectory), events, radius=3, maximum=48)
        multiview_indices = evidence_builder.multiview_indices(len(trajectory), events, maximum=12)
        event_frames = {}
        for event_name, output_name in (("all_feet_takeoff", "event_takeoff.png"),
                                        ("all_feet_landing", "event_landing.png")):
            matching = next((item for item in events if item["name"] == event_name), None)
            if matching is not None:
                event_frames[output_name] = matching["frame"]
        # 同一相机的附件共享解码结果，避免重复打开视频和解码重叠帧。
        requested = {
            "front": set(multiview_indices),
            "side": set(multiview_indices) | set(critical_indices) | set(event_frames.values()),
            "overview": set(multiview_indices) | set(uniform_indices),
        }
        decoded = {camera: dict(FrameSampler.decode(paths[camera], sorted(indices)))
                   for camera, indices in requested.items()}
        overview_frames = [(index, decoded["overview"][index]) for index in uniform_indices
                           if index in decoded["overview"]]
        dense_frames = [(index, decoded["side"][index]) for index in critical_indices
                        if index in decoded["side"]]
        annotation = evidence_builder.annotations(trajectory, events)
        builder = ContactSheetBuilder()
        clean = builder.build(overview_frames, rollout_dir / "contact_sheet_clean.png")
        annotated = ContactSheetBuilder(tile_size=(480, 360), columns=3).build(
            dense_frames, rollout_dir / "contact_sheet_annotated.png", annotation)
        camera_frames = {camera: [(index, frames[index]) for index in multiview_indices if index in frames]
                         for camera, frames in decoded.items()}
        multiview = builder.build_multi_camera(
            camera_frames, rollout_dir / "contact_sheet_multiview.png")
        metadata = read_json(paths["metadata"]) if paths.get("metadata") and paths["metadata"].is_file() else {}
        behavior_evidence = evidence_builder.build(
            task, trajectory, events, rollout_dir / "behavior_evidence.json", metadata)
        event_images: List[Path] = []
        for output_name, frame_index in event_frames.items():
            if frame_index in decoded["side"]:
                output = rollout_dir / output_name
                if not cv2.imwrite(str(output), decoded["side"][frame_index]):
                    raise IOError("无法写入事件帧：%s" % output.name)
                event_images.append(output)
        atomic_write_text(rollout_dir / "visual_prompt.txt", task.original_instruction + "\n")
        attachment_manifest = rollout_dir / "visual_attachment_manifest.json"
        write_json(attachment_manifest, {
            "contact_sheet_clean.png": {"camera": "overview", "frames": uniform_indices,
                                         "purpose": "全程均匀采样"},
            "contact_sheet_annotated.png": {"camera": "side", "frames": critical_indices,
                                             "purpose": "关键事件前后连续帧和同步物理标注"},
            "contact_sheet_multiview.png": {"cameras": ["front", "side", "overview"],
                                             "frames_per_camera": multiview_indices,
                                             "purpose": "同帧三视角足端和身体姿态消歧"},
            "behavior_evidence.json": {
                "purpose": "完整轨迹 command、速度、接触力和安全扫描；不含奖励或 PPO"},
        })
        visual_files = [clean, annotated, multiview, behavior_evidence,
                        attachment_manifest] + event_images
        return VisualEvidenceArtifacts(
            trajectory=trajectory, visual_files=visual_files, clean_sheet=clean,
            annotated_sheet=annotated, multiview_sheet=multiview,
            behavior_evidence=behavior_evidence, attachment_manifest=attachment_manifest)
