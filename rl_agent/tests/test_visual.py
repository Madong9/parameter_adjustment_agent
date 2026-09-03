import json

import cv2
import numpy as np
import pandas as pd
import pytest

from rl_training_agent.visual.contact_sheet import ContactSheetBuilder
from rl_training_agent.visual.cropper import crop_frame
from rl_training_agent.visual.event_detector import EventDetector
from rl_training_agent.visual.evidence_builder import SynchronizedEvidenceBuilder
from rl_training_agent.visual.frame_sampler import FrameSampler
from rl_training_agent.visual.rollout_recorder import DryRunRolloutRecorder
from rl_training_agent.visual.video_metadata import VideoMetadata, read_video_metadata, time_to_frame
from rl_training_agent.providers.mock_provider import MockLLMReasoningProvider
from rl_training_agent.metrics.trajectory_metrics import TrajectoryMetrics
from rl_training_agent.schemas.task import TaskSpec


@pytest.fixture
def rollout(tmp_path):
    """构造测试所需的 rollout 数据。"""
    directory = tmp_path / "rollout"
    files = DryRunRolloutRecorder().record(directory, fps=10, seconds=1.0)
    return directory, files


def test_metadata_time_sync_and_sampling(rollout):
    """验证“metadata time sync and sampling”场景的预期行为。"""
    directory, files = rollout
    metadata = read_video_metadata(files["side"])
    assert metadata.video_fps == 10 and metadata.frame_count == 10
    assert time_to_frame(0.5, metadata) == 5
    assert FrameSampler.uniform_indices(10, 4)[0] == 0
    assert FrameSampler.event_indices(10, [0, 9], radius=1) == [0, 1, 8, 9]


def test_bad_video_and_zero_fps(tmp_path, monkeypatch):
    """验证“bad video and zero fps”场景的预期行为。"""
    with pytest.raises(ValueError, match="cannot be opened"):
        read_video_metadata(tmp_path / "bad.mp4")
    class FakeCapture:
        def isOpened(self):
            """执行 isOpened 对应的业务逻辑并返回结果。"""
            return True
        def get(self, key):
            """执行 get 对应的业务逻辑并返回结果。"""
            return 0
        def release(self):
            """执行 release 对应的业务逻辑并返回结果。"""
            return None
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: FakeCapture())
    with pytest.raises(ValueError, match="FPS"):
        read_video_metadata(tmp_path / "zero.mp4")


def test_crop_and_empty_frame():
    """验证“crop and empty frame”场景的预期行为。"""
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    assert crop_frame(frame, (10, 10, 20, 30)).shape == (30, 20, 3)
    with pytest.raises(ValueError):
        crop_frame(np.array([]), None)
    with pytest.raises(ValueError):
        crop_frame(frame, (190, 0, 20, 20))


def test_contact_sheets_and_multi_camera(rollout):
    """验证“contact sheets and multi camera”场景的预期行为。"""
    directory, files = rollout
    frames = FrameSampler.decode(files["side"], [0, 1, 2])
    clean = ContactSheetBuilder().build(frames, directory / "clean.png")
    annotated = ContactSheetBuilder().build(frames, directory / "annotated.png", {0: {"event": "start"}})
    multi = ContactSheetBuilder().build_multi_camera({"side": frames, "front": frames}, directory / "multi.png")
    assert clean.is_file() and annotated.is_file() and multi.is_file()
    with pytest.raises(ValueError, match="at least one"):
        ContactSheetBuilder().build([], directory / "empty.png")


def test_trajectory_parquet_and_events(rollout):
    """验证“trajectory parquet and events”场景的预期行为。"""
    directory, files = rollout
    data = pd.read_parquet(files["trajectory"])
    events = EventDetector().detect(data)
    names = {event["name"] for event in events}
    assert {"all_feet_takeoff", "jump_peak", "all_feet_landing", "episode_end"} <= names
    assert len(data) == read_video_metadata(files["side"]).frame_count


def test_synchronized_evidence_covers_commands_contacts_and_continuous_frames(rollout):
    """验证证据文件完整扫描轨迹且不包含奖励或 PPO 指标。"""
    directory, files = rollout
    data = pd.read_parquet(files["trajectory"])
    data.at[3, "command"] = np.array([0.5, 0.0, 0.0, 0.0])
    data.at[3, "penalized_contact_force_max"] = 2.5
    data.at[3, "forbidden_contact_force_max"] = 2.5
    data.at[3, "body_collision"] = True
    events = EventDetector().detect(data)
    task = TaskSpec.parse_obj(MockLLMReasoningProvider(1).design_task_and_rewards(
        "向前行走", "go2", {})["task_spec"])
    output = SynchronizedEvidenceBuilder().build(
        task, data, events, directory / "behavior_evidence.json")
    content = output.read_text(encoding="utf-8")
    evidence = json.loads(content)
    assert evidence["coverage"]["analyzed_frame_count"] == len(data)
    assert evidence["command_tracking"]["target_lin_vel_x_mps"]["max"] == 0.5
    assert evidence["safety_scan"]["max_forbidden_body_contact_force_n"] == 2.5
    assert 3 in evidence["safety_scan"]["body_collision_frames"]
    assert '"reward_total":' not in content and '"loss":' not in content.lower()


def test_trajectory_metrics_use_measured_collision_and_tracking_data(rollout):
    """验证最终数值验收不会再用固定零值掩盖身体碰撞。"""
    _, files = rollout
    data = pd.read_parquet(files["trajectory"])
    data["command"] = [np.array([0.5, 0.0, 0.0, 0.0]) for _ in range(len(data))]
    data["base_vx"] = 0.4
    data["body_collision"] = False
    data.at[2, "body_collision"] = True
    data["forbidden_contact_force_max"] = 0.0
    data.at[2, "forbidden_contact_force_max"] = 12.5
    data["termination_reason"] = ""
    data.at[3, "termination_reason"] = "reset"
    metrics = TrajectoryMetrics().compute(data)
    assert metrics["tracking_error"] == pytest.approx(0.1)
    assert metrics["forbidden_collisions"] == 1.0
    assert metrics["forbidden_contact_force"] == 12.5
    assert metrics["abnormal_terminations"] == 1.0
    assert metrics["episode_survival"] == 0.0
    assert "rear_leg_stand_duration" in metrics
    assert "rear_leg_walk_completion" in metrics
    assert "rear_leg_walk_velocity_tracking" in metrics
    assert "front_leg_off_ground_ratio" in metrics
    assert metrics["forbidden_body_contact"] == 12.5
    assert "stable_stand_duration" in metrics
    assert "walking_speed_tracking" in metrics
    assert "body_pitch_within_limit" in metrics
    assert "front_leg_stand_duration" in metrics
