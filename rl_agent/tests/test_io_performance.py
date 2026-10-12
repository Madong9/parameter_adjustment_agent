"""验证长训练日志、指标采集和视觉附件的有界读取与证据一致性。"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from rl_training_agent.metrics.ppo_collector import PPOCollector
from rl_training_agent.metrics.tensorboard_reader import TensorBoardReader
from rl_training_agent.providers.mock_provider import MockLLMReasoningProvider
from rl_training_agent.schemas.task import TaskSpec
from rl_training_agent.training.early_stopping import EarlyStoppingMonitor
from rl_training_agent.visual.contact_sheet import ContactSheetBuilder
from rl_training_agent.visual.evaluation_pipeline import VisualEvaluationPipeline
from rl_training_agent.visual.frame_sampler import FrameSampler
from rl_training_agent.visual.rollout_recorder import DryRunRolloutRecorder


def test_large_log_reads_bounded_tail_and_preserves_unicode_window(tmp_path, monkeypatch):
    """大日志只读取有限尾部，同时仍能识别十万字符窗口内的异常。"""
    log = tmp_path / "stdout.log"
    with log.open("wb") as handle:
        handle.write(b"old nan\n")
        handle.seek(8 * 1024 * 1024)
        handle.write(("CUDA out of memory\n" + "中文" * 40000).encode("utf-8"))
    original_open = Path.open
    reads = []

    class ReadSpy:
        """记录尾部读取的实际长度并代理二进制文件。"""

        def __init__(self, handle):
            """保存真实文件句柄。"""
            self.handle = handle

        def __enter__(self):
            """进入原文件的上下文。"""
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            """关闭原文件并传递异常。"""
            return self.handle.__exit__(*args)

        def __getattr__(self, name):
            """代理定位、描述符等文件接口。"""
            return getattr(self.handle, name)

        def read(self, size=-1):
            """记录单次读取长度，禁止全量读取。"""
            reads.append(size)
            assert 0 <= size <= 400000
            return self.handle.read(size)

    def tracked_open(path, *args, **kwargs):
        """仅包装被检查日志的二进制读取。"""
        handle = original_open(path, *args, **kwargs)
        return ReadSpy(handle) if path == log and args == ("rb",) else handle

    monkeypatch.setattr(Path, "open", tracked_open)
    assert EarlyStoppingMonitor().inspect_log(log) == "gpu_oom"
    assert reads == [400000]


@pytest.mark.parametrize("reward,expected", [
    ("-1.2e+7", "reward_exploded"), (".5", None), ("1e", None),
    ("+", None), ("NaN", "nan_or_inf"), ("Inf", "nan_or_inf"),
])
def test_reward_log_handles_scientific_and_incomplete_values(tmp_path, reward, expected):
    """科学计数法仍触发数值上界检查，未写完的数字不会导致监控崩溃。"""
    log = tmp_path / "stdout.log"
    log.write_text("Mean reward: " + reward + "\n", encoding="utf-8")
    assert EarlyStoppingMonitor().inspect_log(log) == expected


def test_ppo_collect_uses_one_snapshot_and_keeps_iteration(tmp_path, monkeypatch):
    """最新指标与最大迭代号来自同一次扫描，包含空标签和未知标签。"""
    calls = []

    def read_snapshot(self, directory):
        """返回不同迭代的指标并记录磁盘扫描次数。"""
        calls.append(directory)
        return {
            "Train/mean_reward": [{"step": 1, "value": 1.0}, {"step": 7, "value": 4.0}],
            "Loss/value_function": [{"step": 6, "value": 0.25}],
            "Episode/custom": [{"step": 9, "value": 0.8}],
            "empty": [],
        }

    monkeypatch.setattr(TensorBoardReader, "read", read_snapshot)
    stats = PPOCollector().collect(tmp_path)
    assert calls == [tmp_path]
    assert stats.iteration == 9 and stats.mean_reward == 4.0 and stats.value_loss == 0.25
    assert stats.fps == 0.0


def test_ppo_collect_reads_real_tensorboard_events(tmp_path):
    """真实事件文件排序后保留最大步数和同一步数的最新标量。"""
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    writer = EventFileWriter(str(tmp_path))
    try:
        for step, wall_time, value in ((7, 3.0, 5.0), (1, 1.0, 2.0), (7, 2.0, 4.0)):
            writer.add_event(Event(step=step, wall_time=wall_time,
                                   summary=Summary(value=[Summary.Value(
                                       tag="Train/mean_reward", simple_value=value)])))
    finally:
        writer.close()
    stats = PPOCollector().collect(tmp_path)
    assert stats.iteration == 7 and stats.mean_reward == 5.0


class TrackingCapture:
    """用帧内容编码帧号，检测定位次数、失败定位和资源释放。"""

    def __init__(self, fail_read=False, fail_seek=False):
        """初始化模拟解码器的失败模式。"""
        self.position = 0
        self.seeks = []
        self.released = False
        self.fail_read = fail_read
        self.fail_seek = fail_seek

    def isOpened(self):
        """声明模拟视频已打开。"""
        return True

    def set(self, prop, index):
        """记录定位，按需模拟不支持跳转。"""
        self.seeks.append(index)
        if self.fail_seek:
            return False
        self.position = index
        return True

    def read(self):
        """读取帧号对应的图像，按需模拟底层异常。"""
        if self.fail_read:
            raise RuntimeError("decode failed")
        frame = np.full((2, 2, 3), self.position, dtype=np.uint8)
        self.position += 1
        return True, frame

    def release(self):
        """记录句柄是否释放。"""
        self.released = True


def test_dense_frame_decode_keeps_indices_with_fewer_seeks(tmp_path, monkeypatch):
    """乱序、重复和负索引不改变有效帧号，连续帧只需顺序读取。"""
    capture = TrackingCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)
    frames = FrameSampler.decode(tmp_path / "video.mp4", [8, 2, -1, 1, 0, 2, 7])
    assert [index for index, _ in frames] == [0, 1, 2, 7, 8]
    assert [int(frame[0, 0, 0]) for _, frame in frames] == [0, 1, 2, 7, 8]
    assert capture.seeks == [7] and capture.released


def test_frame_decode_releases_capture_after_exception(tmp_path, monkeypatch):
    """解码器抛出异常时仍关闭视频句柄。"""
    capture = TrackingCapture(fail_read=True)
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)
    with pytest.raises(RuntimeError, match="decode failed"):
        FrameSampler.decode(tmp_path / "video.mp4", [0])
    assert capture.released


def test_failed_seek_cannot_mislabel_frame(tmp_path, monkeypatch):
    """定位失败时忽略该帧，不能把当前位置的图像冒充目标帧。"""
    capture = TrackingCapture(fail_seek=True)
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)
    frames = FrameSampler.decode(tmp_path / "video.mp4", [0, 7, 8])
    assert [index for index, _ in frames] == [0]
    assert capture.released


def test_contact_sheet_bounds_allocation_and_preserves_all_tiles(tmp_path, monkeypatch):
    """缩图之前的画布也遵守尺寸限制，每一帧仍出现在对应位置。"""
    frames = [(index, np.full((4, 4, 3), index, dtype=np.uint8)) for index in range(32)]
    original_full = np.full
    allocations = []

    def bounded_full(shape, *args, **kwargs):
        """记录实际画布分配并核对边长上界。"""
        allocations.append(shape)
        assert max(shape[:2]) <= 400
        return original_full(shape, *args, **kwargs)

    monkeypatch.setattr(np, "full", bounded_full)
    output = ContactSheetBuilder(columns=2, max_dimension=400).build(frames, tmp_path / "sheet.png")
    image = cv2.imread(str(output))
    assert allocations and image.shape == allocations[0]
    height, width = image.shape[:2]
    for index in range(32):
        row, column = divmod(index, 2)
        assert int(image[row * (height // 16) + 1, column * (width // 2) + 1, 0]) == index


@pytest.mark.parametrize("kwargs", [{"columns": 0}, {"tile_size": (0, 240)}, {"max_dimension": 0}])
def test_contact_sheet_rejects_invalid_dimensions(kwargs):
    """不合法尺寸在分配内存前被拒绝。"""
    with pytest.raises(ValueError, match="positive"):
        ContactSheetBuilder(**kwargs)


def test_visual_pipeline_decodes_each_camera_once_and_reuses_events(tmp_path, monkeypatch):
    """所有附件共享每个相机的一次解码，事件图仍来自准确的侧视帧。"""
    media = DryRunRolloutRecorder().record(tmp_path, fps=10, seconds=1.0)
    task = TaskSpec.parse_obj(MockLLMReasoningProvider(1).design_task_and_rewards(
        "向前行走", "go2", {})["task_spec"])
    original_decode = FrameSampler.decode
    decoded = {}

    def tracked_decode(path, indices):
        """记录各相机的帧内容并拒绝重复解码。"""
        assert path.stem not in decoded
        frames = original_decode(path, indices)
        decoded[path.stem] = dict(frames)
        return frames

    monkeypatch.setattr(FrameSampler, "decode", tracked_decode)
    result = VisualEvaluationPipeline().build(task, tmp_path, media)
    assert set(decoded) == {"front", "side", "overview"}
    assert all(path.is_file() for path in result.visual_files)
    from rl_training_agent.visual.event_detector import EventDetector
    events = EventDetector().detect(result.trajectory)
    for name, filename in (("all_feet_takeoff", "event_takeoff.png"),
                            ("all_feet_landing", "event_landing.png")):
        frame = next(event["frame"] for event in events if event["name"] == name)
        assert np.array_equal(cv2.imread(str(tmp_path / filename)), decoded["side"][frame])
