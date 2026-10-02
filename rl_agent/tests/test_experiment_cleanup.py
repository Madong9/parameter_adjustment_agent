import json
import pytest

from rl_training_agent.maintenance.experiment_cleanup import ExperimentCleaner


@pytest.mark.parametrize("state", ["COMPLETED", "HUMAN_REVIEW", "FAILED", "FULL_TRAINING"])
def test_success_cleanup_preserves_learning_data_and_other_tasks(tmp_path, state):
    task = tmp_path / "task-action"
    video = _write(task / "candidates" / "old" / "rollouts" / "front.mp4")
    selected = _write(task / "candidates" / "selected" / "rollouts" / "side.MP4")
    probe = _write(task / "feasibility" / "probe.webm")
    other = _write(tmp_path / "task-other" / "front.mp4")
    retained = [_write(task / name) for name in (
        "final/checkpoint.pt", "final/config.yaml", "final/reward_plan.json",
        "memory/reward_experience/evidence.json", "trajectory.parquet", "rewards.parquet",
        "contact_sheet_annotated.png", "visual_report.json", "metadata.json")]
    for name in ("summary.json", "state.json"):
        (task / name).write_text(json.dumps({"state": state}))
    link = task / "external.mp4"
    link.symlink_to(other)
    cleaner = ExperimentCleaner(tmp_path)
    result = cleaner.cleanup_completed_videos(task.name)
    assert all(path.is_file() for path in retained)
    assert other.is_file() and link.is_symlink()
    if state == "COMPLETED":
        assert result["video_files"] == 3 and result["reclaim_bytes"] == 48
        assert not any(path.exists() for path in (video, selected, probe))
        assert cleaner.cleanup_completed_videos(task.name)["video_files"] == 0
    else:
        assert result["status"] == "skipped"
        assert all(path.is_file() for path in (video, selected, probe))


def test_success_cleanup_requires_matching_persisted_terminal_state(tmp_path):
    task = tmp_path / "task-action"
    video = _write(task / "front.mp4")
    (task / "summary.json").write_text(json.dumps({"state": "COMPLETED"}))
    (task / "state.json").write_text(json.dumps({"state": "MEMORY_CURATING"}))
    assert ExperimentCleaner(tmp_path).cleanup_completed_videos(task.name)["status"] == "skipped"
    assert video.is_file()


def _write(path, size=16):
    """创建指定大小的测试产物并返回路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def test_cleanup_previews_then_removes_only_redundant_artifacts(tmp_path):
    """验证预览不改文件，应用后仍保留恢复点、末次模型和三个视觉样本。"""
    task = tmp_path / "task-example"
    run = task / "candidates" / "candidate" / "seed-1"
    protected = _write(run / "model_50.pt")
    redundant = _write(run / "model_100.pt")
    latest = _write(run / "model_150.pt")
    (task / "candidates" / "candidate" / "manifest.json").write_text(json.dumps({
        "checkpoint": "seed-1/model_50.pt",
    }))
    round_root = task / "candidates" / "candidate" / "rollouts" / "round_01"
    records = []
    for index in range(1, 6):
        name = "rollout_%03d" % index
        _write(round_root / name / "front.mp4")
        _write(round_root / name / "trajectory.parquet")
        records.append({"rollout": name, "seed": index, "score": float(index)})
    (round_root / "rollout_metrics.json").write_text(json.dumps({"records": records}))
    counterfactual = _write(round_root / "counterfactual_zero_seed_1" / "front.mp4")
    cleaner = ExperimentCleaner(tmp_path)

    preview = cleaner.run(keep_per_run=1)

    assert preview["mode"] == "preview" and redundant.is_file()
    assert preview["checkpoint_files"] == 1
    assert preview["video_files"] == 3
    applied = cleaner.run(keep_per_run=1, apply=True)
    assert applied["mode"] == "applied"
    assert protected.is_file() and latest.is_file() and not redundant.exists()
    assert not counterfactual.exists()
    assert (round_root / "rollout_001" / "front.mp4").is_file()
    assert (round_root / "rollout_003" / "front.mp4").is_file()
    assert (round_root / "rollout_005" / "front.mp4").is_file()
    assert (round_root / "rollout_002" / "trajectory.parquet").is_file()


def test_video_only_cleanup_removes_obsolete_candidate_without_touching_checkpoints(tmp_path):
    """验证仅视频模式会清除旧候选视频，但保留当前候选视频和全部 checkpoint。"""
    task = tmp_path / "task-video"
    selected_video = _write(task / "candidates" / "selected" / "rollouts" / "front.mp4")
    obsolete_video = _write(task / "candidates" / "failed" / "rollouts" / "front.mp4")
    checkpoint = _write(task / "candidates" / "failed" / "seed-1" / "model_50.pt")
    (task / "summary.json").write_text(json.dumps({
        "state": "HUMAN_REVIEW", "selected_experiment": "selected",
    }))

    result = ExperimentCleaner(tmp_path).run(
        apply=True, cleanup_checkpoints=False, cleanup_videos=True)

    assert result["checkpoint_files"] == 0
    assert selected_video.is_file() and not obsolete_video.exists()
    assert checkpoint.is_file()
