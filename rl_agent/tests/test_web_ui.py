import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from rl_training_agent.settings import Settings, load_settings
from rl_training_agent.web_ui import (
    JobManager,
    JobValidationError,
    create_server,
    interpret_job_outcome,
    state_presentation,
    training_stage_progress,
    validate_job_request,
)


def _settings(tmp_path):
    """创建把界面作业和实验隔离到临时目录的测试配置。"""
    base = load_settings()
    return Settings(**{
        **base.dict(),
        "artifact_root": str(tmp_path / "artifacts"),
        "experiment_root": str(tmp_path / "experiments"),
    })


def test_validate_job_request_enforces_whitelists():
    """验证上位机只接受受支持的机器人、模式和合理长度的动作描述。"""
    assert validate_job_request(
        {"task": "训练机器狗稳定小跑", "robot": "GO2", "mode": "dry-run"}, ["go2"]
    ) == ("训练机器狗稳定小跑", "go2", "dry-run")
    with pytest.raises(JobValidationError):
        validate_job_request({"task": "跑", "robot": "go2", "mode": "dry-run"}, ["go2"])
    with pytest.raises(JobValidationError):
        validate_job_request({"task": "训练稳定行走", "robot": "../../etc", "mode": "real"}, ["go2"])
    with pytest.raises(JobValidationError):
        validate_job_request({"task": "训练稳定行走", "robot": "go2", "mode": "shell"}, ["go2"])


def test_state_presentation_uses_chinese_stages():
    """验证内部状态能够映射为稳定的百分比和中文阶段名称。"""
    assert state_presentation("REWARD_CANDIDATES_CREATED") == {
        "state": "REWARD_CANDIDATES_CREATED",
        "progress": 25,
        "stage_label": "生成奖励候选",
    }
    assert state_presentation("COMPLETED")["progress"] == 100


def test_human_review_is_not_reported_as_completed_training():
    """验证退出码为零但 Agent 要求人工复核时，上位机不会显示训练完成。"""
    assert interpret_job_outcome("HUMAN_REVIEW", 0, [
        {"state": "RECEIVED"}, {"state": "TASK_DESIGNED"},
    ]) == ("review", "训练尚未开始，任务设计等待人工复核")
    status, message = interpret_job_outcome("HUMAN_REVIEW", 0, [
        {"state": "FULL_TRAINING"}, {"state": "HUMAN_REVIEW"},
    ])
    assert status == "review" and "未通过最终验收" in message


def test_old_completed_job_is_dynamically_corrected_to_review(tmp_path):
    """验证旧版保存的 completed 作业会按持久化 HUMAN_REVIEW 状态纠正。"""
    settings = _settings(tmp_path)
    manager = JobManager(settings)
    job_id = "012345abcdef"
    task_id = "task-review"
    job_dir = settings.artifacts_path / "ui_jobs" / job_id
    job_dir.mkdir(parents=True)
    manager._jobs[job_id] = {
        "job_id": job_id, "task_id": task_id, "task": "前腿站立行走", "robot": "go2",
        "mode": "real", "status": "completed", "return_code": 0,
        "created_at": "2026-01-01T00:00:00+00:00", "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:01:00+00:00", "message": "训练已完成",
    }
    state_dir = settings.experiments_path / task_id
    state_dir.mkdir(parents=True)
    (state_dir / "state.json").write_text(json.dumps({
        "state": "HUMAN_REVIEW", "updated_at": "2026-01-01T00:00:30+00:00",
        "history": [{"state": "TASK_DESIGNED", "at": "2026-01-01T00:00:20+00:00"}],
    }))
    assert manager.get_job(job_id)["status"] == "review"


def test_job_exposes_closed_loop_round_and_reward_version(tmp_path):
    """验证上位机能够显示当前闭环轮次、奖励版本、诊断决策和预算。"""
    settings = _settings(tmp_path)
    manager = JobManager(settings)
    job_id = "fedcba654321"
    task_id = "task-loop"
    manager._jobs[job_id] = {
        "job_id": job_id, "task_id": task_id, "task": "后腿站立行走", "robot": "go2",
        "mode": "real", "status": "running", "return_code": None,
        "created_at": "2026-01-01T00:00:00+00:00", "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": None, "message": "训练运行中",
    }
    task_dir = settings.experiments_path / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "state.json").write_text(json.dumps({
        "state": "REVISE_REWARD", "updated_at": "2026-01-01T00:01:00+00:00",
        "history": [], "context": {"loop_round": 2, "reward_version": 3},
    }))
    (task_dir / "loop_status.json").write_text(json.dumps({
        "updated_at": "2026-01-01T00:01:00+00:00", "round": 2, "reward_version": 3,
        "decision": "revise_reward", "remaining_iterations": 3000, "remaining_revisions": 1,
    }))
    job = manager.get_job(job_id)
    assert job["loop_detail"]["remaining_iterations"] == 3000
    assert "闭环第2轮" in job["stage_label"]
    assert "奖励 v3" in job["stage_label"]


def test_training_iteration_advances_stage_progress_across_full_seeds():
    """验证完整训练的种子和内部迭代会共同推进上位机总进度。"""
    detail = {"run_name": "candidate-01-v01-seed-2", "percent": 50.0}
    progress = training_stage_progress("FULL_TRAINING", detail, {"candidate_count": 3}, [1, 2, 3])
    assert progress == 72


def test_job_manager_runs_fixed_dry_run_command(tmp_path, monkeypatch):
    """验证作业管理器使用固定参数数组启动演练并记录退出结果和日志。"""
    captured = {}

    class FakeProcess:
        """模拟一个立即成功退出的训练子进程。"""

        pid = 43210

        def wait(self):
            """返回成功退出码。"""
            return 0

        def poll(self):
            """在测试停止能力时表示进程仍可查询。"""
            return 0

    def fake_popen(command, **kwargs):
        """捕获训练命令并返回可控的模拟进程。"""
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr("rl_training_agent.web_ui.subprocess.Popen", fake_popen)
    manager = JobManager(_settings(tmp_path))
    job = manager.start_job({"task": "训练机器狗稳定向前小跑", "robot": "go2", "mode": "dry-run"})
    deadline = time.time() + 2
    while manager.get_job(job["job_id"])["status"] == "running" and time.time() < deadline:
        time.sleep(0.01)
    final = manager.get_job(job["job_id"])
    assert final["status"] == "completed"
    assert final["progress"] == 100
    assert captured["command"][-1] == "--dry-run"
    assert captured["command"][captured["command"].index("--provider") + 1] == "mock"
    assert captured["kwargs"]["shell"] is False
    assert "实验编号" in manager.read_log(job["job_id"])["text"]


def test_http_server_serves_ui_and_rejects_invalid_job(tmp_path):
    """验证本地服务可发送上位机页面，并拒绝不安全的训练请求。"""
    server = create_server("127.0.0.1", 0, JobManager(_settings(tmp_path)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = "http://127.0.0.1:{}".format(server.server_address[1])
    try:
        with urlopen(base_url + "/", timeout=3) as response:
            page = response.read().decode("utf-8")
        assert "强化学习上位机" in page
        request = Request(
            base_url + "/api/jobs",
            data=json.dumps({"task": "跑", "robot": "go2", "mode": "dry-run"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=3)
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_job_manager_prevents_concurrent_training(tmp_path):
    """验证上位机拒绝并行启动多个可能争抢 GPU 的训练作业。"""
    manager = JobManager(_settings(tmp_path))
    manager._jobs["012345abcdef"] = {
        "job_id": "012345abcdef",
        "status": "running",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    with pytest.raises(JobValidationError, match="已有训练作业"):
        manager.start_job({"task": "训练机器狗稳定向前行走", "robot": "go2", "mode": "dry-run"})
