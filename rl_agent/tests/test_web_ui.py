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
    assert state_presentation("TASK_UNDERSTANDING")["stage_label"] == "GPT 理解动作目标"
    assert state_presentation("TASK_FEASIBILITY_CHECK")["stage_label"] == "预检动作可行性"
    assert state_presentation("MEMORY_CURATING")["progress"] == 99
    assert state_presentation("COMPLETED")["progress"] == 100


def test_config_payload_exposes_multi_agent_and_memory_status(tmp_path):
    """验证上位机能够展示分阶段 Provider、RAG 和长期记忆状态。"""
    payload = JobManager(_settings(tmp_path)).config_payload()
    providers = payload["system"]["providers"]
    assert providers["mode"] == "multi-agent"
    assert providers["task_planner"] == "opencli-doubao"
    assert providers["reward_designer"] == "bailian"
    assert payload["system"]["memory"]["records"] == 0
    assert payload["system"]["rag"]["enabled"]


def test_strategy_playback_only_launches_verified_real_success(tmp_path, monkeypatch):
    """验证上位机只为真实联合验收成功策略启动隔离 Viewer 命令。"""
    settings = _settings(tmp_path)
    training_root = tmp_path / "unitree_rl_gym"
    training_root.mkdir()
    settings.training_project = str(training_root)
    manager = JobManager(settings)
    task_id = "task-a1b2c3d4e5"
    task_dir = settings.experiments_path / task_id
    final_dir = task_dir / "final"
    final_dir.mkdir(parents=True)
    (final_dir / "checkpoint.pt").write_bytes(b"test checkpoint")
    (final_dir / "config.yaml").write_text("{}", encoding="utf-8")
    (task_dir / "state.json").write_text(json.dumps({"state": "COMPLETED", "updated_at": "2026-01-01"}),
                                          encoding="utf-8")
    (task_dir / "summary.json").write_text(json.dumps({
        "state": "COMPLETED", "result": "completed", "dry_run": False,
        "checkpoint": "final/checkpoint.pt", "config": "final/config.yaml",
        "selected_experiment": "candidate-01-v01",
    }), encoding="utf-8")
    (task_dir / "task_spec.json").write_text(json.dumps({"robot": "go2"}), encoding="utf-8")

    class FakeProcess:
        """提供不依赖 Isaac Gym 的播放进程桩。"""

        pid = 3456

        def poll(self):
            """保持进程运行以验证界面状态。"""
            return None

    captured = {}

    def fake_popen(command, **kwargs):
        """记录启动参数，确保播放器使用 argv 且不经过 shell。"""
        captured["command"] = command
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr("rl_training_agent.web_ui.subprocess.Popen", fake_popen)
    assert [item["task_id"] for item in manager.list_playable_experiments()] == [task_id]
    result = manager.start_playback(task_id)
    assert result["pid"] == 3456
    assert "--task" in captured["command"] and "go2" in captured["command"]
    assert captured["shell"] is False
    assert manager.playback_status()["running"]

    summary = json.loads((task_dir / "summary.json").read_text(encoding="utf-8"))
    summary["dry_run"] = True
    (task_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert manager.list_playable_experiments() == []
    with pytest.raises(JobValidationError, match="真实训练"):
        manager.start_playback(task_id)


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


def test_review_job_exposes_reason_and_launches_checkpoint_resume(tmp_path, monkeypatch):
    """验证上位机显示复核原因，并从当前 checkpoint 启动恢复命令而非重跑任务。"""
    settings = _settings(tmp_path)
    manager = JobManager(settings)
    source_id = "a1b2c3d4e5f6"
    task_id = "task-resume"
    started = "2026-01-01T00:00:00+00:00"
    manager._jobs[source_id] = {
        "job_id": source_id, "task_id": task_id, "task": "前腿站立行走", "robot": "go2",
        "mode": "real", "status": "review", "return_code": 0, "created_at": started,
        "started_at": started, "finished_at": started, "message": "等待人工复核",
    }
    task_dir = settings.experiments_path / task_id
    candidate = task_dir / "candidates" / "candidate-v03"
    candidate.mkdir(parents=True)
    (candidate / "reward_plan.json").write_text("{}", encoding="utf-8")
    (task_dir / "state.json").write_text(json.dumps({
        "state": "HUMAN_REVIEW", "updated_at": "2026-01-02T00:00:00+00:00",
        "history": [{"state": "FULL_TRAINING", "at": started}],
    }), encoding="utf-8")
    (task_dir / "summary.json").write_text(json.dumps({
        "selected_experiment": "candidate-v03", "reason": "视觉 Provider 暂时不可用",
    }), encoding="utf-8")
    captured = {}

    class RunningProcess:
        """模拟持续运行的恢复子进程，避免测试期间立刻改变作业终态。"""

        pid = 54321

        def wait(self):
            """等待测试释放恢复进程。"""
            time.sleep(0.2)
            return 0

        def poll(self):
            """报告恢复进程仍在运行。"""
            return None

    def fake_popen(command, **kwargs):
        """捕获上位机生成的恢复命令。"""
        captured["command"] = command
        return RunningProcess()

    monkeypatch.setattr("rl_training_agent.web_ui.subprocess.Popen", fake_popen)
    source = manager.get_job(source_id)
    assert source["can_resume"]
    assert source["review_reason"] == "视觉 Provider 暂时不可用"
    resumed = manager.resume_job(source_id)
    assert resumed["resume_of"] == source_id
    assert "resume" in captured["command"] and "train" not in captured["command"]
    assert captured["command"][captured["command"].index("--task-id") + 1] == task_id


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
        assert "动作可行性" in page and "四层记忆" in page
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


def test_job_manager_persists_job_when_gpu_pool_is_full(tmp_path):
    """验证 GPU 池占满时新作业进入持久等待队列，而不是争抢设备。"""
    manager = JobManager(_settings(tmp_path))
    manager._jobs["012345abcdef"] = {
        "job_id": "012345abcdef",
        "status": "running",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    job = manager.start_job({"task": "训练机器狗稳定向前行走", "robot": "go2", "mode": "dry-run"})
    assert job["status"] == "queued" and job["gpu_id"] is None
    persisted = settings_path = manager.root / job["job_id"] / "job.json"
    assert persisted.is_file() and json.loads(settings_path.read_text())["command"]


def test_job_exposes_feasibility_and_memory_summaries(tmp_path):
    """验证上位机只返回当前作业的分层可行性摘要与记忆晋升状态。"""
    settings = _settings(tmp_path)
    manager = JobManager(settings)
    job_id = "abc123abc123"
    task_id = "task-feasible"
    manager._jobs[job_id] = {
        "job_id": job_id, "task_id": task_id, "task": "Go2 倒退走 0.3m/s",
        "robot": "go2", "mode": "real", "status": "completed", "return_code": 0,
        "created_at": "2026-01-01T00:00:00+00:00",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:01:00+00:00", "message": "流程结束",
    }
    task_dir = settings.experiments_path / task_id
    (task_dir / "memory").mkdir(parents=True)
    (task_dir / "feasibility_report.json").write_text(json.dumps({
        "status": "PHYSICS_VALIDATED", "validation_level": "DYNAMIC_PHYSICS_VALIDATED",
        "feasibility_level": "LEVEL_5_PHYSICS", "motion_type": "LOCOMOTION",
        "confidence": 0.9, "motion_constraint_spec": {
            "motion_type": "LOCOMOTION", "duration": 5.0,
            "required_planners": ["gait", "com"],
        },
        "planning_report": {"status": "READY_FOR_SOLVER", "components": [
            {"planner": "gait", "status": "GENERATED"},
            {"planner": "com", "status": "GENERATED"},
        ]},
        "whole_body_report": {"status": "READY_FOR_PHYSICS", "missing_solvers": []},
        "training_admission": {"decision": "ALLOW_TRAINING", "scope": "simulation_only"},
        "physics_report": {"backend": "isaacgym", "validation_level": "DYNAMIC_PHYSICS_VALIDATED"},
    }), encoding="utf-8")
    (task_dir / "memory" / "working_memory.json").write_text(json.dumps({
        "state": "MEMORY_CURATING", "loop_round": 2, "reward_version": 3,
    }), encoding="utf-8")
    (task_dir / "memory" / "promotion.json").write_text(json.dumps({
        "promoted": True, "memory_id": "memory-demo", "reason": "联合验收通过",
    }), encoding="utf-8")

    job = manager.get_job(job_id)
    assert job["feasibility"]["validation_level"] == "DYNAMIC_PHYSICS_VALIDATED"
    assert job["feasibility"]["component_status"]["gait"] == "GENERATED"
    assert job["feasibility"]["viewer_available"]
    assert job["feasibility"]["training_admission"]["decision"] == "ALLOW_TRAINING"
    assert job["memory_detail"] == {
        "working_available": True, "working_state": "MEMORY_CURATING",
        "loop_round": 2, "reward_version": 3, "promoted": True,
        "promotion_reason": "联合验收通过", "memory_id": "memory-demo",
    }


def test_feasibility_view_uses_safe_cli_command(tmp_path, monkeypatch):
    """验证可行性观察窗口只通过固定 argv 启动，并与策略 Viewer 共用互斥状态。"""
    settings = _settings(tmp_path)
    manager = JobManager(settings)
    job_id = "def456def456"
    task_id = "task-viewable"
    manager._jobs[job_id] = {
        "job_id": job_id, "task_id": task_id, "task": "Go2 倒退走 0.3m/s",
        "robot": "go2", "mode": "real", "status": "completed", "return_code": 0,
        "created_at": "2026-01-01T00:00:00+00:00",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:01:00+00:00", "message": "流程结束",
    }
    task_dir = settings.experiments_path / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "feasibility_report.json").write_text(json.dumps({
        "status": "CAPABILITY_SUPPORTED", "validation_level": "CAPABILITY_ONLY",
        "motion_type": "LOCOMOTION", "motion_constraint_spec": {
            "motion_type": "LOCOMOTION", "duration": 4.0,
        },
    }), encoding="utf-8")

    class FakeProcess:
        """模拟持续运行的可行性 Viewer。"""
        pid = 6789

        def poll(self):
            """报告 Viewer 仍在运行。"""
            return None

    captured = {}

    def fake_popen(command, **kwargs):
        """记录安全启动参数。"""
        captured["command"] = command
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr("rl_training_agent.web_ui.subprocess.Popen", fake_popen)
    result = manager.start_feasibility_view(job_id)
    assert result["kind"] == "feasibility" and result["pid"] == 6789
    assert "feasibility-view" in captured["command"]
    assert captured["command"][captured["command"].index("--max-seconds") + 1] == "4.0"
    assert captured["shell"] is False
    assert manager.playback_status()["kind"] == "feasibility"
