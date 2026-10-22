"""预算耗尽上位机回归：只使用临时 JSON 和进程桩，不读取凭据或启动训练。"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from rl_training_agent.settings import Settings
from rl_training_agent.web_ui import JobManager, JobValidationError


@pytest.fixture
def experiment(tmp_path):
    """建立有可恢复候选的临时人工复核实验，冻结额度独立于当前配置。"""
    settings = Settings(artifact_root=str(tmp_path / "artifacts"),
                        experiment_root=str(tmp_path / "experiments"),
                        max_total_iterations=12000, max_reward_revisions=3)
    manager = JobManager(settings)
    job_id, task_id = "012345abcdef", "task-012345abcd"
    job = {"job_id": job_id, "task_id": task_id, "task": "训练 Go2 稳定行走", "robot": "go2",
           "mode": "real", "status": "review", "return_code": 0,
           "created_at": "2026-01-01T00:00:00+00:00", "started_at": "2026-01-01T00:00:00+00:00",
           "finished_at": "2026-01-02T00:00:00+00:00", "message": "等待人工复核"}
    manager._jobs[job_id] = job
    task_dir = settings.experiments_path / task_id
    candidate = task_dir / "candidates" / "candidate-v02"
    candidate.mkdir(parents=True)
    (candidate / "reward_plan.json").write_text('{"version":2}')
    (candidate / "manifest.json").write_text(json.dumps({
        "reward_version": 2, "training_result": "completed", "checkpoint": "checkpoints/model_1000.pt",
    }))
    (candidate / "checkpoints").mkdir()
    (candidate / "checkpoints" / "model_1000.pt").write_text("safe test marker, not a model")
    (task_dir / "state.json").write_text(json.dumps({
        "state": "HUMAN_REVIEW", "updated_at": "2026-01-02T00:00:00+00:00", "history": [],
        "context": {"run_id": "ui-budget-run", "reward_version": 2},
    }))
    (task_dir / "summary.json").write_text(json.dumps({
        "state": "HUMAN_REVIEW", "result": "human_review", "dry_run": False,
        "selected_experiment": "candidate-v02", "reward_version": 2,
        "reason": "视觉 Provider 暂时不可用", "checkpoint": "candidates/candidate-v02/checkpoints/model_1000.pt",
        "used_iterations": 1000, "remaining_iterations": 2000, "used_revisions": 0, "max_revisions": 1,
    }))
    return manager, job_id, task_dir


def write_execution(task_dir, used_iterations=1000, max_iterations=3000,
                    used_revisions=0, max_revisions=1, **changes):
    """写入绑定本次运行的冻结快照，可显式模拟类型错误、错运行和耗尽。"""
    snapshot = {"run_id": "ui-budget-run", "dry_run": False, "phase": "trained",
                "experiment_id": "candidate-v02", "budget": {
                    "used_iterations": used_iterations, "max_iterations": max_iterations,
                    "used_revisions": used_revisions, "max_revisions": max_revisions,
                }}
    snapshot.update(changes)
    (task_dir / "training_execution.json").write_text(json.dumps(snapshot))


@pytest.mark.parametrize("iterations,revisions", [(3000, 0), (1000, 1)])
def test_exhausted_iteration_or_revision_budget_blocks_resume(experiment, monkeypatch, iterations, revisions):
    """任一闭环资源耗尽都隐藏恢复能力，服务端也不能启动 Provider 或训练。"""
    manager, job_id, task_dir = experiment
    write_execution(task_dir, used_iterations=iterations, used_revisions=revisions)
    job = manager.get_job(job_id)
    assert job["budget_detail"]["valid"] and job["budget_detail"]["exhausted"]
    assert job["budget_detail"]["used_iterations"] == iterations
    assert job["budget_detail"]["max_iterations"] == 3000
    assert job["budget_detail"]["max_revisions"] == 1
    assert job["can_resume"] is False
    assert "当前界面没有追加额度功能" in job["resume_block_reason"]
    monkeypatch.setattr(manager, "_launch_job_process", lambda *args: pytest.fail("不能启动训练"))
    with pytest.raises(JobValidationError, match="预算已耗尽"):
        manager.resume_job(job_id)


def test_frozen_limits_ignore_larger_current_settings(experiment):
    """界面显示实验实际冻结的 3000/1，而不是配置中更大的 12000/3。"""
    manager, job_id, task_dir = experiment
    write_execution(task_dir, used_iterations=2600)
    budget = manager.get_job(job_id)["budget_detail"]
    assert budget["source"] == "training_execution"
    assert budget["used_iterations"] == 2600 and budget["max_iterations"] == 3000
    assert budget["remaining_iterations"] == 400 and budget["max_revisions"] == 1


def test_provider_pause_with_remaining_quota_can_resume(experiment, monkeypatch):
    """Provider 暂停且有冻结额度时继续可恢复，但只启动固定恢复命令。"""
    manager, job_id, task_dir = experiment
    write_execution(task_dir)
    job = manager.get_job(job_id)
    assert job["can_resume"] is True and job["resume_block_reason"] is None
    captured = []

    def launch(job, command, logs):
        """捕获 CLI 参数，不创建进程或发送模型请求。"""
        captured.append(command)
        return job

    monkeypatch.setattr(manager, "_launch_job_process", launch)
    resumed = manager.resume_job(job_id)
    assert resumed["resume_of"] == job_id and captured[0][4] == "resume"


@pytest.mark.parametrize("change", [{"run_id": "other-run"}, {"dry_run": True},
                                    {"budget": {"used_iterations": "1000", "max_iterations": 3000,
                                                "used_revisions": 0, "max_revisions": 1}}])
def test_invalid_execution_does_not_fall_back_to_optimistic_summary(experiment, change):
    """错运行、错模式或无效执行快照不得借旧摘要显示无限或充足额度。"""
    manager, job_id, task_dir = experiment
    write_execution(task_dir, **change)
    job = manager.get_job(job_id)
    assert job["budget_detail"]["valid"] is False
    assert job["budget_detail"]["status"] == "unknown"
    assert job["budget_detail"]["max_iterations"] is None and not job["can_resume"]


def test_missing_budget_is_unknown_and_not_resumable(experiment):
    """旧摘要完全没有额度时明确未知，不以 Settings 默认上限开放恢复。"""
    manager, job_id, task_dir = experiment
    summary = json.loads((task_dir / "summary.json").read_text())
    for key in ("used_iterations", "remaining_iterations", "used_revisions", "max_revisions"):
        summary.pop(key)
    (task_dir / "summary.json").write_text(json.dumps(summary))
    job = manager.get_job(job_id)
    assert job["budget_detail"]["valid"] is False and job["can_resume"] is False
    assert "缺失或无效" in job["resume_block_reason"]


def test_legacy_summary_budget_remains_supported(experiment):
    """旧版 used+remaining 的摘要仍能还原冻结上限并保留 Provider 恢复能力。"""
    manager, job_id, _ = experiment
    job = manager.get_job(job_id)
    assert job["budget_detail"]["source"] == "summary"
    assert job["budget_detail"]["max_iterations"] == 3000 and job["can_resume"] is True


def test_backend_explicit_resume_denial_is_respected(experiment):
    """即使数值上还有余额，后端显式冻结恢复许可也必须在 UI 体现。"""
    manager, job_id, task_dir = experiment
    summary = json.loads((task_dir / "summary.json").read_text())
    summary["resume_allowed"] = False
    (task_dir / "summary.json").write_text(json.dumps(summary))
    job = manager.get_job(job_id)
    assert job["budget_detail"]["valid"] is True and job["can_resume"] is False
    assert "后端已禁止" in job["resume_block_reason"]


def test_ppo_checkpoint_and_reward_version_do_not_imply_acceptance(experiment):
    """已完成 PPO 和保存策略仍不能伪装成动作联合验收成功。"""
    manager, job_id, task_dir = experiment
    state = json.loads((task_dir / "state.json").read_text())
    state["context"]["reward_version"] = 3
    (task_dir / "state.json").write_text(json.dumps(state))
    job = manager.get_job(job_id)
    result = job["result_detail"]
    assert result["reward_version"] == 3 and result["trained_reward_version"] == 2
    assert result["ppo_completed"] and result["checkpoint_available"]
    assert result["accepted"] is False and result["acceptance_status"] == "pending"


def test_completed_play_is_not_disabled_by_exhausted_training_budget(experiment, monkeypatch):
    """已联合验收的真实策略在额度耗尽后仍可用原播放白名单启动。"""
    manager, job_id, task_dir = experiment
    write_execution(task_dir, used_iterations=3000, used_revisions=1)
    state = json.loads((task_dir / "state.json").read_text())
    state["state"] = "COMPLETED"
    (task_dir / "state.json").write_text(json.dumps(state))
    summary = json.loads((task_dir / "summary.json").read_text())
    summary.update(state="COMPLETED", result="completed", config="config.yaml",
                   used_iterations=3000, remaining_iterations=0, used_revisions=1)
    (task_dir / "summary.json").write_text(json.dumps(summary))
    (task_dir / "config.yaml").write_text("{}")
    (task_dir / "task_spec.json").write_text('{"robot":"go2"}')
    manager._jobs[job_id]["status"] = "completed"
    job = manager.get_job(job_id)
    assert job["budget_detail"]["exhausted"] and not job["can_resume"]
    assert job["result_detail"]["accepted"] is True
    assert manager._playable_artifacts(manager._jobs[job_id]["task_id"]) is not None

    class PlaybackProcess:
        """模拟运行中的 Viewer，不加载模型。"""
        pid = 1000000000

        def poll(self):
            """返回 Viewer 仍在运行的状态。"""
            return None

    monkeypatch.setattr("rl_training_agent.web_ui.subprocess.Popen", lambda *args, **kwargs: PlaybackProcess())
    assert manager.start_playback(manager._jobs[job_id]["task_id"])["kind"] == "strategy"


def test_resumed_job_displays_inherited_frozen_budget(experiment):
    """恢复作业启动后尚未写新文件时，也能展示同 run_id 继承的冻结额度。"""
    manager, job_id, task_dir = experiment
    write_execution(task_dir)
    resumed_id = "abcdef012345"
    manager._jobs[resumed_id] = dict(manager._jobs[job_id], job_id=resumed_id, resume_of=job_id,
                                   status="running", started_at="2099-01-01T00:00:00+00:00")
    job = manager.get_job(resumed_id)
    assert job["budget_detail"]["valid"] and job["budget_detail"]["max_iterations"] == 3000
    assert job["result_detail"]["checkpoint_available"]


def test_frontend_budget_presentation_separates_success_states():
    """执行真实 JS 纯展示函数，验证耗尽、未知和 Provider 暂停的按钮与文字。"""
    node = shutil.which("node")
    if not node:
        pytest.skip("未安装 Node，后端预算回归仍会执行")
    script_path = Path(__file__).resolve().parents[1] / "rl_training_agent" / "web" / "app.js"
    code = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const context = {localStorage: {getItem: () => null}, document: {querySelector: () => ({})}};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8').replace(/initialize\(\);\s*$/, ''), context);
const base = {status:'review', can_resume:true,
  budget_detail:{valid:true, exhausted:true, source:'training_execution', used_iterations:3000,
  max_iterations:3000, remaining_iterations:0, used_revisions:1, max_revisions:1, remaining_revisions:0},
  result_detail:{training_status:'completed', checkpoint_available:true, trained_reward_version:2,
  acceptance_status:'not_passed', accepted:false}};
const exhausted = context.trainingResultPresentation(base);
assert.equal(exhausted.canResume, false);
assert.equal(exhausted.usage, '3000/3000 iter');
assert.equal(exhausted.budgetLabel, '已耗尽');
assert.equal(exhausted.trainingLabel, 'PPO 训练已完成');
assert.equal(exhausted.acceptanceLabel, '联合验收未通过');
assert(exhausted.notice.includes('已保存策略 v2'));
assert.equal(context.trainingResultPresentation({...base, budget_detail:{}}).canResume, false);
assert.equal(context.trainingResultPresentation({...base,
  budget_detail:{...base.budget_detail, exhausted:false, remaining_iterations:500, remaining_revisions:1}}).canResume, true);
const accepted = context.trainingResultPresentation({...base, result_detail:{...base.result_detail,
  accepted:true, acceptance_status:'passed'}});
assert(accepted.notice.includes('仍可播放'));
const early = {status:'review', progress:100, budget_detail:{valid:false,status:'not_started'},
  result_detail:{training_status:'not_started',acceptance_status:'not_started',accepted:false},
  stage_detail:{environment:'done',intent:'blocked',reward:'pending',smoke:'pending',full:'pending',visual:'pending'}};
const earlyView = context.trainingResultPresentation(early);
assert.equal(earlyView.usage, '尚未冻结');
assert.equal(earlyView.budgetLabel, '尚未进入训练');
assert.equal(earlyView.trainingLabel, '尚未启动 PPO');
assert.equal(earlyView.acceptanceLabel, '尚未进入验收');
assert.equal(earlyView.canResume, false);
assert(!earlyView.notice.includes('缺失或无效'));
assert.equal(context.stageStatus(early, 'environment'), 'done');
assert.equal(context.stageStatus(early, 'intent'), 'blocked');
for (const name of ['reward','smoke','full','visual','acceptance']) {
  assert.equal(context.stageStatus(early, name), 'pending');
}
assert.equal(context.stageStatus({progress:100,status:'completed'}, 'full'), 'pending');
assert.equal(context.stageStatus({stage_detail:{acceptance:'done'},result_detail:{accepted:false}}, 'acceptance'), 'pending');
assert.equal(context.stageStatus({result_detail:{accepted:true}}, 'acceptance'), 'done');
"""
    result = subprocess.run([node, "-e", code, str(script_path)], text=True, capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr
