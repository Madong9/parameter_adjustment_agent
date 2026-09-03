from __future__ import annotations

import json
import mimetypes
import os
import re
import signal
import subprocess
import sys
import threading
import uuid
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .orchestration.orchestrator import TrainingOrchestrator
from .settings import Settings, load_settings
from .utils.io import read_json, utc_now, write_json


ROBOT_CATALOG = [
    {"id": "go2", "name": "Unitree Go2", "kind": "四足机器人", "mark": "G2"},
    {"id": "g1", "name": "Unitree G1", "kind": "人形机器人", "mark": "G1"},
    {"id": "h1", "name": "Unitree H1", "kind": "人形机器人", "mark": "H1"},
    {"id": "h1_2", "name": "Unitree H1-2", "kind": "人形机器人", "mark": "H2"},
]

STATE_PRESENTATION: Dict[str, Tuple[int, str]] = {
    "RECEIVED": (3, "接收任务"),
    "ENVIRONMENT_INSPECTED": (10, "检查训练环境"),
    "TASK_DESIGNED": (18, "解析动作目标"),
    "REWARD_CANDIDATES_CREATED": (25, "生成奖励候选"),
    "CONFIGS_COMPILED": (32, "编译训练配置"),
    "VALIDATED": (38, "验证奖励与配置"),
    "SMOKE_TRAINING": (48, "快速冒烟训练"),
    "CANDIDATE_SCREENING": (60, "筛选候选方案"),
    "FULL_TRAINING": (74, "执行完整训练"),
    "ROLLOUT_COLLECTING": (84, "采集动作回放"),
    "VISUAL_EVALUATING": (89, "视觉效果评估"),
    "NUMERIC_EVALUATING": (93, "数值指标评估"),
    "DIAGNOSING": (97, "综合诊断"),
    "CONTINUE_TRAINING": (76, "继续训练"),
    "REVISE_REWARD": (42, "修订奖励函数"),
    "REVISE_CURRICULUM": (42, "修订课程策略"),
    "ROLLBACK": (40, "回滚候选方案"),
    "RESTART": (40, "重新开始训练"),
    "HUMAN_REVIEW": (100, "等待人工复核"),
    "COMPLETED": (100, "训练完成"),
    "FAILED": (100, "训练失败"),
}

TRAINING_STAGE_RANGES = {
    "SMOKE_TRAINING": (38, 48),
    "CANDIDATE_SCREENING": (48, 60),
    "FULL_TRAINING": (60, 84),
}

TERMINAL_STATUSES = {"completed", "review", "failed", "stopped", "interrupted"}
JOB_ID_PATTERN = re.compile(r"^[a-f0-9]{12}$")


class JobValidationError(ValueError):
    """表示界面请求中的训练参数不符合安全约束。"""


def _parse_time(value: str) -> datetime:
    """解析状态文件中的 ISO 时间，失败时返回最早的 UTC 时间。"""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=timezone.utc)


def state_presentation(state: str) -> Dict[str, Any]:
    """把内部状态映射为界面使用的进度和中文阶段名。"""
    progress, label = STATE_PRESENTATION.get(state, (0, "准备训练"))
    return {"state": state or "PENDING", "progress": progress, "stage_label": label}


def interpret_job_outcome(state: str, return_code: int, history: List[Dict[str, Any]]) -> Tuple[str, str]:
    """结合 Agent 最终状态解释进程退出，避免把正常返回的人工复核误报为训练完成。"""
    if return_code != 0 or state == "FAILED":
        return "failed", "训练异常退出" if return_code != 0 else "Agent 判定训练失败"
    if state == "COMPLETED":
        return "completed", "训练及验收均已完成"
    if state == "HUMAN_REVIEW":
        entered_training = any(item.get("state") in TRAINING_STAGE_RANGES for item in history)
        message = "训练未通过最终验收，等待人工复核" if entered_training else "训练尚未开始，任务设计等待人工复核"
        return "review", message
    if state and state != "PENDING":
        return "review", "流程已结束，但尚未达到训练完成状态：%s" % state
    # 兼容没有持久化状态的旧作业和隔离测试进程。
    return "completed", "训练进程已完成"


def training_stage_progress(state: str, detail: Dict[str, Any], context: Dict[str, Any],
                            evaluation_seeds: List[int]) -> int:
    """把单个训练进程的迭代百分比映射到当前编排阶段的总进度。"""
    if state not in TRAINING_STAGE_RANGES:
        return state_presentation(state)["progress"]
    fraction = min(1.0, max(0.0, float(detail.get("percent", 0.0)) / 100.0))
    run_name = str(detail.get("run_name", ""))
    if state == "FULL_TRAINING":
        seed_match = re.search(r"-seed-(\d+)$", run_name)
        if seed_match and evaluation_seeds:
            seed = int(seed_match.group(1))
            if seed in evaluation_seeds:
                fraction = (evaluation_seeds.index(seed) + fraction) / len(evaluation_seeds)
    else:
        candidate_match = re.search(r"candidate-(\d+)", run_name)
        candidate_count = max(1, int(context.get("candidate_count", 1)))
        if candidate_match:
            candidate_index = min(candidate_count, max(1, int(candidate_match.group(1))))
            fraction = (candidate_index - 1 + fraction) / candidate_count
    start, end = TRAINING_STAGE_RANGES[state]
    return min(end, max(start, round(start + (end - start) * fraction)))


def validate_job_request(payload: Any, allowed_robots: List[str]) -> Tuple[str, str, str]:
    """校验并规范化来自界面的任务、机器人和运行模式。"""
    if not isinstance(payload, dict):
        raise JobValidationError("请求内容必须是 JSON 对象")
    task = str(payload.get("task", "")).strip()
    robot = str(payload.get("robot", "")).strip().lower()
    mode = str(payload.get("mode", "dry-run")).strip().lower()
    if len(task) < 4:
        raise JobValidationError("请至少用 4 个字符描述想训练的动作")
    if len(task) > 2000:
        raise JobValidationError("动作描述不能超过 2000 个字符")
    if robot not in allowed_robots:
        raise JobValidationError("所选机器人不在允许列表中")
    if mode not in ("dry-run", "real"):
        raise JobValidationError("训练模式只能是 dry-run 或 real")
    return task, robot, mode


class JobManager:
    """管理界面启动的受限训练子进程、状态与日志。"""

    def __init__(self, settings: Optional[Settings] = None):
        """初始化作业目录并恢复可展示的历史作业。"""
        self.settings = settings or load_settings()
        self.root = self.settings.artifacts_path / "ui_jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._processes: Dict[str, subprocess.Popen] = {}
        self._load_jobs()

    def _load_jobs(self) -> None:
        """从磁盘加载历史界面作业，并标记无法继续跟踪的旧进程。"""
        for metadata_path in sorted(self.root.glob("*/job.json")):
            try:
                job = read_json(metadata_path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            job_id = str(job.get("job_id", ""))
            if not JOB_ID_PATTERN.fullmatch(job_id):
                continue
            if job.get("status") in ("queued", "running", "stopping"):
                job["status"] = "interrupted"
                job["finished_at"] = utc_now()
                job["message"] = "界面服务曾中断，请根据实验状态决定是否恢复训练"
                write_json(metadata_path, job)
            self._jobs[job_id] = job

    def _job_dir(self, job_id: str) -> Path:
        """返回由服务端生成且已校验的作业目录。"""
        if not JOB_ID_PATTERN.fullmatch(job_id):
            raise JobValidationError("作业编号格式无效")
        return self.root / job_id

    def _save_job(self, job: Dict[str, Any]) -> None:
        """原子保存单个作业的元数据。"""
        write_json(self._job_dir(job["job_id"]) / "job.json", job)

    def _append_log(self, job_id: str, message: str) -> None:
        """向作业日志追加一行由界面服务生成的消息。"""
        log_path = self._job_dir(job_id) / "training.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")

    def _ensure_no_active_job(self) -> None:
        """确保同一上位机服务当前没有其他训练或恢复作业占用 GPU。"""
        with self._lock:
            active = [job for job in self._jobs.values()
                      if job.get("status") in ("queued", "running", "stopping")]
        if active:
            raise JobValidationError("已有训练作业正在运行，请先等待完成或安全停止")

    def _launch_job_process(self, job: Dict[str, Any], command: List[str],
                            log_lines: List[str]) -> Dict[str, Any]:
        """创建界面作业目录并以受限参数数组启动训练或闭环恢复进程。"""
        job_id = job["job_id"]
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=False)
        self._save_job(job)
        for line in log_lines:
            self._append_log(job_id, line)
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        try:
            with (job_dir / "training.log").open("a", encoding="utf-8") as output:
                process = subprocess.Popen(
                    command, cwd=str(self.settings.agent_root), env=environment,
                    stdout=output, stderr=subprocess.STDOUT, shell=False,
                    start_new_session=True)
        except Exception as exc:
            job["status"] = "failed"
            job["finished_at"] = utc_now()
            job["message"] = "训练进程启动失败：{}".format(exc)
            self._save_job(job)
            self._append_log(job_id, "[界面] {}".format(job["message"]))
            with self._lock:
                self._jobs[job_id] = job
            raise
        job["status"] = "running"
        job["pid"] = process.pid
        job["message"] = "自动闭环正在运行" if job.get("resume_of") else "训练正在运行"
        with self._lock:
            self._jobs[job_id] = job
            self._processes[job_id] = process
            self._save_job(job)
        monitor = threading.Thread(target=self._monitor_job, args=(job_id, process), daemon=True)
        monitor.start()
        return self.get_job(job_id)

    def start_job(self, payload: Any) -> Dict[str, Any]:
        """创建作业并通过固定参数数组启动现有训练 CLI。"""
        task, robot, mode = validate_job_request(payload, self.settings.allowed_robots)
        self._ensure_no_active_job()
        job_id = uuid.uuid4().hex[:12]
        task_id = TrainingOrchestrator._task_id(task, robot)
        now = utc_now()
        job: Dict[str, Any] = {
            "job_id": job_id,
            "task_id": task_id,
            "task": task,
            "robot": robot,
            "mode": mode,
            "status": "queued",
            "created_at": now,
            "started_at": now,
            "finished_at": None,
            "return_code": None,
            "message": "训练作业正在启动",
        }
        command = [sys.executable, "-u", "-m", "rl_training_agent", "train", "--task", task,
                   "--robot", robot, "--provider", "mock" if mode == "dry-run" else "opencli"]
        if mode == "dry-run":
            command.append("--dry-run")
        return self._launch_job_process(job, command, [
            "[界面] 已创建训练作业 {}，实验编号 {}".format(job_id, task_id),
            "[界面] 模式：{}；机器人：{}".format(
                "离线演练" if mode == "dry-run" else "真实训练", robot),
        ])

    def resume_job(self, source_job_id: str) -> Dict[str, Any]:
        """从人工审核作业创建恢复作业，复用已有 checkpoint、rollout 和剩余预算。"""
        self._ensure_no_active_job()
        with self._lock:
            source = dict(self._jobs.get(source_job_id) or {})
        if not source:
            raise KeyError(source_job_id)
        if source.get("status") != "review":
            raise JobValidationError("只有等待人工复核的作业可以继续闭环")
        task_dir = self.settings.experiments_path / source["task_id"]
        summary_path = task_dir / "summary.json"
        if not summary_path.is_file() or not read_json(summary_path).get("selected_experiment"):
            raise JobValidationError("该作业没有可恢复的训练候选或 checkpoint")
        job_id = uuid.uuid4().hex[:12]
        now = utc_now()
        mode = source.get("mode", "real")
        job = {
            "job_id": job_id, "task_id": source["task_id"], "task": source["task"],
            "robot": source["robot"], "mode": mode, "resume_of": source_job_id,
            "status": "queued", "created_at": now, "started_at": now,
            "finished_at": None, "return_code": None, "message": "正在恢复自动闭环",
        }
        command = [sys.executable, "-u", "-m", "rl_training_agent", "resume",
                   "--task-id", source["task_id"], "--provider",
                   "mock" if mode == "dry-run" else "opencli"]
        if mode == "dry-run":
            command.append("--dry-run")
        return self._launch_job_process(job, command, [
            "[界面] 正在从作业 {} 恢复实验 {}".format(source_job_id, source["task_id"]),
            "[界面] 将复用已保存的奖励版本、checkpoint 和完整 rollout，不重复初始训练",
        ])

    def _monitor_job(self, job_id: str, process: subprocess.Popen) -> None:
        """等待训练进程退出，并把退出结果同步到作业元数据。"""
        return_code = process.wait()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job.get("status") not in ("stopped", "stopping"):
                state = self._state_for_job(job)
                job["status"], job["message"] = interpret_job_outcome(
                    str(state.get("state", "PENDING")), return_code, state.get("history", []))
            else:
                job["status"] = "stopped"
                job["message"] = "训练已由用户停止"
            job["return_code"] = return_code
            job["finished_at"] = utc_now()
            self._processes.pop(job_id, None)
            self._save_job(job)
        self._append_log(job_id, "[界面] 训练进程结束，退出码：{}".format(return_code))

    def stop_job(self, job_id: str) -> Dict[str, Any]:
        """停止由当前界面服务启动且仍在运行的训练进程组。"""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            process = self._processes.get(job_id)
            if process is None or process.poll() is not None:
                raise JobValidationError("该训练作业当前不可停止")
            job["status"] = "stopping"
            job["message"] = "正在停止训练"
            self._save_job(job)
        self._append_log(job_id, "[界面] 已请求停止训练，正在终止训练进程组")
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (AttributeError, ProcessLookupError, PermissionError):
            process.terminate()
        terminator = threading.Thread(target=self._ensure_stopped, args=(job_id, process), daemon=True)
        terminator.start()
        return self.get_job(job_id)

    def _ensure_stopped(self, job_id: str, process: subprocess.Popen) -> None:
        """等待正常终止，并在超时后强制结束失去响应的训练进程组。"""
        try:
            process.wait(timeout=8)
            return
        except subprocess.TimeoutExpired:
            self._append_log(job_id, "[界面] 正常停止等待超时，正在强制终止训练进程组")
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (AttributeError, ProcessLookupError, PermissionError):
            process.kill()

    def _state_for_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """读取当前作业对应且时间有效的持久化训练状态。"""
        state_path = self.settings.experiments_path / job["task_id"] / "state.json"
        if not state_path.is_file():
            return {"state": "PENDING", "updated_at": None, "history": []}
        try:
            state = read_json(state_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return {"state": "PENDING", "updated_at": None, "history": []}
        if _parse_time(str(state.get("updated_at", ""))) < _parse_time(job["started_at"]):
            return {"state": "PENDING", "updated_at": None, "history": []}
        return state

    def _training_progress_for_job(self, job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """读取当前作业产生的最新候选训练迭代快照。"""
        candidates_root = self.settings.experiments_path / job["task_id"] / "candidates"
        snapshots = list(candidates_root.glob("*/training_progress.json")) if candidates_root.is_dir() else []
        snapshots.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        for path in snapshots:
            try:
                detail = read_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if _parse_time(str(detail.get("updated_at", ""))) < _parse_time(job["started_at"]):
                continue
            return detail
        return None

    def _loop_status_for_job(self, job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """读取当前作业的闭环轮次、奖励版本、决策和剩余预算。"""
        path = self.settings.experiments_path / job["task_id"] / "loop_status.json"
        if not path.is_file():
            return None
        try:
            detail = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        if _parse_time(str(detail.get("updated_at", ""))) < _parse_time(job["started_at"]):
            return None
        return detail

    def _can_resume_job(self, job: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """判断人工审核作业是否具备候选 checkpoint 且当前没有其他活动作业。"""
        if job.get("status") != "review" or state.get("state") != "HUMAN_REVIEW":
            return False
        with self._lock:
            if any(item.get("status") in ("queued", "running", "stopping")
                   for item in self._jobs.values()):
                return False
        summary_path = self.settings.experiments_path / job["task_id"] / "summary.json"
        if not summary_path.is_file():
            return False
        try:
            summary = read_json(summary_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        experiment_id = summary.get("selected_experiment")
        return bool(experiment_id and (
            self.settings.experiments_path / job["task_id"] / "candidates" /
            str(experiment_id) / "reward_plan.json").is_file())

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """返回附带实时状态、进度与阶段历史的界面作业。"""
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = dict(self._jobs[job_id])
        state = self._state_for_job(job)
        if job["status"] == "completed" and state.get("state") not in (None, "", "PENDING", "COMPLETED"):
            # 动态纠正旧版本仅按退出码保存为 completed 的历史作业。
            job["status"], job["message"] = interpret_job_outcome(
                str(state.get("state")), int(job.get("return_code") or 0), state.get("history", []))
        presentation = state_presentation(str(state.get("state", "PENDING")))
        training_detail = self._training_progress_for_job(job)
        loop_detail = self._loop_status_for_job(job)
        if training_detail and presentation["state"] in TRAINING_STAGE_RANGES:
            presentation["progress"] = training_stage_progress(
                presentation["state"], training_detail, state.get("context", {}),
                self.settings.evaluation_seeds)
            presentation["stage_label"] = "{} · {} · {}/{}".format(
                presentation["stage_label"], training_detail.get("run_name", "训练进程"),
                training_detail.get("iteration", 0), training_detail.get("total_iterations", 0))
        if loop_detail:
            loop_label = "闭环第{}轮 · 奖励 v{}".format(
                loop_detail.get("round", 1), loop_detail.get("reward_version", 1))
            if loop_detail.get("decision"):
                loop_label += " · 决策 {}".format(loop_detail["decision"])
            presentation["stage_label"] = "{} · {}".format(presentation["stage_label"], loop_label)
        if job["status"] == "failed" and presentation["state"] == "PENDING":
            presentation = state_presentation("FAILED")
        if job["status"] == "review" and presentation["state"] == "PENDING":
            presentation = state_presentation("HUMAN_REVIEW")
        if job["status"] == "completed" and presentation["state"] == "PENDING":
            presentation = state_presentation("COMPLETED")
        job.update(presentation)
        job["state_updated_at"] = state.get("updated_at")
        job["history"] = state.get("history", [])
        job["training_detail"] = training_detail
        job["loop_detail"] = loop_detail
        job["can_stop"] = job["status"] in ("running", "stopping") and job_id in self._processes
        job["can_resume"] = self._can_resume_job(job, state)
        return job

    def list_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """按创建时间倒序返回最近的界面训练作业。"""
        with self._lock:
            job_ids = sorted(self._jobs, key=lambda item: self._jobs[item].get("created_at", ""), reverse=True)
        return [self.get_job(job_id) for job_id in job_ids[:limit]]

    def list_experiments(self, limit: int = 12) -> List[Dict[str, Any]]:
        """扫描实验目录并返回可用于最近实验卡片的摘要。"""
        experiments: List[Dict[str, Any]] = []
        if not self.settings.experiments_path.is_dir():
            return experiments
        for state_path in self.settings.experiments_path.glob("task-*/state.json"):
            try:
                state = read_json(state_path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            task_dir = state_path.parent
            request_path = task_dir / "task_request.txt"
            spec_path = task_dir / "task_spec.json"
            task = request_path.read_text(encoding="utf-8").strip() if request_path.is_file() else "未记录动作描述"
            robot = "未知"
            if spec_path.is_file():
                try:
                    robot = str(read_json(spec_path).get("robot", robot))
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            presentation = state_presentation(str(state.get("state", "PENDING")))
            experiments.append({
                "task_id": task_dir.name,
                "task": task,
                "robot": robot,
                "updated_at": state.get("updated_at"),
                **presentation,
            })
        experiments.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        return experiments[:limit]

    def read_log(self, job_id: str, offset: int = 0, limit: int = 65536) -> Dict[str, Any]:
        """从字节偏移量开始增量读取训练日志。"""
        self._job_dir(job_id)
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
        log_path = self._job_dir(job_id) / "training.log"
        if not log_path.is_file():
            return {"text": "", "next_offset": 0}
        size = log_path.stat().st_size
        safe_offset = max(0, min(int(offset), size))
        with log_path.open("rb") as handle:
            handle.seek(safe_offset)
            chunk = handle.read(limit)
        return {"text": chunk.decode("utf-8", errors="replace"), "next_offset": safe_offset + len(chunk)}

    def config_payload(self) -> Dict[str, Any]:
        """返回前端初始化所需的机器人列表和运行能力信息。"""
        robots = [robot for robot in ROBOT_CATALOG if robot["id"] in self.settings.allowed_robots]
        return {
            "robots": robots,
            "default_robot": self.settings.default_robot,
            "system": {
                "training_project_ready": self.settings.training_root.is_dir(),
                "training_entry_ready": (self.settings.training_root / "legged_gym" / "scripts" / "train.py").is_file(),
                "scope": "simulation_only",
            },
            "modes": [
                {"id": "dry-run", "name": "离线演练", "description": "使用模拟推理与模拟训练，适合先验证完整流程"},
                {"id": "real", "name": "真实训练", "description": "使用 OpenCLI 与 GPU 仿真执行正式训练"},
            ],
        }


class TrainingUIHandler(BaseHTTPRequestHandler):
    """提供静态界面与训练作业 JSON API 的 HTTP 请求处理器。"""

    manager: JobManager
    static_root: Path
    server_version = "RLTrainingUI/1.0"

    def _send_headers(self, status: int, content_type: str, length: int) -> None:
        """发送带有本地界面安全策略的通用响应头。"""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'")
        self.end_headers()

    def _send_json(self, status: int, value: Any) -> None:
        """以 UTF-8 JSON 返回 API 响应。"""
        body = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
        self._send_headers(status, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str) -> None:
        """返回格式一致且不暴露调用栈的 API 错误。"""
        self._send_json(status, {"error": message})

    def _send_asset(self, filename: str) -> None:
        """仅从固定白名单路由发送内置静态资源。"""
        asset = self.static_root / filename
        if not asset.is_file():
            self._send_error_json(404, "资源不存在")
            return
        body = asset.read_bytes()
        content_type = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type == "application/javascript":
            content_type += "; charset=utf-8"
        self._send_headers(200, content_type, len(body))
        self.wfile.write(body)

    def _read_json_body(self) -> Any:
        """在固定大小限制内读取并解析请求 JSON。"""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise JobValidationError("Content-Length 无效") from exc
        if length <= 0 or length > 65536:
            raise JobValidationError("请求内容为空或超过 64 KiB")
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise JobValidationError("请求不是有效的 UTF-8 JSON") from exc

    def do_GET(self) -> None:
        """处理首页、静态资源和只读 API。"""
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send_asset("index.html")
            return
        if parsed.path in ("/app.css", "/app.js"):
            self._send_asset(parsed.path.lstrip("/"))
            return
        if parsed.path == "/api/config":
            self._send_json(200, self.manager.config_payload())
            return
        if parsed.path == "/api/jobs":
            self._send_json(200, {"jobs": self.manager.list_jobs(), "experiments": self.manager.list_experiments()})
            return
        match = re.fullmatch(r"/api/jobs/([a-f0-9]{12})", parsed.path)
        log_match = re.fullmatch(r"/api/jobs/([a-f0-9]{12})/logs", parsed.path)
        try:
            if match:
                self._send_json(200, self.manager.get_job(match.group(1)))
                return
            if log_match:
                query = parse_qs(parsed.query)
                offset = int(query.get("offset", ["0"])[0])
                self._send_json(200, self.manager.read_log(log_match.group(1), offset))
                return
        except (KeyError, ValueError):
            self._send_error_json(404, "作业不存在或日志偏移量无效")
            return
        self._send_error_json(404, "接口不存在")

    def do_POST(self) -> None:
        """处理创建训练、恢复闭环与停止训练请求。"""
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/jobs":
                job = self.manager.start_job(self._read_json_body())
                self._send_json(201, job)
                return
            match = re.fullmatch(r"/api/jobs/([a-f0-9]{12})/stop", parsed.path)
            if match:
                self._send_json(200, self.manager.stop_job(match.group(1)))
                return
            resume_match = re.fullmatch(r"/api/jobs/([a-f0-9]{12})/resume", parsed.path)
            if resume_match:
                self._send_json(201, self.manager.resume_job(resume_match.group(1)))
                return
        except JobValidationError as exc:
            self._send_error_json(400, str(exc))
            return
        except KeyError:
            self._send_error_json(404, "作业不存在")
            return
        except Exception as exc:
            self._send_error_json(500, "操作失败：{}".format(exc))
            return
        self._send_error_json(404, "接口不存在")

    def log_message(self, format_string: str, *args: Any) -> None:
        """使用简洁格式记录本地 HTTP 访问日志。"""
        sys.stderr.write("[界面] {} - {}\n".format(self.address_string(), format_string % args))


def create_server(host: str = "127.0.0.1", port: int = 8765,
                  manager: Optional[JobManager] = None) -> ThreadingHTTPServer:
    """创建可供 CLI 和测试复用的多线程本地 HTTP 服务。"""
    selected_manager = manager or JobManager()
    static_root = Path(__file__).resolve().parent / "web"

    class BoundTrainingUIHandler(TrainingUIHandler):
        """把当前服务的作业管理器与静态资源目录绑定到处理器。"""

    BoundTrainingUIHandler.manager = selected_manager
    BoundTrainingUIHandler.static_root = static_root
    return ThreadingHTTPServer((host, port), BoundTrainingUIHandler)


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> int:
    """启动训练界面，在需要时打开浏览器，并持续服务到用户中断。"""
    server = create_server(host, port)
    actual_port = server.server_address[1]
    display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = "http://{}:{}/".format(display_host, actual_port)
    print("强化学习训练界面已启动：{}".format(url))
    print("按 Ctrl+C 关闭界面；请在页面内停止仍在运行的训练作业。")
    if open_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.4)
    except KeyboardInterrupt:
        print("\n界面服务已关闭。")
    finally:
        server.server_close()
    return 0
