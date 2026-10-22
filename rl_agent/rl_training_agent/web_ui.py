from __future__ import annotations

import json
import hmac
import ipaddress
import mimetypes
import os
import re
import signal
import secrets
import socket
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .environment.project_adapter import UnitreeProjectAdapter
from .orchestration.orchestrator import TrainingOrchestrator
from .memory.store import LongTermMemoryStore
from .observability import JsonlEventRecorder, sample_gpu
from .providers.bailian_glm import BailianGLMProvider
from .settings import Settings, load_settings
from .training.process_manager import (capture_process_group, registered_child_groups,
                                      signal_process_group, signal_registered_groups, stop_process_group,
                                      wait_registered_groups)
from .utils.io import read_json, utc_now, write_json
from .utils.paths import ensure_within, relative_display


ROBOT_CATALOG = [
    {"id": "go2", "name": "Unitree Go2", "kind": "四足机器人", "mark": "G2"},
    {"id": "g1", "name": "Unitree G1", "kind": "人形机器人", "mark": "G1"},
    {"id": "h1", "name": "Unitree H1", "kind": "人形机器人", "mark": "H1"},
    {"id": "h1_2", "name": "Unitree H1-2", "kind": "人形机器人", "mark": "H2"},
]

STATE_PRESENTATION: Dict[str, Tuple[int, str]] = {
    "RECEIVED": (3, "接收任务"),
    "ENVIRONMENT_INSPECTED": (10, "检查训练环境"),
    "TASK_UNDERSTANDING": (13, "GPT 理解动作目标"),
    "TASK_FEASIBILITY_CHECK": (15, "预检动作可行性"),
    "MOTION_PROTOTYPE_GENERATING": (16, "生成低维动作原型"),
    "STATIC_MOTION_VALIDATION": (17, "静态物理预检"),
    "DYNAMIC_MOTION_VALIDATION": (18, "动态运动预检"),
    "RAG_RETRIEVING": (19, "检索训练经验"),
    "CONTEXT_BUILDING": (20, "构建任务上下文"),
    "PROMPT_COMPILING": (21, "编译奖励提示词"),
    "REWARD_DESIGNING": (22, "设计奖励"),
    "REWARD_REVIEWING": (24, "审查奖励风险"),
    "TASK_DESIGNED": (23, "完成任务设计"),
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
    "MEMORY_CURATING": (99, "整理长期记忆"),
    "CONTINUE_TRAINING": (76, "继续训练"),
    "REVISE_REWARD": (42, "修订奖励函数"),
    "REVISE_CURRICULUM": (42, "修订课程策略"),
    "ROLLBACK": (40, "回滚候选方案"),
    "RESTART": (40, "重新开始训练"),
    "HUMAN_REVIEW": (0, "等待人工复核"),
    "COMPLETED": (100, "训练完成"),
    "FAILED": (0, "训练失败"),
}

TRAINING_STAGE_RANGES = {
    "SMOKE_TRAINING": (38, 48),
    "CANDIDATE_SCREENING": (48, 60),
    "FULL_TRAINING": (60, 84),
}

TERMINAL_STATUSES = {"completed", "review", "failed", "stopped", "interrupted"}
JOB_ID_PATTERN = re.compile(r"^[a-f0-9]{12}$")
TASK_ID_PATTERN = re.compile(r"^task-[A-Za-z0-9_-]{1,128}$")
SESSION_COOKIE = "rl_ui_session"


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


def current_run_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从最新运行边界开始展示历史，防止同任务的旧训练污染新作业。"""
    for index in range(len(history) - 1, -1, -1):
        if history[index].get("state") == "RECEIVED" and history[index].get("reason") == "new_run":
            return history[index:]
    return history


def last_working_state(state: str, history: List[Dict[str, Any]]) -> str:
    """找到本次运行实际到达的工作阶段，终止状态不代表流程完成。"""
    if state not in ("HUMAN_REVIEW", "FAILED", "COMPLETED", "PENDING"):
        return state
    for entry in reversed(current_run_history(history)):
        value = entry.get("state")
        if value in STATE_PRESENTATION and value not in ("HUMAN_REVIEW", "FAILED", "COMPLETED"):
            return value
    return "PENDING"


def state_presentation(state: str, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """把阶段映射为流程进度；人工复核或失败保持实际阶段，不显示完成。"""
    progress, label = STATE_PRESENTATION.get(state, (0, "准备训练"))
    if state in ("HUMAN_REVIEW", "FAILED"):
        active = last_working_state(state, history or [])
        progress, active_label = STATE_PRESENTATION.get(active, (0, ""))
        if active_label:
            label += " · " + active_label
    return {"state": state or "PENDING", "progress": progress, "stage_label": label}


def stage_track_presentation(state: str, history: List[Dict[str, Any]], accepted: bool) -> Dict[str, str]:
    """根据本次状态历史标注已完成、当前、阻塞和未执行步骤，不依赖百分比。"""
    groups = {
        "environment": {"RECEIVED", "ENVIRONMENT_INSPECTED"},
        "intent": {"TASK_UNDERSTANDING", "TASK_FEASIBILITY_CHECK", "MOTION_PROTOTYPE_GENERATING",
                   "STATIC_MOTION_VALIDATION", "DYNAMIC_MOTION_VALIDATION"},
        "reward": {"RAG_RETRIEVING", "CONTEXT_BUILDING", "PROMPT_COMPILING", "REWARD_DESIGNING",
                   "TASK_DESIGNED", "REWARD_REVIEWING", "REWARD_CANDIDATES_CREATED", "CONFIGS_COMPILED", "VALIDATED"},
        "smoke": {"SMOKE_TRAINING", "CANDIDATE_SCREENING"},
        "full": {"FULL_TRAINING", "CONTINUE_TRAINING", "REVISE_REWARD", "REVISE_CURRICULUM", "ROLLBACK", "RESTART"},
        "visual": {"ROLLOUT_COLLECTING", "VISUAL_EVALUATING"},
        "acceptance": {"NUMERIC_EVALUATING", "DIAGNOSING", "MEMORY_CURATING"},
    }
    completion = {
        "environment": {"TASK_UNDERSTANDING"},
        "intent": {"RAG_RETRIEVING", "CONTEXT_BUILDING"},
        "reward": {"VALIDATED", "SMOKE_TRAINING"},
        "smoke": {"FULL_TRAINING"},
        "full": {"ROLLOUT_COLLECTING"},
        "visual": {"NUMERIC_EVALUATING"},
        "acceptance": set(),
    }
    reached = {entry.get("state") for entry in current_run_history(history)} | {state}
    active = last_working_state(state, history)
    blocked = state in ("HUMAN_REVIEW", "FAILED")
    result = {name: "done" if reached & completion[name] else "pending" for name in groups}
    for name, states in groups.items():
        if (active in states and state not in ("COMPLETED", "PENDING") and
                not (name == "reward" and active == "VALIDATED")):
            result[name] = "blocked" if blocked else "active"
    result["acceptance"] = "done" if accepted else result["acceptance"]
    return result


def interpret_job_outcome(state: str, return_code: int, history: List[Dict[str, Any]]) -> Tuple[str, str]:
    """结合 Agent 最终状态解释进程退出，避免把正常返回的人工复核误报为训练完成。"""
    if return_code != 0 or state == "FAILED":
        return "failed", "训练异常退出" if return_code != 0 else "Agent 判定训练失败"
    if state == "COMPLETED":
        return "completed", "训练及验收均已完成"
    if state == "HUMAN_REVIEW":
        entered_training = any(item.get("state") in TRAINING_STAGE_RANGES for item in current_run_history(history))
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
        self.root = ensure_within(self.settings.artifacts_path / "ui_jobs", self.settings.artifacts_path)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._processes: Dict[str, subprocess.Popen] = {}
        self._job_control_tokens: Dict[str, str] = {}
        self._job_child_groups: Dict[str, List[dict]] = {}
        self._playback_process: Optional[subprocess.Popen] = None
        self._playback_task_id: Optional[str] = None
        self._playback_kind: Optional[str] = None
        self._playback_stopping = False
        self._closing = False
        self._load_jobs()
        self._dispatch_queued()

    def _load_jobs(self) -> None:
        """从磁盘加载历史界面作业，并标记无法继续跟踪的旧进程。"""
        for metadata_path in sorted(self.root.glob("*/job.json")):
            try:
                metadata_path = ensure_within(metadata_path, self.root)
                job = read_json(metadata_path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(job, dict):
                continue
            job_id = str(job.get("job_id", ""))
            if (not JOB_ID_PATTERN.fullmatch(job_id) or metadata_path.parent.name != job_id or
                    not TASK_ID_PATTERN.fullmatch(str(job.get("task_id", "")))):
                continue
            try:
                validate_job_request(job, self.settings.allowed_robots)
            except JobValidationError:
                continue
            if not {"status", "created_at", "started_at", "task", "robot"}.issubset(job):
                continue
            if job.get("status") in ("running", "stopping"):
                job["status"] = "interrupted"
                job["finished_at"] = utc_now()
                job["message"] = "界面服务曾中断，请根据实验状态决定是否恢复训练"
                write_json(metadata_path, job)
            elif job.get("status") == "queued" and not job.get("command"):
                job["status"] = "interrupted"
                job["finished_at"] = utc_now()
                job["message"] = "旧队列记录缺少安全命令，不能自动恢复"
                write_json(metadata_path, job)
            elif job.get("status") == "queued":
                job["gpu_id"] = None
                write_json(metadata_path, job)
            self._jobs[job_id] = job

    def _job_dir(self, job_id: str) -> Path:
        """返回由服务端生成且已校验的作业目录。"""
        if not JOB_ID_PATTERN.fullmatch(job_id):
            raise JobValidationError("作业编号格式无效")
        try:
            return ensure_within(self.root / job_id, self.root)
        except ValueError as exc:
            raise JobValidationError("作业路径越过允许目录") from exc

    def _job_artifact(self, job_id: str, filename: str) -> Path:
        """校验作业文件及其符号链接目标，避免越界读取或写入。"""
        return ensure_within(self._job_dir(job_id) / filename, self._job_dir(job_id))

    def _task_dir(self, task_id: str) -> Path:
        """校验历史实验标识和符号链接，拒绝实验目录逃逸。"""
        if not TASK_ID_PATTERN.fullmatch(str(task_id)):
            raise JobValidationError("实验编号格式无效")
        return ensure_within(self.settings.experiments_path / task_id, self.settings.experiments_path)

    def _task_artifact(self, job: Dict[str, Any], filename: str) -> Path:
        """解析仅位于该实验目录中的状态和结果文件。"""
        task_dir = self._task_dir(job["task_id"])
        return ensure_within(task_dir / filename, task_dir)

    def _assert_accepting(self) -> None:
        """关闭开始后禁止启动或恢复任何新作业与 Viewer。"""
        if self._closing:
            raise JobValidationError("界面正在关闭，不能启动新任务")

    def _viewer_running(self) -> bool:
        """在同一锁内检查 Viewer 租约，避免训练绕过资源互斥。"""
        return self._playback_process is not None and (self._playback_stopping or
                                                       self._playback_process.poll() is None)

    def _save_job(self, job: Dict[str, Any]) -> None:
        """原子保存单个作业的元数据。"""
        write_json(self._job_artifact(job["job_id"], "job.json"), job)

    def _append_log(self, job_id: str, message: str) -> None:
        """向作业日志追加一行由界面服务生成的消息。"""
        log_path = self._job_artifact(job_id, "training.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")

    def _allocate_gpu(self) -> Optional[int]:
        """从配置 GPU 池中分配未被活动作业占用的设备，无空闲时返回 None。"""
        with self._lock:
            if self._closing or self._viewer_running():
                return None
            active = [job for job in self._jobs.values()
                      if job.get("status") in ("queued", "running", "stopping")]
        used = {int(job["gpu_id"]) for job in active if job.get("gpu_id") is not None}
        available = [gpu for gpu in self.settings.gpu_ids if gpu not in used]
        legacy_unassigned = sum(
            job.get("gpu_id") is None and job.get("status") in ("running", "stopping")
            for job in active)
        available = available[legacy_unassigned:]
        return available[0] if available else None

    def _launch_job_process(self, job: Dict[str, Any], command: List[str],
                            log_lines: List[str]) -> Dict[str, Any]:
        """持久化新作业；有空闲 GPU 时启动，否则保留在等待队列。"""
        job_id = job["job_id"]
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=False)
        job["command"] = list(command)
        self._save_job(job)
        for line in log_lines:
            self._append_log(job_id, line)
        with self._lock:
            self._jobs[job_id] = job
        if job.get("gpu_id") is None:
            job["message"] = "等待可用 GPU"
            self._save_job(job)
            self._append_log(job_id, "[队列] 当前 GPU 均被占用，作业已持久化等待")
            return self.get_job(job_id)
        return self._start_persisted_job(job_id)

    def _start_persisted_job(self, job_id: str) -> Dict[str, Any]:
        """在资源租约锁内启动持久作业，关闭时不再派发。"""
        with self._lock:
            self._assert_accepting()
            return self._start_persisted_job_locked(job_id)

    def _start_persisted_job_locked(self, job_id: str) -> Dict[str, Any]:
        """启动已经持久化且获得 GPU 租约的队列作业。"""
        with self._lock:
            job = self._jobs[job_id]
            command = list(job["command"])
        prefix = [sys.executable, "-u", "-m", "rl_training_agent"]
        task, robot, mode = validate_job_request(job, self.settings.allowed_robots)
        if job.get("resume_of"):
            self._task_dir(job["task_id"])
            expected = prefix + ["resume", "--task-id", job["task_id"], "--provider",
                                 "mock" if mode == "dry-run" else self.settings.provider]
        else:
            expected = prefix + ["train", "--task", task, "--robot", robot, "--provider",
                                 "mock" if mode == "dry-run" else self.settings.provider]
        if mode == "dry-run":
            expected.append("--dry-run")
        if command != expected:
            job["status"] = "failed"
            job["finished_at"] = utc_now()
            job["message"] = "持久队列命令未通过安全白名单"
            self._save_job(job)
            raise JobValidationError(job["message"])
        job_dir = self._job_dir(job_id)
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        control_token = secrets.token_hex(32)
        environment["RL_AGENT_JOB_CONTROL_FILE"] = str(self._job_artifact(job_id, "job_control.jsonl"))
        environment["RL_AGENT_JOB_CONTROL_TOKEN"] = control_token
        if job.get("mode") != "dry-run":
            environment["CUDA_VISIBLE_DEVICES"] = str(job["gpu_id"])
            JsonlEventRecorder(job_dir / "events.jsonl").emit(
                "gpu_sample", {"phase": "job_start", **sample_gpu(int(job["gpu_id"]))})
        try:
            with self._job_artifact(job_id, "training.log").open("a", encoding="utf-8") as output:
                process = subprocess.Popen(
                    command, cwd=str(self.settings.agent_root), env=environment,
                    stdout=output, stderr=subprocess.STDOUT, shell=False,
                    start_new_session=True)
                capture_process_group(process)
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
        job["started_at"] = utc_now()
        job["pid"] = process.pid
        job["message"] = "自动闭环正在运行" if job.get("resume_of") else "训练正在运行"
        with self._lock:
            self._jobs[job_id] = job
            self._processes[job_id] = process
            self._job_control_tokens[job_id] = control_token
            self._save_job(job)
        monitor = threading.Thread(target=self._monitor_job, args=(job_id, process), daemon=True)
        monitor.start()
        return self.get_job(job_id)

    def _dispatch_queued(self) -> None:
        """原子派发队列；关闭与 Viewer 租约期间不启动训练。"""
        with self._lock:
            if self._closing or self._viewer_running():
                return
            self._dispatch_queued_locked()

    def _dispatch_queued_locked(self) -> None:
        """按创建时间顺序为等待作业分配空闲 GPU 并启动。"""
        while True:
            gpu_id = self._allocate_gpu()
            if gpu_id is None:
                return
            with self._lock:
                queued = sorted(
                    (job for job in self._jobs.values()
                     if job.get("status") == "queued" and job.get("gpu_id") is None),
                    key=lambda item: item.get("created_at", ""))
                if not queued:
                    return
                job = queued[0]
                job["gpu_id"] = gpu_id
                self._save_job(job)
            self._append_log(job["job_id"], "[队列] 已分配 GPU %s，开始运行" % gpu_id)
            self._start_persisted_job(job["job_id"])

    def start_job(self, payload: Any) -> Dict[str, Any]:
        """把参数验证、GPU 预留和进程登记作为单个原子启动操作。"""
        with self._lock:
            self._assert_accepting()
            if self._viewer_running():
                raise JobValidationError("仿真 Viewer 正在运行，请先停止播放")
            return self._start_job_locked(payload)

    def _start_job_locked(self, payload: Any) -> Dict[str, Any]:
        """创建作业并通过固定参数数组启动现有训练 CLI。"""
        task, robot, mode = validate_job_request(payload, self.settings.allowed_robots)
        gpu_id = self._allocate_gpu()
        job_id = uuid.uuid4().hex[:12]
        task_id = TrainingOrchestrator._task_id(task, robot)
        if any(item.get("task_id") == task_id and item.get("status") in ("queued", "running", "stopping")
               for item in self._jobs.values()):
            raise JobValidationError("同一实验已有活动作业，请等待或停止后再提交")
        now = utc_now()
        job: Dict[str, Any] = {
            "job_id": job_id,
            "task_id": task_id,
            "task": task,
            "robot": robot,
            "mode": mode,
            "status": "queued",
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "return_code": None,
            "message": "训练作业正在启动",
            "gpu_id": gpu_id,
        }
        command = [sys.executable, "-u", "-m", "rl_training_agent", "train", "--task", task,
                   "--robot", robot, "--provider",
                   "mock" if mode == "dry-run" else self.settings.provider]
        if mode == "dry-run":
            command.append("--dry-run")
        return self._launch_job_process(job, command, [
            "[界面] 已创建训练作业 {}，实验编号 {}".format(job_id, task_id),
            "[界面] 模式：{}；机器人：{}".format(
                "离线演练" if mode == "dry-run" else "真实训练", robot),
        ])

    def resume_job(self, source_job_id: str) -> Dict[str, Any]:
        """在同一资源锁内校验恢复条件并预留 GPU，防止恢复竞争。"""
        with self._lock:
            self._assert_accepting()
            if self._viewer_running():
                raise JobValidationError("仿真 Viewer 正在运行，请先停止播放")
            return self._resume_job_locked(source_job_id)

    def _resume_job_locked(self, source_job_id: str) -> Dict[str, Any]:
        """从人工审核作业创建恢复作业，复用已有 checkpoint、rollout 和剩余预算。"""
        gpu_id = self._allocate_gpu()
        with self._lock:
            source = dict(self._jobs.get(source_job_id) or {})
        if not source:
            raise KeyError(source_job_id)
        if source.get("status") != "review":
            raise JobValidationError("只有等待人工复核的作业可以继续闭环")
        if any(item.get("task_id") == source.get("task_id") and item.get("status") in
               ("queued", "running", "stopping") for item in self._jobs.values()):
            raise JobValidationError("该实验已有活动恢复作业")
        summary_path = self._task_artifact(source, "summary.json")
        if not summary_path.is_file() or not read_json(summary_path).get("selected_experiment"):
            raise JobValidationError("该作业没有可恢复的训练候选或 checkpoint")
        budget_detail = self._budget_for_job(source, self._state_for_job(source))
        if not budget_detail["valid"] or budget_detail["exhausted"]:
            raise JobValidationError(budget_detail["resume_block_reason"])
        summary = read_json(summary_path)
        if summary.get("resume_allowed") is False or summary.get("recoverable") is False:
            raise JobValidationError("后端已禁止该实验恢复训练，请查看已保存结果与阻塞原因")
        job_id = uuid.uuid4().hex[:12]
        now = utc_now()
        mode = source.get("mode", "real")
        job = {
            "job_id": job_id, "task_id": source["task_id"], "task": source["task"],
            "robot": source["robot"], "mode": mode, "resume_of": source_job_id,
            "status": "queued", "created_at": now, "started_at": None,
            "finished_at": None, "return_code": None, "message": "正在恢复自动闭环",
            "gpu_id": gpu_id,
        }
        command = [sys.executable, "-u", "-m", "rl_training_agent", "resume",
                   "--task-id", source["task_id"], "--provider",
                   "mock" if mode == "dry-run" else self.settings.provider]
        if mode == "dry-run":
            command.append("--dry-run")
        return self._launch_job_process(job, command, [
            "[界面] 正在从作业 {} 恢复实验 {}".format(source_job_id, source["task_id"]),
            "[界面] 将复用已保存的奖励版本、checkpoint 和完整 rollout，不重复初始训练",
        ])

    def _monitor_job(self, job_id: str, process: subprocess.Popen) -> None:
        """等待训练进程退出，并把退出结果同步到作业元数据。"""
        while True:
            capture_process_group(process)
            self._registered_children(job_id, process)
            try:
                return_code = process.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                continue
            except TypeError:
                # 保持旧测试桩和无 timeout 参数的兼容实现可用。
                return_code = process.wait()
                break
        stop_process_group(process, grace_seconds=0.0)
        # CLI 被 SIGKILL 或崩溃时仍清理签名登记的真实训练会话，再释放 GPU。
        groups = self._registered_children(job_id, process)
        own_group = getattr(process, "_rl_group_identity", None)
        if own_group is not None:
            groups.append(own_group)
        signal_registered_groups(groups, signal.SIGTERM)
        signal_registered_groups(groups, signal.SIGKILL)
        while not wait_registered_groups(groups, timeout=1.0):
            with self._lock:
                job = self._jobs.get(job_id)
                if job:
                    job.update(status="stopping", message="等待真实训练子进程退出，GPU 租约尚未释放")
                    self._save_job(job)
            signal_registered_groups(groups, signal.SIGKILL)
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
            self._job_control_tokens.pop(job_id, None)
            self._job_child_groups.pop(job_id, None)
            self._save_job(job)
        self._append_log(job_id, "[界面] 训练进程结束，退出码：{}".format(return_code))
        gpu_id = job.get("gpu_id")
        if gpu_id is not None and job.get("mode") != "dry-run":
            JsonlEventRecorder(self._job_dir(job_id) / "events.jsonl").emit(
                "gpu_sample", {"phase": "job_end", **sample_gpu(int(gpu_id))})
        self._dispatch_queued()

    def stop_job(self, job_id: str) -> Dict[str, Any]:
        """停止由当前界面服务启动且仍在运行的训练进程组。"""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            process = self._processes.get(job_id)
            if job.get("status") == "queued" and process is None:
                job["status"] = "stopped"
                job["finished_at"] = utc_now()
                job["message"] = "等待中的作业已取消"
                self._save_job(job)
                return self.get_job(job_id)
            if process is None or (process.poll() is not None and not self._registered_children(job_id, process)):
                raise JobValidationError("该训练作业当前不可停止")
            job["status"] = "stopping"
            job["message"] = "正在停止训练"
            self._save_job(job)
        self._append_log(job_id, "[界面] 已请求停止训练，正在终止训练进程组")
        self._stop_registered_children(job_id, process, signal.SIGTERM)
        signal_process_group(process, signal.SIGTERM)
        terminator = threading.Thread(target=self._ensure_stopped, args=(job_id, process), daemon=True)
        terminator.start()
        return self.get_job(job_id)

    def _ensure_stopped(self, job_id: str, process: subprocess.Popen) -> None:
        """等待正常终止，并在超时后强制结束失去响应的训练进程组。"""
        groups = self._registered_children(job_id, process)
        stop_process_group(process, grace_seconds=10.0)
        signal_registered_groups(groups, signal.SIGKILL)

    def _registered_children(self, job_id: str, process: subprocess.Popen) -> List[dict]:
        """从受限文件解析该作业本次启动签名的真实训练进程身份。"""
        with self._lock:
            token = self._job_control_tokens.get(job_id)
        if not token:
            return []
        try:
            current = registered_child_groups(self._job_artifact(job_id, "job_control.jsonl"), token, process.pid)
        except ValueError:
            current = []
        with self._lock:
            known = self._job_child_groups.setdefault(job_id, [])
            identities = {(item["pid"], item["start_ticks"]) for item in current}
            known[:] = [item for item in known if (item["pid"], item["start_ticks"]) not in identities] + current
            return list(known)

    def _stop_registered_children(self, job_id: str, process: subprocess.Popen, sig: int) -> None:
        """即使 CLI 已崩溃也只终止经过本次会话签名验证的真实子组。"""
        signal_registered_groups(self._registered_children(job_id, process), sig)

    def _state_for_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """读取当前作业对应且时间有效的持久化训练状态。"""
        state_path = self._task_artifact(job, "state.json")
        if not state_path.is_file():
            return {"state": "PENDING", "updated_at": None, "history": []}
        try:
            state = read_json(state_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return {"state": "PENDING", "updated_at": None, "history": []}
        if _parse_time(str(state.get("updated_at", ""))) < _parse_time(job["started_at"]):
            return {"state": "PENDING", "updated_at": None, "history": []}
        state["history"] = current_run_history(state.get("history", []))
        return state

    def _training_progress_for_job(self, job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """读取当前作业产生的最新候选训练迭代快照。"""
        candidates_root = self._task_artifact(job, "candidates")
        snapshots = []
        for path in candidates_root.glob("*/training_progress.json") if candidates_root.is_dir() else []:
            try:
                snapshots.append(ensure_within(path, self._task_dir(job["task_id"])))
            except ValueError:
                continue
        snapshots.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        for path in snapshots:
            try:
                path = ensure_within(path, self._task_dir(job["task_id"]))
                detail = read_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if _parse_time(str(detail.get("updated_at", ""))) < _parse_time(job["started_at"]):
                continue
            return detail
        return None

    def _loop_status_for_job(self, job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """读取当前作业的闭环轮次、奖励版本、决策和剩余预算。"""
        path = self._task_artifact(job, "loop_status.json")
        if not path.is_file():
            return None
        try:
            detail = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        if _parse_time(str(detail.get("updated_at", ""))) < _parse_time(job["started_at"]):
            return None
        return detail

    def _can_resume_job(self, job: Dict[str, Any], state: Dict[str, Any],
                        budget_detail: Optional[Dict[str, Any]] = None) -> bool:
        """只为冻结额度仍可用的人工复核候选开放恢复，且不能与活动作业竞争。"""
        if job.get("status") != "review" or state.get("state") != "HUMAN_REVIEW":
            return False
        budget_detail = budget_detail or self._budget_for_job(job, state)
        if not budget_detail["valid"] or budget_detail["exhausted"]:
            return False
        with self._lock:
            if any(item.get("status") in ("queued", "running", "stopping")
                   for item in self._jobs.values()):
                return False
        summary_path = self._task_artifact(job, "summary.json")
        if not summary_path.is_file():
            return False
        try:
            summary = read_json(summary_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        if summary.get("resume_allowed") is False or summary.get("recoverable") is False:
            return False
        experiment_id = summary.get("selected_experiment")
        try:
            return bool(experiment_id and self._task_artifact(
                job, "candidates/%s/reward_plan.json" % experiment_id).is_file())
        except ValueError:
            return False

    def _persisted_ui_artifact(self, job: Dict[str, Any], filename: str,
                               state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """读取该运行的冻结记录，恢复作业允许展示继承的产物而不混入旧实验。"""
        try:
            if not job.get("resume_of"):
                payload = self._current_json_artifact(job, filename)
            else:
                path = self._task_artifact(job, filename)
                payload = read_json(path) if path.is_file() else None
            if not isinstance(payload, dict):
                return None
            context = state.get("context", {})
            run_id = context.get("run_id") if isinstance(context, dict) else None
            if job.get("resume_of") and not run_id:
                inherited_state = read_json(self._task_artifact(job, "state.json"))
                run_id = inherited_state.get("context", {}).get("run_id")
            if payload.get("run_id") is not None and payload["run_id"] != run_id:
                return None
            return payload
        except (OSError, ValueError, TypeError, AttributeError):
            return None

    @staticmethod
    def _parse_frozen_budget(payload: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """解析严格整数的已用和冻结上限，兼容旧版已用加剩余额度格式。"""
        budget = {}
        for unit in ("iterations", "revisions"):
            used = payload.get("used_" + unit)
            maximum = payload.get("max_" + unit)
            remaining = payload.get("remaining_" + unit)
            if maximum is None and type(used) is int and type(remaining) is int:
                maximum = used + remaining
            if (type(used) is not int or type(maximum) is not int or
                    used < 0 or maximum < 0 or used > maximum or
                    (remaining is not None and (type(remaining) is not int or remaining < 0 or
                                               used + remaining != maximum))):
                return None
            budget["used_" + unit] = used
            budget["max_" + unit] = maximum
            budget["remaining_" + unit] = maximum - used
        return budget

    def _budget_for_job(self, job: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """只展示持久化运行额度，缺失或无效时不从当前配置猜测可恢复预算。"""
        unknown = {
            "valid": False, "status": "unknown", "source": None,
            "used_iterations": None, "max_iterations": None, "remaining_iterations": None,
            "used_revisions": None, "max_revisions": None, "remaining_revisions": None,
            "exhausted": None, "iteration_exhausted": None, "revision_exhausted": None,
            "resume_block_reason": "冻结预算缺失或无效，不能确认剩余额度；请核对实验记录，不能直接继续训练",
        }
        execution_path = self._task_artifact(job, "training_execution.json")
        execution = self._persisted_ui_artifact(job, "training_execution.json", state)
        summary = self._persisted_ui_artifact(job, "summary.json", state) or {}
        loop = self._persisted_ui_artifact(job, "loop_status.json", state) or {}
        sources = []
        if execution_path.is_file():
            if (not execution or type(execution.get("dry_run")) is not bool or
                    execution["dry_run"] != (job.get("mode") == "dry-run") or
                    not isinstance(execution.get("run_id"), str) or not execution["run_id"].strip() or
                    not isinstance(execution.get("budget"), dict)):
                return unknown
            parsed = self._parse_frozen_budget(execution["budget"])
            if parsed is None:
                return unknown
            sources.append(("training_execution", parsed))
        for name, record in (("summary", summary), ("loop_status", loop)):
            raw = record.get("budget", record)
            parsed = self._parse_frozen_budget(raw) if isinstance(raw, dict) else None
            if parsed is not None:
                sources.append((name, parsed))
        if not sources:
            if state.get("history") and last_working_state(str(state.get("state", "PENDING")), state["history"]) in {
                    "RECEIVED", "ENVIRONMENT_INSPECTED", "TASK_UNDERSTANDING", "TASK_FEASIBILITY_CHECK",
                    "MOTION_PROTOTYPE_GENERATING", "STATIC_MOTION_VALIDATION", "DYNAMIC_MOTION_VALIDATION",
                    "RAG_RETRIEVING", "CONTEXT_BUILDING", "PROMPT_COMPILING", "REWARD_DESIGNING",
                    "TASK_DESIGNED", "REWARD_REVIEWING"}:
                return dict(unknown, status="not_started",
                            resume_block_reason="本次训练尚未开始，预算尚未冻结；请先处理任务阻塞原因")
            return unknown
        # 历史摘要可比执行快照更旧：已用取最大、上限取最小，绝不显示退款或扩额。
        budget = dict(sources[0][1])
        for _, evidence in sources[1:]:
            for unit in ("iterations", "revisions"):
                budget["used_" + unit] = max(budget["used_" + unit], evidence["used_" + unit])
                budget["max_" + unit] = min(budget["max_" + unit], evidence["max_" + unit])
        if any(budget["used_" + unit] > budget["max_" + unit] for unit in ("iterations", "revisions")):
            return unknown
        for unit in ("iterations", "revisions"):
            budget["remaining_" + unit] = budget["max_" + unit] - budget["used_" + unit]
        iteration_exhausted = budget["remaining_iterations"] == 0
        revision_exhausted = budget["remaining_revisions"] == 0
        exhausted = iteration_exhausted or revision_exhausted or summary.get("budget_exhausted") is True
        reason = ("冻结训练预算已耗尽，当前实验不能继续训练；需新实验或显式批准扩额度"
                  "（当前界面没有追加额度功能）；已保存策略与验收结果不会删除") if exhausted else None
        if not exhausted and (summary.get("resume_allowed") is False or summary.get("recoverable") is False):
            reason = "后端已禁止恢复该实验，请查看阻塞原因；不会自动增加训练额度"
        return dict(budget, valid=True, status="exhausted" if exhausted else "available",
                    source=sources[0][0], sources=[name for name, _ in sources], exhausted=exhausted,
                    iteration_exhausted=iteration_exhausted, revision_exhausted=revision_exhausted,
                    resume_block_reason=reason)

    def _result_for_job(self, job: Dict[str, Any], state: Dict[str, Any],
                        loop: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """分别展示奖励版本、实际 PPO 产物和联合验收，不把预算耗尽等同成功。"""
        summary = self._persisted_ui_artifact(job, "summary.json", state) or {}
        execution = self._persisted_ui_artifact(job, "training_execution.json", state) or {}
        loop = loop or {}
        candidate = summary.get("selected_experiment") or execution.get("experiment_id") or loop.get("experiment_id")
        manifest, plan = {}, {}
        if isinstance(candidate, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", candidate):
            prefix = "candidates/%s/" % candidate
            manifest = self._persisted_ui_artifact(job, prefix + "manifest.json", state) or {}
            plan = self._persisted_ui_artifact(job, prefix + "reward_plan.json", state) or {}
        context = state.get("context", {}) if isinstance(state.get("context"), dict) else {}
        versions = (context.get("reward_version"), loop.get("reward_version"),
                    summary.get("reward_version"), plan.get("version"), manifest.get("reward_version"))
        version = next((value for value in versions if type(value) is int and value > 0), None)
        checkpoint_available = False
        checkpoint = summary.get("checkpoint")
        if not checkpoint and manifest.get("checkpoint") and candidate:
            checkpoint = "candidates/%s/%s" % (candidate, manifest["checkpoint"])
        if isinstance(checkpoint, str):
            try:
                checkpoint_available = self._task_artifact(job, checkpoint).is_file()
            except ValueError:
                pass
        training_result = manifest.get("training_result", "unknown")
        training_status = "unknown"
        if checkpoint_available and training_result == "completed":
            training_status = "completed"
        elif checkpoint_available and training_result == "dry_run_completed":
            training_status = "mock_completed"
        elif training_result in ("failed", "compiled", "running", "pending"):
            training_status = training_result
        accepted = (state.get("state") == "COMPLETED" and summary.get("state") == "COMPLETED" and
                    summary.get("result") == "completed" and summary.get("dry_run") is False)
        if accepted:
            acceptance_status = "passed"
        elif summary.get("result") == "completed" and summary.get("dry_run") is True:
            acceptance_status = "mock_passed"
        elif summary.get("result") == "failed" or (loop.get("phase") == "evaluated" and
                                                    loop.get("decision") != "complete"):
            acceptance_status = "not_passed"
        elif summary:
            acceptance_status = "pending"
        else:
            acceptance_status = "unknown"
        if (state.get("history") and not candidate and not checkpoint_available and
                not any(entry.get("state") in TRAINING_STAGE_RANGES or entry.get("state") in {
                    "ROLLOUT_COLLECTING", "VISUAL_EVALUATING", "NUMERIC_EVALUATING", "DIAGNOSING", "MEMORY_CURATING"}
                    for entry in state["history"])):
            training_status = acceptance_status = "not_started"
        return {"reward_version": version, "trained_reward_version": manifest.get("reward_version"),
                "selected_experiment": candidate, "training_status": training_status,
                "ppo_completed": training_status == "completed", "checkpoint_available": checkpoint_available,
                "acceptance_status": acceptance_status, "accepted": accepted,
                "dry_run": job.get("mode") == "dry-run"}

    def _current_json_artifact(self, job: Dict[str, Any], relative_path: str
                               ) -> Optional[Dict[str, Any]]:
        """读取当前作业启动后生成的 JSON，避免展示同 task_id 的旧轮次产物。"""
        path = self._task_artifact(job, relative_path)
        if not path.is_file():
            return None
        started_at = job.get("started_at")
        if started_at:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if modified < _parse_time(str(started_at)):
                return None
        try:
            payload = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _feasibility_for_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """把完整可行性报告压缩为上位机可稳定渲染的分层摘要。"""
        report = self._current_json_artifact(job, "feasibility_report.json")
        if report is None:
            return {"available": False, "status": "PENDING", "viewer_available": False}
        constraint = report.get("motion_constraint_spec", {})
        planning = report.get("planning_report", {})
        whole_body = report.get("whole_body_report", {})
        physics = report.get("physics_report", {})
        components = planning.get("components", []) if isinstance(planning, dict) else []
        component_status = {
            str(item.get("planner")): str(item.get("status", "UNKNOWN"))
            for item in components if isinstance(item, dict) and item.get("planner")
        }
        motion_type = str(report.get("motion_type") or constraint.get("motion_type") or "UNKNOWN")
        return {
            "available": True,
            "status": str(report.get("status", report.get("overall_status", "UNKNOWN"))),
            "validation_level": str(report.get("validation_level", "CAPABILITY_ONLY")),
            "training_admission": report.get("training_admission") or {},
            "training_readiness": report.get("training_readiness") or {},
            "feasibility_level": str(report.get("feasibility_level", "LEVEL_0_LANGUAGE")),
            "motion_type": motion_type,
            "backend": report.get("backend") or physics.get("backend") or "not_run",
            "confidence": report.get("confidence", 0.0),
            "constraint_ready": bool(constraint),
            "required_planners": list(constraint.get("required_planners", [])),
            "planner_status": str(planning.get("status", "NOT_RUN")),
            "component_status": component_status,
            "whole_body_status": str(whole_body.get("status", "NOT_RUN")),
            "missing_solvers": list(whole_body.get("missing_solvers", [])),
            "physics_status": str(physics.get("validation_level", physics.get("status", "NOT_RUN"))),
            "limitations": list(report.get("limitations", []))[:4],
            "recommended_next_step": str(report.get("recommended_next_step", "")),
            "viewer_available": motion_type == "LOCOMOTION" and job.get("robot") == "go2",
            "duration": constraint.get("duration"),
        }

    def _memory_for_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """读取当前任务工作记忆和长期记忆晋升结果。"""
        working = self._current_json_artifact(job, "memory/working_memory.json") or {}
        promotion = self._current_json_artifact(job, "memory/promotion.json") or {}
        return {
            "working_available": bool(working),
            "working_state": working.get("state"),
            "loop_round": working.get("loop_round", 0),
            "reward_version": working.get("reward_version", 0),
            "promoted": bool(promotion.get("promoted")),
            "promotion_reason": promotion.get("reason") or promotion.get("gate"),
            "memory_id": promotion.get("memory_id"),
        }


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
        presentation = state_presentation(str(state.get("state", "PENDING")), state.get("history", []))
        if job["status"] == "failed" and presentation["state"] not in ("FAILED", "HUMAN_REVIEW", "COMPLETED"):
            # 旧进程可能在写入终态前退出；展示阻塞的实际阶段，不改写历史记录。
            presentation = state_presentation("FAILED", state.get("history", []))
        training_detail = self._training_progress_for_job(job)
        loop_detail = self._loop_status_for_job(job)
        working_state = last_working_state(presentation["state"], state.get("history", []))
        if (training_detail and presentation["state"] != "COMPLETED" and
                working_state in TRAINING_STAGE_RANGES):
            presentation["progress"] = training_stage_progress(
                working_state, training_detail, state.get("context", {}),
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
        job["budget_detail"] = self._budget_for_job(job, state)
        job["result_detail"] = self._result_for_job(job, state, loop_detail)
        job["stage_detail"] = stage_track_presentation(
            presentation["state"], state.get("history", []), job["result_detail"]["accepted"])
        job["can_stop"] = job["status"] in ("running", "stopping") and job_id in self._processes
        job["can_resume"] = self._can_resume_job(job, state, job["budget_detail"])
        job["resume_block_reason"] = job["budget_detail"]["resume_block_reason"]
        job["review_reason"] = None
        if job["status"] == "review":
            for result_path in (self._task_artifact(job, "summary.json"),
                                self._task_artifact(job, "blocking_report.json")):
                if not result_path.is_file():
                    continue
                try:
                    job["review_reason"] = read_json(result_path).get("reason")
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                if job["review_reason"]:
                    break
            job["review_reason"] = job["review_reason"] or job.get("message")
        job["feasibility"] = self._feasibility_for_job(job)
        job["memory_detail"] = self._memory_for_job(job)
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
                task_dir = self._task_dir(state_path.parent.name)
                state_path = ensure_within(state_path, task_dir)
                state = read_json(state_path)
                request_path = ensure_within(task_dir / "task_request.txt", task_dir)
                spec_path = ensure_within(task_dir / "task_spec.json", task_dir)
                task = request_path.read_text(encoding="utf-8").strip() if request_path.is_file() else "未记录动作描述"
            except (OSError, ValueError, json.JSONDecodeError):
                continue
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

    def _playable_artifacts(self, task_id: str) -> Optional[Dict[str, Any]]:
        """只为真实联合验收成功且 checkpoint/config 完整的实验解析播放产物。"""
        if not re.fullmatch(r"task-[a-f0-9]{10}", str(task_id)):
            return None
        try:
            task_dir = self._task_dir(task_id)
            state_path = ensure_within(task_dir / "state.json", task_dir)
            summary_path = ensure_within(task_dir / "summary.json", task_dir)
            if not state_path.is_file() or not summary_path.is_file():
                return None
            state = read_json(state_path)
            summary = read_json(summary_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        if (state.get("state") != "COMPLETED" or summary.get("state") != "COMPLETED" or
                summary.get("result") != "completed" or summary.get("dry_run") is not False):
            return None
        try:
            task_spec_path = ensure_within(task_dir / "task_spec.json", task_dir)
            robot = str(read_json(task_spec_path).get("robot", ""))
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        if robot not in self.settings.allowed_robots:
            return None
        checkpoint_value = summary.get("checkpoint") or "final/checkpoint.pt"
        config_value = summary.get("config") or "final/config.yaml"
        if not isinstance(checkpoint_value, str) or not isinstance(config_value, str):
            return None
        try:
            checkpoint = ensure_within(task_dir / checkpoint_value, task_dir)
            config = ensure_within(task_dir / config_value, task_dir)
        except (TypeError, ValueError):
            return None
        if not checkpoint.is_file() or not config.is_file():
            return None
        return {"task_id": task_id, "robot": robot, "task_dir": task_dir,
                "checkpoint": checkpoint, "config": config, "summary": summary}

    def list_playable_experiments(self, limit: int = 50) -> List[Dict[str, Any]]:
        """列出通过真实训练验收且播放文件齐全的策略。"""
        playable: List[Dict[str, Any]] = []
        for experiment in self.list_experiments(limit=10000):
            artifacts = self._playable_artifacts(experiment["task_id"])
            if artifacts is None:
                continue
            experiment["playable"] = True
            experiment["selected_experiment"] = artifacts["summary"].get("selected_experiment", "")
            playable.append(experiment)
            if len(playable) >= limit:
                break
        return playable

    def start_playback(self, task_id: str, seed: int = 1, num_envs: int = 1) -> Dict[str, Any]:
        """把 Viewer 的互斥校验、启动和租约登记放在同一个临界区。"""
        with self._lock:
            self._assert_accepting()
            return self._start_playback_locked(task_id, seed, num_envs)

    def _start_playback_locked(self, task_id: str, seed: int, num_envs: int) -> Dict[str, Any]:
        """启动已验收策略的仿真 Viewer 播放，不允许 dry-run 或训练并发占用 GPU。"""
        artifacts = self._playable_artifacts(task_id)
        if artifacts is None:
            raise JobValidationError("只能播放真实训练且已通过联合验收的策略；请刷新成功策略列表")
        if not 1 <= int(num_envs) <= 16:
            raise JobValidationError("播放并行环境数量必须位于 1 到 16 之间")
        with self._lock:
            active_jobs = [job for job in self._jobs.values()
                           if job.get("status") in ("queued", "running", "stopping")]
            if active_jobs:
                raise JobValidationError("训练任务仍在运行，请等待训练结束后再播放策略")
            if self._viewer_running():
                raise JobValidationError("已有策略正在播放，请先关闭当前仿真 Viewer")
        adapter = UnitreeProjectAdapter(self.settings.training_root, self.settings.agent_root,
                                        self.settings.experiments_path)
        command = adapter.play_command(artifacts["robot"], artifacts["config"], artifacts["checkpoint"],
                                       seed=int(seed), num_envs=int(num_envs))
        environment = os.environ.copy()
        if self.settings.gpu_ids:
            environment["CUDA_VISIBLE_DEVICES"] = str(self.settings.gpu_ids[0])
        log_path = ensure_within(self.root / "playback.log", self.root)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("a", encoding="utf-8") as output:
                process = subprocess.Popen(command, cwd=str(self.settings.training_root), env=environment,
                                           stdout=output, stderr=subprocess.STDOUT, shell=False,
                                           start_new_session=True)
                capture_process_group(process)
        except OSError as exc:
            raise JobValidationError("策略 Viewer 启动失败：{}".format(exc)) from exc
        with self._lock:
            self._playback_process = process
            self._playback_task_id = task_id
            self._playback_kind = "strategy"
            self._playback_stopping = False
        return {"task_id": task_id, "robot": artifacts["robot"], "pid": process.pid,
                "kind": "strategy", "log": relative_display(log_path, self.settings.agent_root)}

    def start_feasibility_view(self, job_id: str) -> Dict[str, Any]:
        """原子启动可行性 Viewer，复用训练和策略播放器的互斥租约。"""
        with self._lock:
            self._assert_accepting()
            return self._start_feasibility_view_locked(job_id)

    def _start_feasibility_view_locked(self, job_id: str) -> Dict[str, Any]:
        """为当前直线步态可行性报告启动 Isaac Gym 可视化探针。"""
        job = self.get_job(job_id)
        feasibility = job.get("feasibility", {})
        if not feasibility.get("available"):
            raise JobValidationError("当前作业尚未生成动作可行性报告")
        if not feasibility.get("viewer_available"):
            raise JobValidationError("当前仅支持在 Viewer 中观察 Go2 直线行走可行性探针")
        with self._lock:
            active_jobs = [item for item in self._jobs.values()
                           if item.get("status") in ("queued", "running", "stopping")]
            if active_jobs:
                raise JobValidationError("训练任务仍在运行，请等待训练结束后再观察可行性探针")
            if self._viewer_running():
                raise JobValidationError("已有仿真 Viewer 正在运行，请先关闭当前窗口")
        try:
            duration = min(30.0, max(0.1, float(feasibility.get("duration") or 5.0)))
        except (TypeError, ValueError):
            duration = 5.0
        command = [
            sys.executable, "-u", "-m", "rl_training_agent", "feasibility-view",
            "--task", str(job["task"]), "--robot", str(job["robot"]),
            "--seed", "1", "--max-seconds", str(duration),
        ]
        environment = os.environ.copy()
        if self.settings.gpu_ids:
            environment["CUDA_VISIBLE_DEVICES"] = str(self.settings.gpu_ids[0])
        log_path = ensure_within(self.root / "feasibility_view.log", self.root)
        try:
            with log_path.open("a", encoding="utf-8") as output:
                process = subprocess.Popen(
                    command, cwd=str(self.settings.agent_root), env=environment,
                    stdout=output, stderr=subprocess.STDOUT, shell=False,
                    start_new_session=True)
                capture_process_group(process)
        except OSError as exc:
            raise JobValidationError("可行性 Viewer 启动失败：{}".format(exc)) from exc
        with self._lock:
            self._playback_process = process
            self._playback_task_id = job["task_id"]
            self._playback_kind = "feasibility"
            self._playback_stopping = False
        return {"task_id": job["task_id"], "robot": job["robot"], "pid": process.pid,
                "kind": "feasibility", "log": relative_display(log_path, self.settings.agent_root)}

    def playback_status(self) -> Dict[str, Any]:
        """返回当前策略 Viewer 是否仍在运行。"""
        with self._lock:
            process = self._playback_process
            running = self._viewer_running()
            if not running:
                self._playback_process = None
                self._playback_task_id = None
                self._playback_kind = None
                self._playback_stopping = False
            return {"running": running, "task_id": self._playback_task_id if running else None,
                    "pid": process.pid if running else None,
                    "kind": self._playback_kind if running else None}

    def stop_playback(self) -> Dict[str, Any]:
        """请求停止当前策略播放进程组，超时后强制结束以释放 GPU。"""
        with self._lock:
            process = self._playback_process
            if process is None or (process.poll() is not None and not self._playback_stopping):
                self._playback_process = None
                self._playback_task_id = None
                self._playback_kind = None
                self._playback_stopping = False
                return {"running": False, "task_id": None, "pid": None, "kind": None}
            if self._playback_stopping:
                return self.playback_status()
            self._playback_stopping = True
        signal_process_group(process, signal.SIGTERM)
        threading.Thread(target=self._ensure_playback_stopped, args=(process,), daemon=True).start()
        return self.playback_status()

    def _ensure_playback_stopped(self, process: subprocess.Popen) -> None:
        """等待策略播放器退出；失去响应时终止整个 Viewer 进程组。"""
        stop_process_group(process, grace_seconds=10.0)
        identity = getattr(process, "_rl_group_identity", None)
        while identity is not None and not wait_registered_groups([identity], timeout=1.0):
            signal_registered_groups([identity], signal.SIGKILL)
        with self._lock:
            if self._playback_process is process:
                self._playback_process = None
                self._playback_task_id = None
                self._playback_kind = None
                self._playback_stopping = False

    def shutdown(self, wait_seconds: float = 12.0) -> None:
        """先禁止派发并取消全部队列，再同步清理所有本次持有的训练和 Viewer。"""
        with self._lock:
            self._closing = True
            processes = list(self._processes.items())
            playback = self._playback_process
            for job in self._jobs.values():
                if job.get("status") == "queued":
                    job.update(status="stopped", finished_at=utc_now(), message="界面关闭，等待作业已取消")
                    self._save_job(job)
                elif job["job_id"] in self._processes:
                    job.update(status="stopping", message="界面关闭，正在停止训练")
                    self._save_job(job)
        deadline = time.monotonic() + max(0.0, wait_seconds)
        for job_id, process in processes:
            groups = self._registered_children(job_id, process)
            signal_registered_groups(groups, signal.SIGTERM)
            stop_process_group(process, max(0.0, deadline - time.monotonic()))
            signal_registered_groups(groups, signal.SIGKILL)
        if playback is not None:
            stop_process_group(playback, max(0.0, deadline - time.monotonic()))

    def read_log(self, job_id: str, offset: int = 0, limit: int = 65536) -> Dict[str, Any]:
        """从字节偏移量开始增量读取训练日志。"""
        self._job_dir(job_id)
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
        log_path = self._job_artifact(job_id, "training.log")
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
        rag_status = {"enabled": self.settings.rag_enabled, "documents": 0, "chunks": 0}
        if self.settings.rag_index_file.is_file():
            try:
                rag_index = read_json(self.settings.rag_index_file)
                rag_status.update({
                    "documents": int(rag_index.get("document_count", 0)),
                    "chunks": int(rag_index.get("chunk_count", 0)),
                    "generated_at": rag_index.get("generated_at"),
                })
            except (OSError, ValueError, json.JSONDecodeError):
                rag_status["error"] = "索引无法读取"
        memory_status = self.memory_status()
        bailian_status = BailianGLMProvider().health()
        return {
            "robots": robots,
            "default_robot": self.settings.default_robot,
            "system": {
                "training_project_ready": self.settings.training_root.is_dir(),
                "training_entry_ready": (self.settings.training_root / "legged_gym" / "scripts" / "train.py").is_file(),
                "scope": "simulation_only",
                "gpu_ids": self.settings.gpu_ids,
                "rag": rag_status,
                "memory": memory_status,
                "providers": {
                    "mode": self.settings.provider,
                    "task_planner": self.settings.task_planner_provider,
                    "reward_designer": self.settings.reward_designer_provider,
                    "visual_critic": self.settings.visual_critic_provider,
                    "diagnosis": self.settings.diagnosis_provider,
                    "bailian_model": bailian_status["model"],
                    "bailian_configured": bailian_status["available"],
                },
            },
            "modes": [
                {"id": "dry-run", "name": "离线演练", "description": "使用模拟推理与模拟训练，适合先验证完整流程"},
                {"id": "real", "name": "真实训练", "description": "使用 OpenCLI 与 GPU 仿真执行正式训练"},
            ],
        }

    def memory_status(self) -> Dict[str, Any]:
        """返回四层记忆库的实时统计，不读取或暴露记忆正文。"""
        status = LongTermMemoryStore(
            self.settings.memory_path, self.settings.agent_root,
            self.settings.memory_max_context_chars).stats()
        status["enabled"] = self.settings.memory_enabled
        return status


class TrainingUIHandler(BaseHTTPRequestHandler):
    """提供静态界面与训练作业 JSON API 的 HTTP 请求处理器。"""

    manager: JobManager
    static_root: Path
    session_token: str
    session_cookie: str
    allowed_hosts: set
    allowed_origins: set
    server_version = "RLTrainingUI/1.0"

    def _send_headers(self, status: int, content_type: str, length: int,
                      initialize_session: bool = False) -> None:
        """发送带有本地界面安全策略的通用响应头。"""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'")
        if initialize_session:
            self.send_header("Set-Cookie", "%s=%s; Path=/; HttpOnly; SameSite=Strict" %
                             (self.session_cookie, self.session_token))
        self.end_headers()

    def _send_json(self, status: int, value: Any) -> None:
        """以 UTF-8 JSON 返回 API 响应。"""
        body = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
        self._send_headers(status, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str) -> None:
        """返回格式一致且不暴露调用栈的 API 错误。"""
        self._send_json(status, {"error": message})

    def _send_asset(self, filename: str, initialize_session: bool = False) -> None:
        """仅从固定白名单路由发送内置静态资源。"""
        asset = self.static_root / filename
        if not asset.is_file():
            self._send_error_json(404, "资源不存在")
            return
        body = asset.read_bytes()
        content_type = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type == "application/javascript":
            content_type += "; charset=utf-8"
        self._send_headers(200, content_type, len(body), initialize_session)
        self.wfile.write(body)

    def _validate_request_context(self, require_session: bool = False) -> bool:
        """统一拒绝非法 Host、跨源请求和缺失会话，首页通过后才允许签发凭据。"""
        hosts = self.headers.get_all("Host", [])
        origins = self.headers.get_all("Origin", [])
        if len(hosts) != 1 or hosts[0].lower() not in self.allowed_hosts:
            self._send_error_json(403, "Host 不在本地界面允许列表中")
            return False
        if (len(origins) > 1 or (origins and origins[0] not in self.allowed_origins) or
                self.headers.get("Sec-Fetch-Site", "") == "cross-site"):
            self._send_error_json(403, "不允许跨来源操作本地训练界面")
            return False
        if require_session:
            cookies = SimpleCookie()
            try:
                cookies.load(self.headers.get("Cookie", ""))
                token = cookies[self.session_cookie].value if self.session_cookie in cookies else ""
            except Exception:
                token = ""
            if not hmac.compare_digest(token.encode("utf-8"), self.session_token.encode("utf-8")):
                self._send_error_json(401, "请从本地界面首页初始化会话")
                return False
        return True

    def _read_json_body(self) -> Any:
        """在固定大小限制内读取并解析请求 JSON。"""
        if self.headers.get_content_type() != "application/json":
            raise JobValidationError("变更请求必须使用 application/json")
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
        """安全处理只读路由，把损坏产物或越界符号链接转换为稳定响应。"""
        try:
            self._do_get()
        except (OSError, ValueError, KeyError):
            self._send_error_json(404, "产物不存在、无效或不在允许目录中")

    def _do_get(self) -> None:
        """处理首页、静态资源和只读 API。"""
        parsed = urlparse(self.path)
        if not self._validate_request_context(require_session=parsed.path.startswith("/api/")):
            return
        if parsed.path in ("/", "/index.html"):
            self._send_asset("index.html", initialize_session=True)
            return
        if parsed.path in ("/app.css", "/app.js"):
            self._send_asset(parsed.path.lstrip("/"))
            return
        if parsed.path == "/api/config":
            self._send_json(200, self.manager.config_payload())
            return
        if parsed.path == "/api/jobs":
            self._send_json(200, {"jobs": self.manager.list_jobs(), "experiments": self.manager.list_experiments(),
                                  "playable_experiments": self.manager.list_playable_experiments(),
                                  "playback": self.manager.playback_status()})
            return
        if parsed.path == "/api/playback":
            self._send_json(200, self.manager.playback_status())
            return
        if parsed.path == "/api/memory":
            self._send_json(200, self.manager.memory_status())
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
        """处理创建训练、恢复闭环、停止训练与策略播放请求。"""
        parsed = urlparse(self.path)
        if not self._validate_request_context(require_session=True):
            return
        try:
            payload = self._read_json_body()
            if parsed.path == "/api/jobs":
                job = self.manager.start_job(payload)
                self._send_json(201, job)
                return
            if parsed.path == "/api/play":
                if not isinstance(payload, dict):
                    raise JobValidationError("播放请求必须是 JSON 对象")
                task_id = str(payload.get("task_id", ""))
                try:
                    seed = int(payload.get("seed", 1))
                    num_envs = int(payload.get("num_envs", 1))
                except (TypeError, ValueError) as exc:
                    raise JobValidationError("播放随机种子和并行环境数必须是整数") from exc
                result = self.manager.start_playback(task_id, seed=seed, num_envs=num_envs)
                self._send_json(201, result)
                return
            if parsed.path == "/api/feasibility-view":
                if not isinstance(payload, dict):
                    raise JobValidationError("可行性 Viewer 请求必须是 JSON 对象")
                result = self.manager.start_feasibility_view(str(payload.get("job_id", "")))
                self._send_json(201, result)
                return
            if parsed.path == "/api/play/stop":
                self._send_json(200, self.manager.stop_playback())
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
    try:
        local_host = host.lower() == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        local_host = False
    if not local_host:
        raise JobValidationError("训练界面只允许绑定 loopback；远程访问请使用 SSH 隧道或受控代理")
    selected_manager = manager or JobManager()
    static_root = Path(__file__).resolve().parent / "web"

    class BoundTrainingUIHandler(TrainingUIHandler):
        """把当前服务的作业管理器与静态资源目录绑定到处理器。"""

    BoundTrainingUIHandler.manager = selected_manager
    BoundTrainingUIHandler.static_root = static_root
    BoundTrainingUIHandler.session_token = secrets.token_urlsafe(32)
    class LoopbackTrainingServer(ThreadingHTTPServer):
        """为 IPv4 或 IPv6 loopback 使用匹配的地址族。"""
        address_family = socket.AF_INET6 if ":" in host else socket.AF_INET

    server = LoopbackTrainingServer((host, port), BoundTrainingUIHandler)
    actual_port = server.server_address[1]
    BoundTrainingUIHandler.session_cookie = "%s_%d" % (SESSION_COOKIE, actual_port)
    hostnames = ("localhost", "[%s]" % host.lower()) if ":" in host else ("127.0.0.1", "localhost", host.lower())
    BoundTrainingUIHandler.allowed_hosts = {"%s:%d" % (name, actual_port) for name in hostnames}
    BoundTrainingUIHandler.allowed_origins = {"http://" + value for value in BoundTrainingUIHandler.allowed_hosts}
    server.job_manager = selected_manager
    return server


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
        server.job_manager.shutdown()
        server.server_close()
    return 0
