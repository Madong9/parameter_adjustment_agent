from __future__ import annotations

import json
import hmac
import os
import re
import signal
import subprocess
import sys
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..utils.io import atomic_write_text, write_json


PROGRESS_PATTERN = re.compile(r"Learning iteration\s+(\d+)\s*/\s*(\d+)")
_POPEN_TYPE = subprocess.Popen


def process_identity(pid: int) -> Optional[dict]:
    """读取 Linux 进程组和启动时钟，使用启动身份避免 PID 重用误伤。"""
    try:
        fields = Path("/proc/%d/stat" % int(pid)).read_text().rsplit(")", 1)[1].split()
        return {"pid": int(pid), "pgid": int(fields[2]), "start_ticks": int(fields[19])}
    except (OSError, ValueError, IndexError):
        return None


def group_member_identities(pgid: int) -> List[dict]:
    """快照受控组当前成员的启动身份，组长退出后仍能验证原有后代。"""
    members = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            identity = process_identity(int(path.name))
            if identity and identity["pgid"] == pgid:
                members.append(identity)
    return members


def capture_process_group(process: subprocess.Popen) -> None:
    """在新会话启动后保存组长及已知成员身份，禁止后续仅凭旧 PID 杀组。"""
    if not isinstance(process, _POPEN_TYPE):
        return
    identity = process_identity(process.pid)
    previous = getattr(process, "_rl_group_identity", None)
    if identity and identity["pgid"] == process.pid:
        if previous and any(identity[key] != previous[key] for key in ("pid", "pgid", "start_ticks")):
            return
        identity["_members"] = group_member_identities(process.pid)
        process._rl_group_identity = identity


def register_child_group(process: subprocess.Popen) -> None:
    """把受控子会话的签名启动身份追加到 Web 指定的作业控制文件。"""
    filename = os.environ.get("RL_AGENT_JOB_CONTROL_FILE")
    token = os.environ.get("RL_AGENT_JOB_CONTROL_TOKEN")
    if not filename or not token:
        return
    identity = process_identity(process.pid)
    if identity is None:
        return
    identity["owner_pid"] = os.getpid()
    body = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    signature = hmac.new(token.encode(), body.encode(), "sha256").hexdigest()
    record = json.dumps({"identity": identity, "signature": signature}) + "\n"
    fd = os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    try:
        os.write(fd, record.encode())
        os.fsync(fd)
    finally:
        os.close(fd)


def registered_child_groups(path: Path, token: str, owner_pid: int) -> List[dict]:
    """只返回签名有效且启动身份仍匹配的作业子组，拒绝伪造控制记录。"""
    groups = []
    try:
        with path.open("rb") as handle:
            content = handle.read(1024 * 1024 + 1)
        if len(content) > 1024 * 1024:
            return groups
    except OSError:
        return groups
    for line in content.splitlines():
        try:
            record = json.loads(line)
            identity = record["identity"]
            body = json.dumps(identity, sort_keys=True, separators=(",", ":"))
            signature = hmac.new(token.encode(), body.encode(), "sha256").hexdigest()
            if not hmac.compare_digest(signature, str(record["signature"])):
                continue
            if identity["owner_pid"] != owner_pid or identity["pid"] != identity["pgid"]:
                continue
            current = process_identity(identity["pid"])
            if current and all(current[key] == identity[key] for key in ("pid", "pgid", "start_ticks")):
                identity["_members"] = group_member_identities(identity["pgid"])
                groups.append(identity)
        except (ValueError, TypeError, KeyError):
            continue
    return groups


def signal_registered_groups(groups: List[dict], sig: int) -> None:
    """对本次校验过的子会话发信号，PID 身份变化时停止发送。"""
    for identity in groups:
        identities = [identity] + identity.get("_members", [])
        verified = False
        for tracked in identities:
            current = process_identity(tracked["pid"])
            if current and all(current[key] == tracked[key] for key in ("pid", "pgid", "start_ticks")):
                verified = True
                break
        if not verified:
            continue
        try:
            os.killpg(identity["pgid"], sig)
        except (ProcessLookupError, PermissionError):
            continue


def wait_registered_groups(groups: List[dict], timeout: float = 2.0) -> bool:
    """等待已验证成员结束，僵尸不再占用 GPU，身份变化不视为原作业进程。"""
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        running = False
        for group in groups:
            for identity in [group] + group.get("_members", []):
                current = process_identity(identity["pid"])
                if not current or any(current[key] != identity[key] for key in ("pid", "pgid", "start_ticks")):
                    continue
                try:
                    state = Path("/proc/%d/stat" % identity["pid"]).read_text().rsplit(")", 1)[1].split()[0]
                    running = running or state != "Z"
                except (OSError, IndexError):
                    continue
        if not running:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)


def signal_process_group(process: subprocess.Popen, sig: int) -> None:
    """只向由调用方以新会话创建的进程组发送信号，容忍重复清理。"""
    capture_process_group(process)
    identity = getattr(process, "_rl_group_identity", None)
    if identity is not None:
        signal_registered_groups([identity], sig)


def stop_process_group(process: subprocess.Popen, grace_seconds: float = 5.0) -> None:
    """回收整个受控进程组，组长退出后仍强制清理残留后代并回收组长。"""
    signal_process_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=max(0.0, grace_seconds))
    except subprocess.TimeoutExpired:
        pass
    except TypeError:
        process.wait()
    # 不能把组长 wait 成功当成所有后代都已退出；无论组长状态都清理残留组。
    signal_process_group(process, signal.SIGKILL)
    try:
        process.wait(timeout=2)
    except TypeError:
        process.wait()
    identity = getattr(process, "_rl_group_identity", None)
    if identity is not None:
        wait_registered_groups([identity], timeout=2)


@dataclass
class ProcessResult:
    command: List[str]
    exit_code: int
    timed_out: bool
    duration_seconds: float
    pid: int


class ProcessManager:
    """仅运行预构造参数数组，不暴露 shell 命令接口。"""

    _registry = weakref.WeakSet()
    _registry_lock = threading.RLock()

    def __init__(self):
        """初始化 ProcessManager 实例及其运行依赖。"""
        self._process: Optional[subprocess.Popen] = None
        self._run_lock = threading.Lock()
        with self._registry_lock:
            self._registry.add(self)

    @classmethod
    def stop_all_active(cls, grace_seconds: float = 5.0) -> None:
        """先通知所有独立训练会话，再在统一宽限预算内回收全部受控子进程。"""
        with cls._registry_lock:
            processes = [manager._process for manager in list(cls._registry)
                         if manager._process is not None]
        deadline = time.monotonic() + max(0.0, grace_seconds)
        for process in processes:
            signal_process_group(process, signal.SIGTERM)
        for process in processes:
            stop_process_group(process, max(0.0, deadline - time.monotonic()))

    @staticmethod
    def _command_value(command: List[str], option: str, default: str = "") -> str:
        """读取参数数组中指定选项后面的值。"""
        try:
            return command[command.index(option) + 1]
        except (ValueError, IndexError):
            return default

    @staticmethod
    def _read_appended(path: Path, offset: int, pending: str) -> tuple:
        """从日志字节偏移读取新增完整行，并保留尚未换行的尾部。"""
        if not path.is_file():
            return [], offset, pending
        with path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read()
            next_offset = handle.tell()
        if not data:
            return [], next_offset, pending
        combined = pending + data.decode("utf-8", errors="replace")
        parts = combined.splitlines(keepends=True)
        if parts and not parts[-1].endswith(("\n", "\r")):
            pending = parts.pop()
        else:
            pending = ""
        return [line.rstrip("\r\n") for line in parts], next_offset, pending

    @staticmethod
    def _write_training_progress(output_dir: Path, run_name: str, iteration: int, total: int,
                                 status: str) -> None:
        """写入供上位机轮询的轻量训练进度快照。"""
        percent = 0.0 if total <= 0 else min(100.0, max(0.0, iteration * 100.0 / total))
        write_json(output_dir / "training_progress.json", {
            "run_name": run_name,
            "iteration": iteration,
            "total_iterations": total,
            "percent": round(percent, 2),
            "status": status,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    def _consume_training_output(self, lines: List[str], output_dir: Path, run_name: str,
                                 last_reported: int) -> int:
        """解析训练迭代并按约百分之一的频率转发到上位机日志。"""
        for line in lines:
            match = PROGRESS_PATTERN.search(line)
            if not match:
                continue
            iteration, total = int(match.group(1)), int(match.group(2))
            interval = max(1, total // 100)
            if iteration == 0 or iteration >= total - 1 or iteration - last_reported >= interval:
                self._write_training_progress(output_dir, run_name, iteration, total, "running")
                percent = 0.0 if total <= 0 else iteration * 100.0 / total
                print("[训练] {}：迭代 {}/{}（{:.1f}%）".format(
                    run_name, iteration, total, percent), flush=True)
                last_reported = iteration
        return last_reported

    def run(self, command: List[str], cwd: Path, output_dir: Path, timeout: int,
            env: Optional[dict] = None) -> ProcessResult:
        """执行受限进程并在异常、信号或正常退出时清理残留的独立会话。"""
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("同一个进程管理器不能并发启动多个训练")
        try:
            return self._run_process(command, cwd, output_dir, timeout, env)
        finally:
            try:
                self.stop(grace_seconds=1.0)
            finally:
                self._process = None
                self._run_lock.release()

    def _run_process(self, command: List[str], cwd: Path, output_dir: Path, timeout: int,
                     env: Optional[dict] = None) -> ProcessResult:
        """执行 run 对应的业务逻辑并返回结果。"""
        if not command or any(not isinstance(item, str) for item in command):
            raise ValueError("command must be a non-empty string argument list")
        output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path, stderr_path = output_dir / "stdout.log", output_dir / "stderr.log"
        started = time.monotonic()
        timed_out = False
        stdout_offset = stdout_path.stat().st_size if stdout_path.is_file() else 0
        stderr_offset = stderr_path.stat().st_size if stderr_path.is_file() else 0
        stdout_pending = ""
        stderr_pending = ""
        last_reported = -1
        run_name = self._command_value(command, "--run-name", output_dir.name)
        total = int(self._command_value(command, "--max-iterations", "0") or 0)
        self._write_training_progress(output_dir, run_name, 0, total, "starting")
        print("[训练] {}：正在启动，目标迭代 {}".format(run_name, total), flush=True)
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
            self._process = subprocess.Popen(command, cwd=str(cwd), stdout=stdout, stderr=stderr,
                                             text=True, env=env, start_new_session=True, shell=False)
            capture_process_group(self._process)
            register_child_group(self._process)
            atomic_write_text(output_dir / "pid", str(self._process.pid) + "\n")
            identity = process_identity(self._process.pid)
            if identity is not None:
                write_json(output_dir / "process_identity.json", identity)
            deadline = started + timeout
            while self._process.poll() is None:
                lines, stdout_offset, stdout_pending = self._read_appended(
                    stdout_path, stdout_offset, stdout_pending)
                last_reported = self._consume_training_output(lines, output_dir, run_name, last_reported)
                error_lines, stderr_offset, stderr_pending = self._read_appended(
                    stderr_path, stderr_offset, stderr_pending)
                for line in error_lines:
                    print("[训练子进程] " + line, file=sys.stderr, flush=True)
                if time.monotonic() >= deadline:
                    timed_out = True
                    self.stop()
                    break
                time.sleep(0.2)
            exit_code = self._process.wait(timeout=10)
            lines, stdout_offset, stdout_pending = self._read_appended(stdout_path, stdout_offset, stdout_pending)
            self._consume_training_output(lines + ([stdout_pending] if stdout_pending else []),
                                          output_dir, run_name, last_reported)
            error_lines, stderr_offset, stderr_pending = self._read_appended(
                stderr_path, stderr_offset, stderr_pending)
            for line in error_lines + ([stderr_pending] if stderr_pending else []):
                print("[训练子进程] " + line, file=sys.stderr, flush=True)
        self._write_training_progress(output_dir, run_name, total if exit_code == 0 else max(0, last_reported),
                                      total, "completed" if exit_code == 0 else "failed")
        print("[训练] {}：{}，退出码 {}".format(
            run_name, "运行完成" if exit_code == 0 else "运行失败", exit_code), flush=True)
        result = ProcessResult(command=command, exit_code=exit_code, timed_out=timed_out,
                               duration_seconds=time.monotonic() - started, pid=self._process.pid)
        write_json(output_dir / "process_result.json", result.__dict__)
        return result

    def stop(self, grace_seconds: float = 5.0) -> None:
        """幂等清理当前独立进程组，不因组长提前退出而放过残留后代。"""
        process = self._process
        if process is not None:
            stop_process_group(process, grace_seconds)
