from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..utils.io import atomic_write_text, write_json


PROGRESS_PATTERN = re.compile(r"Learning iteration\s+(\d+)\s*/\s*(\d+)")


@dataclass
class ProcessResult:
    command: List[str]
    exit_code: int
    timed_out: bool
    duration_seconds: float
    pid: int


class ProcessManager:
    """仅运行预构造参数数组，不暴露 shell 命令接口。"""

    def __init__(self):
        """初始化 ProcessManager 实例及其运行依赖。"""
        self._process: Optional[subprocess.Popen] = None

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
            atomic_write_text(output_dir / "pid", str(self._process.pid) + "\n")
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
        self._process = None
        return result

    def stop(self) -> None:
        """向当前子进程组发送安全终止信号。"""
        if self._process is None or self._process.poll() is not None:
            return
        os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
