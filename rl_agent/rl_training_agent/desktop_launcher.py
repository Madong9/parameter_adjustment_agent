from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import List, Optional

from .settings import load_settings
from .web_ui import JobManager, create_server


BROWSER_CANDIDATES = ["google-chrome", "chromium", "chromium-browser"]


def find_desktop_browser(candidates: Optional[List[str]] = None) -> Optional[str]:
    """查找能够以独立应用窗口运行上位机的 Chromium 系浏览器。"""
    for name in candidates or BROWSER_CANDIDATES:
        executable = shutil.which(name)
        if executable:
            return executable
    return None


def build_browser_command(executable: str, url: str, profile_dir: Path) -> List[str]:
    """构造无地址栏、隔离用户配置且不加载外部扩展的桌面窗口命令。"""
    return [
        executable,
        "--app={}".format(url),
        "--user-data-dir={}".format(profile_dir),
        "--window-size=1440,900",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-default-apps",
        "--disable-sync",
        "--disable-translate",
        "--disable-extensions",
        "--class=RLTrainingDesktop",
    ]


def stop_active_jobs(manager: JobManager) -> None:
    """在桌面窗口关闭时安全停止仍由本次上位机控制的训练作业。"""
    for job in manager.list_jobs():
        if job.get("can_stop"):
            try:
                manager.stop_job(job["job_id"])
            except (KeyError, ValueError):
                continue


def serve_desktop() -> int:
    """启动本地训练服务和无浏览器工具栏的独立上位机软件窗口。"""
    executable = find_desktop_browser()
    if executable is None:
        from .desktop_ui import serve_tk_desktop
        print("未找到 Chrome 或 Chromium，正在使用兼容版 Tk 桌面界面。")
        return serve_tk_desktop()

    settings = load_settings()
    manager = JobManager(settings)
    server = create_server("127.0.0.1", 0, manager)
    server_thread = threading.Thread(target=server.serve_forever, name="rl-desktop-server", daemon=True)
    server_thread.start()
    url = "http://127.0.0.1:{}/".format(server.server_address[1])
    settings.artifacts_path.mkdir(parents=True, exist_ok=True)
    process: Optional[subprocess.Popen] = None
    print("强化学习训练上位机正在启动……")
    try:
        with tempfile.TemporaryDirectory(prefix="desktop-profile-", dir=str(settings.artifacts_path)) as profile:
            command = build_browser_command(executable, url, Path(profile))
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False)
            return_code = process.wait()
            return 0 if return_code == 0 else return_code
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            process.terminate()
        return 130
    finally:
        stop_active_jobs(manager)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=3)

