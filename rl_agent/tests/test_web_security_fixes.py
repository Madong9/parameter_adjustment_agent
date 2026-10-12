"""验证本地界面鉴权、原子租约与安全子进程回收，不加载真实凭据。"""
import http.client
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from rl_training_agent.desktop_launcher import stop_active_jobs
from rl_training_agent.settings import Settings
from rl_training_agent.training.process_manager import (
    ProcessManager, capture_process_group, process_identity, registered_child_groups,
    signal_registered_groups, stop_process_group, wait_registered_groups,
)
from rl_training_agent.web_ui import JobManager, JobValidationError, create_server


@pytest.fixture(autouse=True)
def no_local_secrets(monkeypatch):
    """测试禁止读取本地环境文件或调用真实 GPU 采样。"""
    monkeypatch.setattr("rl_training_agent.settings._load_local_env_file", lambda: None)
    monkeypatch.setattr("rl_training_agent.web_ui.sample_gpu", lambda ignored: {})


@pytest.fixture
def manager(tmp_path):
    """为每个测试隔离作业、训练、记忆和实验目录。"""
    return JobManager(Settings(artifact_root=str(tmp_path / "artifacts"),
                               experiment_root=str(tmp_path / "experiments"),
                               training_project=str(tmp_path / "training"), gpu_ids=[0]))


@pytest.fixture
def server(manager):
    """只临时启动随机 loopback 端口，测试结束后停止服务和所有受控作业。"""
    instance = create_server("127.0.0.1", 0, manager)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield instance
    finally:
        instance.shutdown()
        manager.shutdown()
        instance.server_close()
        thread.join(timeout=3)


def request(server, path, method="GET", body=None, headers=None):
    """发送有限超时的本地测试请求并返回状态、响应头及响应体。"""
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def session_headers(server):
    """通过合法首页取得 HttpOnly 会话，仅保留 Cookie 供同源 API 测试。"""
    status, headers, body = request(server, "/")
    assert status == 200
    assert "HttpOnly" in headers["Set-Cookie"] and "SameSite=Strict" in headers["Set-Cookie"]
    token = headers["Set-Cookie"].split(";", 1)[0]
    assert token.encode() not in body
    return {"Cookie": token, "Content-Type": "application/json"}


def test_api_requires_session_and_session_is_not_reissued_by_api(server):
    """未经首页初始化的读写 API 均拒绝，不从 API 响应泄露凭据。"""
    for path, method, body in (("/api/playback", "GET", None), ("/api/play/stop", "POST", "{}")):
        status, headers, body = request(server, path, method, body, {"Content-Type": "application/json"})
        assert status == 401
        assert "Set-Cookie" not in headers
    headers = session_headers(server)
    assert request(server, "/api/playback", headers=headers)[0] == 200


@pytest.mark.parametrize("headers", [
    {"Host": "attacker.example"}, {"Origin": "https://attacker.example"},
    {"Sec-Fetch-Site": "cross-site"}, {"Origin": "null"},
])
def test_invalid_host_or_cross_origin_cannot_initialize_cookie(server, headers):
    """非法 Host、跨源及 opaque origin 不能通过首页获得会话。"""
    status, response_headers, _ = request(server, "/", headers=headers)
    assert status == 403 and "Set-Cookie" not in response_headers


def test_foreign_origin_and_plaintext_mutations_are_rejected(server, manager, monkeypatch):
    """即使已有 Cookie，跨源 text/plain 和空停止请求也不能改变服务状态。"""
    called = []
    monkeypatch.setattr(manager, "stop_playback", lambda: called.append(True))
    headers = session_headers(server)
    assert request(server, "/api/play/stop", "POST", "{}",
                   {**headers, "Origin": "https://attacker.example"})[0] == 403
    assert request(server, "/api/play/stop", "POST", "{}",
                   {**headers, "Content-Type": "text/plain"})[0] == 400
    assert request(server, "/api/play/stop", "POST", None, headers)[0] == 400
    assert called == []


def test_same_origin_json_stop_remains_open_and_use(server):
    """桌面首页初始化后同源 JSON 停止请求可直接使用，不需 URL token。"""
    headers = session_headers(server)
    headers["Origin"] = "http://127.0.0.1:%d" % server.server_port
    assert request(server, "/api/play/stop", "POST", "{}", headers)[0] == 200


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.2", "attacker.example"])
def test_non_loopback_binding_is_rejected_before_manager_creation(host, monkeypatch):
    """远程绑定必须先被拒绝，不加载配置或启动历史等待队列。"""
    monkeypatch.setattr("rl_training_agent.web_ui.JobManager", lambda: pytest.fail("不应创建管理器"))
    with pytest.raises(JobValidationError, match="SSH"):
        create_server(host, 0)


class FakeProcess:
    """保持租约有效的虚拟进程，不触碰系统中任何真实 PID。"""
    def __init__(self, pid):
        """为虚拟进程分配只用于测试状态的标识。"""
        self.pid = pid

    def poll(self):
        """模拟进程持续运行。"""
        return None


def fake_launches(manager, monkeypatch):
    """替换启动和监视器，记录资源预留结果但不创建真实训练。"""
    launched = []

    def popen(command, **kwargs):
        """加入短暂启动窗口以验证并发请求不能越过资源锁。"""
        time.sleep(0.02)
        process = FakeProcess(1000000000 + len(launched))
        launched.append(process)
        return process

    monkeypatch.setattr("rl_training_agent.web_ui.subprocess.Popen", popen)
    monkeypatch.setattr(manager, "_monitor_job", lambda *args: None)
    return launched


def run_concurrently(callables):
    """同时触发多个入口并收集成功结果与预期验证错误。"""
    barrier = threading.Barrier(len(callables))
    results, errors = [], []

    def run(call):
        """在统一屏障之后进入真实管理器方法。"""
        barrier.wait(timeout=3)
        try:
            results.append(call())
        except JobValidationError as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(call,)) for call in callables]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
        assert not thread.is_alive()
    return results, errors


def test_concurrent_jobs_reserve_gpu_once(manager, monkeypatch):
    """两个不同作业并发提交时只启动一个，第二个持久等待且无 GPU 租约。"""
    launched = fake_launches(manager, monkeypatch)
    results, errors = run_concurrently([
        lambda: manager.start_job({"task": "safe job alpha", "robot": "go2", "mode": "dry-run"}),
        lambda: manager.start_job({"task": "safe job bravo", "robot": "go2", "mode": "dry-run"}),
    ])
    assert errors == [] and len(launched) == 1
    assert sorted(result["status"] for result in results) == ["queued", "running"]
    assert sum(result["gpu_id"] == 0 for result in results) == 1


def playback_artifacts(manager, monkeypatch):
    """提供受限路径的播放产物桩，不加载真实模型或配置。"""
    task_dir = manager.settings.experiments_path / "task-012345abcd"
    monkeypatch.setattr(manager, "_playable_artifacts", lambda ignored: {
        "robot": "go2", "config": task_dir / "config.yaml", "checkpoint": task_dir / "checkpoint.pt",
    })


def test_concurrent_viewers_have_one_tracked_process(manager, monkeypatch):
    """播放并发请求只创建一个进程，其余请求明确拒绝且不覆盖引用。"""
    playback_artifacts(manager, monkeypatch)
    launched = fake_launches(manager, monkeypatch)
    results, errors = run_concurrently([lambda: manager.start_playback("task-012345abcd")] * 2)
    assert len(results) == len(errors) == len(launched) == 1
    assert manager._playback_process is launched[0]


def test_running_viewer_prevents_training_launch(manager, monkeypatch):
    """Viewer 租约存在时训练入口不得分配相同 GPU 或启动新作业。"""
    manager._playback_process = FakeProcess(1000000000)
    monkeypatch.setattr(manager, "_launch_job_process", lambda *args: pytest.fail("不可启动"))
    with pytest.raises(JobValidationError, match="Viewer"):
        manager.start_job({"task": "safe viewer overlap", "robot": "go2", "mode": "real"})


def test_stopping_viewer_retains_lease_after_leader_exit(manager, monkeypatch):
    """Viewer 组长退出但后代清理未完成时仍保持租约，不提前启动 GPU 训练。"""
    viewer = FakeProcess(1000000000)
    monkeypatch.setattr(viewer, "poll", lambda: 0)
    manager._playback_process = viewer
    manager._playback_stopping = True
    assert manager.playback_status()["running"]
    with pytest.raises(JobValidationError, match="Viewer"):
        manager.start_job({"task": "safe viewer stopping", "robot": "go2", "mode": "real"})


def test_desktop_shutdown_cancels_all_queues_and_viewer(manager, monkeypatch):
    """退出取消所有队列而非仅最近二十条，并回收 Viewer 且禁止后续派发。"""
    for number in range(25):
        job_id = "%012x" % number
        manager._jobs[job_id] = {"job_id": job_id, "status": "queued"}
    viewer = FakeProcess(1000000000)
    manager._playback_process = viewer
    stopped = []
    monkeypatch.setattr("rl_training_agent.web_ui.stop_process_group", lambda process, *args: stopped.append(process))
    monkeypatch.setattr(manager, "_start_persisted_job", lambda *args: pytest.fail("关闭后不可派发"))
    stop_active_jobs(manager)
    manager._dispatch_queued()
    assert all(job["status"] == "stopped" for job in manager._jobs.values())
    assert stopped == [viewer]
    with pytest.raises(JobValidationError, match="关闭"):
        manager.start_job({"task": "safe closed probe", "robot": "go2", "mode": "dry-run"})


def test_log_and_job_directory_symlinks_cannot_escape(manager, tmp_path):
    """日志叶节点及作业目录符号链接均不能越界读取安全标记文件。"""
    job_id = "012345abcdef"
    job_dir = manager.root / job_id
    job_dir.mkdir()
    marker = tmp_path / "outside-marker.txt"
    marker.write_text("outside-root safe marker")
    (job_dir / "training.log").symlink_to(marker)
    manager._jobs[job_id] = {"job_id": job_id}
    with pytest.raises(ValueError):
        manager.read_log(job_id)
    other_id = "abcdef012345"
    (manager.root / other_id).symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(JobValidationError):
        manager._job_dir(other_id)


def test_experiment_symlinks_are_skipped(manager, tmp_path):
    """恶意实验符号链接不会被 API 扫描或作为可播放策略解析。"""
    outside = tmp_path / "outside-experiment"
    outside.mkdir()
    (outside / "state.json").write_text('{"state":"COMPLETED"}')
    manager.settings.experiments_path.mkdir()
    (manager.settings.experiments_path / "task-012345abcd").symlink_to(outside, target_is_directory=True)
    assert manager.list_experiments() == []
    assert manager._playable_artifacts("task-012345abcd") is None


def test_forged_process_registration_cannot_authorize_kill(tmp_path):
    """没有本次进程内会话签名的记录不能授权对当前进程组发信号。"""
    path = tmp_path / "control.jsonl"
    identity = process_identity(os.getpid())
    identity["owner_pid"] = 123
    path.write_text(json.dumps({"identity": identity, "signature": "forged"}) + "\n")
    assert registered_child_groups(path, "test-token", 123) == []


@pytest.mark.parametrize("current", [None, {"pid": 123, "pgid": 123, "start_ticks": 999}])
def test_missing_or_reused_process_identity_cannot_authorize_kill(monkeypatch, current):
    """失踪组长或已重用 PID 不允许仅依据旧 PGID 向未知进程发送信号。"""
    sent = []
    monkeypatch.setattr("rl_training_agent.training.process_manager.process_identity", lambda ignored: current)
    monkeypatch.setattr("rl_training_agent.training.process_manager.os.killpg", lambda *args: sent.append(args))
    signal_registered_groups([{"pid": 123, "pgid": 123, "start_ticks": 1}], signal.SIGKILL)
    assert sent == []


def test_group_leader_exit_still_reclaims_tracked_descendant():
    """组长退出后仍能凭已快照后代启动身份清理忽略 TERM 的短时子进程。"""
    child_code = ("import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                  "print('ready',flush=True); time.sleep(15)")
    parent_code = ("import subprocess,sys,time; child=subprocess.Popen([sys.executable,'-u','-c'," +
                   repr(child_code) + "],stdout=subprocess.PIPE,text=True); child.stdout.readline(); "
                   "print(child.pid,flush=True); time.sleep(15)")
    parent = subprocess.Popen([sys.executable, "-u", "-c", parent_code],
                              start_new_session=True, stdout=subprocess.PIPE, text=True)
    child_pid = None
    try:
        child_pid = int(parent.stdout.readline().strip())
        capture_process_group(parent)
        identities = [parent._rl_group_identity]
        os.killpg(parent.pid, signal.SIGTERM)
        parent.wait(timeout=2)
        assert process_identity(child_pid) is not None
        stop_process_group(parent, grace_seconds=0)
        assert wait_registered_groups(identities, timeout=2)
    finally:
        stop_process_group(parent, grace_seconds=0)
        parent.stdout.close()


def test_stop_all_active_reclaims_real_worker(tmp_path):
    """进程内信号处理入口在统一宽限预算内停止真实安全 worker 并完成 finally。"""
    process_manager = ProcessManager()
    output_dir = tmp_path / "active-worker"
    errors = []

    def run():
        """只执行短时 sleep 来模拟当前正在运行的训练进程。"""
        try:
            process_manager.run([sys.executable, "-c", "import time; time.sleep(15)"], tmp_path, output_dir, 30)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        deadline = time.monotonic() + 3
        while not (output_dir / "process_identity.json").is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (output_dir / "process_identity.json").is_file()
        ProcessManager.stop_all_active(grace_seconds=0.1)
        thread.join(timeout=3)
        assert not thread.is_alive() and process_manager._process is None
        assert errors == []
    finally:
        process_manager.stop(grace_seconds=0)
        thread.join(timeout=3)


def test_registered_child_is_reclaimed_after_cli_sigkill(manager, tmp_path):
    """短时真实 ProcessManager 会话在外层 SIGKILL 后仍由签名记录回收。"""
    job_id = "012345abcdef"
    (manager.root / job_id).mkdir()
    control_path = manager._job_artifact(job_id, "job_control.jsonl")
    output_dir = tmp_path / "safe-worker"
    token = "test-only-control-token"
    environment = os.environ.copy()
    environment.update(RL_AGENT_JOB_CONTROL_FILE=str(control_path), RL_AGENT_JOB_CONTROL_TOKEN=token)
    code = (
        "import sys; from pathlib import Path; "
        "from rl_training_agent.training.process_manager import ProcessManager; "
        "ProcessManager().run([sys.executable,'-u','-c','import time; time.sleep(15)'],"
        "Path('/tmp'),Path(" + repr(str(output_dir)) + "),30)"
    )
    parent = subprocess.Popen([sys.executable, "-u", "-c", code], env=environment,
                              start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    child_pid = None
    try:
        deadline = time.monotonic() + 3
        while not (output_dir / "process_identity.json").is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        identity = json.loads((output_dir / "process_identity.json").read_text())
        child_pid = identity["pid"]
        assert identity == process_identity(child_pid)
        assert child_pid != parent.pid
        os.killpg(parent.pid, signal.SIGKILL)
        parent.wait(timeout=2)
        manager._job_control_tokens[job_id] = token
        groups = manager._registered_children(job_id, parent)
        assert [entry["pid"] for entry in groups] == [child_pid]
        manager._stop_registered_children(job_id, parent, signal.SIGKILL)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            stat_path = Path("/proc/%d/stat" % child_pid)
            if not stat_path.exists() or stat_path.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                break
            time.sleep(0.01)
        else:
            pytest.fail("签名登记的子进程仍在运行")
    finally:
        if child_pid is not None:
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        stop_process_group(parent, grace_seconds=0)
