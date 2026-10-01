import json
import subprocess

import pytest
from pydantic import BaseModel

from rl_training_agent.providers.errors import ProviderError, ProviderNeedsHuman, ProviderResponseError, ProviderTimeout
from rl_training_agent.providers.fallback import FallbackProvider
from rl_training_agent.providers.opencli_chatgpt import OpenCLIChatGPTWebProvider, _compact_capabilities
from rl_training_agent.providers.opencli_doubao import OpenCLIDoubaoWebProvider
from rl_training_agent.schemas.experiments import ConversationHandle
from rl_training_agent.settings import OpenCLISettings, _load_local_env_file


class Reply(BaseModel):
    ok: bool


def test_local_env_file_loads_once_without_overriding_process_environment(tmp_path, monkeypatch):
    """验证本地 .env 能持久提供密钥，但不覆盖当前终端显式设置。"""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# 本地密钥\nDASHSCOPE_API_KEY=stored-key\nBAILIAN_MODEL='stored-model'\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("BAILIAN_MODEL", "terminal-model")
    _load_local_env_file(env_file)
    assert __import__("os").environ["DASHSCOPE_API_KEY"] == "stored-key"
    assert __import__("os").environ["BAILIAN_MODEL"] == "terminal-model"


def result(stdout="", stderr="", code=0):
    """构造测试所需的result数据。"""
    return subprocess.CompletedProcess([], code, stdout, stderr)


def test_doctor_normal(monkeypatch):
    """验证“doctor normal”场景的预期行为。"""
    monkeypatch.setattr("shutil.which", lambda _: "/bin/opencli")
    def runner(args, timeout):
        """执行 runner 对应的业务逻辑并返回结果。"""
        if args[1] == "doctor":
            return result("[OK] Extension: connected")
        return result('[{"Status":"Connected","Login":"Yes"}]')
    health = OpenCLIChatGPTWebProvider(runner=runner).doctor()
    assert health.available and health.chatgpt_logged_in


def test_doctor_failure(monkeypatch):
    """验证“doctor failure”场景的预期行为。"""
    monkeypatch.setattr("shutil.which", lambda _: None)
    health = OpenCLIChatGPTWebProvider().doctor()
    assert not health.available and not health.recoverable


def test_doctor_timeout_returns_unhealthy_result(monkeypatch):
    """验证 OpenCLI 状态命令超时时 doctor 返回诊断结果而不是打印调用栈。"""
    monkeypatch.setattr("shutil.which", lambda _: "/bin/opencli")

    def runner(args, timeout):
        """模拟扩展已连接但 ChatGPT 登录状态查询超时。"""
        if args[1] == "doctor":
            return result("[OK] Extension: connected")
        raise subprocess.TimeoutExpired(args, timeout)

    health = OpenCLIChatGPTWebProvider(runner=runner).doctor()
    assert not health.available
    assert health.opencli_available and health.extension_connected
    assert any("登录状态检查失败" in item for item in health.details)


@pytest.mark.parametrize("state", ["Log in to ChatGPT", "Verify you are human CAPTCHA"])
def test_login_and_captcha_states(state):
    """验证“login and captcha states”场景的预期行为。"""
    provider = OpenCLIChatGPTWebProvider(runner=lambda args, timeout: result(state))
    with pytest.raises(ProviderNeedsHuman):
        provider._state()


@pytest.mark.parametrize("state", [
    "你暂时已用完工作用量，升级或添加额度以立即继续使用工作，或等待用量在 18:22 重置",
    "你已达到使用上限。升级订阅套餐或添加额度即可继续，也可在 18:22 后重试。",
    "You've reached your usage limit. Upgrade or wait before trying again.",
])
def test_usage_limit_state_returns_actionable_error(state):
    """验证 ChatGPT 额度耗尽时立即返回明确提示而不是等待发送或解析超时。"""
    provider = OpenCLIChatGPTWebProvider(runner=lambda args, timeout: result(state))
    with pytest.raises(ProviderNeedsHuman, match="工作用量已耗尽"):
        provider._state()


def test_usage_limit_response_is_not_treated_as_invalid_json(monkeypatch):
    """验证模型用量提示在回复轮询阶段被识别为额度问题。"""
    provider = OpenCLIChatGPTWebProvider(settings=OpenCLISettings(response_timeout=1))
    monkeypatch.setattr(provider, "_state", lambda: "ready")
    monkeypatch.setattr(provider, "_latest_assistant", lambda: "你已达到使用上限，请添加额度。")
    with pytest.raises(ProviderNeedsHuman, match="工作用量已耗尽"):
        provider._wait_for_response("")


def test_json_code_block_and_balanced_extraction():
    """验证“json code block and balanced extraction”场景的预期行为。"""
    provider = OpenCLIChatGPTWebProvider()
    assert provider.parse_json_response("text\n```json\n{\"ok\": true}\n```", Reply).ok
    assert provider.parse_json_response("prefix {\"ok\": false} suffix", Reply).ok is False


def test_json_parser_repairs_only_extra_quote_after_operator_value():
    """验证比较运算符后的单个多余引号可修复，其他非法 JSON 仍会失败。"""
    provider = OpenCLIChatGPTWebProvider()
    repaired = provider._parse_candidate('{"operator":"<="","value":1}')
    assert repaired == {"operator": "<=", "value": 1}
    with pytest.raises(ValueError):
        provider._parse_candidate('{"arbitrary":"broken""}')


def test_invalid_json_validation_error():
    """验证“invalid json validation error”场景的预期行为。"""
    with pytest.raises(ProviderResponseError):
        OpenCLIChatGPTWebProvider().parse_json_response('{"wrong": 1}', Reply)


def test_timeout_is_wrapped():
    """验证“timeout is wrapped”场景的预期行为。"""
    def runner(args, timeout):
        """执行 runner 对应的业务逻辑并返回结果。"""
        raise subprocess.TimeoutExpired(args, timeout)
    with pytest.raises(ProviderTimeout):
        OpenCLIChatGPTWebProvider(runner=runner)._run(["doctor"])


def test_bridge_disconnect_restarts_daemon_and_recovers(monkeypatch):
    """验证扩展断线后会自动重启守护进程并等待连接恢复。"""
    calls = []
    doctor_outputs = iter([
        "[MISSING] Extension: not connected",
        "[OK] Extension: connected",
    ])

    def runner(args, timeout):
        """模拟第一次检查断线并在守护进程重启后恢复连接。"""
        calls.append(args)
        if args[1] == "doctor":
            return result(next(doctor_outputs))
        return result("Daemon restarted")

    monkeypatch.setattr("time.sleep", lambda _: None)
    provider = OpenCLIChatGPTWebProvider(settings=OpenCLISettings(connect_timeout=2), runner=runner)
    provider._ensure_bridge_connected()
    assert any(call[1:3] == ["daemon", "restart"] for call in calls)


def test_bridge_browser_auto_launch_uses_default_chrome(monkeypatch):
    """验证扩展断线时使用默认 Chrome 打开 ChatGPT 以激活桥接。"""
    launched = []

    class ProcessStub:
        """为浏览器启动测试提供无副作用进程占位符。"""

    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr("shutil.which", lambda name: "/opt/google/chrome/chrome"
                        if name == "google-chrome" else None)
    monkeypatch.setattr("subprocess.Popen", lambda command, **kwargs:
                        launched.append((command, kwargs)) or ProcessStub())
    provider = OpenCLIChatGPTWebProvider()
    assert provider._launch_bridge_browser()
    assert launched[0][0] == [
        "/opt/google/chrome/chrome", "--new-window", "https://chatgpt.com/"]
    assert launched[0][1]["shell"] is False


def test_bridge_browser_auto_launch_requires_graphical_session(monkeypatch):
    """验证无图形会话时不会启动浏览器子进程。"""
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    provider = OpenCLIChatGPTWebProvider()
    assert not provider._launch_bridge_browser()


def test_bridge_disconnect_reports_actionable_chinese_error(monkeypatch):
    """验证扩展无法自动重连时返回明确的中文人工恢复指引。"""
    clock = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr("time.monotonic", lambda: next(clock))
    monkeypatch.setattr("time.sleep", lambda _: None)
    provider = OpenCLIChatGPTWebProvider(
        settings=OpenCLISettings(connect_timeout=1),
        runner=lambda args, timeout: result("[MISSING] Extension: not connected"),
    )
    with pytest.raises(ProviderNeedsHuman, match="浏览器扩展未连接"):
        provider._ensure_bridge_connected()


def test_chatgpt_work_mode_is_switched_to_chat_mode(monkeypatch):
    """验证检测到工作模式时会点击稳定的聊天模式开关并确认状态。"""
    active = {"chat": False}
    calls = []

    def runner(args, timeout):
        """模拟 ChatGPT 模式开关从工作切换为聊天。"""
        calls.append(args)
        script = args[-1] if "eval" in args else ""
        if "chat.click()" in script:
            active["chat"] = True
            return result('{"clicked":true}')
        if "data-tpp-toggle-value" in script:
            return result(json.dumps({
                "available": True,
                "chat_active": active["chat"],
                "work_active": not active["chat"],
            }))
        return result("{}")

    provider = OpenCLIChatGPTWebProvider(
        settings=OpenCLISettings(force_chat_mode=True), runner=runner)
    monkeypatch.setattr("time.sleep", lambda _: None)
    provider._ensure_chat_mode()
    assert active["chat"]
    assert any("chat.click()" in str(call[-1]) for call in calls if "eval" in call)


def test_provider_failure_switches_whole_request_to_doubao(tmp_path):
    """验证 ChatGPT Provider 故障时相同高层请求只在豆包重试一次。"""
    class Primary:
        """模拟不可用的 ChatGPT Provider。"""

        def understand_task(self, instruction, robot):
            """始终报告 ChatGPT 不可用。"""
            raise ProviderNeedsHuman("ChatGPT 不可用")

        def close(self):
            """提供关闭兼容接口。"""

    class Backup:
        """模拟正常返回的豆包 Provider。"""

        def understand_task(self, instruction, robot):
            """返回可识别的备用结果。"""
            return {"provider": "doubao", "instruction": instruction, "robot": robot}

        def close(self):
            """提供关闭兼容接口。"""

    provider = FallbackProvider(Primary(), Backup(), record_dir=tmp_path)
    response = provider.understand_task("倒退行走", "go2")
    assert response["provider"] == "doubao"
    assert "自动切换到豆包" in (tmp_path / "fallback.log").read_text(encoding="utf-8")


def test_doubao_provider_uses_dialogue_mode_and_stable_message_selectors(monkeypatch):
    """验证豆包 Provider 使用对话模式及 send/receive 消息测试标识。"""
    provider = OpenCLIDoubaoWebProvider(
        settings=OpenCLISettings(doubao_session="doubao-test"),
        runner=lambda args, timeout: result("{}"))
    monkeypatch.setattr(provider, "_parse_eval_result", lambda output: {
        "ok": True, "changed": False})
    provider._ensure_dialogue_mode()
    assert "send_message" in provider.USER_SELECTOR
    assert "receive_message" in provider.ASSISTANT_SELECTOR
    assert provider.settings.session == "doubao-test"


def test_doubao_json_reply_is_preserved_as_text():
    """验证豆包正文自身是合法 JSON 时不会被二次解析并丢成空字符串。"""
    provider = OpenCLIDoubaoWebProvider(
        settings=OpenCLISettings(doubao_session="doubao-test"),
        runner=lambda args, timeout: result(json.dumps('{"ok":true}')))
    assert provider._latest_assistant() == '{"ok":true}'


def test_doubao_composer_is_cleared_before_verified_fill(monkeypatch):
    """验证重试前清空豆包草稿，并用 textContent 校验一次完整写入。"""
    calls = []

    def runner(args, timeout):
        """模拟豆包页面状态、模式检查、填入和页面内全文校验。"""
        calls.append(args)
        if "state" in args:
            return result("chat_input 新对话")
        if "fill" in args:
            return result('{"filled":true,"verified":false,"actual":""}')
        if "eval" in args and "actual_length" in args[-1]:
            return result('{"found":true,"matches":true,"actual_length":4,"expected_length":4}')
        if "eval" in args:
            return result('{"ok":true,"changed":false}')
        return result('{"clicked":true}')

    provider = OpenCLIDoubaoWebProvider(
        settings=OpenCLISettings(doubao_session="doubao-test"), runner=runner)
    provider._fill_prompt("完整提示")
    assert any(call[-1] == "Control+a" for call in calls if "keys" in call)
    assert any(call[-1] == "Backspace" for call in calls if "keys" in call)


def test_bind_timeout_is_converted_to_human_recovery_instruction(monkeypatch):
    """验证标签页绑定超时不再暴露底层堆栈，而是提示用户恢复浏览器。"""
    provider = OpenCLIChatGPTWebProvider()
    monkeypatch.setattr(provider, "_ensure_bridge_connected", lambda: None)
    monkeypatch.setattr(provider, "_browser", lambda *args, **kwargs: (_ for _ in ()).throw(ProviderTimeout("超时")))
    with pytest.raises(ProviderNeedsHuman, match="无法绑定当前浏览器标签页"):
        provider.open_or_bind()


def test_command_failure_keeps_stdout_and_stderr():
    """验证 OpenCLI 失败时真实标准输出不会被代理警告遮挡。"""
    provider = OpenCLIChatGPTWebProvider(
        runner=lambda args, timeout: result("send button not found", "UNDICI proxy warning", 1)
    )
    with pytest.raises(ProviderError) as error:
        provider._run(["browser", "test", "click"])
    assert "send button not found" in str(error.value)
    assert "UNDICI proxy warning" in str(error.value)


def test_fill_uses_semantic_locator_and_verifies():
    """验证“fill uses semantic locator and verifies”场景的预期行为。"""
    calls = []
    def runner(args, timeout):
        """执行 runner 对应的业务逻辑并返回结果。"""
        calls.append(args)
        if "state" in args:
            return result("Message ChatGPT textbox")
        if "find" in args:
            if "--css" in args:
                return result('{"matches_n":0,"entries":[]}')
            return result('{"matches_n":1,"entries":[]}')
        if "fill" in args:
            return result('{"filled":true,"verified":true}')
        return result("{}")
    provider = OpenCLIChatGPTWebProvider(runner=runner)
    provider._fill_prompt("hello")
    assert any("--role" in call and "textbox" in call for call in calls)
    assert not any("click" in call for call in calls)


def test_fill_accepts_prosemirror_whitespace_after_full_verification():
    """验证富文本编辑器只增加段落空白时仍能通过严格规范化全文校验。"""
    calls = []

    def runner(args, timeout):
        """模拟 OpenCLI 精确校验失败但页面内规范化全文一致。"""
        calls.append(args)
        if "state" in args:
            return result("Message ChatGPT textbox")
        if "find" in args:
            return result('{"matches_n":1,"entries":[{"visible":true}]}')
        if "fill" in args:
            return result('{"filled":true,"verified":false,"actual":"第一段\\n\\n\\n第二段"}')
        if "eval" in args:
            return result('{"found":true,"matches":true,"actual_length":9,"expected_length":7,'
                          '"actual_normalized_length":7,"expected_normalized_length":7,'
                          '"prefix_matches":true,"suffix_matches":true}')
        return result("{}")

    provider = OpenCLIChatGPTWebProvider(runner=runner)
    provider._fill_prompt("第一段\n第二段")
    assert any("eval" in call for call in calls)


def test_compact_capabilities_removes_duplicate_and_path_fields():
    """验证发送给网页的环境清单保留设计字段并去除重复观察与源码路径。"""
    capabilities = {
        "project": "../unitree_rl_gym",
        "robot": "go2",
        "robots": ["go2"],
        "observations": [{"name": "base_lin_vel", "source_file": "large.py"}],
        "reward_variables": [{"name": "base_lin_vel", "shape": [1, 3], "unit": "m/s",
                              "source_file": "large.py", "source_symbol": "Large.symbol"}],
        "rewards": [{"name": "tracking", "implementation": "_reward_tracking",
                     "config_key": "tracking", "dependencies": ["base_lin_vel"],
                     "source_file": "large.py"}],
        "terminations": ["fall"],
        "command_space": ["lin_vel_x"],
    }
    compact = _compact_capabilities(capabilities)
    assert "observations" not in compact
    assert compact["reward_variables"][0]["name"] == "base_lin_vel"
    assert "source_file" not in compact["reward_variables"][0]
    assert compact["rewards"][0]["dependencies"] == ["base_lin_vel"]


def test_compact_capabilities_preserves_rag_evidence():
    """验证精简环境清单不会丢失已限长的 RAG 证据。"""
    context = {"enabled": True, "hits": [{"source": "docs/agent/RAG.md", "text": "历史经验"}]}
    compact = _compact_capabilities({"retrieved_experience": context})
    assert compact["retrieved_experience"] == context


def test_submit_explicitly_clicks_enabled_send_button(monkeypatch):
    """验证提交动作显式点击已启用的发送按钮而不是模拟 Enter。"""
    calls = []

    def runner(args, timeout):
        """返回发送按钮查找和点击所需的模拟 OpenCLI 响应。"""
        calls.append(args)
        if "state" in args:
            return result("Message ChatGPT textbox")
        if "find" in args:
            return result('{"matches_n":1,"entries":[{"visible":true}]}')
        if "click" in args:
            return result('{"clicked":true,"matches_n":1}')
        return result("{}")

    provider = OpenCLIChatGPTWebProvider(runner=runner)
    monkeypatch.setattr("time.sleep", lambda _: None)
    provider._click_send_button()
    assert any("click" in call and "send-button" in " ".join(call) for call in calls)
    assert not any("keys" in call for call in calls)


def test_send_button_detects_usage_limit_from_full_page_before_click(monkeypatch):
    """验证精简 state 漏掉额度弹窗时，发送前仍能从完整页面正文识别。"""
    calls = []

    def runner(args, timeout):
        """模拟页面正文显示额度为零，同时记录是否错误执行了点击。"""
        calls.append(args)
        if "eval" in args:
            return result(json.dumps("你暂时已用完工作用量，可用额度：0"))
        return result("{}")

    provider = OpenCLIChatGPTWebProvider(runner=runner)
    monkeypatch.setattr("time.sleep", lambda _: None)
    with pytest.raises(ProviderNeedsHuman, match="工作用量已耗尽"):
        provider._click_send_button()
    assert not any("click" in call for call in calls)


def test_send_verifies_submission_before_waiting_for_reply(monkeypatch):
    """验证发送流程依次填入、点击、确认用户消息并等待一次回复。"""
    provider = OpenCLIChatGPTWebProvider(settings=OpenCLISettings(max_retries=1))
    events = []
    conversation = ConversationHandle(conversation_id="verified", title_hint="verified")
    monkeypatch.setattr(provider, "_latest_assistant", lambda: "old assistant")
    monkeypatch.setattr(provider, "_latest_user", lambda: "old user")
    monkeypatch.setattr(provider, "_fill_prompt", lambda prompt: events.append(("fill", prompt)))
    monkeypatch.setattr(provider, "_click_send_button", lambda: events.append(("click", None)))
    monkeypatch.setattr(provider, "_wait_for_submission",
                        lambda previous, prompt, timeout=None:
                        events.append(("verify", previous, prompt)))
    monkeypatch.setattr(provider, "_wait_for_response",
                        lambda previous: events.append(("wait", previous)) or "new assistant")
    monkeypatch.setattr(provider, "_record", lambda handle, prompt, response: events.append(("record", response)))
    assert provider.send_text("hello", conversation) == "new assistant"
    assert [event[0] for event in events] == ["fill", "click", "verify", "wait", "record"]


def test_submission_accepts_cleared_composer_before_delayed_message_dom(monkeypatch):
    """验证新会话已清空编辑器时，不因用户消息 DOM 延迟而误报。"""
    provider = OpenCLIChatGPTWebProvider(settings=OpenCLISettings(submit_timeout=1))
    monkeypatch.setattr(provider, "_submission_snapshot", lambda: {
        "latest_user": "", "composer_found": True, "composer_text": "",
        "url": "https://chatgpt.com/c/example", "generating": False,
    })
    provider._wait_for_submission("", "请返回严格 JSON")


def test_submission_does_not_accept_prompt_still_in_composer(monkeypatch):
    """验证提示词仍留在编辑器时不会误判为已成功提交。"""
    provider = OpenCLIChatGPTWebProvider(settings=OpenCLISettings(submit_timeout=1))
    clock = iter([0.0, 2.0, 3.0])
    monkeypatch.setattr("time.monotonic", lambda: next(clock))
    monkeypatch.setattr(provider, "_state", lambda: "ready")
    monkeypatch.setattr(provider, "_submission_snapshot", lambda: {
        "latest_user": "", "composer_found": True, "composer_text": "请返回严格 JSON",
        "url": "https://chatgpt.com/", "generating": False,
    })
    with pytest.raises(ProviderTimeout, match="编辑器存在=True"):
        provider._wait_for_submission("", "请返回严格 JSON")


def test_file_submission_uses_extended_timeout_without_reclicking(monkeypatch, tmp_path):
    """验证附件请求使用更长提交等待，而不是重复点击发送。"""
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")
    provider = OpenCLIChatGPTWebProvider(settings=OpenCLISettings(submit_timeout=45))
    monkeypatch.setattr(provider, "_state", lambda: "ready")
    monkeypatch.setattr(provider, "_browser", lambda *args, **kwargs: (
        '{"matches_n":1,"entries":[{"nth":0,"compound":{}}]}'
        if args[0][0] == "find" else '{"uploaded":true}'))
    monkeypatch.setattr(provider, "_wait_for_upload_preview", lambda names: None)
    received = []
    monkeypatch.setattr(provider, "send_text", lambda prompt, conversation,
                        submission_timeout=None: received.append(submission_timeout) or '{"ok":true}')
    raw = provider.send_with_files(
        "inspect", [image], ConversationHandle(conversation_id="files", title_hint="files"))
    assert json.loads(raw)["ok"]
    assert received == [120]


def test_visual_evidence_selection_limits_redundant_images(tmp_path):
    """验证视觉评价只上传两张互补关键图，并继续携带结构化物理证据。"""
    names = [
        "contact_sheet_clean.png", "contact_sheet_annotated.png",
        "contact_sheet_multiview.png", "event_landing.png",
        "behavior_evidence.json", "visual_attachment_manifest.json",
    ]
    files = []
    for name in names:
        path = tmp_path / name
        path.write_text("evidence")
        files.append(path)
    provider = OpenCLIChatGPTWebProvider(
        settings=OpenCLISettings(visual_image_attachment_limit=2))

    selected = provider._select_visual_evidence_files(files)

    assert [path.name for path in selected] == [
        "contact_sheet_annotated.png", "contact_sheet_multiview.png",
        "behavior_evidence.json", "visual_attachment_manifest.json",
    ]


def test_model_request_retries_silent_conversation_without_assistant(monkeypatch):
    """验证消息已提交但页面静默无回复时在新会话安全重试。"""
    provider = OpenCLIChatGPTWebProvider(settings=OpenCLISettings(max_retries=1))
    conversations = []
    sends = []

    def new_conversation(title):
        """记录静默恢复创建的新会话。"""
        conversations.append(title)
        return ConversationHandle(conversation_id=title, title_hint=title)

    def send(handle, prompt, title, files):
        """模拟首个会话静默超时，第二个会话正常返回。"""
        sends.append(handle.conversation_id)
        if len(sends) == 1:
            raise ProviderTimeout("ChatGPT response did not complete before timeout")
        return '{"ok": true}'

    monkeypatch.setattr(provider, "new_conversation", new_conversation)
    monkeypatch.setattr(provider, "_send_model_prompt", send)
    monkeypatch.setattr(provider, "_submission_snapshot", lambda: {
        "user_count": 1, "assistant_count": 0, "latest_assistant": "",
        "composer_found": True, "composer_text": "", "generating": False,
    })
    monkeypatch.setattr("time.sleep", lambda _: None)
    assert provider._request_model("probe", Reply, "silent").ok
    assert len(conversations) == 2
    assert conversations[1].startswith("silent-silent-retry-")


def test_model_request_does_not_retry_while_assistant_is_generating(monkeypatch):
    """验证助手仍在生成时不创建新会话重复请求。"""
    provider = OpenCLIChatGPTWebProvider(settings=OpenCLISettings(max_retries=2))
    conversations = []
    monkeypatch.setattr(provider, "new_conversation", lambda title: (
        conversations.append(title) or ConversationHandle(
            conversation_id=title, title_hint=title)))
    monkeypatch.setattr(provider, "_send_model_prompt", lambda *args: (
        (_ for _ in ()).throw(ProviderTimeout("仍在生成"))))
    monkeypatch.setattr(provider, "_submission_snapshot", lambda: {
        "user_count": 1, "assistant_count": 0, "latest_assistant": "",
        "composer_found": True, "composer_text": "", "generating": True,
    })
    with pytest.raises(ProviderTimeout, match="仍在生成"):
        provider._request_model("probe", Reply, "generating")
    assert conversations == ["generating"]


def test_submitted_prompt_allows_rendered_markdown_but_rejects_truncation():
    """验证用户消息渲染时反引号消失可接受，而正文截断仍被拒绝。"""
    prompt = "只能使用 `ENVIRONMENT_MANIFEST`，返回 `task_spec` 和 `reward_plans`。"
    rendered = "只能使用 ENVIRONMENT_MANIFEST，返回 task_spec 和 reward_plans。"
    assert OpenCLIChatGPTWebProvider._matches_submitted_prompt(rendered, prompt)
    assert not OpenCLIChatGPTWebProvider._matches_submitted_prompt(rendered[:-8], prompt)


def test_submitted_prompt_accepts_unicode_format_and_single_dom_character():
    """验证长提示词的零宽字符和单个 DOM 排版差异不会造成误报。"""
    prompt = ("请根据机器人环境能力生成严格 JSON，不得省略字段。" * 60)
    with_format_characters = "\u200b" + prompt[:500] + "\ufeff" + prompt[500:]
    assert OpenCLIChatGPTWebProvider._matches_submitted_prompt(
        with_format_characters, prompt)
    with_dom_marker = prompt[:700] + "↵" + prompt[700:]
    assert OpenCLIChatGPTWebProvider._matches_submitted_prompt(with_dom_marker, prompt)


def test_submitted_prompt_rejects_different_long_message():
    """验证长度接近但内容不同的新消息不会被误判为已提交。"""
    prompt = "A" * 800
    unrelated = "B" * 801
    assert not OpenCLIChatGPTWebProvider._matches_submitted_prompt(unrelated, prompt)


def test_submitted_prompt_accepts_attachment_metadata_prefix_only():
    """验证附件文件名与“文件”标签前缀可接受，而任意网页文本前缀不可接受。"""
    prompt = "请完整阅读附件并只返回严格 JSON。"
    rendered = "task-reward-design_需求文档.md\n文件\n" + prompt
    assert OpenCLIChatGPTWebProvider._matches_submitted_prompt(rendered, prompt)
    multiple = ("visual-critique_需求文档.md\n文件\nbehavior_evidence.json\n文件\n"
                "visual_attachment_manifest.json\n文件\n" + prompt)
    assert OpenCLIChatGPTWebProvider._matches_submitted_prompt(multiple, prompt)
    assert not OpenCLIChatGPTWebProvider._matches_submitted_prompt("未知页面提示\n" + prompt, prompt)


def test_file_upload(monkeypatch, tmp_path):
    """验证“file upload”场景的预期行为。"""
    uploaded = []
    file = tmp_path / "image.png"
    file.write_bytes(b"png")
    def runner(args, timeout):
        """执行 runner 对应的业务逻辑并返回结果。"""
        if "state" in args:
            return result("Message ChatGPT textbox")
        if "find" in args:
            return result('{"matches_n":1}')
        if "upload" in args:
            uploaded.extend(args)
            return result('{"uploaded":true,"files":1}')
        if "eval" in args:
            return result("true")
        return result("{}")
    provider = OpenCLIChatGPTWebProvider(runner=runner)
    monkeypatch.setattr(provider, "send_text",
                        lambda prompt, conversation, submission_timeout=None: '{"ok":true}')
    raw = provider.send_with_files("inspect", [file], ConversationHandle(conversation_id="x", title_hint="x"))
    assert "input[type=file]" in uploaded and json.loads(raw)["ok"]


def test_image_upload_falls_back_to_chunked_data_transfer(monkeypatch, tmp_path):
    """验证 Chrome 拒绝本地路径上传时改用分块 DataTransfer，并等待附件预览。"""
    image = tmp_path / "contact_sheet.png"
    image.write_bytes(b"image-bytes" * 12000)
    eval_scripts = []

    def runner(args, timeout):
        """模拟 set-file-input 返回 Not allowed 以及页面分块上传成功。"""
        if "state" in args:
            return result("Message ChatGPT textbox")
        if "find" in args:
            return result('{"matches_n":2,"entries":[{"nth":0,"compound":{}},'
                          '{"nth":1,"compound":{"accept":"image/*"}}]}')
        if "upload" in args:
            return result("", '{"code":-32000,"message":"Not allowed"}', 1)
        if "eval" in args:
            script = args[-1]
            eval_scripts.append(script)
            if "media.length" in script:
                return result("true")
            if "input.files = transfer.files" in script:
                return result('{"ok":true,"count":1,"names":["contact_sheet.png"]}')
            return result('{"ok":true}')
        return result("{}")

    provider = OpenCLIChatGPTWebProvider(runner=runner)
    monkeypatch.setattr(provider, "send_text",
                        lambda prompt, conversation, submission_timeout=None: '{"ok":true}')
    raw = provider.send_with_files(
        "评估动作", [image], ConversationHandle(conversation_id="visual", title_hint="visual")
    )
    chunk_scripts = [script for script in eval_scripts if ".chunks.push(" in script]
    assert json.loads(raw)["ok"]
    assert len(chunk_scripts) >= 3
    assert max(len(script) for script in chunk_scripts) < 60 * 1024
    assert any("input.files = transfer.files" in script for script in eval_scripts)


def test_file_chooser_timeout_is_treated_as_recoverable_upload_error():
    """验证 OpenCLI 未捕获文件选择器事件时会自动切换 DataTransfer 上传。"""
    error = ProviderError(
        "stderr: Page.fileChooserOpened not received within 5s — "
        "the input may not have opened a file chooser")
    assert OpenCLIChatGPTWebProvider._recoverable_file_upload_error(error)


def test_image_upload_does_not_hide_unrecoverable_error(tmp_path):
    """验证非安全策略类上传错误会原样抛出，便于上位机显示真实原因。"""
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")

    def runner(args, timeout):
        """模拟浏览器上传命令的不可恢复服务错误。"""
        if "state" in args:
            return result("Message ChatGPT textbox")
        if "find" in args:
            return result('{"matches_n":1,"entries":[{"nth":0,"compound":{"accept":"image/*"}}]}')
        if "upload" in args:
            return result("", "ChatGPT storage quota exceeded", 1)
        return result("{}")

    provider = OpenCLIChatGPTWebProvider(runner=runner)
    with pytest.raises(ProviderError, match="storage quota exceeded"):
        provider.send_with_files(
            "评估动作", [image], ConversationHandle(conversation_id="visual", title_hint="visual")
        )


def test_mixed_visual_request_uploads_document_and_images_separately(monkeypatch, tmp_path):
    """验证长视觉请求的需求文档不会被计入图片预览数量。"""
    document = tmp_path / "visual_需求文档.md"
    document.write_text("视觉要求", encoding="utf-8")
    image = tmp_path / "contact_sheet.png"
    image.write_bytes(b"png")
    uploads = []

    def runner(args, timeout):
        """分别模拟页面文档附件和 OpenCLI 图片附件上传成功。"""
        if "state" in args:
            return result("Message ChatGPT textbox")
        if "find" in args:
            return result('{"matches_n":2,"entries":[{"nth":0,"compound":{}},'
                          '{"nth":1,"compound":{"accept":"image/*"}}]}')
        if "upload" in args:
            uploads.append(args)
            return result('{"uploaded":true,"files":1}')
        if "eval" in args:
            script = args[-1]
            if "const documents" in script:
                return result('{"ok":true,"count":1,"names":["visual_需求文档.md"]}')
            if "media.length" in script:
                assert 'contact_sheet.png' in script
                assert 'visual_需求文档.md' not in script
                return result("true")
        return result("{}")

    provider = OpenCLIChatGPTWebProvider(runner=runner)
    monkeypatch.setattr(provider, "send_text",
                        lambda prompt, conversation, submission_timeout=None: '{"ok":true}')
    raw = provider.send_with_files(
        "评估动作", [document, image], ConversationHandle(conversation_id="mixed", title_hint="mixed")
    )
    assert json.loads(raw)["ok"]
    assert len(uploads) == 1
    assert uploads[0][uploads[0].index("--nth") + 1] == "0"
    assert str(image.resolve()) in uploads[0]
    assert str(document.resolve()) not in uploads[0]


def test_document_upload_uses_in_page_file_object(monkeypatch, tmp_path):
    """验证 Markdown 文档通过页面文件对象上传，绕过本地路径权限限制。"""
    calls = []
    document = tmp_path / "requirements.md"
    document.write_text("训练要求", encoding="utf-8")

    def runner(args, timeout):
        """模拟页面确认通过 JavaScript File 对象附加的需求文档。"""
        calls.append(args)
        if "state" in args:
            return result("Message ChatGPT textbox")
        if "eval" in args:
            return result('{"ok":true,"count":1,"names":["requirements.md"]}')
        return result("{}")

    provider = OpenCLIChatGPTWebProvider(runner=runner)
    monkeypatch.setattr(provider, "send_text", lambda prompt, conversation: '{"ok":true}')
    provider.send_with_files("读取附件", [document], ConversationHandle(conversation_id="doc", title_hint="doc"))
    assert any("eval" in call and "new File" in " ".join(call) for call in calls)
    assert not any("upload" in call for call in calls)


def test_long_prompt_is_written_to_document_and_sent_as_attachment(monkeypatch, tmp_path):
    """验证超长提示词会落盘为需求文档，输入框只发送简短附件指令。"""
    provider = OpenCLIChatGPTWebProvider(
        settings=OpenCLISettings(prompt_attachment_threshold=20), record_dir=tmp_path
    )
    conversation = ConversationHandle(conversation_id="long", title_hint="long")
    captured = {}
    monkeypatch.setattr(provider, "new_conversation", lambda title: conversation)

    def send_with_files(prompt, files, handle):
        """记录外置后的简短提示词和附件路径并返回合法响应。"""
        captured.update({"prompt": prompt, "files": files, "handle": handle})
        return '{"ok":true}'

    monkeypatch.setattr(provider, "send_with_files", send_with_files)
    long_prompt = "只能使用环境清单。" * 20
    assert provider._request_model(long_prompt, Reply, "task-reward-design").ok
    document = captured["files"][0]
    assert document.name == "task-reward-design_需求文档.md"
    assert long_prompt in document.read_text(encoding="utf-8")
    assert long_prompt not in captured["prompt"]
    assert document.name in captured["prompt"]


def test_long_schema_repair_is_also_sent_as_document(monkeypatch, tmp_path):
    """验证首次响应无效时，较长的 Schema 修复请求也统一通过附件发送。"""
    provider = OpenCLIChatGPTWebProvider(
        settings=OpenCLISettings(prompt_attachment_threshold=1, max_retries=1), record_dir=tmp_path
    )
    conversation = ConversationHandle(conversation_id="repair-doc", title_hint="repair-doc")
    sent_files = []
    replies = iter(['{"wrong":1}', '{"ok":true}'])
    monkeypatch.setattr(provider, "new_conversation", lambda title: conversation)

    def send_with_files(prompt, files, handle):
        """记录首次请求与修复请求各自生成的附件。"""
        sent_files.append([path.name for path in files])
        return next(replies)

    monkeypatch.setattr(provider, "send_with_files", send_with_files)
    assert provider._request_model("初始请求", Reply, "structured").ok
    assert sent_files[0] == ["structured_需求文档.md"]
    assert sent_files[1] == ["structured-repair-01_需求文档.md"]


def test_dom_change_falls_back_to_unnamed_textbox():
    """验证“dom change falls back to unnamed textbox”场景的预期行为。"""
    find_count = 0
    def runner(args, timeout):
        """执行 runner 对应的业务逻辑并返回结果。"""
        nonlocal find_count
        if "state" in args:
            return result("textbox")
        if "find" in args:
            if "--css" in args:
                return result('{"matches_n":0}')
            find_count += 1
            return result('{"matches_n":0}') if find_count == 1 else result('{"matches_n":1}')
        if "fill" in args:
            return result('{"filled":true,"verified":true}')
        return result("{}")
    OpenCLIChatGPTWebProvider(runner=runner)._fill_prompt("hello")
    assert find_count == 2


def test_answer_wait_and_timeout(monkeypatch):
    """验证“answer wait and timeout”场景的预期行为。"""
    provider = OpenCLIChatGPTWebProvider(settings=OpenCLISettings(response_timeout=5))
    values = iter(['{"ok":true}', '{"ok":true}', '{"ok":true}'])
    monkeypatch.setattr(provider, "_state", lambda: "ok")
    monkeypatch.setattr(provider, "_latest_assistant", lambda: next(values))
    monkeypatch.setattr(provider, "_is_generating", lambda: False)
    monkeypatch.setattr("time.sleep", lambda _: None)
    assert provider._wait_for_response("old") == '{"ok":true}'
    timeout_provider = OpenCLIChatGPTWebProvider(settings=OpenCLISettings(response_timeout=0))
    with pytest.raises(ProviderTimeout):
        timeout_provider._wait_for_response("")


def test_response_waits_until_visual_analysis_stops(monkeypatch):
    """验证助手文本短暂稳定但视觉工具仍运行时不会提前返回。"""
    provider = OpenCLIChatGPTWebProvider(settings=OpenCLISettings(response_timeout=10))
    responses = iter(["{", "{", "{", '{"ok":true}', '{"ok":true}', '{"ok":true}'])
    generating = iter([True, False])
    monkeypatch.setattr(provider, "_state", lambda: "ok")
    monkeypatch.setattr(provider, "_latest_assistant", lambda: next(responses))
    monkeypatch.setattr(provider, "_is_generating", lambda: next(generating))
    monkeypatch.setattr("time.sleep", lambda _: None)
    assert provider._wait_for_response("") == '{"ok":true}'


def test_response_does_not_return_stable_incomplete_json(monkeypatch):
    """验证停止按钮缺失时，短暂稳定的半段 JSON 仍不会被当作完整回复。"""
    provider = OpenCLIChatGPTWebProvider(settings=OpenCLISettings(response_timeout=10))
    responses = iter(["{", "{", "{", '{"ok":true}', '{"ok":true}', '{"ok":true}'])
    monkeypatch.setattr(provider, "_state", lambda: "ok")
    monkeypatch.setattr(provider, "_latest_assistant", lambda: next(responses))
    monkeypatch.setattr(provider, "_is_generating", lambda: False)
    monkeypatch.setattr("time.sleep", lambda _: None)
    assert provider._wait_for_response("") == '{"ok":true}'


def test_model_request_uses_completed_dom_response_before_repair(monkeypatch):
    """验证首次读取到流式前缀时优先采用随后补齐的 DOM 回复，不重复发送修复消息。"""
    provider = OpenCLIChatGPTWebProvider(settings=OpenCLISettings(max_retries=1))
    conversation = ConversationHandle(conversation_id="stream", title_hint="stream")
    sends = []
    monkeypatch.setattr(provider, "new_conversation", lambda title: conversation)
    monkeypatch.setattr(provider, "_send_model_prompt",
                        lambda *args: sends.append(args) or "{")
    monkeypatch.setattr(provider, "_latest_assistant", lambda: '{"ok":true}')
    assert provider._request_model("first", Reply, "stream").ok
    assert len(sends) == 1


def test_invalid_json_repair_uses_same_conversation(monkeypatch):
    """验证“invalid json repair uses same conversation”场景的预期行为。"""
    provider = OpenCLIChatGPTWebProvider(settings=OpenCLISettings(max_retries=1))
    conversation = ConversationHandle(conversation_id="same", title_hint="repair")
    sent = []
    monkeypatch.setattr(provider, "new_conversation", lambda title: conversation)
    def send(prompt, handle):
        """执行 send 对应的业务逻辑并返回结果。"""
        sent.append((prompt, handle.conversation_id))
        return '{"wrong": 1}' if len(sent) == 1 else '{"ok": true}'
    monkeypatch.setattr(provider, "send_text", send)
    assert provider._request_model("first", Reply, "title").ok
    assert [item[1] for item in sent] == ["same", "same"]


def test_schema_repair_submission_failure_recovers_in_new_conversation(monkeypatch):
    """验证同会话修复消息提交失败后会携带原回复在新会话继续修复。"""
    provider = OpenCLIChatGPTWebProvider(settings=OpenCLISettings(max_retries=1))
    conversations = []
    calls = []

    def new_conversation(title):
        """为每次新建会话生成可追踪标识。"""
        handle = ConversationHandle(conversation_id="conversation-%d" % (len(conversations) + 1),
                                    title_hint=title)
        conversations.append(handle)
        return handle

    def send(handle, prompt, title, files=None):
        """依次模拟无效回复、同会话提交超时和新会话修复成功。"""
        calls.append((handle.conversation_id, prompt, title, files))
        if len(calls) == 1:
            return '{"wrong": 1}'
        if len(calls) == 2:
            raise ProviderTimeout("修复消息未确认发送")
        assert 'Invalid JSON response to repair' in prompt
        assert '{"wrong": 1}' in prompt
        return '{"ok": true}'

    monkeypatch.setattr(provider, "new_conversation", new_conversation)
    monkeypatch.setattr(provider, "_send_model_prompt", send)
    assert provider._request_model("first", Reply, "title").ok
    assert [item[0] for item in calls] == ["conversation-1", "conversation-1", "conversation-2"]
