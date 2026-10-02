"""覆盖浏览器桥接中的消息完整性、归属和幂等提交边界。"""

import subprocess

import pytest

from rl_training_agent.providers.errors import ProviderError, ProviderTimeout
from rl_training_agent.providers.opencli_chatgpt import OpenCLIChatGPTWebProvider
from rl_training_agent.providers.opencli_doubao import OpenCLIDoubaoWebProvider
from rl_training_agent.schemas.experiments import ConversationHandle
from rl_training_agent.settings import OpenCLISettings


def test_same_prompt_requires_new_message_identity():
    """同文重发必须出现新消息标识，不能复用上一轮用户消息。"""
    provider = OpenCLIChatGPTWebProvider()
    provider._submission_baseline = {
        "latest_user": "相同提示", "latest_user_id": "u1", "user_count": 1}
    assert not provider._submission_confirmed(provider._submission_baseline, "相同提示", "相同提示")
    assert provider._submission_confirmed({
        "latest_user": "相同提示", "latest_user_id": "u2", "user_count": 2}, "相同提示", "相同提示")
    assert provider._confirmed_user_id == "u2"


def test_submission_waits_for_delayed_message(monkeypatch):
    """草稿先清空时继续等待，直至对应用户消息真实出现。"""
    provider = OpenCLIChatGPTWebProvider()
    snapshots = iter([
        {"latest_user": "", "composer_found": True, "composer_text": ""},
        {"latest_user": "请回复", "latest_user_id": "u1", "user_count": 1},
    ])
    monkeypatch.setattr(provider, "_submission_snapshot", lambda: next(snapshots))
    monkeypatch.setattr("time.sleep", lambda _: None)
    provider._wait_for_submission("", "请回复")
    assert provider._confirmed_user_id == "u1"


def test_submission_timeout_does_not_resend(monkeypatch):
    """点击成功后无法确认消息时只报错，不重复点击或读取旧回复。"""
    provider = OpenCLIChatGPTWebProvider()
    events = []
    monkeypatch.setattr(provider, "_submission_snapshot", lambda: {})
    monkeypatch.setattr(provider, "_fill_prompt", lambda _: events.append("fill"))
    monkeypatch.setattr(provider, "_click_send_button", lambda: events.append("click"))
    monkeypatch.setattr(provider, "_collect_page_debug_artifacts", lambda *args: None)
    monkeypatch.setattr(provider, "_wait_for_response", lambda _: events.append("response"))

    def timeout(*args, **kwargs):
        """模拟无法确认提交的超时。"""
        raise ProviderTimeout("未确认消息")

    monkeypatch.setattr(provider, "_wait_for_submission", timeout)
    with pytest.raises(ProviderTimeout):
        provider.send_text("请求", ConversationHandle(conversation_id="test", title_hint="test"))
    assert events == ["fill", "click"]


def test_response_rejects_previous_turn_and_accepts_identical_new_reply(monkeypatch):
    """旧助手消息不能充当新回复，但新一轮同文 JSON 应正常返回。"""
    provider = OpenCLIChatGPTWebProvider()
    provider._submission_baseline = {"latest_assistant_id": "a1"}
    provider._confirmed_user_id = "u2"
    previous_turn = {"latest_assistant": '{"ok":false}', "latest_assistant_id": "a1",
                     "assistant_user_id": "u1"}
    new_turn = {"latest_assistant": '{"ok":true}', "latest_assistant_id": "a2",
                "assistant_user_id": "u2"}
    snapshots = iter([previous_turn] * 4 + [new_turn] * 3)
    calls = []

    def snapshot():
        """先返回仍显示的旧回复，再返回本次请求对应的新回复。"""
        calls.append(True)
        return next(snapshots)

    monkeypatch.setattr(provider, "_submission_snapshot", snapshot)
    monkeypatch.setattr(provider, "_state", lambda: "ready")
    monkeypatch.setattr(provider, "_is_generating", lambda: False)
    monkeypatch.setattr("time.sleep", lambda _: None)
    assert provider._wait_for_response('{"ok":true}') == '{"ok":true}'
    assert len(calls) == 7


@pytest.mark.parametrize("replacement", ["速度=0.9", "", "错误内容"])
def test_prompt_comparison_does_not_ignore_middle(replacement):
    """相同长前后缀也不能掩盖关键速度值被替换或正文被截断。"""
    prefix, suffix = "任务说明" * 100, "返回JSON" * 100
    assert not OpenCLIChatGPTWebProvider._matches_submitted_prompt(
        prefix + replacement + suffix, prefix + "速度=0.5" + suffix)


@pytest.mark.parametrize("prefix", [
    "attachment_check.md文档\n", "requirements.md\nDocument\n",
    "attachment_check.md文档\nbehavior_evidence.json文档\n", "requirements.md\n文件\n",
    "attachment_check.md文档\nbehavior_evidence.json代码\n",
])
def test_submitted_prompt_accepts_document_attachment_label(prefix):
    """新页面内联的文档标签不能让已提交且已回复的消息被误报超时。"""
    prompt = "请读取附件，只返回 JSON。"
    assert OpenCLIChatGPTWebProvider._matches_submitted_prompt(prefix + prompt, prompt)
    assert not OpenCLIChatGPTWebProvider._matches_submitted_prompt(prefix + prompt, "请读取附件，只返回 XML。")


def test_doubao_locator_exact_does_not_mean_text_exact():
    """定位器 exact 和 verified 标签不能代替实际全文比较。"""
    provider = OpenCLIDoubaoWebProvider()
    assert not provider._fill_result_matches_prompt({
        "filled": True, "verified": True, "match_level": "exact", "actual": "速度=3"}, "速度=1")
    assert provider._fill_result_matches_prompt({
        "filled": True, "verified": False, "actual": "第一段\n\n第二段"}, "第一段\n第二段")


def test_unreadable_page_address_is_not_accepted(monkeypatch):
    """扩展未返回地址时立即失败，不能继续在未知页面输入任务。"""
    provider = OpenCLIChatGPTWebProvider()
    monkeypatch.setattr(provider, "_current_page_url", lambda: "")
    with pytest.raises(ProviderError, match="无法读取"):
        provider._assert_expected_page()


def test_profile_is_forwarded_to_opencli():
    """显式浏览器配置必须透传到命令，防止误用默认登录账户。"""
    commands = []

    def runner(args, timeout):
        """捕获参数并返回无副作用结果。"""
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, "{}", "")

    provider = OpenCLIChatGPTWebProvider(settings=OpenCLISettings(profile="robot"), runner=runner)
    provider._browser(["state"])
    command = commands[0]
    assert command[command.index("--profile") + 1] == "robot"
    assert "browser" in command and provider.settings.session in command


@pytest.mark.parametrize("active_user,finish,expected", [
    ("u2", 390, "success"),
    ("u1", 390, "timeout"),
    ("u2", 2000, "timeout"),
])
def test_long_visual_analysis_has_bounded_request_scoped_wait(monkeypatch, active_user, finish, expected):
    provider = OpenCLIChatGPTWebProvider(settings=OpenCLISettings(
        response_timeout=300, response_generation_timeout=900))
    provider._confirmed_user_id = "u2"
    clock = [0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    monkeypatch.setattr("time.sleep", lambda _: clock.__setitem__(0, clock[0] + 10))
    monkeypatch.setattr(provider, "_state", lambda: "ready")
    monkeypatch.setattr(provider, "_is_generating", lambda: clock[0] < finish)

    def response():
        provider._last_message_snapshot = {
            "latest_user_id": active_user, "generating": clock[0] < finish,
            "assistant_user_id": "u2", "latest_assistant_id": "a2"}
        return '{"ok":true}' if clock[0] >= finish else ""

    monkeypatch.setattr(provider, "_latest_assistant", response)
    if expected == "success":
        assert provider._wait_for_response("") == '{"ok":true}'
        assert 390 <= clock[0] < 900
    else:
        with pytest.raises(ProviderTimeout, match="request_confirmed=True"):
            provider._wait_for_response("")
        assert clock[0] == (900 if active_user == "u2" else 300)


@pytest.mark.parametrize("check,blocked", [
    ({"blocked": False, "ready": True}, False),
    ({"blocked": True, "ready": True}, True),
    ({"blocked": False, "ready": False}, True),
    (None, True),
])
def test_doubao_verification_requires_visible_challenge(monkeypatch, check, blocked):
    import json
    from rl_training_agent.providers.errors import ProviderNeedsHuman
    provider = OpenCLIDoubaoWebProvider()
    monkeypatch.setattr(provider, "_assert_expected_page", lambda: None)
    monkeypatch.setattr(provider, "_browser", lambda args, **kwargs:
                        "history title: 安全验证 captcha; chat_input" if args == ["state"] else json.dumps(check))
    if blocked:
        with pytest.raises(ProviderNeedsHuman, match="完成安全验证"):
            provider._state()
    else:
        assert "chat_input" in provider._state()
