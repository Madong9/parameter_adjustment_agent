"""覆盖网页登录误报、跨请求回复与隐私安全诊断，不连接个人浏览器。"""
import json
import subprocess

import pytest

from rl_training_agent.providers.chatgpt_dom import CHATGPT_ACCESS_STATE_SCRIPT
from rl_training_agent.providers.errors import ProviderNeedsHuman, ProviderResponseError
from rl_training_agent.providers.opencli_chatgpt import OpenCLIChatGPTWebProvider
from rl_training_agent.providers.opencli_doubao import DOUBAO_ACCESS_STATE_SCRIPT, OpenCLIDoubaoWebProvider
from rl_training_agent.schemas.agent_workflow import TaskIntentSpec
from rl_training_agent.schemas.experiments import ConversationHandle
from rl_training_agent.settings import OpenCLISettings
from test_chatgpt_dom import _run_dom_script


def _provider_for_page(body, access):
    """为控件状态和整页正文提供互相独立的模拟响应。"""
    def runner(args, timeout):
        """只响应本地模拟的浏览器状态，不启动外部命令。"""
        if args[-1] == "JSON.stringify(location.href || '')":
            output = json.dumps("https://chatgpt.com/c/test")
        elif args[-1] == CHATGPT_ACCESS_STATE_SCRIPT:
            output = json.dumps(dict(access, probe_version="chatgpt-access-v1"))
        else:
            output = body
        return subprocess.CompletedProcess(args, 0, output, "")
    return OpenCLIChatGPTWebProvider(OpenCLISettings(), runner=runner)


@pytest.mark.parametrize("text", [
    "奖励注册表中已有奖励", "登录检测是错误的", "请不要 sign up 或 log in",
    "captcha 和 cloudflare 是旧错误", "You've reached your usage limit 是之前的故障记录",
])
def test_message_keywords_cannot_expire_authenticated_page(text):
    """正文讨论登录、注册、验证码或额度均不能触发站点故障。"""
    provider = _provider_for_page(text, {"composer_found": True, "account_controls_found": True})
    assert provider._state() == text
    provider._check_page_usage_limit()


def test_real_controls_still_require_human():
    """明确的登录控件和验证码分别暂停，而不是关闭安全检查。"""
    for field in ("login_required", "challenge_required"):
        with pytest.raises(ProviderNeedsHuman):
            _provider_for_page("正常正文", {field: True})._state()


def test_access_probe_ignores_conversation_and_hidden_controls(tmp_path):
    """在隔离 Chrome 中执行真实 DOM 探针，验证控件与消息边界。"""
    message = ('<div data-message-author-role="assistant">奖励注册表 登录 cloudflare '
               '<button>Log in</button><div role="alert">你已达到使用上限</div></div>')
    composer = '<div id="prompt-textarea" contenteditable="true" role="textbox">注册奖励</div>'
    fixtures = [
        '<button data-testid="profile-button">账号</button>' + composer + message,
        composer + message + '<button style="display:none">Log in</button>',
        '<header><button data-testid="login-button">Log in</button></header>' + composer,
        '<form><input type="password"></form>',
        '<div id="challenge-stage">Verify you are human</div>',
        composer + '<div role="alert">你已达到使用上限</div>',
    ]
    script = ("const values=[]; for(const fixture of " + json.dumps(fixtures) + ") {"
              "document.body.innerHTML=fixture; values.push(JSON.parse(eval(" +
              json.dumps(CHATGPT_ACCESS_STATE_SCRIPT) + ")));}"
              "document.body.innerHTML='<pre id=\"probe-result\"></pre>';"
              "document.querySelector('#probe-result').textContent=JSON.stringify(values);")
    values = _run_dom_script(script, tmp_path)
    assert not values[0]["login_required"] and not values[0]["challenge_required"]
    assert not values[0]["usage_notice"] and values[0]["composer_found"]
    assert not values[1]["login_required"]
    assert values[2]["login_required"] and values[3]["login_required"]
    assert values[4]["challenge_required"]
    assert "使用上限" in values[5]["usage_notice"]


@pytest.mark.parametrize("same_request", [False, True])
def test_partial_json_recheck_requires_current_message_identity(monkeypatch, same_request):
    """修复前补读 DOM 只能接受当前已提交请求的完整回复。"""
    provider = OpenCLIChatGPTWebProvider(OpenCLISettings(max_retries=0))
    monkeypatch.setattr(provider, "new_conversation", lambda title: ConversationHandle(
        conversation_id="fixture", title_hint=title))
    monkeypatch.setattr(provider, "_send_model_prompt", lambda *args: "{")
    provider._confirmed_user_id = "current-user"
    provider._submission_baseline = {"latest_assistant_id": "old-assistant"}
    reply = TaskIntentSpec(original_instruction="倒退走", robot="go2", action_name="倒退走",
                           normalized_goal="倒退走").json(ensure_ascii=False)
    monkeypatch.setattr(provider, "_submission_snapshot", lambda: {
        "latest_assistant": reply, "latest_assistant_id": "current-assistant",
        "latest_user_id": "current-user",
        "assistant_user_id": "current-user" if same_request else "old-user",
    })
    if same_request:
        assert provider._request_model("倒退走", TaskIntentSpec, "fixture").action_name == "倒退走"
    else:
        with pytest.raises(ProviderResponseError):
            provider._request_model("倒退走", TaskIntentSpec, "fixture")


def test_debug_capture_does_not_save_private_page_content(tmp_path, monkeypatch):
    """诊断仅落盘白名单状态、长度和身份摘要，不保存凭据、正文或查询参数。"""
    provider = OpenCLIChatGPTWebProvider(record_dir=tmp_path)
    monkeypatch.setattr(provider, "_submission_snapshot", lambda: {
        "latest_user": "private-secret", "latest_assistant": "private-answer",
        "composer_text": "private-key", "url": "https://chatgpt.com/c/private?token=secret-token",
        "latest_user_id": "private-user", "latest_assistant_id": "private-assistant",
        "assistant_user_id": "private-user", "user_count": 1,
    })
    provider._collect_page_debug_artifacts(ConversationHandle(conversation_id="fixture", title_hint="fixture"),
                                           "timeout")
    files = list(tmp_path.rglob("*.*"))
    assert len(files) == 1 and files[0].suffix == ".json"
    saved = files[0].read_text(encoding="utf-8")
    assert "private-" not in saved and "secret-token" not in saved
    assert json.loads(saved)["page_origin"] == "https://chatgpt.com"


def test_doubao_message_keywords_cannot_expire_page():
    """备用豆包的登录、注册及验证码旧正文同样不得成为站点故障证据。"""
    def runner(args, timeout):
        """区分站点地址、可见控件与正文，不连接个人浏览器。"""
        if args[-1] == "JSON.stringify(location.href || '')":
            value = json.dumps("https://www.doubao.com/chat/")
        elif args[-1] == DOUBAO_ACCESS_STATE_SCRIPT:
            value = json.dumps({"probe_version": "doubao-access-v1", "composer_found": True})
        else:
            value = "旧故障：登录 注册 验证码 安全验证 captcha"
        return subprocess.CompletedProcess(args, 0, value, "")
    provider = OpenCLIDoubaoWebProvider(OpenCLISettings(), runner=runner)
    assert "旧故障" in provider._state()
    provider._check_page_usage_limit()


def test_doubao_access_probe_uses_visible_site_controls(tmp_path):
    """真实 DOM 验证豆包正文和隐藏验证不误报，实际控件仍被识别。"""
    composer = ('<div data-testid="chat_input_input">'
                '<div contenteditable="true">登录 验证码</div></div>')
    message = ('<div data-testid="receive_message"><button>登录</button>'
               '<div role="dialog">安全验证 captcha</div>'
               '<div role="alert">你已达到使用上限</div></div>')
    fixtures = [
        composer + message,
        composer + '<div id="captcha-hidden" style="display:none">验证码</div>',
        '<button>立即登录</button>',
        '<form><input type="password"></form>',
        '<div role="dialog">安全验证</div>',
        composer + '<div role="alert">你已达到使用上限</div>',
    ]
    script = ("const values=[]; for(const fixture of " + json.dumps(fixtures) + ") {"
              "document.body.innerHTML=fixture; values.push(JSON.parse(eval(" +
              json.dumps(DOUBAO_ACCESS_STATE_SCRIPT) + ")));}"
              "document.body.innerHTML='<pre id=\"probe-result\"></pre>';"
              "document.querySelector('#probe-result').textContent=JSON.stringify(values);")
    values = _run_dom_script(script, tmp_path)
    assert values[0]["composer_found"]
    assert not values[0]["login_required"] and not values[0]["challenge_required"]
    assert not values[0]["usage_notice"] and not values[1]["challenge_required"]
    assert values[2]["login_required"] and values[3]["login_required"]
    assert values[4]["challenge_required"] and "使用上限" in values[5]["usage_notice"]
