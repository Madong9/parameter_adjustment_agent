"""在隔离的无头 Chrome 中验证实际 DOM 提取，而非只模拟 Python 返回值。"""

import html
import json
import re
import shutil
import subprocess

import pytest

from rl_training_agent.providers.chatgpt_dom import CHATGPT_ATTACH_FILES_SCRIPT, CHATGPT_SNAPSHOT_SCRIPT
from rl_training_agent.providers.opencli_chatgpt import OpenCLIChatGPTWebProvider


def _run_dom_script(script, tmp_path):
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        pytest.skip("DOM 集成测试需要 Chrome 或 Chromium")
    page = tmp_path / "fixture.html"
    page.write_text('<!doctype html><meta charset="utf-8"><body><script>' +
                    script.replace("</", "<\\/") + '</script></body>', encoding="utf-8")
    result = subprocess.run([
        chrome, "--headless=new", "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
        "--no-first-run", "--no-default-browser-check", "--disable-background-networking",
        "--user-data-dir=" + str(tmp_path / "chrome"), "--dump-dom", page.as_uri(),
    ], capture_output=True, text=True, timeout=30, check=True)
    match = re.search(r'<pre id="probe-result">(.*?)</pre>', result.stdout, re.S)
    assert match, result.stderr[-1000:]
    return json.loads(html.unescape(match.group(1)))


def test_chatgpt_old_and_new_message_dom(tmp_path):
    """验证新旧 DOM、嵌套容器、角色归属及无回复场景，不访问网络或个人浏览器。"""
    modern = (
        '<div data-chatgpt-search-unit-key="t0:0:user" data-chatgpt-search-message-ids="u1">'
        '<p>请返回 JSON</p><pre>{"schema":"不是回复"}</pre></div>'
        '<div data-chatgpt-search-unit-key="t0:2:assistant" data-chatgpt-search-message-ids="a1">'
        '<h4 data-conversation-role="assistant">ChatGPT 说：</h4>'
        '<pre><code>{"ok":true}</code></pre><button>复制</button></div>')
    legacy = (
        '<div data-message-author-role="user" data-message-id="u1">问题</div>'
        '<div data-message-author-role="assistant" data-message-id="a1"><p>{"ok":true}</p></div>')
    nested = (
        '<div data-chatgpt-search-unit-key="t:0:user"><div data-message-author-role="user">问题</div></div>'
        '<div data-chatgpt-search-unit-key="t:1:assistant">'
        '<div data-message-author-role="assistant"><pre>{"ok":true}</pre></div></div>')
    user_only = ('<div data-chatgpt-search-unit-key="t:0:user">'
                 '<h4>ChatGPT 说：</h4><pre>{"schema":"不是回复"}</pre></div>')
    new_question = '<div data-chatgpt-search-unit-key="t1:0:user" data-chatgpt-search-message-ids="u2">新问题</div>'
    fixtures = [modern, legacy, nested, user_only, modern + new_question,
                modern + '<button data-testid="stop-button">停止</button>',
                modern + '<button aria-label="停止"></button>',
                modern + '<button aria-label="Stop"></button>',
                modern + '<button aria-label="停止听写"></button>']
    script = (
        "const results=[]; const fixtures=" + json.dumps(fixtures) + ";"
        "for (const fixture of fixtures) {document.body.innerHTML=fixture;"
        "results.push(JSON.parse(eval(" + json.dumps(CHATGPT_SNAPSHOT_SCRIPT) + ")));}"
        "document.body.innerHTML='<pre id=\"probe-result\"></pre>';"
        "document.querySelector('#probe-result').textContent=JSON.stringify(results);")
    snapshots = _run_dom_script(script, tmp_path)
    for snapshot in snapshots[:3]:
        assert snapshot["user_count"] == snapshot["assistant_count"] == 1
        assert json.loads(snapshot["latest_assistant"]) == {"ok": True}
        assert snapshot["assistant_user_id"] == snapshot["latest_user_id"]
    assert snapshots[3]["latest_assistant"] == ""
    assert snapshots[3]["assistant_count"] == 0
    assert snapshots[4]["latest_user_id"] == "u2"
    assert snapshots[4]["assistant_user_id"] == "u1"
    assert snapshots[5]["generating"] is True
    assert snapshots[6]["generating"] is True
    assert snapshots[7]["generating"] is True
    assert snapshots[8]["generating"] is False


def test_attachment_handler_may_clear_file_input(tmp_path):
    """React 和原生 change 处理器清空 input.files 时仍保留实际交付文件清单。"""
    script = CHATGPT_ATTACH_FILES_SCRIPT + r"""
const results = [];
for (const react of [true, false]) {
  const input = document.createElement('input');
  input.type = 'file';
  let calls = 0;
  let received = [];
  const handler = event => {
    calls++;
    received = Array.from(event.target.files).map(file => file.name);
    event.target.value = '';
  };
  if (react) input.__reactProps$fixture = {onChange: handler};
  else input.addEventListener('change', handler);
  const transfer = new DataTransfer();
  transfer.items.add(new File(['requirements'], 'requirements.md', {type:'text/markdown'}));
  transfer.items.add(new File(['image'], 'frame.png', {type:'image/png'}));
  const result = attachFiles(input, transfer);
  results.push({result, calls, received, remaining: input.files.length});
}
document.body.innerHTML = '<pre id="probe-result"></pre>';
document.querySelector('#probe-result').textContent = JSON.stringify(results);
"""
    for item in _run_dom_script(script, tmp_path):
        assert item["calls"] == 1
        assert item["remaining"] == 0
        assert item["received"] == ["requirements.md", "frame.png"]
        assert item["result"] == {"ok": True, "count": 2, "names": item["received"]}


def test_document_preview_rejects_old_message_and_unrelated_image(tmp_path):
    """旧消息里的文件名和当前图片预览均不能充当需求文档预览。"""
    scripts = []

    def runner(args, timeout):
        scripts.append(args[-1])
        return subprocess.CompletedProcess(args, 0, "true", "")

    provider = OpenCLIChatGPTWebProvider(runner=runner)
    provider._wait_for_upload_preview(["requirements.md"], allow_media=False)
    fixtures = [
        '<main>requirements.md</main><form><div id="prompt-textarea" contenteditable="true"></div></form>',
        '<form><img style="width:100px;height:100px" src="data:image/png;base64,AA==">'
        '<div id="prompt-textarea" contenteditable="true"></div></form>',
        '<form><span>requirements.md</span><div id="prompt-textarea" contenteditable="true"></div></form>',
    ]
    script = (
        "const results=[];for(const fixture of " + json.dumps(fixtures) + ") {"
        "document.body.innerHTML=fixture;results.push(JSON.parse(eval(" +
        json.dumps(scripts[0]) + ")));}"
        "document.body.innerHTML='<pre id=\"probe-result\"></pre>';"
        "document.querySelector('#probe-result').textContent=JSON.stringify(results);")
    assert _run_dom_script(script, tmp_path) == [False, False, True]


def test_send_click_ignores_hidden_disabled_and_busy_buttons(tmp_path):
    """附件布局改变时，在页面内点击正确按钮一次，跳过无法发送的按钮。"""
    scripts = []

    def runner(args, timeout):
        if "button.click()" in args[-1]:
            scripts.append(args[-1])
            return subprocess.CompletedProcess(args, 0, '{"clicked":true}', '')
        return subprocess.CompletedProcess(args, 0, '{}', '')

    provider = OpenCLIChatGPTWebProvider(runner=runner)
    provider._state = lambda: "ready"
    provider._check_page_usage_limit = lambda: None
    provider._click_send_button()
    script = r"""
document.body.innerHTML = '<button data-testid="send-button" style="display:none"></button>' +
  '<button id="composer-submit-button" disabled></button>' +
  '<button aria-label="发送" aria-disabled="true"></button>' +
  '<button aria-label="发送" aria-busy="true"></button>' +
  '<button aria-label="发送" id="real-send"></button>';
const clicks = [];
document.querySelectorAll('button').forEach(button =>
  button.addEventListener('click', () => clicks.push(button.id)));
const result = JSON.parse(eval(SEND_SCRIPT));
document.body.innerHTML = '<pre id="probe-result"></pre>';
document.querySelector('#probe-result').textContent = JSON.stringify({result, clicks});
""".replace("SEND_SCRIPT", json.dumps(scripts[0]))
    result = _run_dom_script(script, tmp_path)
    assert result["result"]["clicked"] is True
    assert result["clicks"] == ["real-send"]
