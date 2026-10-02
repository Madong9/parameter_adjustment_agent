"""在隔离的无头 Chrome 中验证实际 DOM 提取，而非只模拟 Python 返回值。"""

import html
import json
import re
import shutil
import subprocess

import pytest

from rl_training_agent.providers.chatgpt_dom import CHATGPT_SNAPSHOT_SCRIPT


def test_chatgpt_old_and_new_message_dom(tmp_path):
    """验证新旧 DOM、嵌套容器、角色归属及无回复场景，不访问网络或个人浏览器。"""
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        pytest.skip("DOM 集成测试需要 Chrome 或 Chromium")
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
    snapshots = json.loads(html.unescape(match.group(1)))
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
