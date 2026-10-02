"""通过独立 OpenCLI 浏览器会话驱动豆包网页，作为 GPT 的多模态备用 Provider。"""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..schemas.experiments import ConversationHandle, ProviderHealth
from ..settings import OpenCLISettings, load_opencli_settings
from .errors import ProviderError, ProviderNeedsHuman, ProviderTimeout
from .opencli_chatgpt import (
    OpenCLIChatGPTWebProvider,
    Runner,
    _default_runner,
    _extract_value,
    _json_from_output,
)


class OpenCLIDoubaoWebProvider(OpenCLIChatGPTWebProvider):
    """复用结构化提示词与 Schema 校验，只替换为豆包网页的稳定 DOM 交互。"""

    INPUT_SELECTOR = '[data-testid="chat_input_input"] [contenteditable="true"]'
    SEND_SELECTOR = '[data-testid="chat_input_send_button"]'
    USER_SELECTOR = '[data-testid="send_message"] [data-testid="message_text_content"]'
    ASSISTANT_SELECTOR = '[data-testid="receive_message"] [data-testid="message_text_content"]'

    def __init__(self, settings: Optional[OpenCLISettings] = None,
                 runner: Runner = _default_runner, record_dir: Optional[Path] = None):
        """创建与 ChatGPT 会话隔离的豆包 OpenCLI Provider。"""
        source = settings or load_opencli_settings()
        values = source.dict()
        values.update({
            "session": source.doubao_session,
            "chatgpt_url": source.doubao_url,
            "bind_existing_tab": False,
            "owned_session": False,
            "force_chat_mode": False,
        })
        super().__init__(OpenCLISettings(**values), runner=runner, record_dir=record_dir)

    def _state(self) -> str:
        """刷新豆包页面状态并识别未登录或安全验证页面。"""
        state = self._browser(["state"])
        lowered = state.lower()
        if any(item in lowered for item in ("验证码", "安全验证", "verify", "captcha")):
            # state 包含隐藏节点、脚本属性和历史标题；单个关键字不能证明
            # 当前页面被验证弹窗阻断。只确认可见验证控件，无法检查时保守停止。
            check = self._parse_eval_result(self._browser(["eval", """
(() => {
  const visible = node => !!node.getClientRects().length &&
    getComputedStyle(node).visibility !== 'hidden' &&
    getComputedStyle(node).display !== 'none';
  const candidates = document.querySelectorAll(
    '[role="dialog"], iframe, [id*="captcha"], [class*="captcha"], [id*="verify"], [class*="verify"]');
  const blocked = Array.from(candidates).some(node => visible(node) &&
    /验证码|安全验证|verify you are human|captcha/i.test(
      (node.innerText || '') + ' ' + (node.getAttribute('src') || '') + ' ' +
      (node.getAttribute('aria-label') || '')));
  const editor = document.querySelector('[data-testid="chat_input_input"] [contenteditable="true"]');
  return JSON.stringify({blocked, ready: !!editor && visible(editor)});
})()
"""], allow_failure=True))
            if not isinstance(check, dict) or check.get("blocked") or not check.get("ready"):
                raise ProviderNeedsHuman("豆包页面需要完成安全验证，请在浏览器完成验证后重试。")
        if "登录" in state and "chat_input" not in state and "新对话" not in state:
            raise ProviderNeedsHuman("豆包登录已失效，请在浏览器中重新登录。")
        self._assert_expected_page()
        return state

    def doctor(self) -> ProviderHealth:
        """检查 OpenCLI 扩展和豆包网页登录状态。"""
        try:
            self.open_or_bind()
            state = self._state()
            available = "chat_input" in state or "新对话" in state
            return ProviderHealth(
                available=available,
                opencli_available=True,
                extension_connected=True,
                chatgpt_logged_in=available,
                image_upload_supported=available,
                recoverable=True,
                details=["豆包网页已连接并使用独立 OpenCLI 会话"],
            )
        except ProviderError as exc:
            return ProviderHealth(
                available=False,
                opencli_available=True,
                extension_connected=False,
                chatgpt_logged_in=False,
                image_upload_supported=False,
                recoverable=True,
                details=["豆包网页检查失败：%s" % exc],
            )

    def open_or_bind(self) -> None:
        """打开独立豆包标签页并强制使用普通“对话”模式。"""
        self._ensure_bridge_connected()
        self._open_expected_page(self.settings.doubao_url, "豆包")
        self._state()
        self._ensure_dialogue_mode()
        self._opened = True

    def new_conversation(self, title_hint: str) -> ConversationHandle:
        """打开豆包新对话页并返回本地可审计句柄。"""
        if not self._opened:
            self.open_or_bind()
        else:
            self._open_expected_page(self.settings.doubao_url, "豆包")
            self._state()
            self._ensure_dialogue_mode()
        return ConversationHandle(
            conversation_id="doubao-" + uuid.uuid4().hex,
            title_hint=title_hint,
            owned=True,
        )

    def _ensure_dialogue_mode(self) -> None:
        """验证并选择豆包的“对话”模式，禁止使用“工作”模式。"""
        script = """
(() => {
  const current = document.querySelector('[data-testid="chat_input_action_mode"] [title]');
  if (current && (current.getAttribute('title') || '').trim() === '对话') {
    return JSON.stringify({ok: true, changed: false});
  }
  const root = document.querySelector('[data-testid="conversation-mode-switch"]');
  const button = root && Array.from(root.querySelectorAll('button')).find(item =>
    (item.innerText || '').trim() === '对话');
  if (!button) return JSON.stringify({ok: false, reason: 'dialogue_button_not_found'});
  button.click();
  return JSON.stringify({ok: true, changed: true});
})()
"""
        result = self._parse_eval_result(self._browser(["eval", script], allow_failure=True))
        if not (isinstance(result, dict) and result.get("ok")):
            raise ProviderNeedsHuman("豆包无法切换到“对话”模式，请手动选择“对话”后重试。")
        if result.get("changed"):
            time.sleep(0.5)

    def _fill_prompt(self, prompt: str) -> None:
        """填写豆包富文本输入框并严格验证全文。"""
        self._state()
        self._ensure_dialogue_mode()
        # 豆包使用 ProseMirror。OpenCLI 1.8.x 能写入它，但其通用 fill
        # 校验只读 innerText，会把已成功写入的内容误报为空。先通过真实
        # 键盘事件清空草稿，再在页面内用 textContent 严格核对全文。
        self._browser(["click", self.INPUT_SELECTOR])
        self._browser(["keys", "Control+a"])
        self._browser(["keys", "Backspace"])
        output = self._browser(["fill", self.INPUT_SELECTOR, prompt], allow_failure=True)
        parsed = _json_from_output(output)
        if not (isinstance(parsed, dict) and parsed.get("filled")):
            raise ProviderError("豆包输入框没有确认提示词写入：%s" %
                                self._safe_fill_diagnostic(parsed, prompt))
        # match_level 只描述元素定位。使用 actual 全文与原文比较；仅容忍
        # ProseMirror 的段落空白差异，不能接受首尾相同但中间缺失的内容。
        if self._fill_result_matches_prompt(parsed, prompt):
            return
        if not self._verify_doubao_composer(prompt):
            raise ProviderError("豆包输入框已写入，但全文校验失败：%s" %
                                self._safe_fill_diagnostic(parsed, prompt))

    @staticmethod
    def _fill_result_matches_prompt(result: Any, prompt: str) -> bool:
        """核验 OpenCLI fill 回传的全文，禁止仅凭 filled 标记放行。"""
        if not isinstance(result, dict) or not result.get("filled"):
            return False
        actual = result.get("actual")
        if not isinstance(actual, str):
            return False
        expected_normalized = re.sub(r"\s+", " ", prompt).strip()
        return bool(expected_normalized and
                    re.sub(r"\s+", " ", actual).strip() == expected_normalized)

    @staticmethod
    def _safe_fill_diagnostic(result: Any, prompt: str) -> str:
        """仅输出长度和匹配元数据，避免把完整任务提示词复制到错误日志。"""
        data = result if isinstance(result, dict) else {}
        actual = data.get("actual")
        diagnostic = {
            "filled": bool(data.get("filled")),
            "verified": bool(data.get("verified")),
            "matches_n": data.get("matches_n"),
            "match_level": data.get("match_level"),
            "actual_length": len(actual) if isinstance(actual, str) else data.get("actual_length"),
            "expected_length": len(prompt),
        }
        return json.dumps(diagnostic, ensure_ascii=False, separators=(",", ":"))

    def _verify_doubao_composer(self, prompt: str) -> bool:
        """使用 ProseMirror 的 textContent 验证豆包编辑器内容没有截断或重复。"""
        expected = json.dumps(prompt, ensure_ascii=False)
        script = """
(() => {
  const element = document.querySelector('%s');
  if (!element) return JSON.stringify({found: false, matches: false});
  const normalize = value => String(value).replace(/\\s+/g, ' ').trim();
  const actual = element.textContent || element.innerText || '';
  const expected = %s;
  return JSON.stringify({
    found: true,
    matches: normalize(actual) === normalize(expected),
    actual_length: normalize(actual).length,
    expected_length: normalize(expected).length
  });
})()
""" % (self.INPUT_SELECTOR, expected)
        result = self._parse_eval_result(
            self._browser(["eval", script], allow_failure=True))
        return bool(isinstance(result, dict) and result.get("found") and
                    result.get("matches") and
                    result.get("actual_length") == result.get("expected_length"))

    def _click_send_button(self) -> None:
        """等待豆包发送按钮可用并执行一次显式点击。"""
        self._ensure_dialogue_mode()
        deadline = time.monotonic() + self.settings.submit_timeout
        while time.monotonic() < deadline:
            found = self._browser(
                ["find", "--css", self.SEND_SELECTOR, "--limit", "2"], allow_failure=True)
            data = _json_from_output(found)
            if isinstance(data, dict) and int(data.get("matches_n", 0)) > 0:
                clicked = _json_from_output(
                    self._browser(["click", self.SEND_SELECTOR], allow_failure=True))
                if isinstance(clicked, dict) and clicked.get("clicked"):
                    return
            time.sleep(0.25)
        raise ProviderTimeout("豆包发送按钮在 %s 秒内未变为可点击状态" % self.settings.submit_timeout)

    def _latest_text(self, selector: str) -> str:
        """提取指定豆包消息类型的最后一条正文。"""
        selector_json = json.dumps(selector)
        script = "JSON.stringify(Array.from(document.querySelectorAll(%s)).map(e=>e.innerText||'').filter(Boolean).slice(-1)[0]||'')" % selector_json
        # 这里只解开 OpenCLI 最外层 JSON 字符串。助手正文自身可能正好是
        # 合法 JSON，若沿用 _parse_eval_result 会把正文继续解析成 dict，
        # 随后丢失文本并一直等待到超时。
        value = _json_from_output(self._browser(["eval", script], allow_failure=True))
        return value if isinstance(value, str) else _extract_value(value)

    def _latest_assistant(self) -> str:
        """读取豆包最新一条助手消息正文。"""
        self._last_message_snapshot = self._submission_snapshot()
        return str(self._last_message_snapshot.get("latest_assistant", ""))

    def _latest_user(self) -> str:
        """读取豆包最新一条已提交用户消息正文。"""
        return self._latest_text(self.USER_SELECTOR)

    def _submission_snapshot(self) -> Dict[str, Any]:
        """一次读取豆包用户消息、助手消息、输入框和生成状态。"""
        script = """
(() => {
  const userNodes = Array.from(document.querySelectorAll('%s')).filter(e=>(e.innerText||'').trim());
  const assistantNodes = Array.from(document.querySelectorAll('%s')).filter(e=>(e.innerText||'').trim());
  const users = userNodes.map(e=>e.innerText);
  const assistants = assistantNodes.map(e=>e.innerText);
  const composer = document.querySelector('%s');
  const latestReceive = Array.from(document.querySelectorAll('[data-testid="receive_message"]')).slice(-1)[0];
  return JSON.stringify({
    latest_user: users.slice(-1)[0] || '', user_count: users.length,
    latest_user_id: users.length ? 'doubao-user-' + users.length : '',
    latest_assistant: assistants.slice(-1)[0] || '', assistant_count: assistants.length,
    latest_assistant_id: assistants.length ? 'doubao-assistant-' + assistants.length : '',
    assistant_user_id: latestReceive ? (() => {
      const latestAssistant = assistantNodes.slice(-1)[0];
      if (!latestAssistant) return '';
      const preceding = userNodes.filter(node => Boolean(node.compareDocumentPosition(latestAssistant) & 4));
      return preceding.length ? 'doubao-user-' + preceding.length : '';
    })() : '',
    composer_found: Boolean(composer),
    composer_text: composer ? (composer.textContent || composer.innerText || '') : '',
    url: location.href,
    generating: Boolean(latestReceive && !latestReceive.querySelector('[data-testid="message_action_copy"]'))
  });
})()
""" % (self.USER_SELECTOR, self.ASSISTANT_SELECTOR, self.INPUT_SELECTOR)
        value = self._parse_eval_result(self._browser(["eval", script], allow_failure=True))
        return value if isinstance(value, dict) else {}

    def _is_generating(self) -> bool:
        """判断豆包最新回复是否仍未出现完成后的复制操作栏。"""
        snapshot = self._submission_snapshot()
        return bool(snapshot.get("generating"))

    def _wait_for_doubao_upload(self, file_names: List[str]) -> None:
        """等待豆包输入区显示全部附件名称。"""
        names = json.dumps(file_names, ensure_ascii=False)
        script = """
(() => {
  const names = %s;
  const body = document.body ? (document.body.innerText || '') : '';
  const selected = Array.from(document.querySelectorAll('input[type="file"]'))
    .flatMap(input => Array.from(input.files || []).map(file => file.name));
  const markup = document.querySelector('[data-testid="chat_input"]')?.innerHTML || '';
  return JSON.stringify(names.every(name =>
    selected.includes(name) || body.includes(name) || markup.includes(name)));
})()
""" % names
        deadline = time.monotonic() + self.settings.submit_timeout
        while time.monotonic() < deadline:
            result = self._parse_eval_result(
                self._browser(["eval", script], allow_failure=True))
            if result is True:
                return
            time.sleep(0.5)
        raise ProviderTimeout("豆包在 %s 秒内没有显示全部附件：%s" % (
            self.settings.submit_timeout, ", ".join(file_names)))

    def send_with_files(self, prompt: str, files: List[Path],
                        conversation: ConversationHandle) -> str:
        """向豆包上传视觉/文档附件，再发送结构化评估提示词。"""
        if not files:
            return self.send_text(prompt, conversation)
        resolved = [path.resolve() for path in files]
        missing = [str(path) for path in resolved if not path.is_file()]
        if missing:
            raise FileNotFoundError("豆包附件不存在：%s" % missing)
        self._state()
        self._ensure_dialogue_mode()
        selector = self.settings.doubao_upload_input_selector
        # 展开附件菜单后豆包才挂载真正带 onChange 处理器的上传框。
        self._browser(
            ["click", '[data-testid="upload_file_button"]'], allow_failure=True)
        found = _json_from_output(
            self._browser(["find", "--css", selector, "--limit", "10"], allow_failure=True))
        entries = found.get("entries", []) if isinstance(found, dict) else []
        # 豆包只有在附件菜单展开时才挂载带 testid 的第二个 input；页面
        # 始终存在的第一个隐藏文件框同样接受图片和文档，优先作为兜底。
        if not entries:
            selector = 'input[type="file"]'
            found = _json_from_output(
                self._browser(["find", "--css", selector, "--limit", "10"],
                              allow_failure=True))
            entries = found.get("entries", []) if isinstance(found, dict) else []
        if not entries:
            raise ProviderNeedsHuman("豆包附件输入框不存在，请刷新豆包页面后重试。")
        nth = str(entries[-1].get("nth", 0)) if entries else "0"
        try:
            output = self._browser(
                ["upload", "--nth", nth, selector] + [str(path) for path in resolved])
            uploaded = _json_from_output(output)
            if not (isinstance(uploaded, dict) and uploaded.get("uploaded")):
                raise ProviderError("豆包未确认附件上传：%s" % output[-2000:])
        except ProviderError as exc:
            if not self._recoverable_file_upload_error(exc):
                raise
            self._upload_binary_files_via_data_transfer(
                resolved, int(nth), selector=selector)
        self._wait_for_doubao_upload([path.name for path in resolved])
        return self.send_text(
            prompt, conversation, submission_timeout=max(self.settings.submit_timeout, 120))

    def close(self) -> None:
        """关闭 Agent 自己创建的豆包浏览器会话。"""
        if self._opened:
            self._browser(["close"], allow_failure=True)
            self._opened = False
