from __future__ import annotations

import base64
import difflib
import json
import os
import re
import shutil
import subprocess
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel, ValidationError

from ..agents.context_builder import ContextBuilder
from ..schemas.agent_workflow import TaskIntentSpec, TaskRewardBundle
from ..feasibility.motion_prototype.schema import MotionPrototype
from ..feasibility.prompts import MOTION_PROTOTYPE_PROMPT
from ..schemas.decisions import TrainingDiagnosis
from ..schemas.experiments import ConversationHandle, ProviderHealth
from ..schemas.rewards import RewardPlan
from ..schemas.task import TaskSpec
from ..schemas.visual import VisualBehaviorReport
from ..memory.reward_experience.prompts import REWARD_EXPERIENCE_PROMPT
from ..memory.reward_experience.schema import RewardExperienceNarrative
from ..settings import OpenCLISettings, load_opencli_settings
from ..utils.io import atomic_write_text, json_safe
from .errors import ProviderError, ProviderNeedsHuman, ProviderResponseError, ProviderTimeout
from .chatgpt_dom import CHATGPT_ATTACH_FILES_SCRIPT, CHATGPT_SNAPSHOT_SCRIPT

ModelT = TypeVar("ModelT", bound=BaseModel)
Runner = Callable[[Sequence[str], int], subprocess.CompletedProcess]


def _default_runner(args: Sequence[str], timeout: int) -> subprocess.CompletedProcess:
    """以参数数组执行 OpenCLI 子进程并返回执行结果。"""
    return subprocess.run(list(args), text=True, capture_output=True, timeout=timeout, check=False)


def _json_from_output(text: str) -> Any:
    """尝试把命令输出解析为 JSON 对象。"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_value(value: Any) -> str:
    """从 OpenCLI 的嵌套响应信封中提取文本值。"""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("value", "response", "text", "result"):
            if key in value:
                return _extract_value(value[key])
    if isinstance(value, list) and value:
        return _extract_value(value[-1])
    return ""


def _balanced_json_object(text: str) -> Optional[str]:
    """从混合文本中提取首个括号平衡的 JSON 对象。"""
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
        start = text.find("{", start + 1)
    return None


def _repair_common_json_syntax(text: str) -> str:
    """修复模型在 operator 枚举值后偶发添加的单个多余引号。"""
    return re.sub(r'("operator"\s*:\s*"(?:<=|>=|==|<|>)")"', r'\1', text)


def _compact_capabilities(capabilities: Dict[str, Any]) -> Dict[str, Any]:
    """保留奖励设计必需字段并移除环境清单中的路径与重复观察描述。"""
    compact = ContextBuilder.compact_manifest(capabilities)
    compact["retrieved_experience"] = capabilities.get(
        "retrieved_experience", {"enabled": False, "hits": []})
    compact["long_term_memory"] = capabilities.get(
        "long_term_memory", {"enabled": False, "hits": []})
    return compact


class OpenCLIChatGPTWebProvider:
    """完全通过 OpenCLI browser 命令驱动的 ChatGPT 网页 Provider。"""

    def __init__(self, settings: Optional[OpenCLISettings] = None, runner: Runner = _default_runner,
                 record_dir: Optional[Path] = None):
        """初始化 OpenCLIChatGPTWebProvider 实例及其运行依赖。"""
        self.settings = settings or load_opencli_settings()
        self.runner = runner
        self.record_dir = record_dir
        self._opened = False
        self._request_index = 0
        self._submission_baseline: Dict[str, Any] = {}
        self._confirmed_user_id = ""
        self._last_message_snapshot: Dict[str, Any] = {}

    def _run(self, args: Sequence[str], timeout: Optional[int] = None, allow_failure: bool = False) -> str:
        """执行受限子流程并返回结构化结果。"""
        profile_args = ["--profile", self.settings.profile] if self.settings.profile else []
        command = ["opencli"] + profile_args + list(args)
        try:
            result = self.runner(command, timeout or self.settings.command_timeout)
        except subprocess.TimeoutExpired as exc:
            raise ProviderTimeout("OpenCLI 命令执行超时：%s" % " ".join(command[:4])) from exc
        if result.returncode != 0 and not allow_failure:
            outputs = []
            if result.stdout and result.stdout.strip():
                outputs.append("stdout:\n" + result.stdout.strip())
            if result.stderr and result.stderr.strip():
                outputs.append("stderr:\n" + result.stderr.strip())
            safe_error = ("\n".join(outputs) or "unknown OpenCLI failure")[-6000:]
            raise ProviderError(safe_error)
        return result.stdout

    def _browser(self, command: Sequence[str], timeout: Optional[int] = None, allow_failure: bool = False) -> str:
        """在固定会话中执行 OpenCLI browser 子命令。"""
        return self._run(["browser", self.settings.session] + list(command), timeout, allow_failure)

    def _state(self) -> str:
        """刷新页面状态并检测登录失效或验证码。"""
        state = self._browser(["state"])
        self._raise_for_usage_limit(state)
        lowered = state.lower()
        if any(item in lowered for item in ("captcha", "verify you are human", "cloudflare")):
            raise ProviderNeedsHuman("ChatGPT page requires CAPTCHA verification")
        if any(item in lowered for item in ("log in", "sign up", "登录", "注册")) and "message chatgpt" not in lowered:
            raise ProviderNeedsHuman("ChatGPT login has expired")
        self._assert_expected_page()
        return state

    @staticmethod
    def _url_matches_expected(current_url: str, expected_url: str) -> bool:
        """只接受与配置地址同域的 HTTP(S) 页面，拒绝 about:blank 和错误站点。"""
        current = urlparse(str(current_url).strip())
        expected = urlparse(str(expected_url).strip())
        current_host = (current.hostname or "").lower()
        expected_host = (expected.hostname or "").lower()
        return bool(
            current.scheme in ("http", "https") and expected_host and
            (current_host == expected_host or current_host.endswith("." + expected_host))
        )

    def _current_page_url(self) -> str:
        """读取当前会话真实 URL；OpenCLI 无法读取时返回空串交由调用方处理。"""
        output = self._browser(
            ["eval", "JSON.stringify(location.href || '')"], allow_failure=True)
        parsed = self._parse_eval_result(output)
        if isinstance(parsed, str):
            return parsed.strip()
        if isinstance(parsed, dict):
            return str(parsed.get("url", "")).strip()
        return ""

    def _assert_expected_page(self) -> None:
        """要求实际 URL 可读且属于目标站点，避免错误网页或读取失败被放行。"""
        current_url = self._current_page_url()
        if not current_url:
            raise ProviderError("无法读取 OpenCLI 会话地址，请检查浏览器扩展连接。")
        if current_url and not self._url_matches_expected(
                current_url, self.settings.chatgpt_url):
            expected_host = urlparse(self.settings.chatgpt_url).hostname or "目标站点"
            raise ProviderNeedsHuman(
                "OpenCLI 会话没有停留在 %s，当前页面为 %s；请确认浏览器标签页可正常访问目标站点。" %
                (expected_host, current_url[:300]))

    def _open_expected_page(self, url: str, label: str) -> None:
        """导航至指定 Provider 页面，并在继续 DOM 操作前确认真实地址已生效。"""
        self._browser(["open", url], timeout=self.settings.connect_timeout)
        deadline = time.monotonic() + self.settings.connect_timeout
        last_url = ""
        while time.monotonic() < deadline:
            last_url = self._current_page_url()
            if self._url_matches_expected(last_url, url):
                return
            time.sleep(0.25)
        raise ProviderNeedsHuman(
            "%s 页面在 %s 秒内未成功打开；当前页面为 %s。请检查网络、登录状态和浏览器扩展后重试。" %
            (label, self.settings.connect_timeout, last_url or "无法读取"))

    @staticmethod
    def _raise_for_usage_limit(text: str) -> None:
        """识别 ChatGPT 额度耗尽页面或回复，并给出可执行的人工恢复提示。"""
        normalized = re.sub(r"\s+", " ", str(text)).strip()
        lowered = normalized.lower()
        exhausted_markers = (
            "你暂时已用完工作用量",
            "你已达到使用上限",
            "升级订阅套餐或添加额度",
            "you've reached your usage limit",
            "you have reached your usage limit",
            "usage limit reached",
        )
        if not any(marker in lowered for marker in exhausted_markers):
            return
        reset = re.search(r"(?:在\s*)?(\d{1,2}:\d{2})(?:\s*后)?(?:\s*重置|\s*重试)", normalized)
        reset_hint = "；页面显示可在 %s 后重试" % reset.group(1) if reset else ""
        raise ProviderNeedsHuman(
            "ChatGPT 工作用量已耗尽%s。请等待额度重置或添加额度后，再从当前策略继续闭环；"
            "已有训练 checkpoint 不受影响。" % reset_hint)

    def _check_page_usage_limit(self) -> None:
        """读取完整页面可见正文，弥补 OpenCLI 精简状态可能遗漏的额度弹窗。"""
        output = self._browser(
            ["eval", "JSON.stringify((document.body&&document.body.innerText||'').slice(0,30000))"],
            allow_failure=True,
        )
        page_text = self._parse_eval_result(output)
        self._raise_for_usage_limit(page_text if isinstance(page_text, str) else "")

    @staticmethod
    def _bridge_connected(doctor_output: str) -> bool:
        """判断 OpenCLI doctor 输出是否确认浏览器扩展已经连接。"""
        normalized = re.sub(r"\s+", " ", doctor_output).strip().lower()
        return bool(re.search(r"\[ok\]\s+extension:\s*connected", normalized))

    def _launch_bridge_browser(self) -> bool:
        """在图形会话中启动默认 Chrome 的 ChatGPT 标签页，以激活 OpenCLI 扩展。"""
        if not self.settings.auto_launch_browser or self.runner is not _default_runner:
            return False
        if not (os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")):
            return False
        candidates = ([self.settings.bridge_browser_executable]
                      if self.settings.bridge_browser_executable else
                      ["google-chrome", "chromium", "chromium-browser"])
        executable = next((shutil.which(name) for name in candidates if name), None)
        if executable is None:
            return False
        try:
            subprocess.Popen(
                [executable, "--new-window", self.settings.chatgpt_url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                shell=False,
            )
            return True
        except OSError:
            return False

    def _ensure_bridge_connected(self) -> None:
        """检查浏览器桥接，断线时重启守护进程、启动默认 Chrome 并等待重连。"""
        doctor = self._run(["doctor"], timeout=self.settings.connect_timeout, allow_failure=True)
        if self._bridge_connected(doctor):
            return

        self._run(["daemon", "restart"], timeout=self.settings.connect_timeout, allow_failure=True)
        browser_started = self._launch_bridge_browser()
        deadline = time.monotonic() + self.settings.connect_timeout
        while time.monotonic() < deadline:
            time.sleep(1.0)
            doctor = self._run(["doctor"], timeout=self.settings.connect_timeout, allow_failure=True)
            if self._bridge_connected(doctor):
                return
        raise ProviderNeedsHuman(
            "OpenCLI 浏览器扩展未连接。程序已自动重启守护进程%s，但扩展仍未重连；" %
            ("并启动默认 Chrome" if browser_started else "") +
            "请打开安装了 Browser Bridge 扩展的 Chrome，确认扩展已启用，并保持一个已登录 ChatGPT 的标签页，"
            "看到 opencli doctor 显示 Extension: connected 后再重新下发任务。"
        )

    def doctor(self) -> ProviderHealth:
        """检查运行环境、外部依赖和服务健康状态。"""
        details: List[str] = []
        if shutil.which("opencli") is None:
            return ProviderHealth(available=False, opencli_available=False, extension_connected=False,
                                  chatgpt_logged_in=False, image_upload_supported=False,
                                  recoverable=False, details=["opencli executable not found"])
        try:
            doctor = self._run(["doctor"], allow_failure=True)
        except ProviderError as exc:
            return ProviderHealth(
                available=False, opencli_available=True, extension_connected=False,
                chatgpt_logged_in=False, image_upload_supported=False, recoverable=True,
                details=["OpenCLI 健康检查失败：%s" % exc])
        extension = "extension: connected" in doctor.lower()
        details.append(doctor.strip()[-1000:])
        try:
            status_text = self._run(["chatgpt", "status", "-f", "json"], allow_failure=True)
        except ProviderError as exc:
            details.append("ChatGPT 登录状态检查失败：%s" % exc)
            return ProviderHealth(
                available=False, opencli_available=True, extension_connected=extension,
                chatgpt_logged_in=False, image_upload_supported=False, recoverable=True,
                details=details)
        status = _json_from_output(status_text)
        logged_in = "\"login\":\"yes\"" in re.sub(r"\s+", "", status_text.lower())
        if isinstance(status, list) and status:
            logged_in = str(status[0].get("Login", "")).lower() == "yes"
        return ProviderHealth(available=extension and logged_in, opencli_available=True,
                              extension_connected=extension, chatgpt_logged_in=logged_in,
                              image_upload_supported=extension and logged_in, details=details)

    def open_or_bind(self) -> None:
        """打开或绑定配置指定的 ChatGPT 浏览器会话。"""
        self._ensure_bridge_connected()
        if self.settings.bind_existing_tab:
            try:
                self._browser(["bind"], timeout=self.settings.connect_timeout)
            except ProviderTimeout as exc:
                raise ProviderNeedsHuman(
                    "OpenCLI 扩展已经连接，但无法绑定当前浏览器标签页。请把已登录的 ChatGPT 标签页切到前台，"
                    "确认没有验证码或登录弹窗，然后重新下发任务。"
                ) from exc
        else:
            self.settings.owned_session = True
        # bind 只建立会话映射，并不保证被绑定标签页就是 ChatGPT。始终显式
        # 导航并核对 URL，防止 about:blank 被旧版 state 误判为聊天页。
        self._open_expected_page(self.settings.chatgpt_url, "ChatGPT")
        self._state()
        self._ensure_chat_mode()
        self._opened = True

    def new_conversation(self, title_hint: str) -> ConversationHandle:
        """创建新会话并返回可持久化的会话句柄。"""
        if not self._opened:
            self.open_or_bind()
        # 打开根地址会创建全新网页会话，同时强制验证没有落到空白页。
        self._open_expected_page(self.settings.chatgpt_url, "ChatGPT")
        self._state()
        self._ensure_chat_mode()
        return ConversationHandle(conversation_id=uuid.uuid4().hex, title_hint=title_hint,
                                  owned=self.settings.owned_session)

    def _ensure_chat_mode(self) -> None:
        """发现 ChatGPT 的聊天/工作切换器时，强制选择普通聊天模式并验证结果。"""
        if not self.settings.force_chat_mode:
            return
        inspect_script = """
(() => {
  const chat = document.querySelector('[data-tpp-toggle-value="chatgpt"]');
  const work = document.querySelector('[data-tpp-toggle-value="work"]');
  if (!chat && !work) return JSON.stringify({available: false, active: 'legacy_chat'});
  const chatActive = Boolean(chat && (chat.getAttribute('aria-checked') === 'true' ||
    chat.getAttribute('data-state') === 'on'));
  const workActive = Boolean(work && (work.getAttribute('aria-checked') === 'true' ||
    work.getAttribute('data-state') === 'on'));
  return JSON.stringify({available: true, chat_active: chatActive, work_active: workActive});
})()
"""
        state = self._parse_eval_result(self._browser(["eval", inspect_script], allow_failure=True))
        if not isinstance(state, dict):
            raise ProviderNeedsHuman("无法读取 ChatGPT 聊天/工作模式，请刷新页面后重试。")
        if not state.get("available"):
            # 兼容尚未提供双模式切换器的旧版纯聊天页面。
            return
        if state.get("chat_active") and not state.get("work_active"):
            return
        click_script = """
(() => {
  const chat = document.querySelector('[data-tpp-toggle-value="chatgpt"]');
  if (!chat) return JSON.stringify({clicked: false});
  chat.click();
  return JSON.stringify({clicked: true});
})()
"""
        clicked = self._parse_eval_result(self._browser(["eval", click_script], allow_failure=True))
        if not (isinstance(clicked, dict) and clicked.get("clicked")):
            raise ProviderNeedsHuman("无法从 ChatGPT 工作模式切换到聊天模式，请手动选择“聊天”后重试。")
        deadline = time.monotonic() + min(10, self.settings.submit_timeout)
        while time.monotonic() < deadline:
            time.sleep(0.25)
            state = self._parse_eval_result(
                self._browser(["eval", inspect_script], allow_failure=True))
            if isinstance(state, dict) and state.get("chat_active") and not state.get("work_active"):
                return
        raise ProviderNeedsHuman("ChatGPT 模式切换未生效，请手动选择“聊天”后重试。")

    def _record(self, conversation: ConversationHandle, prompt: str, response: str) -> None:
        """将提示词和原始回复写入会话记录目录。"""
        if self.record_dir is None:
            return
        self._request_index += 1
        directory = self.record_dir / conversation.conversation_id
        atomic_write_text(directory / ("request_%03d.txt" % self._request_index), prompt)
        atomic_write_text(directory / ("response_%03d.txt" % self._request_index), response)

    def _write_prompt_document(self, conversation: ConversationHandle, title: str, prompt: str) -> Path:
        """把超长提示词写成可上传、可审计的 Markdown 需求文档。"""
        base = self.record_dir or (Path.cwd() / "artifacts" / "provider_records")
        safe_title = re.sub(r"[^A-Za-z0-9_-]+", "-", title).strip("-") or "request"
        path = base / conversation.conversation_id / (safe_title + "_需求文档.md")
        document = (
            "# 强化学习任务设计要求\n\n"
            "以下内容是本次请求的完整且唯一要求。请完整读取环境能力清单和 JSON Schema，"
            "不得虚构清单中不存在的物理量或奖励函数。\n\n"
            "## 完整请求\n\n" + prompt.rstrip() + "\n"
        )
        atomic_write_text(path, document)
        return path

    def _send_model_prompt(self, conversation: ConversationHandle, prompt: str, title: str,
                           files: Optional[List[Path]] = None) -> str:
        """统一发送首次请求和修复请求，并把达到阈值的正文外置为附件。"""
        request_files = list(files or [])
        submitted_prompt = prompt
        if len(prompt) >= self.settings.prompt_attachment_threshold:
            document = self._write_prompt_document(conversation, title, prompt)
            request_files.insert(0, document)
            submitted_prompt = (
                "请完整阅读附件《%s》，严格按照其中的环境能力约束和 JSON Schema 完成本次任务。"
                "只返回文档要求的严格 JSON，不要添加解释，也不要省略任何必填字段。" % document.name
            )
        return (self.send_with_files(submitted_prompt, request_files, conversation)
                if request_files else self.send_text(submitted_prompt, conversation))

    def _fill_prompt(self, prompt: str) -> None:
        """动态定位输入框、填入提示词并验证实际内容。"""
        self._state()
        direct_selectors = [
            "#prompt-textarea",
            '[data-testid="prompt-textarea"]',
            '[contenteditable="true"][role="textbox"]',
        ]
        last_output = ""
        for selector in direct_selectors:
            found = self._browser(["find", "--css", selector, "--limit", "2"], allow_failure=True)
            found_data = _json_from_output(found)
            if not (isinstance(found_data, dict) and int(found_data.get("matches_n", 0)) > 0):
                continue
            output = self._browser(["fill", selector, prompt], allow_failure=True)
            last_output = output
            parsed = _json_from_output(output)
            if isinstance(parsed, dict) and parsed.get("filled") and parsed.get("verified"):
                return
            if isinstance(parsed, dict) and parsed.get("filled") and self._verify_composer_content(prompt):
                return
        find = self._browser(["find", "--role", "textbox", "--name", "Message", "--limit", "5"],
                             allow_failure=True)
        if '"matches_n":0' in re.sub(r"\s+", "", find):
            find = self._browser(["find", "--role", "textbox", "--limit", "5"], allow_failure=True)
        if '"matches_n":0' in re.sub(r"\s+", "", find):
            raise ProviderNeedsHuman("ChatGPT message textbox was not found after checking page state")
        find_data = _json_from_output(find)
        entries = find_data.get("entries", []) if isinstance(find_data, dict) else []
        candidates = [entry for entry in entries if entry.get("visible") and
                      str(entry.get("attrs", {}).get("type", "")).lower() != "file"]
        chosen = candidates[0] if candidates else None
        element_id = str(chosen.get("attrs", {}).get("id", "")) if chosen else ""
        if element_id and re.match(r"^[A-Za-z_][A-Za-z0-9_-]*$", element_id):
            output = self._browser(["fill", "#" + element_id, prompt], allow_failure=True)
        else:
            output = self._browser(["fill", "--role", "textbox", prompt], allow_failure=True)
        last_output = output
        parsed = _json_from_output(output)
        if isinstance(parsed, dict) and parsed.get("filled") and parsed.get("verified"):
            return
        if isinstance(parsed, dict) and parsed.get("filled") and self._verify_composer_content(prompt):
            return
        detail = last_output[-2000:] if last_output else "输入框定位命令没有返回结果"
        raise ProviderError("OpenCLI 找到了 ChatGPT 输入框，但无法验证提示词已经完整填入：%s" % detail)

    def _verify_composer_content(self, prompt: str) -> bool:
        """规范化富文本段落空白后，在页面内严格比较编辑器全文。"""
        expected = json.dumps(prompt, ensure_ascii=False)
        script = """
(() => {
  const selectors = ['#prompt-textarea', '[data-testid="prompt-textarea"]',
    '[contenteditable="true"][role="textbox"]'];
  const candidates = selectors.flatMap(selector => Array.from(document.querySelectorAll(selector)));
  const element = candidates.find(item => item.getClientRects().length > 0 &&
    (item.isContentEditable || item instanceof HTMLTextAreaElement || item instanceof HTMLInputElement));
  if (!element) return JSON.stringify({found: false, matches: false});
  const actual = element.isContentEditable ? (element.innerText || element.textContent || '') : String(element.value || '');
  const normalize = value => String(value).replace(/\\s+/g, ' ').trim();
  const expected = %s;
  const actualNormalized = normalize(actual);
  const expectedNormalized = normalize(expected);
  return JSON.stringify({
    found: true,
    matches: actualNormalized === expectedNormalized,
    actual_length: actual.length,
    expected_length: expected.length,
    actual_normalized_length: actualNormalized.length,
    expected_normalized_length: expectedNormalized.length,
    prefix_matches: actualNormalized.slice(0, 160) === expectedNormalized.slice(0, 160),
    suffix_matches: actualNormalized.slice(-160) === expectedNormalized.slice(-160)
  });
})()
""" % expected
        output = self._browser(["eval", script], allow_failure=True)
        parsed = _json_from_output(output)
        if isinstance(parsed, str):
            parsed = _json_from_output(parsed)
        return bool(isinstance(parsed, dict) and parsed.get("found") and parsed.get("matches") and
                    parsed.get("prefix_matches") and parsed.get("suffix_matches") and
                    parsed.get("actual_normalized_length") == parsed.get("expected_normalized_length"))

    def _latest_assistant(self) -> str:
        """通过共用的新旧 DOM 解析器读取助手正文，并保留消息身份。"""
        self._last_message_snapshot = self._submission_snapshot()
        return str(self._last_message_snapshot.get("latest_assistant", ""))

    def _latest_user(self) -> str:
        """通过共用解析器读取已提交用户消息，避免与回复读取使用不同选择器。"""
        return str(self._submission_snapshot().get("latest_user", ""))

    def _submission_snapshot(self) -> Dict[str, Any]:
        """一次读取最新用户消息和编辑器状态，避免高频 state 调用干扰页面时序。"""
        output = self._browser(["eval", CHATGPT_SNAPSHOT_SCRIPT], allow_failure=True)
        parsed = self._parse_eval_result(output)
        return parsed if isinstance(parsed, dict) else {}

    def _collect_page_debug_artifacts(self, conversation: Optional[ConversationHandle], label: str) -> None:
        """采集页面快照、主要 DOM / 按钮状态、性能资源与有限的控制台线索，写入记录目录以便离线分析。"""
        try:
            base = self.record_dir or (Path.cwd() / "artifacts" / "provider_records")
            convo_dir = base / (conversation.conversation_id if conversation else "no-conversation")
            convo_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            # 1) 结构化 submission snapshot
            try:
                snapshot = self._submission_snapshot()
            except Exception as exc:
                snapshot = {"error": "failed_to_capture_submission_snapshot", "detail": str(exc)}
            atomic_write_text(convo_dir / (f"debug_snapshot_{label}_{timestamp}.json"), json.dumps(snapshot, ensure_ascii=False, indent=2))

            # 2) Lightweight DOM / buttons / perf summary
            script = r'''
(() => {
  const body = document.body ? String(document.body.innerText || '').slice(0, 20000) : '';
  const buttons = Array.from(document.querySelectorAll('button')).slice(0,200).map(b=>({label: b.getAttribute('aria-label')||b.innerText||'', disabled: !!b.disabled, dataset: b.dataset?Object.fromEntries(Object.entries(b.dataset)):{}}));
  const stop = !!document.querySelector('button[data-testid="stop-button"]');
  const imgs = Array.from(document.querySelectorAll('img')).slice(0,10).map(i=>i.src);
  const perf = (window.performance && typeof window.performance.getEntries === 'function') ? window.performance.getEntries().slice(-50).map(e=>({name: e.name, initiatorType: e.initiatorType, duration: e.duration, transferSize: e.transferSize||0})) : [];
  return JSON.stringify({body_prefix: body, buttons, stop, imgs, perf, url: location.href, title: document.title});
})()
'''
            try:
                output = self._browser(["eval", script], allow_failure=True)
                parsed = self._parse_eval_result(output)
            except Exception as exc:
                parsed = {"error": "eval_failed", "detail": str(exc)}
            atomic_write_text(convo_dir / (f"debug_dom_{label}_{timestamp}.json"), json.dumps(parsed, ensure_ascii=False, indent=2))

            # 3) Try to capture visible HTML head/body (trimmed)
            try:
                html_out = self._browser(["eval", "JSON.stringify((document.documentElement&&document.documentElement.outerHTML||'').slice(0,200000))"], allow_failure=True)
                html_parsed = _json_from_output(html_out)
                if isinstance(html_parsed, str):
                    atomic_write_text(convo_dir / (f"debug_page_html_{label}_{timestamp}.html"), html_parsed)
            except Exception:
                # best-effort
                pass

            # 4) File inputs info
            try:
                found = self._browser(["find", "--css", "input[type=file]", "--limit", "10"], allow_failure=True)
                found_data = _json_from_output(found)
                atomic_write_text(convo_dir / (f"debug_file_inputs_{label}_{timestamp}.json"), json.dumps(found_data, ensure_ascii=False, indent=2))
            except Exception:
                pass
        except Exception:
            # never fail the provider because debug capture failed
            return

    @staticmethod
    def _can_retry_silent_response(snapshot: Dict[str, Any]) -> bool:
        """仅在消息已提交但页面静默无回复时允许换新会话重试。"""
        composer = OpenCLIChatGPTWebProvider._normalize_submitted_text(
            str(snapshot.get("composer_text", "")))
        return bool(
            int(snapshot.get("user_count", 0)) > 0 and
            int(snapshot.get("assistant_count", 0)) == 0 and
            not str(snapshot.get("latest_assistant", "")).strip() and
            snapshot.get("composer_found") and
            not composer and
            not snapshot.get("generating")
        )

    @staticmethod
    def _normalize_submitted_text(value: str) -> str:
        """消除网页渲染引入的 Unicode 格式字符、Markdown 反引号和空白差异。"""
        normalized = unicodedata.normalize("NFKC", value).replace("`", "")
        normalized = "".join(
            character for character in normalized if unicodedata.category(character) != "Cf")
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _matches_submitted_prompt(actual: str, prompt: str) -> bool:
        """比较已渲染用户消息与原文，容忍受限的 DOM 渲染差异。"""
        actual_normalized = OpenCLIChatGPTWebProvider._normalize_submitted_text(actual)
        prompt_normalized = OpenCLIChatGPTWebProvider._normalize_submitted_text(prompt)
        if not actual_normalized or not prompt_normalized:
            return False
        if actual_normalized == prompt_normalized:
            return True
        if actual_normalized.endswith(prompt_normalized):
            attachment_prefix = actual_normalized[:-len(prompt_normalized)].strip()
            safe_attachment = (r'[^<>:"/\\|?*\x00-\x1f]{1,240}'
                               r'\.(?:md|txt|json|ya?ml|png|jpe?g|webp|gif)'
                               r'\s*(?:文件|文档|代码|File|Document|Code)')
            if re.fullmatch(r"(?:%s\s*)+" % safe_attachment, attachment_prefix,
                            flags=re.IGNORECASE):
                return True
        return False

    def _submission_confirmed(self, snapshot: Dict[str, Any], previous_user: str,
                              prompt: str) -> bool:
        """仅在本次新用户消息完整匹配时确认提交；清空草稿不代表服务端接收。"""
        latest = str(snapshot.get("latest_user", ""))
        if not self._matches_submitted_prompt(latest, prompt):
            return False
        baseline = self._submission_baseline
        if baseline:
            new_id = snapshot.get("latest_user_id")
            fresh = (bool(new_id and new_id != baseline.get("latest_user_id")) or
                     int(snapshot.get("user_count", 0)) > int(baseline.get("user_count", 0)))
        else:
            fresh = latest != previous_user
        if fresh:
            self._confirmed_user_id = str(snapshot.get("latest_user_id", ""))
        return bool(fresh)

    def _click_send_button(self) -> None:
        """等待 ChatGPT 发送按钮可用并显式点击，避免依赖输入框焦点。"""
        # OpenCLI 的精简 state 有时不包含侧栏中的额度弹窗；发送前直接读取
        # 一次可见正文，避免额度为零时仍等待按钮和提交确认超时。
        self._check_page_usage_limit()
        selectors = [
            'button[data-testid="send-button"]:not([disabled])',
            '#composer-submit-button:not([disabled])',
            'button[aria-label="Send prompt"]:not([disabled])',
            'button[aria-label="发送提示"]:not([disabled])',
            'button[aria-label="发送"]:not([disabled])',
        ]
        # 定位和点击在同一次页面执行中完成。扩展的坐标点击可能返回
        # clicked=true 却没有触发发送，尤其在附件改变编辑器布局之后。
        script = """
(() => {
  const selectors = %s;
  for (const selector of selectors) {
    const button = Array.from(document.querySelectorAll(selector)).find(item => {
      const style = getComputedStyle(item);
      return item.getClientRects().length > 0 && style.visibility !== 'hidden' &&
        !item.disabled && item.getAttribute('aria-disabled') !== 'true' &&
        item.getAttribute('aria-busy') !== 'true';
    });
    if (!button) continue;
    button.click();
    return JSON.stringify({clicked: true, selector});
  }
  return JSON.stringify({clicked: false});
})()
""" % json.dumps(selectors)
        deadline = time.monotonic() + self.settings.submit_timeout
        last_output = ""
        while time.monotonic() < deadline:
            self._state()
            last_output = self._browser(["eval", script])
            clicked_data = self._parse_eval_result(last_output)
            if isinstance(clicked_data, dict) and clicked_data.get("clicked"):
                return
            time.sleep(0.25)
        self._check_page_usage_limit()
        detail = last_output[-1000:] if last_output else "没有发现已启用的发送按钮"
        raise ProviderTimeout("ChatGPT 发送按钮在 %s 秒内未变为可点击状态：%s" %
                              (self.settings.submit_timeout, detail))

    def _wait_for_submission(self, previous_user: str, prompt: str,
                             timeout: Optional[int] = None) -> None:
        """确认新用户消息已进入会话，防止把仅填入输入框误判为发送成功。"""
        wait_seconds = int(timeout or self.settings.submit_timeout)
        deadline = time.monotonic() + wait_seconds
        latest = ""
        snapshot: Dict[str, Any] = {}
        while time.monotonic() < deadline:
            snapshot = self._submission_snapshot()
            latest = str(snapshot.get("latest_user", ""))
            if self._submission_confirmed(snapshot, previous_user, prompt):
                return
            time.sleep(0.5)
        # 边界时刻检查页面健康状态并再读取一次。
        self._state()
        snapshot = self._submission_snapshot()
        latest = str(snapshot.get("latest_user", ""))
        if self._submission_confirmed(snapshot, previous_user, prompt):
            return
        latest_normalized = self._normalize_submitted_text(latest)
        prompt_normalized = self._normalize_submitted_text(prompt)
        similarity = difflib.SequenceMatcher(
            None, latest_normalized, prompt_normalized, autojunk=False).ratio() \
            if latest_normalized and prompt_normalized else 0.0
        raise ProviderTimeout(
            "已点击发送按钮，但未在会话中确认新的用户消息；"
            "原始长度=%s，规范化长度=%s，期望长度=%s，相似度=%.4f，编辑器存在=%s" % (
                len(latest), len(latest_normalized), len(prompt_normalized), similarity,
                bool(snapshot.get("composer_found"))))

    def _wait_for_response(self, previous: str) -> str:
        """等待助手回复稳定，并防止把流式 JSON 的前缀误判为完整结果。"""
        started = time.monotonic()
        deadline = started + self.settings.response_timeout
        hard_deadline = started + max(self.settings.response_timeout,
                                      self.settings.response_generation_timeout)
        last = ""
        stable_count = 0
        while time.monotonic() < deadline:
            self._state()
            current = self._latest_assistant()
            self._raise_for_usage_limit(current)
            snapshot = self._last_message_snapshot
            # 视觉附件分析可能超过初始五分钟。仅为已确认提交的本轮请求
            # 续等，不重发，也不让上一轮的生成状态延长本轮等待。
            if (snapshot.get("generating") and self._confirmed_user_id and
                    snapshot.get("latest_user_id") == self._confirmed_user_id):
                deadline = min(hard_deadline, max(deadline, time.monotonic() + 60))
            new_message = (current != previous or bool(
                snapshot.get("latest_assistant_id") and
                snapshot.get("latest_assistant_id") != self._submission_baseline.get("latest_assistant_id")))
            belongs_to_request = (not self._confirmed_user_id or
                                  snapshot.get("assistant_user_id") == self._confirmed_user_id)
            if current and new_message and belongs_to_request:
                if current == last:
                    stable_count += 1
                else:
                    stable_count = 0
                    last = current
                if stable_count >= 2 and not self._is_generating():
                    # OpenCLI 偶尔无法识别新版 ChatGPT 的流式生成按钮。结构化
                    # 回复必须形成完整 JSON 后才能返回，避免只记录 "{" 或半段对象。
                    if self._response_contains_complete_json(current):
                        return current
                    # 非 JSON 的拒答或错误信息仍允许在更长静默期后交给上层处理。
                    if stable_count >= 8:
                        return current
            time.sleep(1.0)
        raise ProviderTimeout(
            "ChatGPT response did not complete before timeout; "
            "initial_limit=%ss, generation_limit=%ss, generating=%s, "
            "assistant_count=%s, request_confirmed=%s" % (
                self.settings.response_timeout, self.settings.response_generation_timeout,
                self._last_message_snapshot.get("generating", False),
                self._last_message_snapshot.get("assistant_count", 0),
                bool(self._confirmed_user_id)))

    @classmethod
    def _response_contains_complete_json(cls, response: str) -> bool:
        """判断回复中是否已经出现可完整解析的 JSON 对象或数组。"""
        stripped = str(response).strip()
        if not stripped:
            return False
        try:
            cls._parse_candidate(stripped)
            return True
        except ValueError:
            return False

    def _is_generating(self) -> bool:
        """检测 ChatGPT 是否仍在思考、调用视觉工具或流式生成回复。"""
        script = """
(() => {
  if (document.querySelector('button[data-testid="stop-button"]')) return JSON.stringify(true);
  return JSON.stringify(Array.from(document.querySelectorAll('button')).some(button => {
    const label = button.getAttribute('aria-label') || '';
    return /Stop generating|Stop responding|停止生成|停止回答|Thinking|正在思考|^(?:停止|Stop)$/i.test(label.trim());
  }));
})()
"""
        output = self._browser(["eval", script], allow_failure=True)
        return self._parse_eval_result(output) is True

    def send_text(self, prompt: str, conversation: ConversationHandle,
                  submission_timeout: Optional[int] = None) -> str:
        """向指定网页会话发送文本并返回最新助手回复。"""
        self._submission_baseline = self._submission_snapshot()
        self._confirmed_user_id = ""
        self._last_message_snapshot = {}
        previous_assistant = str(self._submission_baseline.get("latest_assistant", ""))
        previous_user = str(self._submission_baseline.get("latest_user", ""))
        error: Optional[Exception] = None
        for _ in range(self.settings.max_retries + 1):
            try:
                self._fill_prompt(prompt)
                self._click_send_button()
                break
            except ProviderNeedsHuman:
                raise
            except (ProviderError, ProviderTimeout) as exc:
                error = exc
                time.sleep(1.0)
        else:
            raise ProviderError("OpenCLI 提交消息失败，重试后仍未确认发送：%s" % error)
        # 点击动作一旦成功便不再重复点击。大附件可能需要较长时间才在会话中
        # 生成用户消息，重复点击可能造成重复请求或中断已开始的回复。
        try:
            self._wait_for_submission(previous_user, prompt, timeout=submission_timeout)
        except ProviderTimeout as exc:
            # 在提交确认超时时收集页面调试信息以便事后分析，但不要影响提交的重试语义
            try:
                self._collect_page_debug_artifacts(conversation, "submission_timeout")
            except Exception:
                pass
            raise
        try:
            response = self._wait_for_response(previous_assistant)
        except ProviderTimeout as exc:
            # 响应等待超时（静默无回复），采集详细调试信息并再抛出以由上层决定恢复策略
            try:
                self._collect_page_debug_artifacts(conversation, "response_timeout")
            except Exception:
                pass
            raise
        self._record(conversation, prompt, response)
        return response

    def _upload_text_documents(self, files: List[Path]) -> None:
        """在页面内构造文本文件对象并附加，绕过扩展对本地文件路径的限制。"""
        mime_types = {
            ".md": "text/markdown", ".txt": "text/plain", ".json": "application/json",
            ".yaml": "application/yaml", ".yml": "application/yaml",
        }
        documents = [{"name": path.name, "type": mime_types.get(path.suffix.lower(), "text/plain"),
                      "content": path.read_text(encoding="utf-8")} for path in files]
        payload = json.dumps(documents, ensure_ascii=False)
        script = """
(() => {
  const input = document.querySelector('#upload-files') ||
    Array.from(document.querySelectorAll('input[type=file]')).find(item =>
      !String(item.getAttribute('accept') || '').toLowerCase().includes('image/'));
  if (!input) return JSON.stringify({ok: false, reason: 'general_file_input_not_found'});
  const documents = %s;
  const transfer = new DataTransfer();
  for (const document of documents) {
    transfer.items.add(new File([document.content], document.name, {type: document.type}));
  }
  %s
  return JSON.stringify(attachFiles(input, transfer));
})()
""" % (payload, CHATGPT_ATTACH_FILES_SCRIPT)
        output = self._browser(["eval", script])
        parsed = _json_from_output(output)
        if isinstance(parsed, str):
            parsed = _json_from_output(parsed)
        expected_names = [path.name for path in files]
        if not (isinstance(parsed, dict) and parsed.get("ok") and
                int(parsed.get("count", 0)) == len(files) and parsed.get("names") == expected_names):
            raise ProviderError("ChatGPT 页面未确认需求文档附件：%s" % output[-2000:])
        self._wait_for_upload_preview(expected_names, allow_media=False)

    @staticmethod
    def _recoverable_file_upload_error(error: Exception) -> bool:
        """判断本地路径上传是否被扩展能力或 Chrome 安全策略拒绝。"""
        message = str(error).lower()
        return any(fragment in message for fragment in (
            "not allowed", "unknown action", "not supported", "setfileinput",
            "set-file-input", "no element found", "filechooseropened", "file chooser",
            "may not have opened a file chooser",
        ))

    @staticmethod
    def _parse_eval_result(output: str) -> Any:
        """解析 browser eval 可能返回的一层或两层 JSON 信封。"""
        parsed = _json_from_output(output)
        if isinstance(parsed, str):
            nested = _json_from_output(parsed)
            return nested if nested is not None else parsed
        return parsed

    def _upload_binary_files_via_data_transfer(
            self, files: List[Path], nth: int,
            selector: str = "input[type=file]") -> None:
        """分块传输二进制文件并用 DataTransfer 附加到指定文件框。"""
        mime_types = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp", ".gif": "image/gif", ".mp4": "video/mp4",
        }
        token = "rl_agent_%s" % uuid.uuid4().hex
        descriptors = [{"name": path.name,
                        "type": mime_types.get(path.suffix.lower(), "application/octet-stream"),
                        "chunks": []} for path in files]
        token_json = json.dumps(token)
        init_script = """
(() => {
  window.__rlAgentFileUploads = window.__rlAgentFileUploads || {};
  window.__rlAgentFileUploads[%s] = {files: %s};
  return JSON.stringify({ok: true});
})()
""" % (token_json, json.dumps(descriptors, ensure_ascii=False))
        init_output = self._browser(["eval", init_script])
        init_result = self._parse_eval_result(init_output)
        if not (isinstance(init_result, dict) and init_result.get("ok")):
            raise ProviderError("无法初始化 ChatGPT 图片分块上传：%s" % init_output[-1000:])

        try:
            # 单个 Linux 命令行参数通常限制在 128 KiB；48 KiB 块留出脚本和 JSON 转义余量。
            chunk_size = 48 * 1024
            for file_index, path in enumerate(files):
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                for offset in range(0, len(encoded), chunk_size):
                    chunk = encoded[offset:offset + chunk_size]
                    append_script = """
(() => {
  const upload = window.__rlAgentFileUploads && window.__rlAgentFileUploads[%s];
  if (!upload) return JSON.stringify({ok: false, reason: 'upload_session_missing'});
  upload.files[%d].chunks.push(%s);
  return JSON.stringify({ok: true, chunks: upload.files[%d].chunks.length});
})()
""" % (token_json, file_index, json.dumps(chunk), file_index)
                    append_output = self._browser(["eval", append_script])
                    append_result = self._parse_eval_result(append_output)
                    if not (isinstance(append_result, dict) and append_result.get("ok")):
                        raise ProviderError("ChatGPT 图片分块传输失败：%s" % append_output[-1000:])

            commit_script = """
(() => {
  const uploads = window.__rlAgentFileUploads || {};
  const upload = uploads[%s];
  const inputs = Array.from(document.querySelectorAll(%s));
  const input = inputs[%d];
  if (!upload) return JSON.stringify({ok: false, reason: 'upload_session_missing'});
  if (!(input instanceof HTMLInputElement)) {
    delete uploads[%s];
    return JSON.stringify({ok: false, reason: 'file_input_not_found', input_count: inputs.length});
  }
  const transfer = new DataTransfer();
  for (const item of upload.files) {
    const binary = atob(item.chunks.join(''));
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
    transfer.items.add(new File([bytes], item.name, {type: item.type}));
  }
  %s
  const result = attachFiles(input, transfer);
  delete uploads[%s];
  return JSON.stringify(result);
})()
""" % (token_json, json.dumps(selector), nth, token_json, CHATGPT_ATTACH_FILES_SCRIPT, token_json)
            commit_output = self._browser(["eval", commit_script])
            commit_result = self._parse_eval_result(commit_output)
            expected_names = [path.name for path in files]
            if not (isinstance(commit_result, dict) and commit_result.get("ok") and
                    int(commit_result.get("count", 0)) == len(files) and
                    commit_result.get("names") == expected_names):
                raise ProviderError("ChatGPT 页面未确认 DataTransfer 图片附件：%s" % commit_output[-2000:])
        except Exception:
            cleanup = """
(() => {
  if (window.__rlAgentFileUploads) delete window.__rlAgentFileUploads[%s];
  return JSON.stringify({ok: true});
})()
""" % token_json
            self._browser(["eval", cleanup], allow_failure=True)
            raise

    def _upload_files_via_doubao(self, files: List[Path], conversation: Optional[ConversationHandle]) -> Optional[List[str]]:
        """当 ChatGPT 存储配额受限时的回退：通过 OpenCLI 打开豆包页面并上传，返回可分享的链接列表。

        此方法为 best-effort，使用配置的 `doubao_upload_input_selector` 和 `doubao_result_selector`。
        """
        if not self.settings.use_doubao_on_quota:
            return None
        # 以 opencli browser open 指向豆包上传页，然后尝试填充文件输入并读取分享链接
        try:
            self._browser(["open", self.settings.doubao_url])
        except Exception:
            # best-effort, do not raise here
            pass
        # find file input and perform upload via opencli upload
        try:
            found = self._browser(["find", "--css", self.settings.doubao_upload_input_selector, "--limit", "5"], allow_failure=True)
            found_data = _json_from_output(found) or {}
            entries = found_data.get("entries", []) if isinstance(found_data, dict) else []
            nth = str(entries[0].get("nth", 0)) if entries else "0"
            output = self._browser(["upload", "--nth", nth, self.settings.doubao_upload_input_selector] + [str(p) for p in files], allow_failure=True)
            parsed = _json_from_output(output)
            # wait for result selector to appear and extract link(s)
            deadline = time.monotonic() + int(self.settings.doubao_max_wait)
            links = []
            while time.monotonic() < deadline:
                try:
                    res = self._browser(["eval", "JSON.stringify(Array.from(document.querySelectorAll('%s')).map(e=>e.href||e.value||e.innerText))" % self.settings.doubao_result_selector], allow_failure=True)
                    parsed_links = _json_from_output(res)
                    if isinstance(parsed_links, list) and parsed_links:
                        links = [str(item) for item in parsed_links if item]
                        break
                except Exception:
                    pass
                time.sleep(1.0)
            return links if links else None
        except Exception:
            return None

    def _wait_for_upload_preview(self, file_names: List[str], allow_media: bool = True) -> None:
        """等待 ChatGPT 编辑器显示全部附件名称或对应数量的媒体预览。"""
        names_json = json.dumps(file_names, ensure_ascii=False)
        deadline = time.monotonic() + self.settings.submit_timeout
        while time.monotonic() < deadline:
            script = """
(() => {
  const names = %s;
  const composer = document.querySelector('#prompt-textarea, [data-testid="prompt-textarea"], [contenteditable="true"][role="textbox"]');
  if (!composer) return JSON.stringify(false);
  let root = composer;
  for (let index = 0; index < 6 && root.parentElement &&
       root.parentElement !== document.body; index += 1) root = root.parentElement;
  const scope = composer.closest('form') || root;
  if (!scope) return JSON.stringify(false);
  const text = scope.innerText || '';
  if (names.every(name => text.includes(name))) return JSON.stringify(true);
  if (!%s) return JSON.stringify(false);
  const visible = node => {
    if (!(node instanceof HTMLElement)) return false;
    const style = window.getComputedStyle(node);
    if (style.display === 'none' || style.visibility === 'hidden') return false;
    const rect = node.getBoundingClientRect();
    const width = node.naturalWidth || node.videoWidth || rect.width || 0;
    const height = node.naturalHeight || node.videoHeight || rect.height || 0;
    return (width > 32 && height > 32) ||
      (/url\\(/.test(style.backgroundImage || '') && rect.width > 32 && rect.height > 32);
  };
  const media = Array.from(scope.querySelectorAll('img[src], canvas, video, [style*="background-image"]')).filter(visible);
  return JSON.stringify(media.length >= names.length);
})()
""" % (names_json, "true" if allow_media else "false")
            output = self._browser(["eval", script], allow_failure=True)
            result = self._parse_eval_result(output)
            if result is True:
                return
            time.sleep(0.5)
        raise ProviderTimeout("ChatGPT 在 %s 秒内没有显示全部附件预览：%s" %
                              (self.settings.submit_timeout, ", ".join(file_names)))

    def send_with_files(self, prompt: str, files: List[Path], conversation: ConversationHandle) -> str:
        """上传本地文件、发送提示词并返回最新助手回复。"""
        if not files:
            return self.send_text(prompt, conversation)
        resolved = [path.resolve() for path in files]
        missing = [str(path) for path in resolved if not path.is_file()]
        if missing:
            raise FileNotFoundError("files for ChatGPT upload do not exist: %s" % missing)
        self._state()
        # 额度耗尽时不再执行耗时且注定无效的附件上传。
        self._check_page_usage_limit()
        text_suffixes = (".md", ".txt", ".json", ".yaml", ".yml")
        text_files = [path for path in resolved if path.suffix.lower() in text_suffixes]
        binary_files = [path for path in resolved if path.suffix.lower() not in text_suffixes]
        if text_files:
            self._upload_text_documents(text_files)
        if not binary_files:
            return self.send_text(prompt, conversation)
        found = self._browser(["find", "--css", "input[type=file]", "--limit", "10"])
        found_data = _json_from_output(found)
        entries = found_data.get("entries", []) if isinstance(found_data, dict) else []
        image_only = all(path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".gif")
                         for path in binary_files)
        if image_only:
            # ChatGPT 当前的 upload-photos 输入框会在 React onChange 后清空 DataTransfer；
            # 无 accept 的通用 upload-files 输入框既支持图片，也能稳定保留附件。
            preferred = [entry for entry in entries
                         if "image/" not in str(entry.get("compound", {}).get("accept", "")).lower()]
        else:
            # ChatGPT 的通用文件输入框通常不可见且没有 accept；可见输入框反而只接受图片。
            preferred = [entry for entry in entries
                         if "image/" not in str(entry.get("compound", {}).get("accept", "")).lower()]
        chosen = (preferred or entries or [{"nth": 0}])[0]
        nth = str(chosen.get("nth", 0))
        try:
            output = self._browser(["upload", "--nth", nth, "input[type=file]"] +
                                   [str(path) for path in binary_files])
            parsed = _json_from_output(output)
            if not isinstance(parsed, dict) or not parsed.get("uploaded"):
                raise ProviderError("OpenCLI 未确认视觉文件上传：%s" % output[-1000:])
        except ProviderError as exc:
            msg = str(exc).lower()
            if "storage quota exceeded" in msg:
                # 尝试豆包回退上传（仅在配置允许时）
                links = None
                try:
                    links = self._upload_files_via_doubao(binary_files, conversation)
                except Exception:
                    links = None
                if links:
                    # 将分享链接作为短文本替代附件发送
                    links_text = "\n".join(links)
                    prompt = prompt + "\n\n[附件已回退至外部分享链接，按需下载：]\n" + links_text
                    return self.send_text(prompt, conversation, submission_timeout=max(self.settings.submit_timeout, 120))
                # 若回退失败则继续按旧逻辑判断是否可恢复
            if not self._recoverable_file_upload_error(exc):
                raise
            self._upload_binary_files_via_data_transfer(binary_files, int(nth))
        self._wait_for_upload_preview([path.name for path in binary_files])
        return self.send_text(
            prompt, conversation, submission_timeout=max(self.settings.submit_timeout, 120))

    @staticmethod
    def _parse_candidate(raw_response: str) -> Any:
        """依次从代码块、原文和平衡对象中解析 JSON。"""
        fences = re.findall(r"```(?:json)?\s*(.*?)```", raw_response, flags=re.IGNORECASE | re.DOTALL)
        for candidate in fences + [raw_response]:
            stripped = candidate.strip()
            for syntax_candidate in (stripped, _repair_common_json_syntax(stripped)):
                try:
                    return json.loads(syntax_candidate)
                except json.JSONDecodeError:
                    continue
            balanced = _balanced_json_object(_repair_common_json_syntax(candidate))
            if balanced:
                try:
                    return json.loads(balanced)
                except json.JSONDecodeError:
                    continue
        raise ValueError("no complete JSON object found")

    def parse_json_response(self, raw_response: str, schema: Type[ModelT]) -> ModelT:
        """提取并校验网页回复中的结构化 JSON。"""
        try:
            return schema.parse_obj(self._parse_candidate(raw_response))
        except (ValueError, ValidationError) as exc:
            raise ProviderResponseError(str(exc)) from exc

    def _request_model(self, prompt: str, schema: Type[ModelT], title: str,
                       files: Optional[List[Path]] = None) -> ModelT:
        """发送结构化请求并在校验失败时进行有限修复。"""
        conversation: Optional[ConversationHandle] = None
        raw = ""
        for request_attempt in range(self.settings.max_retries + 1):
            conversation = self.new_conversation(
                title if request_attempt == 0 else "%s-silent-retry-%02d" % (
                    title, request_attempt))
            try:
                raw = self._send_model_prompt(conversation, prompt, title, files)
                break
            except ProviderTimeout:
                snapshot = self._submission_snapshot()
                if (request_attempt >= self.settings.max_retries or
                        not self._can_retry_silent_response(snapshot)):
                    raise
                time.sleep(1.0)
        if conversation is None:
            raise ProviderResponseError("ChatGPT 会话未创建")
        for attempt in range(self.settings.max_retries + 1):
            try:
                return self.parse_json_response(raw, schema)
            except ProviderResponseError as exc:
                # 页面 DOM 可能在 OpenCLI 返回后又补齐了流式回复。发送修复消息前
                # 再读取一次；若已成为合法结果，直接使用，避免重复消息与额度消耗。
                latest = ""
                try:
                    self._parse_candidate(raw)
                except ValueError:
                    latest = self._latest_assistant()
                if latest and len(latest) > len(raw):
                    try:
                        return self.parse_json_response(latest, schema)
                    except ProviderResponseError:
                        # 页面可能仍停留在上一项任务的较长回复。它不满足本次 Schema
                        # 就不能覆盖本次原始响应，否则恢复会把旧会话内容发给新任务。
                        pass
                if attempt >= self.settings.max_retries:
                    raise
                repair = ("Return only corrected strict JSON matching this JSON Schema. Validation error: %s\nSchema: %s\n"
                          "Do not add commentary." % (exc, schema.schema_json()))
                repair_title = "%s-repair-%02d" % (title, attempt + 1)
                try:
                    raw = self._send_model_prompt(conversation, repair, repair_title)
                except ProviderNeedsHuman:
                    raise
                except (ProviderError, ProviderTimeout):
                    # 页面偶发会在点击修复请求后仍显示上一条用户消息。换新会话携带原回复，
                    # 可避免重复点击同一会话，同时保留严格 Schema 修复能力。
                    conversation = self.new_conversation(repair_title + "-recovery")
                    recovery = repair + "\nInvalid JSON response to repair:\n" + raw
                    raw = self._send_model_prompt(conversation, recovery, repair_title + "-recovery")
        raise ProviderResponseError("unreachable response validation state")

    def design_task_and_rewards(self, instruction: str, robot: str, capabilities: Dict[str, Any]) -> Dict[str, Any]:
        """依据任务描述和环境能力生成任务规格与奖励候选。"""
        template = (Path(__file__).parents[1] / "prompts" / "task_reward_design.md").read_text(encoding="utf-8")
        compact_capabilities = _compact_capabilities(capabilities)
        prompt = (template + "\nRobot: %s\nInstruction: %s\nENVIRONMENT_MANIFEST: %s\nJSON Schema: %s" %
                  (robot, instruction, json.dumps(compact_capabilities, ensure_ascii=False, separators=(",", ":")),
                   TaskRewardBundle.schema_json()))
        return self._request_model(prompt, TaskRewardBundle, "task-reward-design").dict()

    def understand_task(self, instruction: str, robot: str) -> TaskIntentSpec:
        """把上位机动作输入转换为结构化任务意图，不在此阶段设计奖励。"""
        template = (Path(__file__).parents[1] / "prompts" / "task_intent.md").read_text(encoding="utf-8")
        prompt = (template + "\nRobot: %s\nInstruction: %s\nJSON Schema: %s" %
                  (robot, instruction, TaskIntentSpec.schema_json()))
        return self._request_model(prompt, TaskIntentSpec, "task-intent")

    def generate_motion_prototype(self, intent: TaskIntentSpec) -> MotionPrototype:
        """生成高层动作阶段语义，不允许输出关节角或控制策略。"""
        prompt = (MOTION_PROTOTYPE_PROMPT + "\nTASK_INTENT_SPEC: " + intent.json(ensure_ascii=False) +
                  "\nJSON Schema: " + MotionPrototype.schema_json())
        return self._request_model(prompt, MotionPrototype, "motion-prototype")

    def design_task_bundle(self, compiled_prompt: str) -> Dict[str, Any]:
        """执行已经由本地固定模板编译的奖励设计请求。"""
        return self._request_model(
            compiled_prompt, TaskRewardBundle, "compiled-reward-design").dict()

    def design_visual_evaluation(self, task: TaskSpec) -> Dict[str, Any]:
        """为任务生成视觉评估输入与事件设计。"""
        template = (Path(__file__).parents[1] / "prompts" / "visual_spec_design.md").read_text(encoding="utf-8")
        conversation = self.new_conversation("visual-evaluation-design")
        raw = self.send_text(template + "\nTaskSpec: " + task.json(), conversation)
        value = self._parse_candidate(raw)
        if not isinstance(value, dict):
            raise ProviderResponseError("visual design must be a JSON object")
        return value

    def critique_visual_behavior(self, task: TaskSpec, files: List[Path]) -> VisualBehaviorReport:
        """基于视觉材料生成不受奖励数值锚定的行为评论。"""
        template = (Path(__file__).parents[1] / "prompts" / "visual_critic.md").read_text(encoding="utf-8")
        selected_files = self._select_visual_evidence_files(files)
        prompt = (template + "\nTaskSpec: " + task.json() +
                  "\n已按可靠上传上限选择关键证据附件：" +
                  ", ".join(path.name for path in selected_files) +
                  "\nJSON Schema: " + VisualBehaviorReport.schema_json())
        return self._request_model(
            prompt, VisualBehaviorReport, "visual-critique", selected_files)

    def _select_visual_evidence_files(self, files: List[Path]) -> List[Path]:
        """保留关键连续帧、三视角图和物理证据，避免冗余图片导致网页上传超时。"""
        available = {path.name: path for path in files}
        image_priority = (
            "contact_sheet_annotated.png", "contact_sheet_multiview.png",
            "contact_sheet_clean.png", "event_takeoff.png", "event_landing.png",
        )
        document_priority = ("behavior_evidence.json", "visual_attachment_manifest.json")
        images = [available[name] for name in image_priority if name in available]
        documents = [available[name] for name in document_priority if name in available]
        selected = images[:self.settings.visual_image_attachment_limit] + documents
        return selected or list(files)

    def diagnose_training(self, payload: Dict[str, Any]) -> TrainingDiagnosis:
        """融合视觉、物理和 PPO 证据生成训练诊断。"""
        template = (Path(__file__).parents[1] / "prompts" / "training_diagnosis.md").read_text(encoding="utf-8")
        prompt = (template + "\nEvidence: " +
                  json.dumps(json_safe(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False) +
                  "\nJSON Schema: " + TrainingDiagnosis.schema_json())
        return self._request_model(prompt, TrainingDiagnosis, "training-diagnosis")

    def summarize_reward_experience(self, payload: Dict[str, Any]) -> RewardExperienceNarrative:
        """使用普通聊天模式归纳受证据 ID 限定的奖励实验叙述。"""
        prompt = (REWARD_EXPERIENCE_PROMPT + "\n\nEvidence: " +
                  json.dumps(json_safe(payload), ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False) +
                  "\nJSON Schema: " + RewardExperienceNarrative.schema_json())
        return self._request_model(prompt, RewardExperienceNarrative, "reward-experience")

    def close(self) -> None:
        """释放 Provider 持有或绑定的浏览器资源。"""
        if not self._opened:
            return
        if self.settings.owned_session:
            self._browser(["close"], allow_failure=True)
        else:
            self._browser(["unbind"], allow_failure=True)
        self._opened = False
