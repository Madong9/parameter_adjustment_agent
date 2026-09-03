from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError

from ..schemas.decisions import TrainingDiagnosis
from ..schemas.experiments import ConversationHandle, ProviderHealth
from ..schemas.rewards import RewardPlan
from ..schemas.task import TaskSpec
from ..schemas.visual import VisualBehaviorReport
from ..settings import OpenCLISettings, load_opencli_settings
from ..utils.io import atomic_write_text, json_safe
from .errors import ProviderError, ProviderNeedsHuman, ProviderResponseError, ProviderTimeout

ModelT = TypeVar("ModelT", bound=BaseModel)
Runner = Callable[[Sequence[str], int], subprocess.CompletedProcess]


class TaskRewardBundle(BaseModel):
    task_spec: TaskSpec
    reward_plans: List[RewardPlan]
    reward_hacking_risks: List[str] = Field(default_factory=list)
    termination_suggestions: List[str] = Field(default_factory=list)


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
    variable_fields = ("name", "shape", "unit", "coordinate_frame", "normalized",
                       "available_to_policy", "available_to_reward", "simulation_only", "derivation")
    reward_fields = ("name", "implementation", "config_key", "parameters", "expected_raw_range",
                     "default_weight", "sign", "dependencies", "supported_phases")

    def select_fields(item: Any, fields: Sequence[str]) -> Dict[str, Any]:
        """从单条能力记录中选择模型决策所需字段。"""
        if not isinstance(item, dict):
            return {}
        return {field: item[field] for field in fields if field in item}

    return {
        "project": capabilities.get("project"),
        "robot": capabilities.get("robot"),
        "robots": capabilities.get("robots", []),
        "reward_variables": [select_fields(item, variable_fields)
                             for item in capabilities.get("reward_variables", [])],
        "rewards": [select_fields(item, reward_fields) for item in capabilities.get("rewards", [])],
        "terminations": capabilities.get("terminations", []),
        "command_space": capabilities.get("command_space", []),
        "evaluation_metrics": capabilities.get("evaluation_metrics", []),
        "unsupported": capabilities.get("unsupported", []),
    }


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

    def _run(self, args: Sequence[str], timeout: Optional[int] = None, allow_failure: bool = False) -> str:
        """执行受限子流程并返回结构化结果。"""
        command = ["opencli"] + list(args)
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
        lowered = state.lower()
        if any(item in lowered for item in ("captcha", "verify you are human", "cloudflare")):
            raise ProviderNeedsHuman("ChatGPT page requires CAPTCHA verification")
        if any(item in lowered for item in ("log in", "sign up", "登录", "注册")) and "message chatgpt" not in lowered:
            raise ProviderNeedsHuman("ChatGPT login has expired")
        return state

    @staticmethod
    def _bridge_connected(doctor_output: str) -> bool:
        """判断 OpenCLI doctor 输出是否确认浏览器扩展已经连接。"""
        normalized = re.sub(r"\s+", " ", doctor_output).strip().lower()
        return bool(re.search(r"\[ok\]\s+extension:\s*connected", normalized))

    def _ensure_bridge_connected(self) -> None:
        """检查浏览器桥接，断线时重启守护进程并等待扩展自动重连。"""
        doctor = self._run(["doctor"], timeout=self.settings.connect_timeout, allow_failure=True)
        if self._bridge_connected(doctor):
            return

        self._run(["daemon", "restart"], timeout=self.settings.connect_timeout, allow_failure=True)
        deadline = time.monotonic() + self.settings.connect_timeout
        while time.monotonic() < deadline:
            time.sleep(1.0)
            doctor = self._run(["doctor"], timeout=self.settings.connect_timeout, allow_failure=True)
            if self._bridge_connected(doctor):
                return
        raise ProviderNeedsHuman(
            "OpenCLI 浏览器扩展未连接。程序已自动重启守护进程，但扩展仍未重连；"
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
        doctor = self._run(["doctor"], allow_failure=True)
        extension = "extension: connected" in doctor.lower()
        details.append(doctor.strip()[-1000:])
        status_text = self._run(["chatgpt", "status", "-f", "json"], allow_failure=True)
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
            self._browser(["open", self.settings.chatgpt_url])
            self.settings.owned_session = True
        self._state()
        self._opened = True

    def new_conversation(self, title_hint: str) -> ConversationHandle:
        """创建新会话并返回可持久化的会话句柄。"""
        if not self._opened:
            self.open_or_bind()
        if self.settings.owned_session:
            self._browser(["open", self.settings.chatgpt_url])
        else:
            # 在已绑定标签页打开根地址，以创建新的网页会话。
            self._browser(["open", self.settings.chatgpt_url])
        self._state()
        return ConversationHandle(conversation_id=uuid.uuid4().hex, title_hint=title_hint,
                                  owned=self.settings.owned_session)

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
        """提取页面中最新一条助手消息。"""
        script = "JSON.stringify(Array.from(document.querySelectorAll('[data-message-author-role=assistant]')).map(e=>e.innerText).filter(Boolean).slice(-1)[0]||'')"
        output = self._browser(["eval", script], allow_failure=True)
        parsed = _json_from_output(output)
        value = _extract_value(parsed)
        if value:
            nested = _json_from_output(value)
            return nested if isinstance(nested, str) else value
        return ""

    def _latest_user(self) -> str:
        """提取页面中最新一条已经提交的用户消息。"""
        script = "JSON.stringify(Array.from(document.querySelectorAll('[data-message-author-role=user]')).map(e=>e.innerText).filter(Boolean).slice(-1)[0]||'')"
        output = self._browser(["eval", script], allow_failure=True)
        parsed = _json_from_output(output)
        value = _extract_value(parsed)
        if value:
            nested = _json_from_output(value)
            return nested if isinstance(nested, str) else value
        return ""

    @staticmethod
    def _matches_submitted_prompt(actual: str, prompt: str) -> bool:
        """比较已渲染用户消息与原文，并识别 ChatGPT 添加的附件元数据前缀。"""
        normalize = lambda value: re.sub(r"\s+", " ", value.replace("`", "")).strip()
        actual_normalized = normalize(actual)
        prompt_normalized = normalize(prompt)
        if not actual_normalized or not prompt_normalized:
            return False
        if actual_normalized == prompt_normalized:
            return True
        if actual_normalized.endswith(prompt_normalized):
            attachment_prefix = actual_normalized[:-len(prompt_normalized)].strip()
            safe_attachment = r'[^<>:"/\\|?*\x00-\x1f]{1,240}\.(?:md|txt|json|ya?ml|png|jpe?g|webp|gif)\s+文件'
            if re.fullmatch(r"(?:%s\s*)+" % safe_attachment, attachment_prefix,
                            flags=re.IGNORECASE):
                return True
        return (
            len(prompt_normalized) > 200 and
            actual_normalized.startswith(prompt_normalized[:120]) and
            actual_normalized.endswith(prompt_normalized[-120:])
        )

    def _click_send_button(self) -> None:
        """等待 ChatGPT 发送按钮可用并显式点击，避免依赖输入框焦点。"""
        selectors = [
            'button[data-testid="send-button"]:not([disabled])',
            '#composer-submit-button:not([disabled])',
            'button[aria-label="Send prompt"]:not([disabled])',
            'button[aria-label="发送提示"]:not([disabled])',
            'button[aria-label="发送"]:not([disabled])',
        ]
        deadline = time.monotonic() + self.settings.submit_timeout
        last_output = ""
        while time.monotonic() < deadline:
            self._state()
            for selector in selectors:
                found = self._browser(["find", "--css", selector, "--limit", "2"], allow_failure=True)
                found_data = _json_from_output(found)
                if not (isinstance(found_data, dict) and int(found_data.get("matches_n", 0)) > 0):
                    continue
                clicked = self._browser(["click", selector], allow_failure=True)
                last_output = clicked
                clicked_data = _json_from_output(clicked)
                if isinstance(clicked_data, dict) and clicked_data.get("clicked"):
                    return
            time.sleep(0.25)
        detail = last_output[-1000:] if last_output else "没有发现已启用的发送按钮"
        raise ProviderTimeout("ChatGPT 发送按钮在 %s 秒内未变为可点击状态：%s" %
                              (self.settings.submit_timeout, detail))

    def _wait_for_submission(self, previous_user: str, prompt: str) -> None:
        """确认新用户消息已进入会话，防止把仅填入输入框误判为发送成功。"""
        deadline = time.monotonic() + self.settings.submit_timeout
        latest = ""
        while time.monotonic() < deadline:
            self._state()
            latest = self._latest_user()
            if latest != previous_user and self._matches_submitted_prompt(latest, prompt):
                return
            time.sleep(0.25)
        # 边界时刻再读取一次；若回复已经很快生成，后续等待逻辑会直接接续该回复。
        self._state()
        latest = self._latest_user()
        if latest != previous_user and self._matches_submitted_prompt(latest, prompt):
            return
        raise ProviderTimeout("已点击发送按钮，但未在会话中确认新的用户消息；最新用户消息长度=%s" % len(latest))

    def _wait_for_response(self, previous: str) -> str:
        """轮询页面直到最新助手回复稳定或超时。"""
        deadline = time.monotonic() + self.settings.response_timeout
        last = ""
        stable_count = 0
        while time.monotonic() < deadline:
            self._state()
            current = self._latest_assistant()
            if current and current != previous:
                if current == last:
                    stable_count += 1
                else:
                    stable_count = 0
                    last = current
                if stable_count >= 2 and not self._is_generating():
                    return current
            time.sleep(1.0)
        raise ProviderTimeout("ChatGPT response did not complete before timeout")

    def _is_generating(self) -> bool:
        """检测 ChatGPT 是否仍在思考、调用视觉工具或流式生成回复。"""
        script = """
(() => {
  if (document.querySelector('button[data-testid="stop-button"]')) return JSON.stringify(true);
  return JSON.stringify(Array.from(document.querySelectorAll('button')).some(button => {
    const label = button.getAttribute('aria-label') || '';
    return /Stop generating|Stop responding|停止生成|停止回答|Thinking|正在思考/.test(label);
  }));
})()
"""
        output = self._browser(["eval", script], allow_failure=True)
        return self._parse_eval_result(output) is True

    def send_text(self, prompt: str, conversation: ConversationHandle) -> str:
        """向指定网页会话发送文本并返回最新助手回复。"""
        previous_assistant = self._latest_assistant()
        previous_user = self._latest_user()
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
        # 点击动作一旦成功便不再重复提交，避免页面状态读取异常时发送重复消息。
        self._wait_for_submission(previous_user, prompt)
        response = self._wait_for_response(previous_assistant)
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
  input.files = transfer.files;
  input.dispatchEvent(new Event('change', {bubbles: true}));
  return JSON.stringify({ok: true, count: input.files.length,
    names: Array.from(input.files).map(file => file.name)});
})()
""" % payload
        output = self._browser(["eval", script])
        parsed = _json_from_output(output)
        if isinstance(parsed, str):
            parsed = _json_from_output(parsed)
        expected_names = [path.name for path in files]
        if not (isinstance(parsed, dict) and parsed.get("ok") and
                int(parsed.get("count", 0)) == len(files) and parsed.get("names") == expected_names):
            raise ProviderError("ChatGPT 页面未确认需求文档附件：%s" % output[-2000:])

    @staticmethod
    def _recoverable_file_upload_error(error: Exception) -> bool:
        """判断本地路径上传是否被扩展能力或 Chrome 安全策略拒绝。"""
        message = str(error).lower()
        return any(fragment in message for fragment in (
            "not allowed", "unknown action", "not supported", "setfileinput",
            "set-file-input", "no element found",
        ))

    @staticmethod
    def _parse_eval_result(output: str) -> Any:
        """解析 browser eval 可能返回的一层或两层 JSON 信封。"""
        parsed = _json_from_output(output)
        if isinstance(parsed, str):
            nested = _json_from_output(parsed)
            return nested if nested is not None else parsed
        return parsed

    def _upload_binary_files_via_data_transfer(self, files: List[Path], nth: int) -> None:
        """分块传输二进制文件并用 DataTransfer 附加，规避 Chrome 的 Not allowed。"""
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
  const inputs = Array.from(document.querySelectorAll('input[type=file]'));
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
  input.files = transfer.files;
  const propsKey = Object.keys(input).find(key => key.startsWith('__reactProps$'));
  if (propsKey && input[propsKey] && typeof input[propsKey].onChange === 'function') {
    const nativeEvent = new Event('change', {bubbles: true});
    input[propsKey].onChange({
      target: input, currentTarget: input, nativeEvent,
      preventDefault() {}, stopPropagation() {},
      isDefaultPrevented() { return false; },
      isPropagationStopped() { return false; }, persist() {}
    });
  } else {
    input.dispatchEvent(new Event('input', {bubbles: true}));
    input.dispatchEvent(new Event('change', {bubbles: true}));
  }
  const names = Array.from(input.files).map(file => file.name);
  delete uploads[%s];
  return JSON.stringify({ok: true, count: names.length, names});
})()
""" % (token_json, nth, token_json, token_json)
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

    def _wait_for_upload_preview(self, file_names: List[str]) -> None:
        """等待 ChatGPT 编辑器显示全部附件名称或对应数量的媒体预览。"""
        names_json = json.dumps(file_names, ensure_ascii=False)
        deadline = time.monotonic() + self.settings.submit_timeout
        while time.monotonic() < deadline:
            script = """
(() => {
  const names = %s;
  const text = document.body ? (document.body.innerText || '') : '';
  if (names.filter(name => text.includes(name)).length >= names.length) return JSON.stringify(true);
  const composer = document.querySelector('#prompt-textarea, [data-testid="prompt-textarea"], [contenteditable="true"][role="textbox"]');
  let root = composer;
  for (let index = 0; index < 6 && root && root.parentElement; index += 1) root = root.parentElement;
  const scope = root || document.body;
  if (!scope) return JSON.stringify(false);
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
""" % names_json
            output = self._browser(["eval", script], allow_failure=True)
            result = self._parse_eval_result(output)
            if result is True:
                return
            time.sleep(0.5)
        raise ProviderTimeout("ChatGPT 在 %s 秒内没有显示全部视觉评估附件预览：%s" %
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
            if not self._recoverable_file_upload_error(exc):
                raise
            self._upload_binary_files_via_data_transfer(binary_files, int(nth))
        self._wait_for_upload_preview([path.name for path in binary_files])
        return self.send_text(prompt, conversation)

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
        conversation = self.new_conversation(title)
        raw = self._send_model_prompt(conversation, prompt, title, files)
        for attempt in range(self.settings.max_retries + 1):
            try:
                return self.parse_json_response(raw, schema)
            except ProviderResponseError as exc:
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
        prompt = template + "\nTaskSpec: " + task.json() + "\nJSON Schema: " + VisualBehaviorReport.schema_json()
        return self._request_model(prompt, VisualBehaviorReport, "visual-critique", files)

    def diagnose_training(self, payload: Dict[str, Any]) -> TrainingDiagnosis:
        """融合视觉、物理和 PPO 证据生成训练诊断。"""
        template = (Path(__file__).parents[1] / "prompts" / "training_diagnosis.md").read_text(encoding="utf-8")
        prompt = (template + "\nEvidence: " +
                  json.dumps(json_safe(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False) +
                  "\nJSON Schema: " + TrainingDiagnosis.schema_json())
        return self._request_model(prompt, TrainingDiagnosis, "training-diagnosis")

    def close(self) -> None:
        """释放 Provider 持有或绑定的浏览器资源。"""
        if not self._opened:
            return
        if self.settings.owned_session:
            self._browser(["close"], allow_failure=True)
        else:
            self._browser(["unbind"], allow_failure=True)
        self._opened = False
