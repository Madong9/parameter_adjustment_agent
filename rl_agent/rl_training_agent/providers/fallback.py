"""为外部推理 Provider 提供显式、可审计的主备切换。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from ..schemas.experiments import ProviderHealth
from .errors import ProviderError


class FallbackProvider:
    """主 Provider 抛出可识别故障时，以相同高层请求调用备用 Provider。"""

    def __init__(self, primary: Any, fallback: Any, primary_name: str = "ChatGPT",
                 fallback_name: str = "豆包", record_dir: Optional[Path] = None):
        """保存主备 Provider 和用于上位机日志的可读名称。"""
        self.primary = primary
        self.fallback = fallback
        self.primary_name = primary_name
        self.fallback_name = fallback_name
        self.record_dir = record_dir
        self._active = primary

    def _record_fallback(self, operation: str, error: Exception) -> None:
        """输出切换日志，并将不含提示词和密钥的原因追加到审计文件。"""
        message = "[Provider] %s 的 %s 不可用，自动切换到%s：%s" % (
            self.primary_name, operation, self.fallback_name, str(error).strip()[-1000:])
        print(message, flush=True)
        if self.record_dir is not None:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            with (self.record_dir / "fallback.log").open("a", encoding="utf-8") as handle:
                handle.write(message.replace("\n", " ") + "\n")

    def _call(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        """调用主 Provider；只有 Provider 故障才切换，业务校验异常保持原样。"""
        if self._active is self.fallback:
            return getattr(self.fallback, operation)(*args, **kwargs)
        try:
            return getattr(self.primary, operation)(*args, **kwargs)
        except ProviderError as primary_error:
            self._record_fallback(operation, primary_error)
            self._active = self.fallback
            try:
                return getattr(self.fallback, operation)(*args, **kwargs)
            except ProviderError as fallback_error:
                raise ProviderError(
                    "%s 不可用：%s；%s 回退也失败：%s" % (
                        self.primary_name, primary_error, self.fallback_name, fallback_error)
                ) from fallback_error

    def doctor(self) -> ProviderHealth:
        """主服务健康即就绪；否则检查备用豆包并报告可恢复状态。"""
        primary = self.primary.doctor()
        if primary.available:
            return primary
        fallback = self.fallback.doctor()
        details = list(primary.details) + list(fallback.details)
        return ProviderHealth(
            available=fallback.available,
            opencli_available=primary.opencli_available or fallback.opencli_available,
            extension_connected=primary.extension_connected or fallback.extension_connected,
            chatgpt_logged_in=primary.chatgpt_logged_in,
            image_upload_supported=(primary.image_upload_supported or
                                    fallback.image_upload_supported),
            recoverable=primary.recoverable or fallback.recoverable,
            details=details,
        )

    def open_or_bind(self) -> None:
        """打开当前主 Provider，失败时自动改为豆包。"""
        self._call("open_or_bind")

    def new_conversation(self, title_hint: str) -> Any:
        """在当前可用 Provider 中创建新会话。"""
        return self._call("new_conversation", title_hint)

    def send_text(self, prompt: str, conversation: Any) -> str:
        """发送纯文本；低层兼容调用失败时切换到豆包新会话。"""
        if self._active is self.primary:
            try:
                return self.primary.send_text(prompt, conversation)
            except ProviderError as error:
                self._record_fallback("send_text", error)
                self._active = self.fallback
                conversation = self.fallback.new_conversation("fallback-text")
        return self.fallback.send_text(prompt, conversation)

    def send_with_files(self, prompt: str, files: List[Path], conversation: Any) -> str:
        """发送附件请求；主服务失败时在豆包新会话重新上传相同附件。"""
        if self._active is self.primary:
            try:
                return self.primary.send_with_files(prompt, files, conversation)
            except ProviderError as error:
                self._record_fallback("send_with_files", error)
                self._active = self.fallback
                conversation = self.fallback.new_conversation("fallback-files")
        return self.fallback.send_with_files(prompt, files, conversation)

    def parse_json_response(self, raw_response: str, schema: Any) -> Any:
        """使用当前 Provider 的兼容严格 JSON 解析器。"""
        return self._active.parse_json_response(raw_response, schema)

    def understand_task(self, instruction: str, robot: str) -> Any:
        """执行任务理解，ChatGPT 故障时整次请求交给豆包。"""
        return self._call("understand_task", instruction, robot)

    def generate_motion_prototype(self, intent: Any) -> Any:
        """生成动作阶段语义；主 Provider 故障时转交备用 Provider。"""
        return self._call("generate_motion_prototype", intent)

    def design_task_bundle(self, compiled_prompt: str) -> Any:
        """执行编译后的奖励设计兼容接口。"""
        return self._call("design_task_bundle", compiled_prompt)

    def design_task_and_rewards(self, instruction: str, robot: str, capabilities: Any) -> Any:
        """执行旧版任务和奖励联合设计兼容接口。"""
        return self._call("design_task_and_rewards", instruction, robot, capabilities)

    def design_visual_evaluation(self, task: Any) -> Any:
        """执行视觉规范设计，主服务失败时交给豆包。"""
        return self._call("design_visual_evaluation", task)

    def critique_visual_behavior(self, task: Any, files: List[Path]) -> Any:
        """执行完整多模态视觉评价，主服务失败时整次交给豆包。"""
        return self._call("critique_visual_behavior", task, files)

    def diagnose_training(self, payload: Any) -> Any:
        """执行训练诊断兼容接口。"""
        return self._call("diagnose_training", payload)

    def summarize_reward_experience(self, payload: Any) -> Any:
        """总结奖励经验；网页 Provider 故障时由主备机制切换。"""
        return self._call("summarize_reward_experience", payload)

    def close(self) -> None:
        """关闭主备 Provider 持有的全部浏览器会话。"""
        for provider in (self.primary, self.fallback):
            try:
                provider.close()
            except ProviderError:
                continue
