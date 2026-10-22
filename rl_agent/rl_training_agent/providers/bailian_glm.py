"""实现阿里云百炼 OpenAI 兼容接口的 GLM 结构化推理 Provider。"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from ..schemas.agent_workflow import TaskRewardBundle
from ..schemas.rewards import RewardPlan
from ..schemas.reward_repairs import RewardRepairBundle
from ..schemas.decisions import TrainingDiagnosis
from ..memory.reward_experience.prompts import REWARD_EXPERIENCE_PROMPT
from ..memory.reward_experience.schema import RewardExperienceNarrative
from ..settings import BailianSettings, load_bailian_settings
from ..utils.io import atomic_write_text, json_safe, write_json
from .errors import ProviderError, ProviderResponseError, ProviderTimeout
from .schema_repair import (apply_schema_field_patches, merge_schema_repairs, normalize_reward_response,
                            project_legacy_schema_reply, render_schema_patch_request, repair_preservation_errors)

ModelT = TypeVar("ModelT", bound=BaseModel)


def _balanced_json(text: str) -> str:
    """从可能含解释文字的回复中提取首个完整 JSON 对象。"""
    start = text.find("{")
    while start >= 0:
        depth = 0
        quoted = False
        escaped = False
        for index in range(start, len(text)):
            character = text[index]
            if quoted:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    quoted = False
            elif character == '"':
                quoted = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
        start = text.find("{", start + 1)
    raise ProviderResponseError("百炼回复中没有完整 JSON 对象")


class BailianGLMProvider:
    """通过标准库 HTTP 客户端调用百炼 GLM，并严格校验 Pydantic Schema。"""

    def __init__(self, settings: Optional[BailianSettings] = None,
                 record_dir: Optional[Path] = None):
        """初始化百炼模型、超时、重试和审计目录。"""
        self.settings = settings or load_bailian_settings()
        self.record_dir = record_dir
        self._request_index = 0

    @property
    def api_key(self) -> str:
        """从配置指定的环境变量读取 API Key，避免密钥落盘。"""
        return os.getenv(self.settings.api_key_env, "").strip()

    def health(self) -> Dict[str, Any]:
        """返回无需产生模型费用的本地百炼配置健康状态。"""
        return {
            "available": bool(self.api_key),
            "model": self.settings.model,
            "base_url": self.settings.base_url,
            "api_key_env": self.settings.api_key_env,
            "details": [] if self.api_key else [
                "未设置环境变量 %s" % self.settings.api_key_env],
        }

    def _record(self, title: str, request_body: Dict[str, Any], response_text: str) -> None:
        """保存不含鉴权头的请求与原始回复，便于审计模型行为。"""
        if self.record_dir is None:
            return
        self._request_index += 1
        prefix = "%03d_%s" % (self._request_index, title.replace("/", "-")[:60])
        write_json(self.record_dir / (prefix + "_request.json"), request_body)
        atomic_write_text(self.record_dir / (prefix + "_response.txt"), response_text + "\n")

    def _http_request(self, body: Dict[str, Any]) -> str:
        """向百炼 Chat Completions 端点发送一次鉴权请求。"""
        if not self.api_key:
            raise ProviderError(
                "百炼奖励设计不可用：请先设置环境变量 %s" % self.settings.api_key_env)
        endpoint = self.settings.base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[-4000:]
            raise ProviderError("百炼 API 返回 HTTP %d：%s" % (exc.code, detail)) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProviderTimeout("百炼 API 连接或响应超时：%s" % exc) from exc

    def _completion(self, prompt: str, title: str) -> str:
        """执行带有限重试的 GLM 文本生成并提取回复正文。"""
        body: Dict[str, Any] = {
            "model": self.settings.model,
            "messages": [{"role": "user", "content": prompt}],
            "enable_thinking": self.settings.enable_thinking,
        }
        if self.settings.response_format_json:
            body["response_format"] = {"type": "json_object"}
        last_error: Optional[Exception] = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                envelope_text = self._http_request(body)
                envelope = json.loads(envelope_text)
                content = envelope["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item)
                                      for item in content)
                text = str(content).strip()
                self._record(title, body, text)
                return text
            except (ProviderError, ProviderTimeout, KeyError, IndexError, TypeError,
                    json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self.settings.max_retries:
                    break
                time.sleep(min(2 ** attempt, 4))
        if isinstance(last_error, ProviderError):
            raise last_error
        raise ProviderResponseError("百炼响应信封无法解析：%s" % last_error)

    def _parse_schema_candidate(self, raw: str, schema: Type[ModelT], title: str) -> Any:
        """规范无损奖励表示，并将转换与原始回复一起保留供审计。"""
        data, adjustments = normalize_reward_response(json.loads(_balanced_json(raw)), schema)
        if self.record_dir is not None and adjustments:
            prefix = "%03d_%s" % (self._request_index, title.replace("/", "-")[:60])
            write_json(self.record_dir / (prefix + "_schema_normalization.json"), {
                "scope": "lossless_reward_response_types_v3", "adjustments": adjustments,
            })
        return data

    def request_model(self, prompt: str, schema: Type[ModelT], title: str) -> ModelT:
        """无损规范奖励回复；一次格式重试仅能修改校验失败的位置。"""
        raw = self._completion(prompt, title)
        original_data = None
        try:
            original_data = self._parse_schema_candidate(raw, schema, title)
            return schema.parse_obj(original_data)
        except (ValueError, ValidationError, json.JSONDecodeError, ProviderResponseError) as exc:
            validation_errors = exc.errors() if isinstance(exc, ValidationError) else [
                {"loc": (), "type": "invalid_json"}]
            repair = (
                prompt + "\n\n修复要求：保持上述原始目标、环境能力和附件约束不变。\n"
                "只修复 validation_errors 指定的位置；保留其他已有字段、全部候选、课程、奖励 purpose、任务身份和阈值。不得输出简版。\n"
                "VALIDATION_ERRORS: " + json.dumps(validation_errors, ensure_ascii=False, default=str) + "\n"
                "修复下面的 JSON，使其严格符合 Schema。只返回 JSON。\n"
                "Schema: %s\n校验错误: %s\n原回复: %s" %
                (schema.schema_json(), exc, raw)
            )
            patch_request = None
            if schema in (TaskRewardBundle, RewardPlan) and isinstance(original_data, dict):
                repair, patch_request = render_schema_patch_request(prompt, original_data, schema, validation_errors)
            repair_title = title + "-repair"
            repaired = self._completion(repair, repair_title)
            try:
                if patch_request is not None:
                    patch_data = json.loads(_balanced_json(repaired))
                    if isinstance(patch_data, dict) and ('patches' in patch_data or 'base_sha256' in patch_data):
                        merged, patch_audit = apply_schema_field_patches(original_data, patch_data, schema, validation_errors)
                        if self.record_dir is not None:
                            prefix = '%03d_%s' % (self._request_index, repair_title.replace('/', '-')[:60])
                            write_json(self.record_dir / (prefix + '_schema_error_patches.json'), patch_audit)
                        return schema.parse_obj(merged)
                    repaired_data = self._parse_schema_candidate(repaired, schema, repair_title)
                    merged, legacy_audit = project_legacy_schema_reply(original_data, repaired_data, schema, validation_errors)
                    if self.record_dir is not None and legacy_audit['adjustments']:
                        prefix = '%03d_%s' % (self._request_index, repair_title.replace('/', '-')[:60])
                        write_json(self.record_dir / (prefix + '_schema_field_repair.json'), legacy_audit)
                    if not legacy_audit['adjustments'] and legacy_audit['ignored_repair_changes']:
                        raise ProviderResponseError(str(exc) + '\n已保留最完整来源；拒绝删除或改写有效字段：' +
                            '；'.join(legacy_audit['ignored_repair_changes'][:8]))
                    return schema.parse_obj(merged)
                repaired_data = self._parse_schema_candidate(repaired, schema, repair_title)
                regressions = repair_preservation_errors(
                    original_data, repaired_data, schema, validation_errors)
                if regressions:
                    merged, patches = merge_schema_repairs(
                        original_data, repaired_data, schema, validation_errors)
                    if patches:
                        result = schema.parse_obj(merged)
                        if self.record_dir is not None:
                            prefix = "%03d_%s" % (self._request_index, repair_title.replace("/", "-")[:60])
                            write_json(self.record_dir / (prefix + "_schema_field_repair.json"), {
                                "scope": "validated_reward_field_repairs_v1", "adjustments": patches,
                                "ignored_repair_changes": regressions, "effective_response": merged,
                            })
                        return result
                    raise ProviderResponseError(
                        str(exc) + "\n已保留最完整来源；拒绝删除或改写有效字段：" + "；".join(regressions[:8]))
                return schema.parse_obj(repaired_data)
            except (ValueError, ValidationError, json.JSONDecodeError,
                    ProviderResponseError) as repair_exc:
                raise ProviderResponseError("百炼结构化回复修复失败：%s" % repair_exc) from repair_exc

    def design_task_bundle(self, compiled_prompt: str) -> Dict[str, Any]:
        """根据本地编译提示词生成 TaskRewardBundle。"""
        return self.request_model(
            compiled_prompt, TaskRewardBundle, "bailian-reward-design").dict()

    def repair_reward_plans(self, payload: Dict[str, Any]) -> RewardRepairBundle:
        """使用与网页入口相同的局部补丁协议。"""
        from ..agents.reward_prompt_policy import render_reward_policy
        from ..agents.schema_contract import render_schema_contract
        template = (Path(__file__).parents[1] / 'prompts' / 'reward_field_repair.md').read_text(encoding='utf-8')
        prompt = template + '\n' + render_reward_policy()
        prompt += '\nREPAIR_REQUEST: ' + json.dumps(json_safe(payload), ensure_ascii=False, allow_nan=False)
        prompt += '\nJSON Schema: ' + RewardRepairBundle.schema_json()
        prompt += '\n' + render_schema_contract(RewardRepairBundle)
        return self.request_model(prompt, RewardRepairBundle, 'bailian-reward-field-repair')

    def redesign_rewards(self, payload: Dict[str, Any]) -> RewardPlan:
        """首次训练失败后的新方案仍由本地审核，后续格式/业务修复走补丁接口。"""
        from ..agents.reward_prompt_policy import render_reward_policy
        from ..agents.schema_contract import render_schema_contract
        template = (Path(__file__).parents[1] / 'prompts' / 'failure_reward_redesign.md').read_text(encoding='utf-8')
        prompt = template + '\n' + render_reward_policy()
        prompt += '\nEvidence: ' + json.dumps(json_safe(payload), ensure_ascii=False, allow_nan=False)
        prompt += '\nJSON Schema: ' + RewardPlan.schema_json()
        prompt += '\n' + render_schema_contract(RewardPlan)
        return self.request_model(prompt, RewardPlan, 'bailian-failure-reward-redesign')

    def diagnose_training(self, payload: Dict[str, Any]) -> TrainingDiagnosis:
        """使用 GLM 融合训练证据并生成结构化修订决策。"""
        template = (Path(__file__).parents[1] / "prompts" / "training_diagnosis.md").read_text(
            encoding="utf-8")
        prompt = (template + "\nEvidence: " + json.dumps(
            json_safe(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False) +
            "\nJSON Schema: " + TrainingDiagnosis.schema_json())
        return self.request_model(prompt, TrainingDiagnosis, "bailian-training-diagnosis")

    def summarize_reward_experience(self, payload: Dict[str, Any]) -> RewardExperienceNarrative:
        """仅总结本地提供且带来源 ID 的训练奖励经验，不返回任何配置事实。"""
        prompt = (REWARD_EXPERIENCE_PROMPT + "\n\nEvidence: " + json.dumps(
            json_safe(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False) +
            "\nJSON Schema: " + RewardExperienceNarrative.schema_json())
        return self.request_model(prompt, RewardExperienceNarrative, "bailian-reward-experience")

    def close(self) -> None:
        """百炼 HTTP Provider 不持有长期连接，无需额外释放资源。"""
        return None
