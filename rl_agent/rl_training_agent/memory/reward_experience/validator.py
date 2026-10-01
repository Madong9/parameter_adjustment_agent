"""对 Reward Experience 执行来源、数值和因果措辞的确定性校验。"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .evidence_builder import RewardExperienceEvidenceBuilder
from .schema import RewardExperience, RewardExperienceEvidence


class RewardExperienceValidationError(ValueError):
    """表示奖励经验与真实文件或证据引用不一致。"""


class RewardExperienceValidator:
    """验证经验输出的证据来源、奖励差异、任务身份和保守措辞。"""

    CAUSAL_TERMS = re.compile(
        r"导致|造成|使得|因此(?:成功|失败|改善|下降|提升)|证明了?|必然|因为|由于|促使|引发|从而|带来|"
        r"caused?|resulted in|proves?|due to|because of|as a result|improved by",
        re.IGNORECASE,
    )
    UNCERTAINTY_TERMS = re.compile(r"可能|推测|假设|或许|may|might|could|hypothesis", re.IGNORECASE)
    UNIVERSAL_TERMS = re.compile(r"所有机器人|任何机器人|普遍适用|通用规律|all robots|universally", re.IGNORECASE)

    @staticmethod
    def _inside(root: Path, candidate: Path) -> bool:
        """判断解析后的证据路径是否仍位于实验任务目录内。"""
        try:
            return os.path.commonpath([str(root.resolve()), str(candidate.resolve())]) == str(root.resolve())
        except (OSError, ValueError):
            return False

    @staticmethod
    def _json_file(path: Path) -> Dict[str, Any]:
        """读取用于复核的 JSON 对象。"""
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, json.JSONDecodeError):
            return {}

    def _verify_sources(self, evidence: RewardExperienceEvidence, task_dir: Path) -> None:
        """复核 evidence ID 对应文件仍存在且内容哈希未改变。"""
        for evidence_id, source in evidence.evidence_index.items():
            path = task_dir / source.source
            if not self._inside(task_dir, path) or not path.is_file():
                raise RewardExperienceValidationError("evidence source missing or unsafe: %s" % evidence_id)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != source.sha256:
                raise RewardExperienceValidationError("evidence source changed: %s" % evidence_id)

    @staticmethod
    def _source_for_id(evidence: RewardExperienceEvidence, evidence_id: str) -> str:
        """将 evidence ID 映射回证据包记录的来源文件。"""
        source = evidence.evidence_index.get(evidence_id)
        if source is None:
            raise RewardExperienceValidationError("unknown evidence_id: %s" % evidence_id)
        return source.source

    def _verify_evaluation_payloads(self, evidence: RewardExperienceEvidence,
                                    task_dir: Path) -> None:
        """确认数值和视觉摘要逐字段来自其声明的原始评估文件。"""
        numeric_sources = [self._source_for_id(evidence, item)
                           for item in evidence.numeric_evaluation.get("evidence_ids", [])]
        evaluation_source = next((item for item in numeric_sources if item.endswith("evaluation.json")), None)
        numeric_source = next((item for item in numeric_sources if item.endswith("numeric_summary.json")), None)
        if evaluation_source is None or numeric_source is None:
            raise RewardExperienceValidationError("numeric evaluation must cite evaluation.json and numeric_summary.json")
        evaluation = self._json_file(task_dir / evaluation_source)
        numeric = self._json_file(task_dir / numeric_source)
        expected_metrics = evaluation.get("metrics", [])
        if evidence.numeric_evaluation.get("metrics") != numeric:
            raise RewardExperienceValidationError("numeric metrics do not match numeric_summary.json")
        for field in ("hard_constraints_passed", "task_metrics_passed", "completed", "violations"):
            if evidence.numeric_evaluation.get(field) != evaluation.get(field):
                raise RewardExperienceValidationError("numeric evaluation field does not match evaluation.json: %s" % field)
        if evidence.numeric_evaluation.get("evaluation_metrics") != expected_metrics:
            raise RewardExperienceValidationError("metric results do not match evaluation.json")
        visual_id = evidence.visual_evaluation.get("visual_evidence_id")
        if not visual_id:
            raise RewardExperienceValidationError("visual evaluation evidence ID is missing")
        visual_source = self._source_for_id(evidence, visual_id)
        if not visual_source.endswith("visual_report.json"):
            raise RewardExperienceValidationError("visual evidence must reference visual_report.json")
        visual = self._json_file(task_dir / visual_source)
        expected_visual = dict(visual)
        expected_visual["evaluation_alignment_passed"] = evaluation.get("visual_alignment_passed")
        expected_visual["evidence_ids"] = evidence.visual_evaluation.get("evidence_ids")
        expected_visual["visual_evidence_id"] = visual_id
        if evidence.visual_evaluation != expected_visual:
            raise RewardExperienceValidationError("visual evaluation does not match source files")

    def _verify_task_payload(self, evidence: RewardExperienceEvidence, task_dir: Path) -> None:
        """确认任务机器人和动作描述确实来自 task_spec.json。"""
        task_id = evidence.task.get("evidence_ids", [])
        if not task_id:
            raise RewardExperienceValidationError("task spec evidence ID is missing")
        task_source = self._source_for_id(evidence, task_id[0])
        if not task_source.endswith("task_spec.json"):
            raise RewardExperienceValidationError("task evidence must reference task_spec.json")
        actual = self._json_file(task_dir / task_source)
        if evidence.task.get("robot") != actual.get("robot"):
            raise RewardExperienceValidationError("robot does not match task_spec.json")
        actual_goal = str(actual.get("normalized_description") or actual.get("original_instruction") or "")
        if evidence.task.get("goal") != actual_goal:
            raise RewardExperienceValidationError("goal does not match task_spec.json")
        if evidence.task.get("action_name") != str(actual.get("task_name", "")):
            raise RewardExperienceValidationError("action name does not match task_spec.json")
        if evidence.task_id != str(actual.get("task_id", evidence.task_id)):
            raise RewardExperienceValidationError("task ID does not match task_spec.json")

    def _verify_training_payload(self, evidence: RewardExperienceEvidence, task_dir: Path) -> None:
        """确认训练状态、配置指纹、随机种子和 checkpoint 路径来自快照及 manifest。"""
        sources = [self._source_for_id(evidence, item)
                   for item in evidence.training_result.get("evidence_ids", [])]
        snapshot_source = next((item for item in sources if item.endswith("training_result_snapshot.json")), None)
        manifest_source = next((item for item in sources if item.endswith("manifest.json")), None)
        if snapshot_source is None or manifest_source is None:
            raise RewardExperienceValidationError("training evidence must cite result snapshot and manifest")
        snapshot = self._json_file(task_dir / snapshot_source)
        manifest = self._json_file(task_dir / manifest_source)
        training = evidence.training_result
        if training.get("state") != snapshot.get("state") or \
                training.get("result") != snapshot.get("result") or \
                training.get("dry_run") != bool(snapshot.get("dry_run", False)) or \
                training.get("provider_error") != bool(snapshot.get("provider_error", False)):
            raise RewardExperienceValidationError("training result does not match immutable result snapshot")
        if training.get("selected_experiment") != snapshot.get("selected_experiment") or \
                training.get("selected_experiment") != manifest.get("experiment_id"):
            raise RewardExperienceValidationError("selected experiment does not match snapshot and manifest")
        if training.get("training_result") != manifest.get("training_result") or \
                training.get("iteration") != manifest.get("iteration", 0):
            raise RewardExperienceValidationError("PPO completion status does not match manifest")
        if evidence.git_commit != manifest.get("git_commit") or \
                evidence.config_hash != manifest.get("config_hash") or \
                evidence.task.get("environment_version") != manifest.get("config_hash"):
            raise RewardExperienceValidationError("environment fingerprint does not match manifest")
        if training.get("success") != (snapshot.get("state") == "COMPLETED"):
            raise RewardExperienceValidationError("training success flag does not match terminal state")
        if training.get("seeds") != list(snapshot.get("evaluation_seeds", [])):
            raise RewardExperienceValidationError("evaluation seeds do not match result snapshot")
        checkpoint = training.get("checkpoint")
        checkpoint_path = task_dir / str(checkpoint or "__missing_checkpoint__")
        if not checkpoint or not self._inside(task_dir, checkpoint_path):
            raise RewardExperienceValidationError("checkpoint path is missing or unsafe")
        if training.get("checkpoint_exists") != checkpoint_path.is_file():
            raise RewardExperienceValidationError("checkpoint existence does not match the recorded path")

    def _verify_reward_diffs(self, evidence: RewardExperienceEvidence, task_dir: Path) -> None:
        """重建父子奖励计划链及逐版本差异，拒绝模型编造或篡改。"""
        initial_ids = evidence.initial_reward.get("evidence_ids", [])
        final_ids = evidence.final_reward.get("evidence_ids", [])
        if not initial_ids or not final_ids or not evidence.reward_history:
            raise RewardExperienceValidationError("initial/final reward evidence is missing")
        history = []
        for snapshot in evidence.reward_history:
            sources = [self._source_for_id(evidence, item) for item in snapshot.evidence_ids]
            plan_source = next((item for item in sources if item.endswith("reward_plan.json")), None)
            manifest_source = next((item for item in sources if item.endswith("manifest.json")), None)
            audit_source = next((item for item in sources if item.endswith("revision_audit.json")), None)
            if plan_source is None or manifest_source is None:
                raise RewardExperienceValidationError("each reward version must cite its real plan and manifest")
            plan = self._json_file(task_dir / plan_source)
            manifest = self._json_file(task_dir / manifest_source)
            if snapshot.experiment_id != manifest.get("experiment_id") or \
                    snapshot.parent_experiment_id != manifest.get("parent_experiment_id") or \
                    snapshot.reward_version != plan.get("version"):
                raise RewardExperienceValidationError("reward history does not match its manifest/config")
            match = re.search(r"revision-(\d+)", Path(plan_source).parent.name)
            actual_iteration = int(match.group(1)) if match else None
            if snapshot.iteration != actual_iteration:
                raise RewardExperienceValidationError("reward revision iteration does not match experiment ID")
            plan_id = next(item for item in snapshot.evidence_ids
                           if self._source_for_id(evidence, item).endswith("reward_plan.json"))
            manifest_id = next(item for item in snapshot.evidence_ids
                               if self._source_for_id(evidence, item).endswith("manifest.json"))
            audit_id = next((item for item in snapshot.evidence_ids
                             if self._source_for_id(evidence, item).endswith("revision_audit.json")), None)
            expected_sources = [plan_id, manifest_id] + ([audit_id] if audit_id else [])
            if snapshot.evidence_ids != expected_sources:
                raise RewardExperienceValidationError("reward version evidence IDs are not canonical")
            if audit_source:
                audit = self._json_file(task_dir / audit_source)
                if audit.get("parent_experiment") != manifest.get("parent_experiment_id"):
                    raise RewardExperienceValidationError("revision audit does not match manifest parent")
            if audit_source is None and len(expected_sources) != 2:
                raise RewardExperienceValidationError("reward history has an invalid audit evidence reference")
            if audit_source is not None and len(expected_sources) != 3:
                raise RewardExperienceValidationError("reward history audit reference is incomplete")
            history.append({
                "snapshot": snapshot,
                "plan": plan,
                "terms": RewardExperienceEvidenceBuilder._terms(plan),
                "plan_id": plan_id,
                "audit_id": audit_id,
            })
        if history[0]["snapshot"].parent_experiment_id is not None:
            raise RewardExperienceValidationError("reward history is missing its root parent plan")
        for previous, current in zip(history, history[1:]):
            if current["snapshot"].parent_experiment_id != previous["snapshot"].experiment_id:
                raise RewardExperienceValidationError("reward history parent-child chain is broken")

        initial_plan = history[0]["plan"]
        final_plan = history[-1]["plan"]
        if initial_ids != [history[0]["plan_id"]] or final_ids != [history[-1]["plan_id"]]:
            raise RewardExperienceValidationError("initial/final reward references do not match reward history")
        initial_terms = evidence.initial_reward.get("terms", {})
        final_terms = evidence.final_reward.get("terms", {})
        actual_initial = history[0]["terms"]
        actual_final = history[-1]["terms"]
        if initial_terms != actual_initial or final_terms != actual_final:
            raise RewardExperienceValidationError("reward designs do not match real reward_plan.json files")
        if evidence.initial_reward.get("version") != initial_plan.get("version") or \
                evidence.final_reward.get("version") != final_plan.get("version"):
            raise RewardExperienceValidationError("reward versions do not match reward_plan.json files")
        if not isinstance(actual_initial, dict) or not isinstance(actual_final, dict):
            raise RewardExperienceValidationError("reward terms are malformed")
        expected = []
        for previous, current in zip(history, history[1:]):
            transition_ids = [previous["plan_id"], current["plan_id"]]
            if current["audit_id"]:
                transition_ids.append(current["audit_id"])
            for name in sorted(set(previous["terms"]) | set(current["terms"])):
                before = previous["terms"].get(name)
                after = current["terms"].get(name)
                if before != after:
                    expected.append((name, before, after,
                                     current["snapshot"].iteration,
                                     current["snapshot"].reward_version,
                                     transition_ids))
        reported = [(item.reward_name, item.before, item.after, item.iteration,
                     item.reward_version, item.evidence_ids) for item in evidence.reward_changes]
        if reported != expected:
            raise RewardExperienceValidationError("reward changes do not match real initial/final configurations")

    def _verify_claims(self, experience: RewardExperience,
                       evidence: RewardExperienceEvidence) -> None:
        """检查每条文字结论都有来源且没有把假设冒充因果事实。"""
        evidence_ids = set(evidence.evidence_index)
        statements = list(experience.observed_facts) + list(experience.hypotheses) + list(experience.limitations)
        hypotheses = experience.hypotheses
        for statement in statements:
            if not statement.evidence_ids or not set(statement.evidence_ids).issubset(evidence_ids):
                raise RewardExperienceValidationError("every narrative statement must cite known evidence IDs")
            is_hypothesis = statement in hypotheses
            if self.CAUSAL_TERMS.search(statement.statement) and not (
                    is_hypothesis and self.UNCERTAINTY_TERMS.search(statement.statement)):
                raise RewardExperienceValidationError("causal wording is not allowed as an observed fact")
            if is_hypothesis and not self.UNCERTAINTY_TERMS.search(statement.statement):
                raise RewardExperienceValidationError("hypotheses must use explicitly uncertain wording")
        expected_changes = [(item.reward_name, item.iteration, item.reward_version,
                             item.before, item.after) for item in evidence.reward_changes]
        reported_changes = [(item.reward_name, item.iteration, item.reward_version,
                             item.before, item.after) for item in experience.reward_evolution.changes]
        if reported_changes != expected_changes:
            raise RewardExperienceValidationError("reward evolution must include exactly the real configuration diffs")
        for change in experience.reward_evolution.changes:
            source_change = next((item for item in evidence.reward_changes
                                  if item.reward_name == change.reward_name and
                                  item.iteration == change.iteration and
                                  item.reward_version == change.reward_version and
                                  item.before == change.before and item.after == change.after), None)
            if source_change is None:
                raise RewardExperienceValidationError("experience contains an invented reward change")
            if not set(source_change.evidence_ids).issubset(change.evidence_ids):
                raise RewardExperienceValidationError("reward change omits its configuration evidence IDs")
            if not set(change.evidence_ids).issubset(evidence_ids):
                raise RewardExperienceValidationError("reward change cites an unknown evidence ID")
            behavior_evidence = set(change.evidence_ids) - set(source_change.evidence_ids)
            if change.observed_behavior_change and self.CAUSAL_TERMS.search(change.observed_behavior_change):
                raise RewardExperienceValidationError("reward behavior summary overstates causality")
            if change.observed_behavior_change and not behavior_evidence:
                raise RewardExperienceValidationError("behavior change needs separate evaluation evidence")
            for evidence_id in behavior_evidence:
                source = self._source_for_id(evidence, evidence_id)
                if not source.endswith(("evaluation.json", "numeric_summary.json", "visual_report.json")):
                    raise RewardExperienceValidationError("behavior evidence must come from numeric or visual evaluation")
        expected_scope = {
            "robot": str(evidence.task.get("robot", "unknown")),
            "action": str(evidence.task.get("action_name", "unknown")),
            "goal": str(evidence.task.get("goal", "")),
            "environment_version": str(evidence.config_hash),
        }
        patterns = experience.successful_patterns + experience.failure_patterns
        for pattern in patterns:
            if pattern.applicability != expected_scope:
                raise RewardExperienceValidationError("patterns must be scoped to the current robot/task/environment")
            if not pattern.evidence_ids or not set(pattern.evidence_ids).issubset(evidence_ids):
                raise RewardExperienceValidationError("patterns must cite existing evidence IDs")
            if self.UNIVERSAL_TERMS.search(pattern.statement):
                raise RewardExperienceValidationError("single-experiment pattern cannot claim universal applicability")
            if self.CAUSAL_TERMS.search(pattern.statement):
                raise RewardExperienceValidationError("pattern overstates causality")

    def validate(self, experience: RewardExperience, evidence: RewardExperienceEvidence,
                 task_dir: Path) -> None:
        """校验来源文件、任务身份、奖励差异和文字结论，失败时抛出异常。"""
        self._verify_sources(evidence, task_dir)
        self._verify_task_payload(evidence, task_dir)
        self._verify_training_payload(evidence, task_dir)
        self._verify_evaluation_payloads(evidence, task_dir)
        from .eligibility import ExperienceEligibilityChecker
        eligibility = ExperienceEligibilityChecker().check(evidence)
        if not eligibility.eligible or experience.outcome != eligibility.outcome:
            raise RewardExperienceValidationError("experience outcome does not pass deterministic eligibility gate")
        self._verify_reward_diffs(evidence, task_dir)
        if experience.outcome not in ("SUCCESS", "VERIFIED_FAILURE"):
            raise RewardExperienceValidationError("INCONCLUSIVE cannot be persisted as an experience")
        if experience.task_context.get("robot") != evidence.task.get("robot") or \
                experience.task_context.get("task_id") != evidence.task_id:
            raise RewardExperienceValidationError("robot or task identity does not match evidence")
        if experience.reward_evolution.initial_design != evidence.initial_reward or \
                experience.reward_evolution.final_design != evidence.final_reward:
            raise RewardExperienceValidationError("reward designs must be copied from evidence package")
        if experience.evidence_index != evidence.evidence_index:
            raise RewardExperienceValidationError("evidence ID/source index must be copied without alteration")
        self._verify_claims(experience, evidence)
