"""实现成功经验与明确失败模式的严格情景记忆晋升门控。"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..schemas.agent_workflow import LongTermMemoryRecord, TaskIntentSpec
from ..schemas.rewards import RewardPlan
from ..schemas.task import TaskSpec
from ..schemas.metrics import EvaluationResult
from ..utils.io import read_json, utc_now
from .reward_experience.schema import RewardExperience
from .reward_experience.schema import RewardExperienceEvidence
from .reward_experience.validator import RewardExperienceValidator
from .evidence_snapshot import create_evidence_snapshot, file_sha256, safe_evidence_path


class MemoryCuratorAgent:
    """从真实多种子实验提取紧凑经验，拒绝故障和未经证实的结论。"""

    def __init__(self) -> None:
        """初始化最近一次门控说明，供任务产物和上位机展示。"""
        self.last_reason = "尚未执行记忆整理"

    @staticmethod
    def _safe_mapping(value: Any) -> Dict[str, Any]:
        """把 Pydantic 对象或字典转换为普通字典。"""
        if hasattr(value, "dict"):
            return value.dict()
        if hasattr(value, "__dict__"):
            return dict(vars(value))
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _evidence_object(path: Path) -> Dict[str, Any]:
        """读取证据对象并把损坏原因绑定到来源文件名，便于定位而不输出绝对路径。"""
        try:
            value = read_json(path)
            if not isinstance(value, dict):
                raise ValueError("必须是 JSON 对象")
            return value
        except (OSError, ValueError) as exc:
            raise ValueError("%s 无效：%s" % (path.name, exc)) from exc

    @staticmethod
    def _reward_diff(plan: RewardPlan, selected: Dict[str, Any]) -> Dict[str, Any]:
        """生成可审计的最终奖励版本及相对父版本修改摘要。"""
        audit_path = Path(selected.get("dir", ".")) / "revision_audit.json"
        audit = read_json(audit_path) if audit_path.is_file() else {}
        diagnosis = audit.get("diagnosis", {}) if isinstance(audit.get("diagnosis", {}), dict) else {}
        return {
            "parent_experiment": audit.get("parent_experiment"),
            "parent_reward_version": audit.get(
                "parent_reward_version", plan.version - 1 if audit_path.is_file() else None),
            "final_reward_version": plan.version,
            "changes": audit.get(
                "reward_changes", diagnosis.get("reward_changes", audit.get("changes", []))),
            "normalization_adjustments": audit.get("adjustments", []),
            "initial_design": not audit_path.is_file(),
            "term_count": len(plan.terms),
        }

    def curate(self, intent: TaskIntentSpec, task: TaskSpec, plan: RewardPlan,
               summary: Dict[str, Any], outcome: Dict[str, Any], selected: Dict[str, Any],
               task_dir: Path, require_multi_seed: bool = True,
               experience: Optional[RewardExperience] = None,
               require_validated_experience: bool = False,
               memory_root: Optional[Path] = None) -> Optional[LongTermMemoryRecord]:
        """仅为真实、多种子一致且证据完整的成功或明确失败生成情景记忆。"""
        if summary.get("dry_run", False):
            self.last_reason = "dry-run 使用模拟证据，不得晋升长期记忆"
            return None
        final_state = str(summary.get("state", ""))
        if final_state in ("HUMAN_REVIEW", "") or outcome.get("provider_error"):
            self.last_reason = "人工复核、Provider 超时或格式错误只保留原始实验"
            return None
        evaluation = outcome.get("evaluation")
        if evaluation is None:
            self.last_reason = "缺少确定性评估，不能验证经验"
            return None
        evaluation_data = self._safe_mapping(evaluation)
        completed = bool(evaluation_data.get("completed"))
        success = (final_state == "COMPLETED" and completed and
                   evaluation_data.get("hard_constraints_passed") and
                   evaluation_data.get("task_metrics_passed") and
                   evaluation_data.get("visual_alignment_passed"))
        violations = [str(item) for item in evaluation_data.get("violations", []) if str(item)]
        verified_failure = final_state == "FAILED" and not completed and bool(violations)
        if not success and not verified_failure:
            self.last_reason = "结果既未联合验收通过，也不是具有确定性违规证据的明确失败"
            return None
        if require_validated_experience and (
                experience is None or experience.outcome != ("SUCCESS" if success else "VERIFIED_FAILURE")):
            self.last_reason = "缺少与真实证据相符且通过 Validator 的 Reward Experience"
            return None
        seeds = list(selected.get("checkpoint_seeds", []))
        if require_multi_seed and len(set(seeds)) < 2:
            self.last_reason = "长期经验至少需要两个不同随机种子的结果"
            return None
        selected_id = str(summary.get("selected_experiment", "unknown"))
        if not re.fullmatch(r"[A-Za-z0-9_-]+", selected_id):
            self.last_reason = "实验标识不能用于安全证据快照"
            return None
        task_dir = task_dir.resolve()
        rollout = summary.get("rollout")
        local_evidence = [
            task_dir / "task_spec.json",
            task_dir / "final" / "reward_plan.json",
            task_dir / "loop_history.json",
        ]
        if experience is not None:
            experience_dir = task_dir / "memory" / "reward_experience" / selected_id
            experience_files = ["evidence.json", "eligibility.json", "experience.json", "validation.json"]
            local_evidence.extend(experience_dir / name for name in experience_files)
        if rollout:
            local_evidence.append(task_dir / str(rollout) / "evaluation.json")
        if not rollout or not all(path.is_file() for path in local_evidence):
            self.last_reason = "任务、奖励、闭环历史或评估证据文件不完整"
            return None
        memory_id = "memory-" + hashlib.sha1(
            (task.task_id + selected_id + str(plan.version) + final_state).encode("utf-8")).hexdigest()[:12]
        try:
            for path in local_evidence:
                safe_evidence_path(task_dir, path.relative_to(task_dir).as_posix())
            actual_task = self._evidence_object(task_dir / "task_spec.json")
            for field in ("task_id", "robot", "task_name", "original_instruction", "normalized_description"):
                if actual_task.get(field) != getattr(task, field):
                    raise ValueError("任务身份与实际 task_spec.json 不符：" + field)
            try:
                actual_plan = RewardPlan.parse_obj(self._evidence_object(task_dir / "final" / "reward_plan.json"))
            except ValueError as exc:
                raise ValueError("reward_plan.json 无效：%s" % exc) from exc
            if actual_plan.dict() != plan.dict():
                raise ValueError("最终奖励与实际 reward_plan.json 不符")
            try:
                actual_evaluation = EvaluationResult.parse_obj(
                    self._evidence_object(task_dir / str(rollout) / "evaluation.json"))
            except ValueError as exc:
                raise ValueError("evaluation.json 无效：%s" % exc) from exc
            if actual_evaluation.dict() != EvaluationResult.parse_obj(evaluation_data).dict():
                raise ValueError("完成判定与实际 evaluation.json 不符")
            expected_hashes: Dict[str, str] = {}
            if experience is not None:
                saved_experience = RewardExperience.parse_obj(read_json(experience_dir / "experience.json"))
                evidence_package = RewardExperienceEvidence.parse_obj(read_json(experience_dir / "evidence.json"))
                if saved_experience.dict() != experience.dict() or read_json(
                        experience_dir / "validation.json").get("valid") is not True:
                    raise ValueError("Reward Experience 与已通过校验的落盘产物不符")
                RewardExperienceValidator().validate(experience, evidence_package, task_dir)
                for source in experience.evidence_index.values():
                    local_evidence.append(safe_evidence_path(task_dir, source.source))
                    expected_hashes[source.source] = source.sha256
            for name in ("environment_compatibility.json", "acceptance_contract.json",
                         "environment_manifest.json", "training_admission.json", "training_readiness.json", "state.json"):
                path = task_dir / name
                if path.is_file():
                    local_evidence.append(safe_evidence_path(task_dir, name))
            local_evidence.extend((task_dir / str(rollout)).glob("*.parquet"))
            checkpoint = Path(selected.get("checkpoint") or task_dir / str(summary.get("checkpoint", "")))
            checkpoint = safe_evidence_path(task_dir, checkpoint.relative_to(task_dir).as_posix())
            if not checkpoint.is_file():
                raise ValueError("长期经验缺少真实 checkpoint 产物")
            checkpoint_hash = file_sha256(checkpoint)
            final_checkpoint = safe_evidence_path(task_dir, str(summary.get("checkpoint") or "final/checkpoint.pt"))
            if not final_checkpoint.is_file() or file_sha256(final_checkpoint) != checkpoint_hash:
                raise ValueError("最终归档 checkpoint 与所选策略不符")
            checkpoint_provenance = {
                "task_id": task.task_id,
                "source": checkpoint.relative_to(task_dir).as_posix(),
                "recorded_checkpoint": str(summary.get("checkpoint") or "final/checkpoint.pt"),
                "sha256": checkpoint_hash, "exists_at_promotion": True,
                "selected_experiment": selected_id,
            }
            state_path = task_dir / "state.json"
            state_data = self._evidence_object(state_path) if state_path.is_file() else {}
            context = state_data.get("context", {})
            run_match = re.match(r"^(run-[A-Za-z0-9]+)", selected_id)
            checkpoint_provenance["run_id"] = str(summary.get("run_id") or (
                context.get("run_id") if isinstance(context, dict) else None) or (
                run_match.group(1) if run_match else "unknown"))
            snapshot_root, snapshot_hashes = create_evidence_snapshot(
                memory_root or task_dir / "memory", memory_id, task_dir, local_evidence,
                checkpoint_provenance=checkpoint_provenance, expected_hashes=expected_hashes)
        except (OSError, ValueError) as exc:
            self.last_reason = "经验证据校验或不可变快照失败：%s" % exc
            return None
        evidence = [snapshot_root + "/" + source for source in sorted(snapshot_hashes)]
        lessons: List[str] = list(plan.design_rationale)
        if experience is not None:
            lessons.extend(item.statement for item in experience.observed_facts)
            lessons.extend(item.statement for item in experience.successful_patterns)
            lessons.extend(item.statement for item in experience.failure_patterns)
        if success:
            lessons.append("该奖励版本已通过视觉、任务指标和硬安全约束联合验收。")
        else:
            lessons.append("已由多种子确定性评估确认失败模式：%s。" % "；".join(violations))
        if summary.get("loop_rounds", 0) > 1:
            lessons.append("经过%d轮训练诊断与复评后得到结论。" % summary["loop_rounds"])
        physical = self._safe_mapping(outcome.get("physical", {}))
        safe_metrics = {
            key: value for key, value in physical.items()
            if isinstance(value, (int, float, bool)) or value is None
        }
        visual = self._safe_mapping(outcome.get("visual", {}))
        visual_conclusion = {
            key: visual.get(key) for key in (
                "visual_success", "alignment_score", "confidence", "summary",
                "failure_modes", "unintended_behaviors", "uncertain_items", "requires_human_review")
            if key in visual
        }
        manifest = selected.get("manifest")
        manifest_data = self._safe_mapping(manifest)
        git_commit = str(manifest_data.get("git_commit", "unknown"))
        config_hash = str(manifest_data.get(
            "config_hash", self._safe_mapping(selected.get("metadata", {})).get("config_hash", "unknown")))
        checkpoint = str(summary.get("checkpoint", ""))
        now = utc_now()
        compatibility_path = task_dir / "environment_compatibility.json"
        compatibility = read_json(compatibility_path) if compatibility_path.is_file() else {}
        failure_patterns = (
            [item.statement for item in experience.failure_patterns] if experience is not None else [])
        failure_pattern = ("；".join(failure_patterns) or "；".join(sorted(violations))
                           if verified_failure else None)
        self.last_reason = "已通过成功经验门控" if success else "已通过明确失败模式门控"
        return LongTermMemoryRecord(
            memory_id=memory_id,
            task_id=task.task_id,
            robot=task.robot,
            action_name=intent.action_name,
            normalized_goal=intent.normalized_goal,
            outcome="completed" if success else "verified_failure",
            final_state=final_state,
            reward_version=plan.version,
            reward_terms=[{
                "name": term.name,
                "weight": term.weight,
                "parameters": term.parameters,
                "active_phases": term.active_phases,
            } for term in plan.terms],
            reward_diff=self._reward_diff(plan, selected),
            reward_experience=experience.dict() if experience is not None else None,
            lessons=lessons,
            metrics=safe_metrics,
            deterministic_metrics=safe_metrics,
            visual_conclusion=visual_conclusion,
            evidence_sources=evidence,
            evidence_snapshot_root=snapshot_root,
            evidence_snapshot_hashes=snapshot_hashes,
            environment_fingerprint="%s:%s" % (git_commit, config_hash),
            simulation_platform="unitree_rl_gym / Isaac Gym",
            environment_version=compatibility.get("environment_version", "unknown"),
            reward_registry_version=compatibility.get("reward_registry_version", "unknown"),
            git_commit=git_commit,
            checkpoint=checkpoint,
            seed_count=len(set(seeds)),
            consistent_across_seeds=True,
            failure_pattern=failure_pattern,
            confidence=min(experience.confidence, 0.65) if experience is not None else (
                0.98 if success else 0.82),
            created_at=now,
            updated_at=now,
        )
