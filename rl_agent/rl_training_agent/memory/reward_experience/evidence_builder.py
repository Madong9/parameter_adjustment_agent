"""只从实验目录中的真实产物构造奖励经验证据包。"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...utils.io import read_json
from .schema import (
    EvidenceSource, RewardChangeEvidence, RewardExperienceEvidence, RewardPlanSnapshot,
)


class RewardExperienceEvidenceBuilder:
    """从任务、奖励、训练和评估文件中抽取可追溯的紧凑事实。"""

    @staticmethod
    def _mapping(value: Any) -> Dict[str, Any]:
        """把 Pydantic 对象或字典安全转换为普通字典。"""
        if hasattr(value, "dict"):
            return value.dict()
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _terms(plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """把奖励计划规范化为按奖励名索引的可比较配置。"""
        result: Dict[str, Dict[str, Any]] = {}
        for term in plan.get("terms", []) if isinstance(plan.get("terms", []), list) else []:
            if not isinstance(term, dict) or not term.get("name"):
                continue
            result[str(term["name"])] = {
                key: term.get(key) for key in (
                    "weight", "implementation", "parameters", "active_phases",
                    "activation_condition", "normalization") if key in term
            }
        return result

    @staticmethod
    def _safe_relative(path: Path, task_dir: Path) -> Optional[str]:
        """仅返回位于当前任务目录内的 POSIX 相对文件名。"""
        try:
            return path.resolve().relative_to(task_dir.resolve()).as_posix()
        except ValueError:
            return None

    @staticmethod
    def _json_file(path: Path) -> Dict[str, Any]:
        """读取 JSON 对象；文件缺失或损坏时返回空对象。"""
        if not path.is_file():
            return {}
        try:
            value = read_json(path)
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, json.JSONDecodeError):
            return {}

    def _register(self, path: Path, task_dir: Path,
                  index: Dict[str, EvidenceSource]) -> Optional[str]:
        """计算真实文件哈希并登记稳定 evidence ID。"""
        if not path.is_file():
            return None
        relative = self._safe_relative(path, task_dir)
        if relative is None:
            return None
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        evidence_id = "ev-" + hashlib.sha256((relative + "\n" + digest).encode("utf-8")).hexdigest()[:16]
        index[evidence_id] = EvidenceSource(source=relative, sha256=digest)
        return evidence_id

    def _reward_history(self, task_dir: Path, selected_dir: Path,
                        evidence_index: Dict[str, EvidenceSource]) -> List[Dict[str, Any]]:
        """沿父实验关系读取奖励计划链，并拒绝断裂、越界或循环的谱系。"""
        current_dir = selected_dir
        seen = set()
        reversed_history: List[Dict[str, Any]] = []
        reached_root = False
        while current_dir.is_dir() and current_dir != task_dir and current_dir not in seen:
            if self._safe_relative(current_dir, task_dir) is None:
                return []
            seen.add(current_dir)
            manifest_path = current_dir / "manifest.json"
            manifest = self._json_file(manifest_path)
            plan_path = current_dir / "reward_plan.json"
            plan = self._json_file(plan_path)
            if not manifest or not plan:
                return []
            experiment_id = str(manifest.get("experiment_id") or current_dir.name)
            plan_evidence_id = self._register(plan_path, task_dir, evidence_index)
            manifest_evidence_id = self._register(manifest_path, task_dir, evidence_index)
            audit_path = current_dir / "revision_audit.json"
            audit_evidence_id = self._register(audit_path, task_dir, evidence_index)
            match = re.search(r"revision-(\d+)", current_dir.name)
            iteration = int(match.group(1)) if match else None
            reversed_history.append({
                "experiment_id": experiment_id,
                "parent_experiment_id": manifest.get("parent_experiment_id"),
                "reward_version": int(plan.get("version", 0)),
                "iteration": iteration,
                "plan": plan,
                "plan_evidence_id": plan_evidence_id,
                "manifest_evidence_id": manifest_evidence_id,
                "audit_evidence_id": audit_evidence_id,
                "directory": current_dir,
            })
            parent = manifest.get("parent_experiment_id")
            if not parent:
                reached_root = True
                break
            if not re.match(r"^[A-Za-z0-9_-]+$", str(parent)):
                return []
            parent_dir = task_dir / "candidates" / str(parent)
            if not parent_dir.is_dir():
                return []
            current_dir = parent_dir
        if not reached_root:
            return []
        return list(reversed(reversed_history))

    def build(self, task_dir: Path, selected: Dict[str, Any], summary: Dict[str, Any],
              outcome: Dict[str, Any]) -> RewardExperienceEvidence:
        """从本次所选实验和任务目录的产物生成不可臆造的证据包。"""
        task_dir = task_dir.resolve()
        selected_dir = Path(selected.get("dir", task_dir)).resolve()
        if self._safe_relative(selected_dir, task_dir) is None:
            selected_dir = task_dir / "__missing_selected_experiment__"
        evidence_index: Dict[str, EvidenceSource] = {}
        source_ids: Dict[str, Any] = {"index": evidence_index}

        task_path = task_dir / "task_spec.json"
        selected_id_hint = str(summary.get("selected_experiment") or selected.get("id") or "unknown")
        if not re.match(r"^[A-Za-z0-9_-]+$", selected_id_hint):
            selected_id_hint = "unknown"
        summary_path = (task_dir / "memory" / "reward_experience" / selected_id_hint /
                        "training_result_snapshot.json")
        if not summary_path.is_file():
            summary_path = task_dir / "summary.json"
        task_spec = self._json_file(task_path)
        saved_summary = self._json_file(summary_path)
        if not saved_summary:
            saved_summary = dict(summary)
        source_ids["task"] = self._register(task_path, task_dir, evidence_index)
        source_ids["training_snapshot"] = self._register(summary_path, task_dir, evidence_index)

        final_plan_path = selected_dir / "reward_plan.json"
        if not final_plan_path.is_file():
            final_plan_path = task_dir / "final" / "reward_plan.json"
        final_plan = self._json_file(final_plan_path)
        source_ids["final_reward"] = self._register(final_plan_path, task_dir, evidence_index)
        reward_chain = self._reward_history(task_dir, selected_dir, evidence_index)
        initial_plan = reward_chain[0]["plan"] if reward_chain else {}
        if reward_chain:
            final_plan = reward_chain[-1]["plan"]
            source_ids["initial_reward"] = reward_chain[0]["plan_evidence_id"]
            source_ids["final_reward"] = reward_chain[-1]["plan_evidence_id"]

        selected_id = str(saved_summary.get("selected_experiment", summary.get("selected_experiment", "")))
        manifest_path = selected_dir / "manifest.json"
        manifest = self._json_file(manifest_path)
        if not manifest:
            manifest = self._mapping(selected.get("manifest"))
        source_ids["manifest"] = self._register(manifest_path, task_dir, evidence_index)

        rollout_relative = saved_summary.get("rollout") or summary.get("rollout")
        rollout_dir = task_dir / str(rollout_relative) if rollout_relative else task_dir / "__missing_rollout__"
        if self._safe_relative(rollout_dir, task_dir) is None:
            rollout_dir = task_dir / "__missing_rollout__"
        evaluation_path = rollout_dir / "evaluation.json"
        numeric_path = rollout_dir / "numeric_summary.json"
        visual_path = rollout_dir / "visual_report.json"
        evaluation = self._json_file(evaluation_path)
        numeric = self._json_file(numeric_path)
        visual = self._json_file(visual_path)
        source_ids["evaluation"] = self._register(evaluation_path, task_dir, evidence_index)
        source_ids["numeric"] = self._register(numeric_path, task_dir, evidence_index)
        source_ids["visual"] = self._register(visual_path, task_dir, evidence_index)
        loop_path = task_dir / "loop_history.json"
        source_ids["loop_history"] = self._register(loop_path, task_dir, evidence_index)

        initial_terms = self._terms(initial_plan)
        final_terms = self._terms(final_plan)
        changes: List[RewardChangeEvidence] = []
        reward_history = []
        for item in reward_chain:
            reward_history.append(RewardPlanSnapshot(
                experiment_id=item["experiment_id"],
                parent_experiment_id=item["parent_experiment_id"],
                reward_version=item["reward_version"],
                iteration=item["iteration"],
                evidence_ids=[value for value in (
                    item["plan_evidence_id"], item["manifest_evidence_id"],
                    item["audit_evidence_id"]) if value],
            ))
        for previous, current in zip(reward_chain, reward_chain[1:]):
            before_terms = self._terms(previous["plan"])
            after_terms = self._terms(current["plan"])
            transition_evidence = [value for value in (
                previous["plan_evidence_id"], current["plan_evidence_id"],
                current["audit_evidence_id"]) if value]
            for name in sorted(set(before_terms) | set(after_terms)):
                before = before_terms.get(name)
                after = after_terms.get(name)
                if before == after:
                    continue
                changes.append(RewardChangeEvidence(
                    reward_name=name, before=before, after=after,
                    iteration=current["iteration"],
                    reward_version=current["reward_version"],
                    evidence_ids=transition_evidence,
                ))

        metrics = evaluation.get("metrics", []) if isinstance(evaluation.get("metrics", []), list) else []
        failed_metrics = [
            "%s=%s 未通过阈值" % (item.get("name", "unknown"), item.get("value", "unknown"))
            for item in metrics if isinstance(item, dict) and item.get("passed") is False
        ]
        failure_cases = [str(item) for item in evaluation.get("violations", []) if str(item)]
        failure_cases.extend(failed_metrics)
        for item in visual.get("failure_modes", []) if isinstance(visual.get("failure_modes", []), list) else []:
            if isinstance(item, dict):
                failure_cases.append(str(item.get("description") or item.get("failure_mode") or "视觉失败模式"))
            elif item:
                failure_cases.append(str(item))
        failure_cases.extend(str(item) for item in visual.get("unintended_behaviors", []) if item)

        state = str(saved_summary.get("state", summary.get("state", "UNKNOWN")))
        checkpoint_reference = saved_summary.get("checkpoint") or summary.get("checkpoint")
        if not checkpoint_reference and selected.get("checkpoint"):
            checkpoint_reference = self._safe_relative(Path(selected["checkpoint"]), task_dir)
        training_result = {
            "state": state,
            "success": state == "COMPLETED",
            "result": saved_summary.get("result", summary.get("result", "unknown")),
            "training_result": manifest.get("training_result", "unknown"),
            "iteration": manifest.get("iteration", 0),
            "dry_run": bool(saved_summary.get("dry_run", summary.get("dry_run", False))),
            "selected_experiment": selected_id or manifest.get("experiment_id", "unknown"),
            "checkpoint": checkpoint_reference,
            "checkpoint_exists": bool((task_dir / "final" / "checkpoint.pt").is_file() or
                                       (selected.get("checkpoint") and Path(selected["checkpoint"]).is_file())),
            "seeds": list(saved_summary.get("evaluation_seeds", [])),
            "provider_error": bool(saved_summary.get("provider_error", outcome.get("provider_error"))),
            "training_failure_reason": manifest.get("failure_reason"),
            "evidence_ids": [item for item in (
                source_ids.get("training_snapshot"), source_ids.get("manifest")) if item],
        }
        numeric_evaluation = {
            "metrics": numeric,
            "evaluation_metrics": metrics,
            "hard_constraints_passed": evaluation.get("hard_constraints_passed"),
            "task_metrics_passed": evaluation.get("task_metrics_passed"),
            "completed": evaluation.get("completed"),
            "violations": evaluation.get("violations", []),
            "evidence_ids": [item for item in (
                source_ids.get("evaluation"), source_ids.get("numeric")) if item],
        }
        visual_evaluation = dict(visual)
        visual_evaluation["evaluation_alignment_passed"] = evaluation.get("visual_alignment_passed")
        visual_evaluation["evidence_ids"] = [item for item in (
            source_ids.get("visual"), source_ids.get("evaluation")) if item]
        visual_evaluation["visual_evidence_id"] = source_ids.get("visual")

        task = {
            "task_id": str(task_spec.get("task_id") or saved_summary.get("task_id") or task_dir.name),
            "robot": str(task_spec.get("robot") or manifest.get("robot") or "unknown"),
            "goal": str(task_spec.get("normalized_description") or
                        task_spec.get("original_instruction") or ""),
            "action_name": str(task_spec.get("task_name") or ""),
            "environment_version": str(manifest.get("config_hash") or "unknown"),
            "environment_commit": str(manifest.get("git_commit") or "unknown"),
            "evidence_ids": [source_ids["task"]] if source_ids.get("task") else [],
        }
        return RewardExperienceEvidence(
            task_id=task["task_id"], task=task,
            initial_reward={"version": initial_plan.get("version"), "terms": initial_terms,
                            "evidence_ids": [source_ids["initial_reward"]] if source_ids.get("initial_reward") else []},
            final_reward={"version": final_plan.get("version"), "terms": final_terms,
                          "evidence_ids": [source_ids["final_reward"]] if source_ids.get("final_reward") else []},
            reward_history=reward_history,
            reward_changes=changes, training_result=training_result,
            visual_evaluation=visual_evaluation, numeric_evaluation=numeric_evaluation,
            failure_cases=sorted(set(failure_cases)),
            git_commit=str(manifest.get("git_commit") or "unknown"),
            config_hash=str(manifest.get("config_hash") or "unknown"),
            evidence_index=evidence_index,
        )
