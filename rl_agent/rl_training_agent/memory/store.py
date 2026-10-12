"""实现工作、情景、语义和程序四层训练记忆的持久化与检索。"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..rag.knowledge_base import TrainingKnowledgeBase, tokenize
from ..schemas.agent_workflow import (
    LongTermMemoryRecord,
    ProceduralMemoryRecord,
    SemanticMemoryRecord,
    WorkingMemorySnapshot,
)
from ..utils.io import read_json, utc_now, write_json
from ..utils.paths import relative_display
from ..utils.paths import ensure_within
from ..storage.locking import FileLock
from .evidence_snapshot import SNAPSHOT_MANIFEST, file_sha256, safe_evidence_path
from .reward_experience.schema import RewardExperience, RewardExperienceEvidence
from .reward_experience.validator import RewardExperienceValidator


class LongTermMemoryStore:
    """管理四层记忆，并只把活跃的可信长期记忆送入模型上下文。"""

    def __init__(self, root: Path, agent_root: Path, max_context_chars: int = 4000):
        """初始化各层目录和检索上下文长度限制。"""
        self.root = root.resolve()
        self.agent_root = agent_root.resolve()
        self.records_dir = self.root / "records"
        self.semantic_dir = self.root / "semantic"
        self.procedural_dir = self.root / "procedural"
        self.max_context_chars = max_context_chars

    def _record_path(self, directory: Path, identifier: str) -> Path:
        """校验记忆标识和写入边界，防止不可信历史 ID 产生路径跳转。"""
        if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
            raise ValueError("不安全的记忆标识")
        path = safe_evidence_path(self.root, (directory / (identifier + ".json")).relative_to(self.root).as_posix())
        ensure_within(path, directory)
        return path

    def evidence_valid(self, record: LongTermMemoryRecord) -> bool:
        """复核不可变来源哈希与本地验收真值，缺失或损坏快照的记录不得注入。"""
        if not record.evidence_snapshot_root or not record.evidence_snapshot_hashes:
            return False
        try:
            if record.evidence_snapshot_root != "evidence/" + record.memory_id:
                return False
            snapshot = safe_evidence_path(self.root, record.evidence_snapshot_root)
            hashes = record.evidence_snapshot_hashes
            if SNAPSHOT_MANIFEST not in hashes or any(
                    file_sha256(safe_evidence_path(snapshot, source)) != digest
                    for source, digest in hashes.items()):
                return False
            manifest = read_json(snapshot / SNAPSHOT_MANIFEST)
            if manifest.get("memory_id") != record.memory_id or manifest.get("hashes") != {
                    key: value for key, value in hashes.items() if key != SNAPSHOT_MANIFEST}:
                return False
            task = read_json(snapshot / "task_spec.json")
            if task.get("task_id") != record.task_id or task.get("robot") != record.robot:
                return False
            plan = read_json(safe_evidence_path(snapshot, "final/reward_plan.json"))
            if plan.get("version") != record.reward_version or record.reward_terms != [{
                    key: term[key] for key in ("name", "weight", "parameters", "active_phases") if key in term
                    } for term in plan.get("terms", [])]:
                return False
            if record.reward_experience is not None:
                experience = RewardExperience.parse_obj(record.reward_experience)
                prefix = "memory/reward_experience/" + str(
                    experience.task_context.get("training_result", {}).get("selected_experiment", ""))
                evidence = RewardExperienceEvidence.parse_obj(read_json(
                    safe_evidence_path(snapshot, prefix + "/evidence.json")))
                validation = read_json(safe_evidence_path(snapshot, prefix + "/validation.json"))
                saved = RewardExperience.parse_obj(read_json(safe_evidence_path(snapshot, prefix + "/experience.json")))
                if validation.get("valid") is not True or saved.dict() != experience.dict():
                    return False
                RewardExperienceValidator().validate(
                    experience, evidence, snapshot,
                    checkpoint_provenance=manifest.get("checkpoint_provenance", {}))
                training = evidence.training_result
                if (record.outcome != ("completed" if experience.outcome == "SUCCESS" else "verified_failure") or
                        record.final_state != training.get("state") or
                        record.seed_count != len(set(training.get("seeds", [])))):
                    return False
            else:
                evaluation_sources = [source for source in hashes if source.endswith("/evaluation.json")]
                if len(evaluation_sources) != 1:
                    return False
                evaluation = read_json(safe_evidence_path(snapshot, evaluation_sources[0]))
                if record.outcome == "completed":
                    if not all(evaluation.get(field) is True for field in (
                            "completed", "hard_constraints_passed", "task_metrics_passed", "visual_alignment_passed")):
                        return False
                elif record.outcome != "verified_failure" or evaluation.get("completed") is not False or not evaluation.get("violations"):
                    return False
            return True
        except (OSError, ValueError, TypeError, AttributeError, KeyError):
            return False

    @staticmethod
    def _parse_time(value: Optional[str]) -> datetime:
        """把 ISO 时间转换为 UTC 时间，损坏值按 Unix 起点处理。"""
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return datetime.fromtimestamp(0, tz=timezone.utc)

    def records(self, active_only: bool = False) -> List[LongTermMemoryRecord]:
        """读取合法情景记忆，可排除已归档记录。"""
        values: List[LongTermMemoryRecord] = []
        if not self.records_dir.is_dir():
            return values
        for path in sorted(self.records_dir.glob("*.json")):
            try:
                if path.is_symlink():
                    continue
                ensure_within(path, self.root)
                record = LongTermMemoryRecord.parse_obj(read_json(path))
                if self._record_path(self.records_dir, record.memory_id) != path:
                    continue
                if not active_only or record.status == "active":
                    values.append(record)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return values

    def semantic_records(self, active_only: bool = False) -> List[SemanticMemoryRecord]:
        """读取合法语义记忆，可只返回已经完成升级的活跃规律。"""
        values: List[SemanticMemoryRecord] = []
        if not self.semantic_dir.is_dir():
            return values
        for path in sorted(self.semantic_dir.glob("*.json")):
            try:
                if path.is_symlink():
                    continue
                ensure_within(path, self.root)
                record = SemanticMemoryRecord.parse_obj(read_json(path))
                if self._record_path(self.semantic_dir, record.semantic_id) != path:
                    continue
                if not active_only or record.status == "active":
                    values.append(record)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return values

    def promote(self, record: LongTermMemoryRecord) -> Path:
        """原子保存已通过晋升门控的情景记忆。"""
        path = self._record_path(self.records_dir, record.memory_id)
        write_json(path, record)
        return path

    def save_semantic(self, record: SemanticMemoryRecord) -> Path:
        """保存候选或活跃语义规律，不覆盖其证据关系。"""
        path = self._record_path(self.semantic_dir, record.semantic_id)
        write_json(path, record)
        return path

    def save_working(self, task_dir: Path, snapshot: WorkingMemorySnapshot) -> Path:
        """把当前任务工作记忆写入任务目录，而非全局可信经验库。"""
        path = task_dir / "memory" / "working_memory.json"
        write_json(path, snapshot)
        return path

    def save_procedural(self, record: ProceduralMemoryRecord) -> Path:
        """保存当前程序记忆，同时按版本保留不可变快照。"""
        version_path = self._record_path(self.procedural_dir, record.procedure_id)
        with FileLock(self.root / ".procedure.lock"):
            if version_path.is_file():
                immutable = ProceduralMemoryRecord.parse_obj(read_json(version_path))
                ignored = {"created_at", "updated_at"}
                if immutable.dict(exclude=ignored) != record.dict(exclude=ignored):
                    raise ValueError("同一程序快照 ID 不能覆盖不同执行规则")
                record.created_at = immutable.created_at
            else:
                write_json(version_path, record)
            write_json(self._record_path(self.procedural_dir, "current"), record)
        return version_path

    def retrieve(self, query: str, robot: str, top_k: int = 4,
                 exclude_task_id: Optional[str] = None,
                 environment_version: Optional[str] = None,
                 reward_registry_version: Optional[str] = None) -> Dict[str, Any]:
        """先硬过滤机器人、环境与奖励版本，再应用相关性、可信度和时间衰减。"""
        def compatible(record: Any) -> bool:
            """指定版本时拒绝未知或不同版本，不让提示词代替确定性兼容判断。"""
            return ((environment_version is None or record.environment_version == environment_version) and
                    (reward_registry_version is None or record.reward_registry_version == reward_registry_version))

        all_records = self.records()
        trusted = {record.memory_id: record for record in all_records if self.evidence_valid(record)}
        episodic = [record for record in all_records
                    if record.status == "active" and record.memory_id in trusted and
                    record.robot == robot and compatible(record) and
                    (not exclude_task_id or record.task_id != exclude_task_id)]
        semantic = [record for record in self.semantic_records(active_only=True)
                    if (not record.robot_scope or robot in record.robot_scope) and compatible(record) and
                    record.supporting_memory_ids and all(
                        identifier in trusted and
                        trusted[identifier].environment_version == record.environment_version and
                        trusted[identifier].reward_registry_version == record.reward_registry_version
                        for identifier in record.supporting_memory_ids)]
        objects: List[Tuple[str, Any, str]] = []
        for record in episodic:
            text = " ".join([
                record.robot, record.action_name, record.normalized_goal,
                " ".join(record.lessons), record.failure_pattern or "",
                json.dumps(record.reward_terms, ensure_ascii=False),
                json.dumps(record.reward_experience or {}, ensure_ascii=False),
            ])
            objects.append(("episodic", record, text))
        for record in semantic:
            text = " ".join(record.robot_scope + record.action_scope + [record.statement, record.rule_key])
            objects.append(("semantic", record, text))
        scores = TrainingKnowledgeBase._bm25_scores(
            tokenize(query), [tokenize(item[2]) for item in objects])
        ranked: List[Tuple[float, str, Any]] = []
        now = datetime.now(timezone.utc)
        for score, (kind, record, _) in zip(scores, objects):
            if score <= 0:
                continue
            confidence = float(record.confidence)
            if kind == "episodic":
                if record.robot == robot:
                    score *= 1.20
                age_days = max(0.0, (now - self._parse_time(record.created_at)).total_seconds() / 86400.0)
                score *= math.pow(0.5, age_days / 365.0)
            elif not record.robot_scope or robot in record.robot_scope:
                score *= 1.10
            score *= 0.75 + 0.25 * confidence
            ranked.append((score, kind, record))
        ranked.sort(key=lambda item: (-item[0], getattr(item[2], "memory_id", getattr(
            item[2], "semantic_id", ""))))
        hits: List[Dict[str, Any]] = []
        used = 0
        for score, kind, record in ranked:
            if len(hits) >= max(0, top_k) or used >= self.max_context_chars:
                break
            payload = record.dict()
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            remaining = self.max_context_chars - used
            if len(encoded) > remaining:
                payload = self._compact_payload(kind, record)
                encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            if len(encoded) > remaining:
                break
            path = (self.records_dir / (record.memory_id + ".json") if kind == "episodic" else
                    self.semantic_dir / (record.semantic_id + ".json"))
            hits.append({
                "memory_type": kind,
                "source": relative_display(path, self.agent_root),
                "score": round(score, 6),
                "record": payload,
            })
            if kind == "episodic":
                record.access_count += 1
                record.last_accessed_at = utc_now()
                record.updated_at = record.last_accessed_at
                self.promote(record)
            used += len(encoded)
        return {
            "enabled": True,
            "notice": "只检索已通过门控的活跃记忆；候选语义和归档记录不得指导奖励修改。",
            "query": query,
            "records": len(episodic),
            "semantic_rules": len(semantic),
            "unverified_records": sum(record.status == "active" and record.memory_id not in trusted
                                      for record in all_records),
            "compatibility": {"robot": robot, "environment_version": environment_version,
                              "reward_registry_version": reward_registry_version},
            "hits": hits,
        }

    @staticmethod
    def _compact_payload(kind: str, record: Any) -> Dict[str, Any]:
        """在上下文预算不足时保留最关键的可追溯字段。"""
        if kind == "semantic":
            return {
                "semantic_id": record.semantic_id, "statement": record.statement,
                "robot_scope": record.robot_scope, "action_scope": record.action_scope,
                "evidence_count": record.evidence_count, "confidence": record.confidence,
            }
        experience = record.reward_experience or {}
        evolution = experience.get("reward_evolution", {}) if isinstance(experience, dict) else {}
        compact_changes = evolution.get("changes", [])[:4]
        compact_facts = experience.get("observed_facts", [])[:2]
        compact_hypotheses = experience.get("hypotheses", [])[:2]
        compact_successes = experience.get("successful_patterns", [])[:2]
        compact_failures = experience.get("failure_patterns", [])[:2]
        compact_limitations = experience.get("limitations", [])[:1]
        retained = [compact_changes, compact_facts, compact_hypotheses,
                    compact_successes, compact_failures, compact_limitations]
        referenced_ids = set()

        def collect_evidence_ids(value: Any) -> None:
            """收集压缩后保留的陈述和奖励差异所引用的证据编号。"""
            if isinstance(value, dict):
                referenced_ids.update(str(item) for item in value.get("evidence_ids", [])
                                      if item is not None)
                for child in value.values():
                    collect_evidence_ids(child)
            elif isinstance(value, list):
                for child in value:
                    collect_evidence_ids(child)

        for item in retained:
            collect_evidence_ids(item)
        evidence_index = experience.get("evidence_index", {})
        compact_evidence_index = {
            key: value for key, value in evidence_index.items() if key in referenced_ids
        } if isinstance(evidence_index, dict) else {}
        compact_experience = {
            "outcome": experience.get("outcome"),
            "evidence_index": compact_evidence_index,
            "reward_changes": compact_changes,
            "observed_facts": compact_facts,
            "hypotheses": compact_hypotheses,
            "successful_patterns": compact_successes,
            "failure_patterns": compact_failures,
            "limitations": compact_limitations,
        }
        return {
            "memory_id": record.memory_id, "task_id": record.task_id,
            "robot": record.robot, "action_name": record.action_name,
            "normalized_goal": record.normalized_goal, "outcome": record.outcome,
            "lessons": record.lessons, "failure_pattern": record.failure_pattern,
            "reward_experience": compact_experience,
            "evidence_snapshot_root": getattr(record, "evidence_snapshot_root", ""),
            "confidence": record.confidence,
        }

    def maintain(self, max_records: int, max_age_days: int,
                 min_confidence: float) -> Dict[str, Any]:
        """按年龄、可信度和容量归档情景记忆，保留原文件供审计。"""
        now = datetime.now(timezone.utc)
        active = self.records(active_only=True)
        reasons: Dict[str, str] = {}
        for record in active:
            age_days = (now - self._parse_time(record.created_at)).total_seconds() / 86400.0
            if record.confidence < min_confidence:
                reasons[record.memory_id] = "可信度低于保留阈值"
            elif age_days > max_age_days:
                reasons[record.memory_id] = "超过活跃记忆保留期限"
        survivors = [item for item in active if item.memory_id not in reasons]
        if len(survivors) > max_records:
            ranked = sorted(survivors, key=lambda item: (
                item.confidence, item.access_count, self._parse_time(item.last_accessed_at or item.created_at)))
            for record in ranked[:len(survivors) - max_records]:
                reasons[record.memory_id] = "超过活跃记忆容量，按可信度和使用频率归档"
        for record in active:
            if record.memory_id in reasons:
                record.status = "archived"
                record.archive_reason = reasons[record.memory_id]
                record.updated_at = utc_now()
                self.promote(record)
        report = {"archived": len(reasons), "reasons": reasons, "deleted": 0, "updated_at": utc_now()}
        write_json(self.root / "maintenance.json", report)
        return report

    def stats(self) -> Dict[str, Any]:
        """返回上位机可展示的四层记忆统计。"""
        records = self.records()
        semantic = self.semantic_records()
        trusted = {record.memory_id for record in records if self.evidence_valid(record)}
        active = [record for record in records if record.status == "active" and record.memory_id in trusted]
        trusted_semantic = [record for record in semantic if record.status == "active" and
                           record.supporting_memory_ids and set(record.supporting_memory_ids).issubset(trusted)]
        current = self.procedural_dir / "current.json"
        return {
            "root": relative_display(self.root, self.agent_root),
            "records": len(active),
            "episodic_active": len(active),
            "episodic_quarantined": sum(record.status == "active" and record.memory_id not in trusted for record in records),
            "episodic_archived": sum(record.status == "archived" for record in records),
            "completed": sum(record.outcome == "completed" for record in active),
            "verified_failures": sum(record.outcome == "verified_failure" for record in active),
            "semantic_active": len(trusted_semantic),
            "semantic_candidates": sum(record.status == "candidate" for record in semantic),
            "procedural_snapshot": relative_display(current, self.agent_root) if current.is_file() else None,
        }
