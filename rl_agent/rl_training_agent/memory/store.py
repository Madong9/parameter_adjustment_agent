"""实现工作、情景、语义和程序四层训练记忆的持久化与检索。"""
from __future__ import annotations

import json
import math
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
                record = LongTermMemoryRecord.parse_obj(read_json(path))
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
                record = SemanticMemoryRecord.parse_obj(read_json(path))
                if not active_only or record.status == "active":
                    values.append(record)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return values

    def promote(self, record: LongTermMemoryRecord) -> Path:
        """原子保存已通过晋升门控的情景记忆。"""
        path = self.records_dir / (record.memory_id + ".json")
        write_json(path, record)
        return path

    def save_semantic(self, record: SemanticMemoryRecord) -> Path:
        """保存候选或活跃语义规律，不覆盖其证据关系。"""
        path = self.semantic_dir / (record.semantic_id + ".json")
        write_json(path, record)
        return path

    def save_working(self, task_dir: Path, snapshot: WorkingMemorySnapshot) -> Path:
        """把当前任务工作记忆写入任务目录，而非全局可信经验库。"""
        path = task_dir / "memory" / "working_memory.json"
        write_json(path, snapshot)
        return path

    def save_procedural(self, record: ProceduralMemoryRecord) -> Path:
        """保存当前程序记忆，同时按版本保留不可变快照。"""
        version_path = self.procedural_dir / (record.procedure_id + ".json")
        if version_path.is_file():
            immutable = ProceduralMemoryRecord.parse_obj(read_json(version_path))
            record.created_at = immutable.created_at
        else:
            write_json(version_path, record)
        write_json(self.procedural_dir / "current.json", record)
        return version_path

    def retrieve(self, query: str, robot: str, top_k: int = 4,
                 exclude_task_id: Optional[str] = None) -> Dict[str, Any]:
        """联合检索活跃情景与语义记忆，并应用可信度和时间衰减。"""
        episodic = [record for record in self.records(active_only=True)
                    if not exclude_task_id or record.task_id != exclude_task_id]
        semantic = self.semantic_records(active_only=True)
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
        current = self.procedural_dir / "current.json"
        return {
            "root": relative_display(self.root, self.agent_root),
            "records": sum(record.status == "active" for record in records),
            "episodic_active": sum(record.status == "active" for record in records),
            "episodic_archived": sum(record.status == "archived" for record in records),
            "completed": sum(record.outcome == "completed" and record.status == "active" for record in records),
            "verified_failures": sum(record.outcome == "verified_failure" and record.status == "active"
                                     for record in records),
            "semantic_active": sum(record.status == "active" for record in semantic),
            "semantic_candidates": sum(record.status == "candidate" for record in semantic),
            "procedural_snapshot": relative_display(current, self.agent_root) if current.is_file() else None,
        }
