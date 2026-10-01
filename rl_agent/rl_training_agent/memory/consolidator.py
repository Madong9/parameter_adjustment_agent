"""把多条独立情景证据整理并升级为可复用语义规律。"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Tuple

from ..schemas.agent_workflow import LongTermMemoryRecord, SemanticMemoryRecord
from ..utils.io import utc_now
from .store import LongTermMemoryStore


class SemanticMemoryConsolidatorAgent:
    """使用确定性规则形成语义候选，达到独立证据阈值后才激活。"""

    @staticmethod
    def _candidate(record: LongTermMemoryRecord) -> Optional[Tuple[str, str]]:
        """从一条已验证情景中抽取保守且可追溯的规律候选。"""
        rewards = {str(item.get("name")) for item in record.reward_terms}
        action_text = record.action_name + " " + record.normalized_goal
        if record.outcome == "verified_failure" and record.failure_pattern:
            key = "failure:%s:%s" % (record.robot, record.failure_pattern)
            return key, "%s 在%s任务中应规避已验证失败模式：%s。" % (
                record.robot, record.action_name, record.failure_pattern)
        if ("后腿" in action_text or "rear_leg" in action_text) and {
                "rear_leg_stand", "rear_leg_walk"}.issubset(rewards):
            return (
                "rear-leg-posture-gating",
                "后腿站立行走应同时使用 rear_leg_stand 与 rear_leg_walk 姿态门控，"
                "并避免普通平躯干 orientation 奖励与目标俯仰姿态冲突。",
            )
        if ("前腿" in action_text or "front_leg" in action_text) and {
                "front_leg_stand", "front_leg_walk"}.issubset(rewards):
            return (
                "front-leg-support-gating",
                "前腿支撑站立行走应联合使用 front_leg_stand 与 front_leg_walk，"
                "并明确要求前足支撑、后足离地。",
            )
        return None

    def consolidate(self, store: LongTermMemoryStore, record: LongTermMemoryRecord,
                    min_support: int) -> Optional[SemanticMemoryRecord]:
        """合并同类证据；只有不同任务达到阈值时才把候选升级为活跃规律。"""
        candidate = self._candidate(record)
        if candidate is None:
            return None
        rule_key, statement = candidate
        semantic_id = "semantic-" + hashlib.sha1(rule_key.encode("utf-8")).hexdigest()[:12]
        existing: Optional[SemanticMemoryRecord] = next(
            (item for item in store.semantic_records() if item.semantic_id == semantic_id), None)
        now = utc_now()
        if existing is None:
            existing = SemanticMemoryRecord(
                semantic_id=semantic_id,
                rule_key=rule_key,
                statement=statement,
                created_at=now,
                updated_at=now,
            )
        memories = set(existing.supporting_memory_ids)
        tasks = set(existing.supporting_task_ids)
        memories.add(record.memory_id)
        tasks.add(record.task_id)
        existing.supporting_memory_ids = sorted(memories)
        existing.supporting_task_ids = sorted(tasks)
        existing.robot_scope = sorted(set(existing.robot_scope + [record.robot]))
        existing.action_scope = sorted(set(existing.action_scope + [record.action_name]))
        existing.evidence_count = len(tasks)
        existing.confidence = min(0.99, 0.45 + 0.15 * existing.evidence_count)
        existing.status = "active" if existing.evidence_count >= min_support else "candidate"
        existing.updated_at = now
        store.save_semantic(existing)
        if existing.status == "active" and record.outcome == "verified_failure":
            rewards = {str(item.get("name")) for item in record.reward_terms}
            contradicted = []
            if {"rear_leg_stand", "rear_leg_walk"}.issubset(rewards):
                contradicted.append("rear-leg-posture-gating")
            if {"front_leg_stand", "front_leg_walk"}.issubset(rewards):
                contradicted.append("front-leg-support-gating")
            for semantic in store.semantic_records(active_only=True):
                if semantic.rule_key not in contradicted:
                    continue
                semantic.counterevidence_memory_ids = sorted(set(
                    semantic.counterevidence_memory_ids + [record.memory_id]))
                semantic.status = "superseded"
                semantic.superseded_by = existing.semantic_id
                semantic.archive_reason = "已被达到证据阈值的明确失败模式反驳"
                semantic.updated_at = now
                store.save_semantic(semantic)
        return existing
