"""实现多随机种子视觉样本选择、保守聚合和置信度校准。"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from ..schemas.visual import VisualBehaviorReport


class ConservativeVisualAggregator:
    """选择最差、中央和最好 rollout，并以最差证据决定视觉验收。"""

    @staticmethod
    def select(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从按分数任意排列的记录中选取互不重复的最差、中央和最好样本。"""
        if not records:
            return []
        ranked = sorted(records, key=lambda item: (float(item.get("score", 0.0)),
                                                    str(item.get("seed", ""))))
        indices = sorted(set((0, len(ranked) // 2, len(ranked) - 1)))
        return [ranked[index] for index in indices]

    @staticmethod
    def aggregate(reports: Sequence[VisualBehaviorReport],
                  rollout_names: Sequence[str]) -> VisualBehaviorReport:
        """使用全通过和最低分规则聚合报告，并按跨样本一致性校准可信度。"""
        if not reports:
            raise ValueError("at least one visual report is required")
        successes = [bool(item.visual_success) for item in reports]
        majority = max(sum(successes), len(successes) - sum(successes))
        agreement = majority / float(len(successes))
        worst = min(reports, key=lambda item: (item.alignment_score, item.confidence))
        failures = []
        unintended: List[str] = []
        uncertain: List[str] = []
        findings = []
        evidence_frames: List[int] = []
        for report in reports:
            failures.extend(report.failure_modes)
            unintended.extend(report.unintended_behaviors)
            uncertain.extend(report.uncertain_items)
            findings.extend(report.evidence_findings)
            evidence_frames.extend(report.evidence_frames)
        calibrated_confidence = min(item.confidence for item in reports) * agreement
        requires_review = any(item.requires_human_review for item in reports) or agreement < 1.0
        return VisualBehaviorReport(
            visual_success=all(successes) and not requires_review,
            alignment_score=min(item.alignment_score for item in reports),
            confidence=calibrated_confidence,
            summary="保守聚合%d个视觉样本；最差样本结论：%s" % (len(reports), worst.summary),
            phase_results=worst.phase_results,
            failure_modes=failures,
            unintended_behaviors=sorted(set(unintended)),
            evidence_frames=sorted(set(evidence_frames)),
            evidence_findings=findings,
            uncertain_items=sorted(set(uncertain)),
            requires_human_review=requires_review,
            aggregation_method="worst_median_best_conservative",
            evaluated_rollouts=list(rollout_names),
            agreement_score=agreement,
        )
