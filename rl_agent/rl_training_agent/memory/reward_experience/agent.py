"""实现受确定性证据边界约束的 Reward Experience Agent。"""
from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, List

from .schema import (
    ExperienceEligibility, RewardChangeSummary, RewardEvolution,
    RewardExperience, RewardExperienceEvidence, RewardExperienceNarrative, ScopedPattern,
)


NarrativeProvider = Callable[[Dict[str, Any]], Any]


class RewardExperienceAgent:
    """委托 LLM 归纳文字经验，但由本地证据决定所有结构化事实。"""

    def __init__(self, summarize: NarrativeProvider):
        """注入只负责受限文字归纳的模型调用函数。"""
        self.narrative_provider = summarize

    @staticmethod
    def _scoped_patterns(items: List[str], evidence_ids: List[str],
                         evidence: RewardExperienceEvidence) -> List[ScopedPattern]:
        """把模型提出的模式限制在当前实验的机器人、任务和环境内。"""
        applicability = {
            "robot": str(evidence.task.get("robot", "unknown")),
            "action": str(evidence.task.get("action_name", "unknown")),
            "goal": str(evidence.task.get("goal", "")),
            "environment_version": str(evidence.config_hash),
        }
        return [ScopedPattern(
            statement=str(item).strip(), applicability=applicability,
            evidence_ids=list(evidence_ids)) for item in items if str(item).strip()]

    @staticmethod
    def _narrative_payload(evidence: RewardExperienceEvidence) -> Dict[str, Any]:
        """组装不含原始日志且不授权模型改写事实的紧凑提示数据。"""
        return {
            "task_id": evidence.task_id,
            "task": evidence.task,
            "initial_reward": evidence.initial_reward,
            "final_reward": evidence.final_reward,
            "reward_changes": [item.dict() for item in evidence.reward_changes],
            "training_result": evidence.training_result,
            "visual_evaluation": evidence.visual_evaluation,
            "numeric_evaluation": evidence.numeric_evaluation,
            "failure_cases": evidence.failure_cases,
            "git_commit": evidence.git_commit,
            "config_hash": evidence.config_hash,
            "evidence_index": {key: value.dict() for key, value in evidence.evidence_index.items()},
        }

    def create(self, evidence: RewardExperienceEvidence,
               eligibility: ExperienceEligibility) -> RewardExperience:
        """获取受限模型叙述并用真实证据字段组装最终 RewardExperience。"""
        if not eligibility.eligible or eligibility.outcome == "INCONCLUSIVE":
            raise ValueError("INCONCLUSIVE 实验不能生成可写入记忆的 RewardExperience")
        raw = self.narrative_provider(self._narrative_payload(evidence))
        if isinstance(raw, RewardExperienceNarrative):
            narrative = raw
        else:
            narrative = RewardExperienceNarrative.parse_obj(raw)
        if not narrative.observed_facts:
            raise ValueError("Reward Experience must include at least one evidence-backed observed fact")
        if eligibility.outcome == "SUCCESS" and not narrative.successful_patterns:
            raise ValueError("successful experiments must include at least one locally scoped pattern")
        if eligibility.outcome == "VERIFIED_FAILURE" and not narrative.failure_patterns:
            raise ValueError("verified failures must include at least one failure pattern")
        change_keys = {(item.reward_name, item.reward_version) for item in evidence.reward_changes}
        change_name_counts = {name: sum(1 for item in evidence.reward_changes if item.reward_name == name)
                              for name, _ in change_keys}
        for item in narrative.reward_change_observations:
            key = (item.reward_name, item.reward_version)
            if key not in change_keys and not (
                    item.reward_version is None and change_name_counts.get(item.reward_name) == 1):
                raise ValueError("reward_change_observations does not match a real reward diff/version")
        all_evidence_ids = sorted(evidence.evidence_index)
        change_summaries: List[RewardChangeSummary] = []
        for change in evidence.reward_changes:
            observations = [item for item in narrative.reward_change_observations
                            if item.reward_name == change.reward_name and
                            (item.reward_version == change.reward_version or
                             (item.reward_version is None and change_name_counts.get(item.reward_name) == 1))]
            behavior_change = "；".join(item.observed_behavior_change for item in observations)
            behavior_evidence_ids = [item for observation in observations
                                     for item in observation.evidence_ids]
            change_summaries.append(RewardChangeSummary(
                reward_name=change.reward_name,
                before=change.before,
                after=change.after,
                iteration=change.iteration,
                reward_version=change.reward_version,
                observed_behavior_change=behavior_change,
                evidence_ids=sorted(set(change.evidence_ids + behavior_evidence_ids)),
            ))
        task_context = {
            **evidence.task,
            "git_commit": evidence.git_commit,
            "environment_version": evidence.config_hash,
            "training_result": evidence.training_result,
            "visual_evaluation": evidence.visual_evaluation,
            "numeric_evaluation": evidence.numeric_evaluation,
        }
        experiment_id = str(evidence.training_result.get("selected_experiment", "unknown"))
        experience_key = "%s|%s|%s|%s|%s" % (
            evidence.task_id, experiment_id, eligibility.outcome,
            evidence.git_commit, evidence.config_hash)
        experience_id = "experience-" + hashlib.sha1(experience_key.encode("utf-8")).hexdigest()[:12]
        patterns_evidence = sorted(set(
            list(evidence.task.get("evidence_ids", [])) +
            list(evidence.initial_reward.get("evidence_ids", [])) +
            list(evidence.final_reward.get("evidence_ids", [])) +
            list(evidence.numeric_evaluation.get("evidence_ids", [])) +
            list(evidence.visual_evaluation.get("evidence_ids", [])) +
            [item for change in evidence.reward_changes for item in change.evidence_ids]
        )) or all_evidence_ids
        seed_count = len(set(evidence.training_result.get("seeds", [])))
        confidence = min(0.65, 0.55 + 0.05 * max(0, min(seed_count - 1, 2)))
        return RewardExperience(
            experience_id=experience_id,
            outcome=eligibility.outcome,
            task_context=task_context,
            reward_evolution=RewardEvolution(
                initial_design=evidence.initial_reward,
                final_design=evidence.final_reward,
                changes=change_summaries,
            ),
            evidence_index=evidence.evidence_index,
            observed_facts=narrative.observed_facts,
            hypotheses=narrative.hypotheses,
            successful_patterns=self._scoped_patterns(
                narrative.successful_patterns if eligibility.outcome == "SUCCESS" else [],
                patterns_evidence, evidence),
            failure_patterns=self._scoped_patterns(
                narrative.failure_patterns if eligibility.outcome == "VERIFIED_FAILURE" else [],
                patterns_evidence, evidence),
            limitations=narrative.limitations,
            confidence=confidence,
        )
