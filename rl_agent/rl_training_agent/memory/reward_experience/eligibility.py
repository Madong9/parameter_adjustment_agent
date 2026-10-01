"""对实验产物执行不依赖 LLM 的奖励经验资格门控。"""
from __future__ import annotations

from typing import Dict

from .schema import ExperienceEligibility, RewardExperienceEvidence


class ExperienceEligibilityChecker:
    """只允许训练和验收证据完整的成功或明确失败进入经验总结。"""

    def check(self, evidence: RewardExperienceEvidence) -> ExperienceEligibility:
        """依据任务终态、训练产物、数值与视觉结果判定经验结局。"""
        training = evidence.training_result
        numeric = evidence.numeric_evaluation
        visual = evidence.visual_evaluation
        evaluation_ids = numeric.get("evidence_ids", [])
        visual_ids = visual.get("evidence_ids", [])
        training_ids = training.get("evidence_ids", [])
        try:
            iterations = int(training.get("iteration", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            iterations = 0
        base_checks: Dict[str, bool] = {
            "real_training": not bool(training.get("dry_run")),
            "ppo_completed": training.get("training_result") == "completed" and
                             iterations > 0,
            "checkpoint_present": bool(training.get("checkpoint_exists")),
            "task_evidence_present": bool(evidence.task.get("evidence_ids")),
            "reward_configs_present": bool(evidence.initial_reward.get("evidence_ids")) and
                                      bool(evidence.final_reward.get("evidence_ids")),
            "evaluation_evidence_present": len(evaluation_ids) >= 2 and bool(visual_ids),
            "visual_file_present": bool(visual.get("visual_evidence_id")),
            "provider_healthy": not bool(training.get("provider_error")),
            "robot_known": evidence.task.get("robot", "unknown") != "unknown",
            "task_known": bool(evidence.task.get("goal")),
            "manifest_evidence_present": bool(training_ids),
        }
        if training.get("provider_error"):
            return ExperienceEligibility(
                eligible=False, reason="Provider 失败属于 INCONCLUSIVE，只保存原始实验证据",
                outcome="INCONCLUSIVE", checks=base_checks)
        if training.get("dry_run"):
            return ExperienceEligibility(
                eligible=False, reason="dry-run 不是真实 PPO 实验，不能总结为长期经验",
                outcome="INCONCLUSIVE", checks=base_checks)

        success_checks = dict(base_checks)
        success_checks.update({
            "completed_terminal_state": training.get("state") == "COMPLETED",
            "completed_result_recorded": training.get("result") == "completed",
            "numeric_task_passed": bool(numeric.get("task_metrics_passed")),
            "safety_constraints_passed": bool(numeric.get("hard_constraints_passed")),
            "visual_passed": bool(visual.get("evaluation_alignment_passed")) and
                             bool(visual.get("visual_evidence_id")) and
                             bool(visual.get("visual_success")) and
                             not bool(visual.get("requires_human_review")) and
                             float(visual.get("alignment_score", 0.0) or 0.0) >= 0.7,
            "no_evaluation_violations": not bool(numeric.get("violations")),
        })
        success_required = (
            "real_training", "ppo_completed", "checkpoint_present", "task_evidence_present",
            "reward_configs_present", "evaluation_evidence_present", "provider_healthy",
            "robot_known", "task_known", "manifest_evidence_present", "completed_terminal_state",
            "numeric_task_passed", "safety_constraints_passed", "visual_passed",
            "no_evaluation_violations", "completed_result_recorded",
        )
        if all(success_checks.get(name, False) for name in success_required):
            return ExperienceEligibility(
                eligible=True, reason="真实 PPO、数值任务指标、视觉评价和安全约束全部通过",
                outcome="SUCCESS", checks=success_checks)

        failed_metrics = [
            item for item in numeric.get("evaluation_metrics", [])
            if isinstance(item, dict) and item.get("passed") is False
        ]
        has_metric_failure = bool(failed_metrics or numeric.get("violations"))
        visual_failures = visual.get("failure_modes", [])
        numeric_metrics = numeric.get("metrics", {}) if isinstance(numeric.get("metrics", {}), dict) else {}
        numeric_behavior = any(
            item.get("name") in numeric_metrics and
            numeric_metrics.get(item.get("name")) == item.get("value")
            for item in failed_metrics
        )
        visual_behavior = bool(
            visual.get("visual_success") is False or visual_failures or
            visual.get("unintended_behaviors") or numeric_behavior
        )
        failure_checks = dict(base_checks)
        failure_checks.update({
            "failed_terminal_state": training.get("state") == "FAILED",
            "failed_result_recorded": training.get("result") == "failed",
            "evaluation_indicates_failure": numeric.get("completed") is False,
            "explicit_failure_metric": has_metric_failure,
            "failure_behavior_evidence": visual_behavior,
            "evaluation_artifacts_present": bool(evaluation_ids),
            "visual_file_present": bool(visual.get("visual_evidence_id")),
        })
        failure_required = (
            "real_training", "ppo_completed", "checkpoint_present", "task_evidence_present",
            "reward_configs_present", "provider_healthy", "failed_terminal_state",
            "failed_result_recorded", "evaluation_indicates_failure",
            "explicit_failure_metric", "failure_behavior_evidence", "evaluation_artifacts_present",
            "visual_file_present",
        )
        if all(failure_checks.get(name, False) for name in failure_required):
            return ExperienceEligibility(
                eligible=True, reason="真实 PPO 已完成，且存在失败指标和对应行为证据",
                outcome="VERIFIED_FAILURE", checks=failure_checks)

        return ExperienceEligibility(
            eligible=False,
            reason="证据不足以确认成功或可复现失败；中断、Provider/格式错误等归为 INCONCLUSIVE",
            outcome="INCONCLUSIVE", checks={**base_checks, **{
                "explicit_failure_metric": has_metric_failure,
                "failure_behavior_evidence": visual_behavior,
            }})
