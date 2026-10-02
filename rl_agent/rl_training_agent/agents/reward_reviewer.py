"""实现独立于奖励生成模型的确定性奖励审查 Agent。"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Set

from ..schemas.agent_workflow import RewardCandidateReview, RewardReviewReport, TaskIntentSpec
from ..schemas.rewards import RewardPlan
from ..schemas.task import TaskSpec
from ..utils.task_semantics import is_front_leg_support_text


class RewardReviewAgent:
    """在训练前检查遗漏、冲突、非法奖励和奖励投机风险。"""

    @staticmethod
    def _rear_leg_task(intent: TaskIntentSpec, task: TaskSpec) -> bool:
        """判断动作是否明确要求后腿支撑。"""
        return "后腿" in intent.original_instruction or any(
            "rear_leg" in item.name for item in task.required_behaviors)

    @staticmethod
    def _front_leg_task(intent: TaskIntentSpec) -> bool:
        """判断动作是否明确要求前腿支撑而非抬起前腿。"""
        return is_front_leg_support_text((intent.original_instruction,))

    def review(self, intent: TaskIntentSpec, task: TaskSpec, plans: Sequence[RewardPlan],
               capabilities: Dict[str, Any], global_risks: Sequence[str]) -> RewardReviewReport:
        """逐个审查候选；坏候选被隔离，其余候选仍可进入训练。"""
        omissions: List[str] = []
        conflicts: List[str] = []
        risks: Set[str] = {str(item) for item in global_risks if str(item).strip()}
        candidate_results: List[RewardCandidateReview] = []
        registered = {item.get("name") for item in capabilities.get("rewards", [])}
        required_metrics = {item.name for item in task.success_metrics if item.required}
        if not plans:
            omissions.append("奖励设计没有候选方案")
        for index, plan in enumerate(plans, start=1):
            candidate_omissions: List[str] = []
            candidate_conflicts: List[str] = []
            names = {item.name for item in plan.terms}
            unknown = sorted(name for name in names if name not in registered)
            if unknown:
                candidate_conflicts.append("包含未注册奖励：%s" % ", ".join(unknown))
            provided_metrics = {item.name for item in plan.success_metrics if item.required}
            missing_metrics = sorted(required_metrics - provided_metrics)
            if missing_metrics:
                candidate_omissions.append("缺少验收指标：%s" % ", ".join(missing_metrics))
            for term in plan.terms:
                risks.update(str(item) for item in term.reward_hacking_risks if str(item).strip())
            if self._rear_leg_task(intent, task):
                missing = sorted({"rear_leg_stand", "rear_leg_walk"} - names)
                if missing:
                    candidate_omissions.append("缺少后腿门控奖励：%s" % ", ".join(missing))
                if "orientation" in names:
                    candidate_conflicts.append("水平 orientation 与后腿目标俯仰角冲突")
            if self._front_leg_task(intent):
                missing = sorted({"front_leg_stand", "front_leg_walk"} - names)
                if missing:
                    candidate_omissions.append("缺少前腿门控奖励：%s" % ", ".join(missing))
            candidate_results.append(RewardCandidateReview(
                candidate_index=index,
                approved=not candidate_omissions and not candidate_conflicts,
                omissions=candidate_omissions,
                conflicts=candidate_conflicts,
            ))
            omissions.extend("候选%d%s" % (index, reason) for reason in candidate_omissions)
            conflicts.extend("候选%d%s" % (index, reason) for reason in candidate_conflicts)
        if not risks:
            omissions.append("奖励设计没有记录任何奖励投机风险（所有候选均不予准入）")
            for candidate in candidate_results:
                candidate.approved = False
                candidate.omissions.append("没有记录任何奖励投机风险")
        passed = [item.candidate_index for item in candidate_results if item.approved]
        rejected = [item.candidate_index for item in candidate_results if not item.approved]
        return RewardReviewReport(
            approved=bool(passed),
            omissions=omissions,
            conflicts=conflicts,
            reward_hacking_risks=sorted(risks),
            checked_candidates=len(plans),
            candidate_results=candidate_results,
            passed_candidate_indexes=passed,
            rejected_candidate_indexes=rejected,
            notes=["逐候选审查；仅隔离不合格候选，审查 Agent 不修改奖励方案。"],
        )
