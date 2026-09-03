"""把结构化训练诊断转换为下一版本的受限奖励计划。"""
from __future__ import annotations

from typing import Dict, Iterable, List

from ..environment.capability_manifest import RewardRegistryItem
from ..schemas.decisions import TrainingDiagnosis
from ..schemas.rewards import CurriculumStage, RewardPlan, RewardTerm
from .validator import RewardPlanValidator, RewardValidationError


TERM_UPDATE_FIELDS = {
    "weight", "parameters", "active_phases", "activation_condition", "normalization",
    "expected_training_trend", "purpose", "reward_hacking_risks", "failure_modes_addressed",
}


class RewardPlanReviser:
    """仅允许通过已注册奖励和白名单字段修订奖励计划。"""

    def __init__(self, registry: Iterable[RewardRegistryItem], max_abs_weight: float = 100.0):
        """保存奖励注册表并创建最终安全校验器。"""
        self.registry: Dict[str, RewardRegistryItem] = {item.name: item for item in registry}
        self.validator = RewardPlanValidator(self.registry.values(), max_abs_weight)

    def _new_term(self, name: str, changes: Dict[str, object]) -> RewardTerm:
        """根据注册表和诊断字段构造一个完整的新奖励项。"""
        if name not in self.registry:
            raise RewardValidationError("diagnosis requested unregistered reward: %s" % name)
        item = self.registry[name]
        weight = float(changes.get("weight", item.default_weight))
        if weight == 0.0:
            weight = -1.0 if item.sign == "negative" else 1.0
        return RewardTerm(
            name=name,
            implementation=item.implementation,
            purpose=str(changes.get("purpose", "诊断新增的环境奖励：%s" % name)),
            weight=weight,
            parameters=dict(changes.get("parameters", {})),
            active_phases=list(changes.get("active_phases", ["all"])),
            activation_condition=changes.get("activation_condition"),
            normalization=str(changes.get("normalization", "environment_dt")),
            expected_raw_range=tuple(item.expected_raw_range),
            expected_training_trend=str(changes.get("expected_training_trend", "improve")),
            dependencies=list(item.dependencies),
            failure_modes_addressed=list(changes.get("failure_modes_addressed", [])),
            reward_hacking_risks=list(changes.get("reward_hacking_risks", [])),
        )

    @staticmethod
    def _updated_weight(current: float, changes: Dict[str, object]) -> float:
        """支持绝对权重、增量和倍率三种互斥的常见修订表达。"""
        if "weight" in changes:
            return float(changes["weight"])
        if "weight_delta" in changes:
            return current + float(changes["weight_delta"])
        if "weight_multiplier" in changes:
            return current * float(changes["weight_multiplier"])
        return current

    def _apply_reward_changes(self, plan: RewardPlan, diagnosis: TrainingDiagnosis,
                              audit: List[str]) -> None:
        """按顺序应用奖励新增、删除和白名单更新。"""
        for change in diagnosis.reward_changes:
            index = next((i for i, term in enumerate(plan.terms) if term.name == change.term), None)
            if change.action == "remove":
                if index is not None:
                    plan.terms.pop(index)
                    audit.append("删除奖励项：%s" % change.term)
                continue
            if change.action == "add":
                if index is None:
                    plan.terms.append(self._new_term(change.term, change.changes))
                    audit.append("新增奖励项：%s" % change.term)
                else:
                    audit.append("奖励项已存在，按更新处理：%s" % change.term)
            index = next((i for i, term in enumerate(plan.terms) if term.name == change.term), None)
            if index is None:
                raise RewardValidationError("diagnosis requested update for missing reward: %s" % change.term)
            term = plan.terms[index]
            unknown = set(change.changes) - TERM_UPDATE_FIELDS - {"weight_delta", "weight_multiplier"}
            if unknown:
                raise RewardValidationError("unsupported reward change fields: %s" % ", ".join(sorted(unknown)))
            term.weight = self._updated_weight(term.weight, change.changes)
            for field in TERM_UPDATE_FIELDS - {"weight"}:
                if field in change.changes:
                    value = change.changes[field]
                    if field == "parameters":
                        merged = dict(term.parameters)
                        merged.update(dict(value))
                        value = merged
                    setattr(term, field, value)
            audit.append("更新奖励项：%s" % change.term)

    @staticmethod
    def _apply_curriculum_changes(plan: RewardPlan, diagnosis: TrainingDiagnosis,
                                  audit: List[str]) -> None:
        """把诊断中的课程修改合并到指定阶段的边界或参数变化。"""
        for change in diagnosis.curriculum_changes:
            stage = next((item for item in plan.curriculum if item.name == change.stage), None)
            if stage is None:
                stage = CurriculumStage(name=change.stage, start_iteration=0, end_iteration=0,
                                        parameter_changes={})
                plan.curriculum.append(stage)
            updates = dict(change.changes)
            if "start_iteration" in updates:
                stage.start_iteration = int(updates.pop("start_iteration"))
            if "end_iteration" in updates:
                stage.end_iteration = int(updates.pop("end_iteration"))
            nested = updates.pop("parameter_changes", {})
            stage.parameter_changes.update(dict(nested))
            stage.parameter_changes.update(updates)
            audit.append("更新课程阶段：%s" % change.stage)

    def revise(self, parent: RewardPlan, diagnosis: TrainingDiagnosis) -> tuple:
        """生成、校验并返回下一版本奖励计划及其审计记录。"""
        revised = parent.copy(deep=True)
        revised.parent_version = parent.version
        revised.version = parent.version + 1
        audit: List[str] = []
        self._apply_reward_changes(revised, diagnosis, audit)
        self._apply_curriculum_changes(revised, diagnosis, audit)
        # 赋值发生在已有 Pydantic 对象上，重新解析一次以触发嵌套字段和课程边界校验。
        revised = RewardPlan.parse_obj(revised.dict())
        self.validator.validate(revised)
        return revised, audit
