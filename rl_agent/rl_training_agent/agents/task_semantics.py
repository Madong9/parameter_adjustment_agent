"""在昂贵推理和 PPO 前确定性核对动作理解与任务合同，不替用户选择动作。"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from ..schemas.agent_workflow import TaskIntentSpec
from ..schemas.task import TaskSpec
from ..utils.task_semantics import RollMotionKind, classify_roll_motion, _positive_motion_text


class TaskSemanticReport(BaseModel):
    """保存语义一致性及待澄清问题；通过不代表动作物理可行或训练成功。"""

    version: str = "task-semantic-contract-v1"
    status: Literal["CONSISTENT", "NEEDS_CLARIFICATION", "CONFLICT"]
    motion_family: RollMotionKind
    issues: List[str] = Field(default_factory=list)
    question: Optional[str] = None
    source: str = "original_instruction"


class TaskSemanticValidator:
    """用户原文为唯一动作来源，拒绝否定短语命中和地面/腾空语义漂移。"""

    QUESTION = "请明确：贴地侧滚后站起，还是起跳腾空侧翻后落地？并说明方向和圈数。"

    @classmethod
    def assess_intent(cls, intent: TaskIntentSpec) -> TaskSemanticReport:
        """任务理解后立即校验，含糊侧翻不进入昂贵动作求解和奖励设计。"""
        family = classify_roll_motion(intent.original_instruction)
        if family == RollMotionKind.AMBIGUOUS:
            return TaskSemanticReport(status="NEEDS_CLARIFICATION", motion_family=family,
                issues=["原始动作未区分贴地侧滚与腾空侧翻，不允许自动选择空翻配方"],
                question=cls.QUESTION)
        text = " ".join([intent.action_name, intent.normalized_goal] + intent.required_behaviors)
        generated = classify_roll_motion(text)
        issues = []
        if family == RollMotionKind.GROUND_ROLL and cls._requires_flight(text):
            issues.append("用户要求地面侧滚，但任务理解添加了起跳/腾空要求")
        elif family == RollMotionKind.AERIAL_SIDE_FLIP and generated == RollMotionKind.GROUND_ROLL:
            issues.append("用户要求腾空侧翻，但任务理解替换为地面侧滚")
        elif family == RollMotionKind.NONE and generated in (
                RollMotionKind.GROUND_ROLL, RollMotionKind.AERIAL_SIDE_FLIP):
            issues.append("原始指令未要求侧滚/侧空翻，模型不能凭空替换动作")
        return TaskSemanticReport(status="CONFLICT" if issues else "CONSISTENT",
                                  motion_family=family, issues=issues)

    @staticmethod
    def _requires_flight(text: str) -> bool:
        """只检测肯定的起跳或腾空要求，不采信被否定的动作词。"""
        positive = _positive_motion_text(text)
        return any(token in positive for token in (
            "起跳", "腾空", "侧空翻", "airborne", "takeoff", "aerial side flip"))

    @classmethod
    def assess_task(cls, intent: TaskIntentSpec, task: TaskSpec) -> TaskSemanticReport:
        """奖励设计后校验任务语义及执行阶段，不修改原始阈值或允许接触。"""
        report = cls.assess_intent(intent)
        if report.status != "CONSISTENT":
            return report
        issues = []
        if task.original_instruction != intent.original_instruction or task.robot != intent.robot:
            issues.append("任务合同的原始指令或机器人与用户输入不一致")
        text = " ".join([task.task_name, task.normalized_description] +
                        [item.description for item in task.required_behaviors])
        phases = [phase for phase in task.phases if phase.scope == "execution"]
        flight = cls._requires_flight(text) or any(
            phase.name.upper() in ("TAKEOFF", "FLIGHT") for phase in phases)
        if report.motion_family == RollMotionKind.GROUND_ROLL:
            if flight or any(metric.required and metric.name == "feet_air_time" and
                             metric.operator in (">", ">=") and metric.value > 0
                             for metric in task.success_metrics):
                issues.append("地面侧滚合同不能增加必需的起跳、腾空或全足离地验收")
            if any(metric.required and metric.name in ("forbidden_collisions", "forbidden_contact_force")
                   and metric.operator in ("==", "<=", "<") and metric.value <= 1.0
                   for metric in task.safety_constraints):
                issues.append("地面侧滚的计划机体接触与当前全非足端禁触合同冲突；须明确接触阶段，不能自动豁免安全约束")
        elif report.motion_family == RollMotionKind.AERIAL_SIDE_FLIP:
            if not flight:
                issues.append("腾空侧翻任务合同缺少明确的起跳/腾空动作要求")
        elif classify_roll_motion(text) in (RollMotionKind.GROUND_ROLL, RollMotionKind.AERIAL_SIDE_FLIP):
            issues.append("奖励设计不能把其他动作替换为侧滚或侧空翻")
        return TaskSemanticReport(status="CONFLICT" if issues else "CONSISTENT",
                                  motion_family=report.motion_family, issues=issues)
