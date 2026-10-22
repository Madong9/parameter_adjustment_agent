"""在昂贵推理和 PPO 前确定性核对动作理解与任务合同，不替用户选择动作。"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ..schemas.agent_workflow import TaskIntentSpec
from ..schemas.task import TaskSpec
from ..utils.task_semantics import (ForwardFlipKind, RollMotionKind, classify_forward_flip,
                                    classify_roll_motion, _positive_motion_text)
from ..utils.units import is_dimensionless_unit


class TaskSemanticReport(BaseModel):
    """保存语义一致性及待澄清问题；通过不代表动作物理可行或训练成功。"""

    version: str = "task-semantic-contract-v2"
    status: Literal["CONSISTENT", "NEEDS_CLARIFICATION", "CONFLICT"]
    motion_family: RollMotionKind
    forward_flip_family: ForwardFlipKind = ForwardFlipKind.NONE
    issues: List[str] = Field(default_factory=list)
    question: Optional[str] = None
    source: str = "original_instruction"


class TaskSemanticValidator:
    """用户原文为唯一动作来源，拒绝否定短语命中和地面/腾空语义漂移。"""

    QUESTION = "请明确：贴地侧滚后站起，还是起跳腾空侧翻后落地？并说明方向和圈数。"
    FORWARD_QUESTION = "请明确：允许机体触地的前滚翻，还是起跳腾空前空翻后足端落地？并说明圈数和结束姿态。"

    @classmethod
    def assess_intent(cls, intent: TaskIntentSpec) -> TaskSemanticReport:
        """任务理解后立即校验，含糊前翻或侧翻不进入动作求解和奖励设计。"""
        family = classify_roll_motion(intent.original_instruction)
        forward = classify_forward_flip(intent.original_instruction)
        if forward == ForwardFlipKind.AMBIGUOUS:
            return TaskSemanticReport(status="NEEDS_CLARIFICATION", motion_family=family,
                forward_flip_family=forward,
                issues=["原始动作未区分地面前滚翻与腾空前空翻，不允许自动选择接触或旋转合同"],
                question=cls.FORWARD_QUESTION)
        if family == RollMotionKind.AMBIGUOUS:
            return TaskSemanticReport(status="NEEDS_CLARIFICATION", motion_family=family,
                issues=["原始动作未区分贴地侧滚与腾空侧翻，不允许自动选择空翻配方"],
                question=cls.QUESTION)
        text = " ".join([intent.action_name, intent.normalized_goal] + intent.required_behaviors)
        generated = classify_roll_motion(text)
        generated_forward = classify_forward_flip(text)
        issues = []
        if forward != ForwardFlipKind.NONE and generated != RollMotionKind.NONE:
            issues.append("用户要求前向翻转，但任务理解替换为侧向滚转")
        if family != RollMotionKind.NONE and generated_forward != ForwardFlipKind.NONE:
            issues.append("用户要求侧向滚转，但任务理解替换为前向翻转")
        if forward == ForwardFlipKind.GROUND and cls._requires_flight(text):
            issues.append("用户要求地面前滚翻，但任务理解添加了起跳/腾空要求")
        elif forward == ForwardFlipKind.AERIAL and generated_forward == ForwardFlipKind.GROUND:
            issues.append("用户要求腾空前空翻，但任务理解替换为地面前滚翻")
        elif forward == ForwardFlipKind.NONE and generated_forward != ForwardFlipKind.NONE:
            issues.append("原始指令未要求前翻，模型不能凭空替换动作")
        if family == RollMotionKind.GROUND_ROLL and cls._requires_flight(text):
            issues.append("用户要求地面侧滚，但任务理解添加了起跳/腾空要求")
        elif family == RollMotionKind.AERIAL_SIDE_FLIP and generated == RollMotionKind.GROUND_ROLL:
            issues.append("用户要求腾空侧翻，但任务理解替换为地面侧滚")
        elif family == RollMotionKind.NONE and generated in (
                RollMotionKind.GROUND_ROLL, RollMotionKind.AERIAL_SIDE_FLIP):
            issues.append("原始指令未要求侧滚/侧空翻，模型不能凭空替换动作")
        return TaskSemanticReport(status="CONFLICT" if issues else "CONSISTENT",
                                  motion_family=family, forward_flip_family=forward, issues=issues)

    @staticmethod
    def action_capability_gaps(intent: TaskIntentSpec, capabilities: Dict[str, Any]) -> List[str]:
        """核对前翻需要的轴向信号与验收实现，现有 roll 代理不能冒充支持。"""
        family = classify_forward_flip(intent.original_instruction)
        rewards = {item.get("name") for item in capabilities.get("rewards", [])}
        metrics = set(capabilities.get("evaluation_metrics", []))
        if family == ForwardFlipKind.AERIAL:
            gaps = []
            for name in ('net_pitch_progress', 'pitch_rotation', 'front_flip_recovery'):
                if name not in rewards:
                    gaps.append("reward_signal:" + name)
            for name in ('final_pitch_angle', 'pitch_axis_error'):
                if name not in metrics:
                    gaps.append("evaluation:" + name)
            return gaps
        if family == ForwardFlipKind.GROUND:
            return ["contact_contract:planned_body_contacts（前滚翻的计划机体接触阶段）",
                    "reward_signal:ground_forward_rotation（地面前滚翻进度）",
                    "evaluation:ground_forward_rotation（前滚翻完成与恢复）"]
        return []

    @staticmethod
    def _requires_flight(text: str) -> bool:
        """只检测肯定的起跳或腾空要求，不采信被否定的动作词。"""
        positive = _positive_motion_text(text)
        return any(token in positive for token in (
            "起跳", "腾空", "在空中", "侧空翻", "前空翻", "airborne", "takeoff", "aerial side flip", "front flip", "frontflip"))

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
        generated_forward = classify_forward_flip(text)
        if report.forward_flip_family != ForwardFlipKind.NONE:
            if classify_roll_motion(text) != RollMotionKind.NONE:
                issues.append("前向翻转任务合同不能替换为侧向滚转")
            if any(metric.required and metric.name in ("final_roll_angle", "max_roll_velocity")
                   for metric in task.success_metrics):
                issues.append("前向翻转不能使用 roll 轴净滚转角或纵轴角速度作为完成验收")
            if report.forward_flip_family == ForwardFlipKind.AERIAL and not flight:
                issues.append("腾空前翻任务合同缺少明确的起跳/腾空动作要求")
            if report.forward_flip_family == ForwardFlipKind.AERIAL:
                required = {metric.name: metric for metric in task.success_metrics if metric.required}
                missing = {'final_pitch_angle', 'pitch_axis_error', 'landing_stability', 'stable_stand_duration'} - set(required)
                if missing:
                    issues.append("前空翻完成合同缺少必选旋转、轴向或站稳证据：" + "、".join(sorted(missing)))
                angle = required.get('final_pitch_angle')
                if angle is not None and (angle.unit != 'rad' or angle.operator not in ('>', '>=') or angle.value <= 0):
                    issues.append("前空翻必须验收正向横轴净旋转，使用 rad 单位和正目标下界")
                axis = required.get('pitch_axis_error')
                if axis is not None and (axis.operator not in ('<', '<=') or not 0 <= axis.value < 1):
                    issues.append("前空翻必须独立约束非横轴旋转比例，使用小于 1 的无量纲上界")
                if axis is not None and not is_dimensionless_unit(axis.unit):
                    issues.append("前空翻非横轴旋转比例的单位不兼容：%r；应使用 ratio、1 或 dimensionless 等无量纲单位" % axis.unit)
                for name in ('landing_stability', 'stable_stand_duration'):
                    threshold = required.get(name)
                    if threshold is not None and (threshold.operator not in ('>', '>=') or threshold.value <= 0):
                        issues.append("前空翻落地站稳必须设置正值必选下界：" + name)
                for threshold in task.safety_constraints:
                    if (threshold.required and threshold.operator in ('<', '<=', '==') and
                            ((threshold.name in ('max_abs_pitch', 'pitch_limit') and threshold.value < 1.5707963267948966) or
                             (threshold.name in ('max_abs_roll', 'roll_limit') and threshold.value < 3.141592653589793))):
                        issues.append("完整前空翻与全程欧拉姿态硬约束冲突，不能删除原始硬约束放行：" + threshold.name)
            if report.forward_flip_family == ForwardFlipKind.GROUND and flight:
                issues.append("地面前滚翻合同不能增加必需的起跳/腾空要求")
        elif generated_forward != ForwardFlipKind.NONE:
            issues.append("奖励设计不能把其他动作替换为前翻")
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
                                  motion_family=report.motion_family,
                                  forward_flip_family=report.forward_flip_family, issues=issues)
