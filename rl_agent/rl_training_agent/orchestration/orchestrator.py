from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ..agents.context_builder import ContextBuilder
from ..agents.prompt_compiler import RewardPromptCompiler
from ..agents.reward_reviewer import RewardReviewAgent
from ..agents.task_semantics import TaskSemanticValidator
from ..environment.inspector import EnvironmentInspector
from ..environment.metric_registry import normalize_task_metrics
from ..environment.project_adapter import UnitreeProjectAdapter
from ..feasibility.agent import FeasibilityPipeline
from ..feasibility.admission import (TrainingAdmissionPolicy, TrainingAdmissionReport,
                                    check_environment_health)
from ..feasibility.motion_constraints import MotionConstraintCompiler
from ..feasibility.schema import FeasibilityStatus, ValidationLevel, training_ready
from ..evaluation.deterministic import DeterministicEvaluator
from ..metrics.report_builder import ReportBuilder
from ..metrics.ppo_collector import PPOCollector
from ..metrics.tensorboard_reader import TensorBoardReader
from ..metrics.trajectory_metrics import TrajectoryMetrics
from ..memory.curator import MemoryCuratorAgent
from ..memory.compatibility import compatibility_fingerprint
from ..memory.consolidator import SemanticMemoryConsolidatorAgent
from ..memory.procedural import ProceduralMemoryAgent
from ..memory.store import LongTermMemoryStore
from ..maintenance.experiment_cleanup import ExperimentCleaner
from ..memory.reward_experience.agent import RewardExperienceAgent
from ..memory.reward_experience.eligibility import ExperienceEligibilityChecker
from ..memory.reward_experience.evidence_builder import RewardExperienceEvidenceBuilder
from ..memory.reward_experience.validator import RewardExperienceValidationError, RewardExperienceValidator
from ..observability.events import JsonlEventRecorder
from ..providers.base import LLMReasoningProvider
from ..providers.errors import ProviderError, ProviderResponseError
from ..providers.mock_provider import MockLLMReasoningProvider
from ..providers.multi_agent import MultiAgentProvider
from ..providers.opencli_chatgpt import OpenCLIChatGPTWebProvider
from ..providers.registry import ProviderRegistry
from ..rag.knowledge_base import TrainingKnowledgeBase
from ..rewards.compiler import RewardCompiler
from ..rewards.reviser import RewardPlanReviser
from ..rewards.validator import RewardValidationError
from ..schemas.decisions import TrainingDiagnosis
from ..schemas.agent_workflow import TaskIntentSpec, WorkingMemorySnapshot
from ..schemas.experiments import ExperimentManifest
from ..schemas.rewards import CurriculumStage, RewardPlan, RewardTerm
from ..schemas.task import TaskSpec
from ..schemas.visual import VisualBehaviorReport
from ..settings import Settings
from ..storage.experiment_store import ExperimentStore
from ..storage.lineage import LineageGraph
from ..storage.locking import FileLock
from ..training.checkpoint_manager import CheckpointManager
from ..training.controller import TrainingController
from ..training.process_manager import process_identity
from ..utils.io import atomic_write_text, read_json, utc_now, write_json
from ..utils.paths import relative_display
from ..utils.task_semantics import ForwardFlipKind, classify_forward_flip, is_front_leg_support_text
from ..visual.evaluation_pipeline import VisualEvaluationPipeline
from ..visual.ensemble import ConservativeVisualAggregator
from ..visual.rollout_recorder import DryRunRolloutRecorder
from .budget import BudgetTracker
from .state_machine import AgentState, PersistentStateMachine


class TrainingOrchestrator:
    def __init__(self, settings: Settings, provider: LLMReasoningProvider):
        """初始化 TrainingOrchestrator 实例及其运行依赖。"""
        self.settings = settings
        self.provider = provider
        self.store = ExperimentStore(settings.experiments_path)
        self.inspector = EnvironmentInspector(settings.training_root)
        self.knowledge = TrainingKnowledgeBase.from_settings(settings)
        self.memory = LongTermMemoryStore(
            settings.memory_path, settings.agent_root, settings.memory_max_context_chars)
        self.context_builder = ContextBuilder()
        self.prompt_compiler = RewardPromptCompiler(
            settings.agent_root / "rl_training_agent" / "prompts" / "reward_design_agent.md")
        self.reward_reviewer = RewardReviewAgent()
        self.feasibility = FeasibilityPipeline(settings.training_root)
        self.memory_curator = MemoryCuratorAgent()
        self.reward_experience_evidence = RewardExperienceEvidenceBuilder()
        self.reward_experience_eligibility = ExperienceEligibilityChecker()
        self.reward_experience_validator = RewardExperienceValidator()
        self.memory_consolidator = SemanticMemoryConsolidatorAgent()
        self.procedural_memory = ProceduralMemoryAgent()
        adapter = UnitreeProjectAdapter(settings.training_root, settings.agent_root, settings.experiments_path)
        self.controller = TrainingController(adapter, timeout_seconds=settings.training_timeout_seconds)

    @staticmethod
    def provider_for(settings: Settings, name: str, record_dir: Optional[Path] = None) -> LLMReasoningProvider:
        """按命令创建多 Agent、兼容 OpenCLI 或显式测试 Provider。"""
        if name == "mock":
            return MockLLMReasoningProvider(settings.num_reward_candidates)
        if name == "opencli":
            return OpenCLIChatGPTWebProvider(record_dir=record_dir)
        if name in ("doubao", "opencli-doubao"):
            return ProviderRegistry().create(name, settings, record_dir)
        if name == "multi-agent":
            return MultiAgentProvider(settings, record_dir=record_dir)
        raise ValueError("provider must be 'multi-agent', 'opencli-doubao', 'opencli', 'doubao', or 'mock'")

    @staticmethod
    def _task_id(instruction: str, robot: str) -> str:
        """根据机器人和原始指令生成稳定任务标识。"""
        return "task-" + hashlib.sha1((robot + instruction).encode("utf-8")).hexdigest()[:10]

    @staticmethod
    def _select_safe_candidate(screened: List[Dict[str, Any]]) -> Dict[str, Any]:
        """只从通过筛选硬门槛且得分有限的候选中选择最优项。"""
        eligible = []
        for item in screened:
            screening = read_json(item["dir"] / "metrics" / "screening.json")
            score = screening.get("composite_score")
            if screening.get("hard_constraints_passed") and isinstance(score, (int, float)) and math.isfinite(score):
                item["screening_score"] = float(score)
                eligible.append(item)
        if not eligible:
            raise RuntimeError("所有奖励候选均未通过数值健康与硬约束筛选，已禁止进入完整训练")
        return max(eligible, key=lambda item: item["screening_score"])

    @staticmethod
    def _validate_plan_metric_coverage(task: TaskSpec, plan: RewardPlan) -> None:
        """确保奖励计划覆盖必选指标，并为后腿任务使用姿态门控奖励。"""
        from ..rewards.action_normalizer import validate_action_semantics
        validate_action_semantics(task, plan)
        required = {item.name for item in task.success_metrics if item.required}
        provided = {item.name for item in plan.success_metrics if item.required}
        missing = sorted(required - provided)
        if missing:
            raise RewardValidationError("reward plan misses required success metrics: %s" % ", ".join(missing))
        rear_leg_task = ("后腿" in task.original_instruction or any(
            "rear_leg" in item.name for item in task.required_behaviors))
        if rear_leg_task:
            reward_names = {item.name for item in plan.terms}
            missing_rewards = sorted({"rear_leg_stand", "rear_leg_walk"} - reward_names)
            if missing_rewards:
                raise RewardValidationError("rear-leg task misses posture-gated rewards: %s" %
                                            ", ".join(missing_rewards))
            if "orientation" in reward_names:
                raise RewardValidationError(
                    "rear-leg task cannot use flat-base orientation cost; rear_leg_stand already controls roll and target pitch")
        if TrainingOrchestrator._is_front_leg_support_task(task):
            reward_names = {item.name for item in plan.terms}
            missing_rewards = sorted({"front_leg_stand", "front_leg_walk"} - reward_names)
            if missing_rewards:
                raise RewardValidationError("front-leg support task misses posture-gated rewards: %s" %
                                            ", ".join(missing_rewards))

    @staticmethod
    def _is_front_leg_support_task(task: TaskSpec) -> bool:
        """判断原始指令是否明确要求以前腿支撑站立或行走。"""
        # 用户原始指令是任务身份的唯一真值，不能让模型生成的规范化描述
        # 通过错误加入“前腿离地”等文字反向覆盖用户意图。
        return is_front_leg_support_text((task.original_instruction,))

    @staticmethod
    def _normalize_task_for_instruction(task: TaskSpec) -> List[str]:
        """纠正自然语言中“用前腿站立”被误解为“抬起前腿”的任务语义。"""
        adjustments: List[str] = []
        if not TrainingOrchestrator._is_front_leg_support_task(task):
            return adjustments
        # 用户描述地面前进速度，不能用倒立机体 x 分量或奖励得分代替。
        if task.velocity_frame != "heading":
            task.velocity_frame = "heading"
            adjustments.append("前腿倒立速度验收统一为水平航向坐标，单位 m/s")
        for metric in task.success_metrics:
            if metric.name == "front_leg_walk_velocity_tracking" and metric.unit == "m/s":
                metric.name = "front_leg_forward_speed"
                adjustments.append("把误标为 m/s 的跟踪得分改为实际水平前进速度，保留阈值")
        # 未明确要求动作内先站后走时，模型生成的学习阶段仅用于训练课程。
        sequential = any(token in task.original_instruction for token in ("先", "然后", "再走", "再前进"))
        if not sequential:
            for phase in task.phases:
                if phase.name in ("front_stand", "front_walk") and phase.scope != "training":
                    phase.scope = "training"
                    adjustments.append("将模型生成的站立/行走学习阶段标记为训练课程")
        for behavior in task.required_behaviors:
            if behavior.name == "front_leg_lifted_posture" or "前腿离地" in behavior.description:
                behavior.name = "front_leg_support_posture"
                behavior.description = "保持两个前足支撑、两个后足离地，并维持目标高度、俯仰角和横滚稳定。"
                adjustments.append("纠正前腿站立语义：前足支撑、后足离地")
        requirement = "必须确认前足支撑、后足离地，不能把前腿站立误判为抬起前腿。"
        if requirement not in task.visual_evaluation_requirements:
            task.visual_evaluation_requirements.append(requirement)
        return adjustments

    def _check_task_semantics(self, task_dir: Path, state: PersistentStateMachine,
                              intent: TaskIntentSpec, task: Optional[TaskSpec] = None
                              ) -> Optional[Dict[str, Any]]:
        """持久化意图/任务语义审核，冲突不得靠奖励重设计或扩大预算掩盖。"""
        report = (TaskSemanticValidator.assess_task(intent, task) if task is not None
                  else TaskSemanticValidator.assess_intent(intent))
        stage = 'task' if task is not None else 'intent'
        write_json(task_dir / (stage + '_semantic_check.json'), report)
        JsonlEventRecorder(task_dir / 'events.jsonl').emit('task_semantic_check', {
            'run_id': state.record.context.get('run_id'), 'stage': stage, **report.dict()})
        if report.status == 'CONSISTENT':
            return None
        reason = ('动作目标需要澄清：' if report.status == 'NEEDS_CLARIFICATION' else
                  '动作语义不一致，已禁止启动训练：') + '；'.join(report.issues)
        if report.question:
            reason += '；' + report.question
        result = {'task_id': task_dir.name, 'state': AgentState.HUMAN_REVIEW.value,
                  'result': 'human_review', 'reason': reason,
                  'semantic_check': report.dict(),
                  'requires_clarification': report.status == 'NEEDS_CLARIFICATION'}
        write_json(task_dir / 'blocking_report.json', result)
        state.transition(AgentState.HUMAN_REVIEW, {'reason': reason},
                         operation_id='semantic-block:%s:%s' % (
                             state.record.context.get('run_id'), stage))
        self._write_working_memory(task_dir, state)
        print('[任务语义] %s' % reason, flush=True)
        return result

    def _repair_reward_candidates(self, task, task_dir, plans, review, manifest,
                                  record_root, attempt, extra_context=None, require_objective_links=True):
        """仅为本地定位的问题请求补丁；不重新生成任务或完整候选。"""
        from ..rewards.repair_scope import build_repair_scope
        from ..rewards.patches import apply_reward_repairs, reward_plan_sha256
        targets, blocked = [], []
        for candidate in review.candidate_results:
            if candidate.approved:
                continue
            index = candidate.candidate_index
            plan = plans[index - 1]
            scope = build_repair_scope(task, plan, dict(manifest.dict(), require_objective_links=require_objective_links,
                                                      max_abs_reward_weight=self.settings.max_abs_reward_weight))
            blocked.extend({'candidate_index': index, 'issue': issue} for issue in scope.get('blocked', []))
            if scope['allowed_patches']:
                targets.append({'candidate_index': index, 'base_sha256': reward_plan_sha256(plan),
                    'reward_plan': plan.dict(), 'allowed_patches': scope['allowed_patches'],
                    'issues': scope['issues'], 'review': candidate.dict()})
        payload = {'immutable_task': task.dict(), 'environment_manifest': manifest.dict(),
                   'targets': targets, 'blocked': blocked, 'attempt': attempt}
        if extra_context:
            payload.update(extra_context)
        record_root.mkdir(parents=True, exist_ok=True)
        write_json(record_root / ('request_%02d.json' % attempt), payload)
        if not targets:
            raise RewardValidationError('没有可安全定位的奖励字段修复范围；保留原始候选')
        repair = getattr(self.provider, 'repair_reward_plans', None)
        if not callable(repair):
            raise RewardValidationError('当前奖励 Provider 不支持局部补丁修复；保留原始候选')
        raw = self._provider_call(task_dir, 'reward_designer', 'repair_reward_plans',
                                  lambda: repair(payload))
        write_json(record_root / ('response_%02d.json' % attempt), raw)
        revised, audit = apply_reward_repairs(plans, targets, raw)
        write_json(record_root / ('patch_audit_%02d.json' % attempt), audit)
        write_json(record_root / ('effective_plans_%02d.json' % attempt), [plan.dict() for plan in revised])
        if not audit['accepted']:
            reasons = '；'.join(item['reason'] for item in audit['rejected'])
            raise RewardValidationError('奖励补丁没有产生有效局部修改：' + (reasons or '空补丁或内容未改变'))
        return revised, audit

    def _block_initial_provider_failure(self, task_dir: Path, state: PersistentStateMachine,
                                        role: str, error: ProviderError) -> Dict[str, Any]:
        """保存训练前推理故障及重新下发方式，防止进程退出后仍显示正在设计。"""
        stage = state.record.state.value
        label = {AgentState.TASK_UNDERSTANDING.value: "任务理解",
                 AgentState.REWARD_DESIGNING.value: "奖励设计"}[stage]
        detail = re.sub(r"\s+", " ", str(error)).strip()
        if len(detail) > 1000:
            detail = detail[:997] + "..."
        failure_kind = 'response_schema' if isinstance(error, ProviderResponseError) else 'provider_unavailable'
        reason = ("%s回复结构不合格：%s" if failure_kind == 'response_schema' else
                  "%s Provider 不可用：%s") % (label, detail or error.__class__.__name__)
        report = {"task_id": task_dir.name, "state": AgentState.HUMAN_REVIEW.value,
                  "result": "human_review", "stage": stage, "recoverable": True,
                  "failure_kind": failure_kind,
                  "resume_allowed": False, "restart_required": True, "reason": reason,
                  "recovery_hint": ("本轮尚未启动 PPO；回复格式未通过校验，修复结构后可重新下发原始任务。"
                                    if failure_kind == 'response_schema' else
                                    "本轮尚未启动 PPO；检查推理页面或连接后重新下发原始任务。")}
        write_json(task_dir / "blocking_report.json", report)
        state.transition(AgentState.HUMAN_REVIEW, {"reason": reason, "provider_stage": role},
                         operation_id="initial-provider-failure:%s:%s" % (
                             state.record.context.get("run_id"), role))
        self._write_working_memory(task_dir, state)
        print("[%s] %s" % (label, reason), flush=True)
        return report

    def _check_action_capabilities(self, task_dir: Path, state: PersistentStateMachine,
                                   intent: TaskIntentSpec, capabilities: Dict[str, Any]
                                   ) -> Optional[Dict[str, Any]]:
        """在动作求解和奖励请求前报告前翻的本地能力缺口。"""
        gaps = TaskSemanticValidator.action_capability_gaps(intent, capabilities)
        if not gaps:
            return None
        reason = "动作所需本地实现尚未补齐，已禁止启动训练：" + "；".join(gaps)
        report = {'task_id': task_dir.name, 'state': AgentState.HUMAN_REVIEW.value,
                  'result': 'human_review', 'stage': 'action_capability_check',
                  'reason': reason, 'capability_gaps': gaps, 'requires_local_implementation': True,
                  'requires_clarification': False}
        write_json(task_dir / 'action_capability_check.json', report)
        write_json(task_dir / 'blocking_report.json', report)
        state.events.emit('action_capability_check', report)
        state.transition(AgentState.HUMAN_REVIEW, {'reason': reason})
        self._write_working_memory(task_dir, state)
        print('[动作能力] %s' % reason, flush=True)
        return report

    def _check_task_contract(self, task_dir: Path, state: PersistentStateMachine,
                             task: TaskSpec) -> Optional[Dict[str, Any]]:
        """冻结验收前检查任务接触单位，不能让奖励重试修复不可修改的坏合同。"""
        from ..training.contact_semantics import contact_contract
        try:
            contact_contract(task.safety_constraints)
        except ValueError as exc:
            detail = str(exc)
            report = {'task_id': task_dir.name, 'state': AgentState.HUMAN_REVIEW.value,
                      'result': 'human_review', 'stage': 'task_contract_check',
                      'approved': False,
                      'reason': '任务接触合同无效，须修正任务设计：' + detail,
                      'errors': [detail], 'requires_task_redesign': True,
                      'requires_clarification': False}
            write_json(task_dir / 'task_contract_check.json', report)
            write_json(task_dir / 'blocking_report.json', report)
            state.events.emit('task_contract_check', report)
            state.transition(AgentState.HUMAN_REVIEW, {'reason': report['reason']})
            self._write_working_memory(task_dir, state)
            print('[任务合同] %s' % report['reason'], flush=True)
            return report
        write_json(task_dir / 'task_contract_check.json', {'approved': True, 'errors': []})
        return None

    @staticmethod
    def _normalize_plan_for_task(task: TaskSpec, plan: RewardPlan) -> List[str]:
        """继承任务验收指标，并移除后腿任务中会诱导爬行的冲突奖励。"""
        adjustments: List[str] = []
        plan.velocity_frame = task.velocity_frame
        from ..rewards.action_normalizer import normalize_action_plan
        adjustments.extend(normalize_action_plan(task, plan))
        for metric in plan.success_metrics:
            if (task.velocity_frame == "heading" and metric.name == "front_leg_walk_velocity_tracking"
                    and metric.unit == "m/s"):
                metric.name = "front_leg_forward_speed"
        existing_metrics = {item.name for item in plan.success_metrics}
        for metric in task.success_metrics:
            if metric.required and metric.name not in existing_metrics:
                plan.success_metrics.append(metric.copy(deep=True))
                adjustments.append("补充任务必选验收指标：%s" % metric.name)
        command_target = TrainingOrchestrator._explicit_locomotion_command(task)
        if command_target is not None:
            stages = plan.curriculum
            if not stages:
                stages.append(CurriculumStage(
                    name="task_command",
                    start_iteration=0,
                    end_iteration=max(0, task.training_budget.max_total_iterations - 1),
                    parameter_changes={},
                ))
                adjustments.append("补充覆盖全程的确定性任务命令阶段")
            for stage in stages:
                token = stage.name.lower()
                standing = "stand" in token or "站" in stage.name
                scale = stage.parameter_changes.get("command_scale", 1.0)
                factor = max(0.0, float(scale)) if isinstance(scale, (int, float)) else 1.0
                stage_target = 0.0 if standing else command_target * factor
                stage.parameter_changes.update({
                    "lin_vel_y": [0.0, 0.0],
                    "ang_vel_yaw": [0.0, 0.0],
                    "heading": [0.0, 0.0],
                })
                requested = stage.parameter_changes.get("lin_vel_x")
                values = list(requested) if isinstance(requested, (list, tuple)) else [requested]
                legitimate_ramp = (values and all(isinstance(v, (int, float)) and
                    0 <= v * command_target <= command_target ** 2 for v in values)
                    and any(v != 0 for v in values))
                if standing or not legitimate_ramp:
                    stage.parameter_changes["lin_vel_x"] = [stage_target, stage_target]
            adjustments.append("按任务方向固定 lin_vel_x 命令：%.3f m/s" % command_target)
            tracking_required = any(
                item.required and item.name == "tracking_lin_vel" for item in task.success_metrics)
            tracking_term = next((item for item in plan.terms if item.name == "tracking_lin_vel"), None)
            if tracking_required and tracking_term is not None and tracking_term.weight <= 0.0:
                tracking_term.weight = 1.0
                tracking_term.active_phases = ["all"]
                adjustments.append("恢复必选速度跟踪奖励的正权重")
        rear_leg_task = ("后腿" in task.original_instruction or any(
            "rear_leg" in item.name for item in task.required_behaviors))
        if rear_leg_task:
            conflicting = {"tracking_lin_vel", "orientation"}
            removed = [item.name for item in plan.terms if item.name in conflicting]
            plan.terms = [item for item in plan.terms if item.name not in conflicting]
            for name in removed:
                adjustments.append("移除后腿任务冲突奖励：%s" % name)
            rear_parameters = {
                "rear_stand_height_target": 0.42, "rear_stand_pitch_target": 0.85,
                "rear_stand_height_sigma": 0.12, "rear_stand_pitch_sigma": 0.30,
            }
            for term in plan.terms:
                if term.name in ("rear_leg_stand", "rear_leg_walk"):
                    before = dict(term.parameters)
                    term.parameters.update(rear_parameters)
                    if term.parameters != before:
                        adjustments.append("规范后腿支撑目标参数：%s" % term.name)
            for stage in plan.curriculum:
                token = stage.name.lower()
                if "stand" in token or "站" in stage.name:
                    stage.parameter_changes["command_scale"] = 0.0
                if "walk" in token or "行走" in stage.name:
                    value = stage.parameter_changes.get("lin_vel_x")
                    if value is None:
                        stage.parameter_changes["lin_vel_x"] = [0.25, 0.35]
                        adjustments.append("为后腿行走阶段补充非零前向命令")
                    elif isinstance(value, (int, float)) and abs(float(value)) <= 0.2:
                        stage.parameter_changes["lin_vel_x"] = (
                            [-0.35, -0.25] if float(value) < 0 else [0.25, 0.35])
                        adjustments.append("将后腿行走命令移出 Unitree 速度死区")
        if TrainingOrchestrator._is_front_leg_support_task(task):
            # front_leg_walk 已经将速度奖励门控在前足支撑姿态内。普通速度奖励
            # 及模型虚构的阶段后缀别名会诱导四足行走，必须在审查前确定性移除。
            conflicting = {
                "tracking_lin_vel", "tracking_lin_vel_stand", "tracking_lin_vel_walk",
                "orientation", "landing_stability",
            }
            removed = [item.name for item in plan.terms if item.name in conflicting]
            plan.terms = [item for item in plan.terms if item.name not in conflicting]
            for name in removed:
                adjustments.append("移除前腿支撑任务冲突奖励：%s" % name)
            names = {item.name for item in plan.terms}
            phase_names = [item.name for item in task.phases] or ["all"]
            walking_phases = [name for name in phase_names if "walk" in name.lower() or "行走" in name]
            common_parameters = {
                "front_stand_height_target": 0.42,
                "front_stand_pitch_target": 0.85,
                "front_stand_height_sigma": 0.12,
                "front_stand_pitch_sigma": 0.30,
            }
            for term in plan.terms:
                if term.name not in ("front_leg_stand", "front_leg_walk"):
                    continue
                before = dict(term.parameters)
                term.parameters.update(common_parameters)
                if term.parameters != before:
                    adjustments.append("规范前腿支撑目标参数：%s" % term.name)
            for stage in plan.curriculum:
                token = stage.name.lower()
                changes = stage.parameter_changes
                # 模型偶发把命令放入嵌套 commands；运行时只接受白名单顶层键。
                # 前面已经从用户目标编译了确定性命令，因此移除不可执行的副本。
                if "commands" in changes:
                    changes.pop("commands", None)
                    adjustments.append("移除不可执行的嵌套课程命令：%s" % stage.name)
                reward_scales = changes.get("reward_scales")
                if isinstance(reward_scales, dict):
                    removed_scales = sorted(set(reward_scales).intersection(conflicting))
                    for name in removed_scales:
                        reward_scales.pop(name, None)
                        adjustments.append("移除课程中的前腿冲突奖励缩放：%s" % name)
                if "stand" in token or "站" in stage.name:
                    changes.update({
                        "command_scale": 0.0,
                        "lin_vel_x": [0.0, 0.0],
                        "lin_vel_y": [0.0, 0.0],
                        "ang_vel_yaw": [0.0, 0.0],
                        "heading": [0.0, 0.0],
                    })
                if "walk" in token or "行走" in stage.name:
                    value = changes.get("lin_vel_x")
                    if value is None:
                        changes["lin_vel_x"] = [0.25, 0.35]
                        adjustments.append("为前腿行走阶段补充非零前向命令")
                    elif isinstance(value, (int, float)) and abs(float(value)) <= 0.2:
                        changes["lin_vel_x"] = (
                            [-0.35, -0.25] if float(value) < 0 else [0.25, 0.35])
                        adjustments.append("将前腿行走命令移出 Unitree 速度死区")
            # 保证学习型站立阶段确实在训练时执行，而非由评估帧0推测。
            if (all(any(p.scope == "training" and p.name == name for p in task.phases)
                    for name in ("front_stand", "front_walk"))
                    and not any("stand" in s.name.lower() or "站" in s.name for s in plan.curriculum)):
                horizon = max(2, max((s.end_iteration + 1 for s in plan.curriculum), default=1000))
                split = max(1, horizon // 5)
                walking = sorted(plan.curriculum, key=lambda s: s.start_iteration)
                if not walking:
                    walking = [CurriculumStage(name="front_walk", start_iteration=0, end_iteration=horizon - 1,
                        parameter_changes={"lin_vel_x": [command_target or 0.3, command_target or 0.3],
                                           "lin_vel_y": [0.0, 0.0], "ang_vel_yaw": [0.0, 0.0],
                                           "heading": [0.0, 0.0]})]
                original = [(s.start_iteration, s.end_iteration) for s in walking]
                for stage, (start, end) in zip(walking, original):
                    stage.start_iteration = min(horizon - 1, split + start * (horizon - split) // horizon)
                    stage.end_iteration = min(horizon - 1, max(stage.start_iteration,
                        split + (end + 1) * (horizon - split) // horizon - 1))
                if len(walking) == 1:
                    walking[0].name = "front_walk"
                plan.curriculum = [CurriculumStage(name="front_stand", start_iteration=0, end_iteration=split - 1,
                        parameter_changes={"lin_vel_x": [0.0, 0.0], "lin_vel_y": [0.0, 0.0],
                                           "ang_vel_yaw": [0.0, 0.0], "heading": [0.0, 0.0]})] + walking
                adjustments.append("补充前腿站立到行走的连续训练课程")
            if "front_leg_stand" not in names:
                plan.terms.append(RewardTerm(
                    name="front_leg_stand", implementation="registry:front_leg_stand",
                    purpose="建立前足支撑、后足离地的稳定倒立姿态。", weight=2.0,
                    parameters=common_parameters, active_phases=phase_names,
                    expected_raw_range=(0.0, 1.0), expected_training_trend="increase",
                    dependencies=["contact_forces", "base_pos", "rpy"],
                    failure_modes_addressed=["后足未离地", "身体失稳"],
                    reward_hacking_risks=["短暂倒立但不能持续"],
                ))
                adjustments.append("补充前腿支撑奖励：front_leg_stand")
            if "front_leg_walk" not in names:
                plan.terms.append(RewardTerm(
                    name="front_leg_walk", implementation="registry:front_leg_walk",
                    purpose="仅在前腿倒立成立时奖励速度跟踪。", weight=2.0,
                    parameters=common_parameters, active_phases=walking_phases or phase_names[-1:],
                    expected_raw_range=(0.0, 1.0), expected_training_trend="increase",
                    dependencies=["contact_forces", "base_pos", "rpy", "base_lin_vel", "commands"],
                    failure_modes_addressed=["四足普通行走", "倒立后无法移动"],
                    reward_hacking_risks=["仅瞬时匹配速度"],
                ))
                adjustments.append("补充前腿门控行走奖励：front_leg_walk")
        for term in plan.terms:
            if term.name in ('front_leg_stand', 'rear_leg_stand'):
                term.active_phases = ['all']
                term.activation_condition = None
            elif term.name in ('front_leg_walk', 'rear_leg_walk'):
                stages = [s.name for s in plan.curriculum if 'walk' in s.name.lower() or '行走' in s.name]
                term.active_phases = stages or ['all']
                term.activation_condition = None
        return adjustments

    @staticmethod
    def _explicit_locomotion_command(task: TaskSpec) -> Optional[float]:
        """从包含方向和速度单位的描述提取目标，坐标系由 task.velocity_frame 决定。"""
        speed_pattern = r"([0-9]+(?:\.[0-9]+)?)\s*(?:m\s*/\s*s|米\s*/\s*秒|米每秒)"
        backward_tokens = ("倒退", "倒着", "后退", "向后", "往后", "backward", "reverse")
        forward_tokens = ("向前", "前进", "forward")
        # 原始用户指令优先。模型生成的 normalized_description 常包含
        # “禁止向前”等反向约束，若把两段文字拼接会同时命中前进与后退，
        # 从而丢失用户明确给出的 -X 命令。
        for text in (task.original_instruction, task.normalized_description):
            match = re.search(speed_pattern, text, flags=re.IGNORECASE)
            if match is None:
                continue
            speed = float(match.group(1))
            if not math.isfinite(speed) or speed <= 0.0:
                continue
            lowered = text.lower()
            backward = any(token in lowered for token in backward_tokens)
            forward = any(token in lowered for token in forward_tokens)
            if backward != forward:
                return -speed if backward else speed
        return None

    def inspect_environment(self, robot: str, output: Optional[Path] = None):
        """检查训练环境并保存能力清单。"""
        destination = output or self.settings.artifacts_path / "environment_manifest.json"
        return self.inspector.write(destination, robot)

    def _retrieve_experience(self, query: str, purpose: str, robot: str,
                             exclude_task_id: Optional[str] = None) -> Dict[str, Any]:
        """检索历史训练经验并以失败开放方式返回可追溯模型上下文。"""
        if not self.settings.rag_enabled:
            return {"enabled": False, "purpose": purpose, "hits": []}
        try:
            self.knowledge.refresh()
            result = self.knowledge.retrieve(
                query, purpose, top_k=self.settings.rag_top_k,
                robot=robot, exclude_task_id=exclude_task_id)
            payload = result.prompt_payload()
            payload.update({
                "enabled": True, "indexed_documents": result.indexed_documents,
                "indexed_chunks": result.indexed_chunks,
            })
            print("[RAG] %s：从 %d 个片段中检索到 %d 条相关经验" % (
                purpose, result.indexed_chunks, len(result.hits)), flush=True)
            return payload
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            # RAG 是增强证据而不是训练安全前提；索引损坏不能阻塞确定性主链路。
            print("[RAG] 检索不可用，继续使用当前任务证据：%s" % exc, flush=True)
            return {"enabled": True, "purpose": purpose, "hits": [], "error": str(exc)}

    def _retrieve_memory(self, query: str, robot: str,
                         exclude_task_id: Optional[str] = None) -> Dict[str, Any]:
        """检索经过验收晋升的长期记忆，并在损坏时安全降级。"""
        if not self.settings.memory_enabled:
            return {"enabled": False, "hits": []}
        try:
            compatibility = compatibility_fingerprint(
                self.settings.training_root, self.inspector.inspect(robot).dict())
            result = self.memory.retrieve(
                query, robot, top_k=self.settings.memory_top_k,
                exclude_task_id=exclude_task_id, **compatibility)
            print("[记忆] 从 %d 条长期记忆中检索到 %d 条相关经验" % (
                result.get("records", 0), len(result.get("hits", []))), flush=True)
            return result
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print("[记忆] 检索不可用，继续使用当前任务证据：%s" % exc, flush=True)
            return {"enabled": True, "hits": [], "error": str(exc)}

    @staticmethod
    def _provider_call(task_dir: Path, role: str, operation: str,
                       callback: Callable[[], Any]) -> Any:
        """记录 Provider 调用耗时与结果，不记录密钥或完整提示词。"""
        recorder = JsonlEventRecorder(task_dir / "events.jsonl")
        started = time.monotonic()
        try:
            result = callback()
        except Exception as exc:
            recorder.emit("provider_call", {
                "role": role, "operation": operation, "ok": False,
                "latency_seconds": round(time.monotonic() - started, 6),
                "error_type": exc.__class__.__name__,
            })
            raise
        recorder.emit("provider_call", {
            "role": role, "operation": operation, "ok": True,
            "latency_seconds": round(time.monotonic() - started, 6),
        })
        return result

    def _write_working_memory(self, task_dir: Path, state: PersistentStateMachine,
                              budget: Optional[BudgetTracker] = None,
                              selected: Optional[Dict[str, Any]] = None,
                              outcome: Optional[Dict[str, Any]] = None,
                              loop_round: int = 0) -> None:
        """将当前状态、预算、奖励版本和最新评估写入任务级工作记忆。"""
        if not self.settings.memory_enabled:
            return
        budget_data = budget or BudgetTracker(max_iterations=0, max_revisions=0)
        evaluation = (outcome or {}).get("evaluation")
        diagnosis = (outcome or {}).get("diagnosis")
        snapshot = WorkingMemorySnapshot(
            task_id=task_dir.name,
            state=state.record.state.value,
            loop_round=loop_round,
            reward_version=int(getattr((selected or {}).get("plan"), "version", 0)),
            used_iterations=budget_data.used_iterations,
            remaining_iterations=max(0, budget_data.max_iterations - budget_data.used_iterations),
            used_revisions=budget_data.used_revisions,
            remaining_revisions=max(0, budget_data.max_revisions - budget_data.used_revisions),
            latest_evaluation=evaluation.dict() if hasattr(evaluation, "dict") else (evaluation or {}),
            latest_diagnosis=diagnosis.dict() if hasattr(diagnosis, "dict") else (diagnosis or {}),
            updated_at=utc_now(),
        )
        self.memory.save_working(task_dir, snapshot)

    def _git_commit(self) -> str:
        """读取训练项目当前 Git 提交标识。"""
        result = subprocess.run(["git", "-C", str(self.settings.training_root), "rev-parse", "HEAD"],
                                text=True, capture_output=True, timeout=10, check=False)
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    @staticmethod
    def _clear_stale_run_outputs(task_dir: Path) -> None:
        """开始同一任务的新运行时移除会误导界面的旧终态摘要，保留候选和 checkpoint。"""
        for name in ("blocking_report.json", "summary.json", "report.md", "loop_status.json",
                     "loop_history.json", "loop_blocking_error.txt", "feasibility_report.json",
                     "training_admission.json", "training_readiness.json", "environment_health.json",
                     "acceptance_contract.json", "evaluation_protocol.json", "training_execution.json",
                     "intent_semantic_check.json", "task_semantic_check.json",
                     "action_capability_check.json", "task_contract_check.json",
                     "local_reward_baseline.json", "reward_review_before_baseline.json",
                     "reward_review_with_baseline.json", "task_identity_normalization.json"):
            path = task_dir / name
            if path.is_file():
                path.unlink()
        for pattern in ("reward_review_attempt_*.json", "design_response_attempt_*.json",
                        "plan_normalization_attempt_*.json"):
            for path in task_dir.glob(pattern):
                if path.is_file():
                    path.unlink()

    def plan(self, instruction: str, robot: str) -> Dict[str, Any]:
        """生成并验证任务与候选配置但不启动训练。"""
        return self._run(instruction, robot, dry_run=True, stop_after_plan=True)

    def train(self, instruction: str, robot: str, dry_run: bool = False) -> Dict[str, Any]:
        """执行自然语言任务的完整训练编排流程。"""
        return self._run(instruction, robot, dry_run=dry_run, stop_after_plan=False)

    @staticmethod
    def _checkpoint_for_seed(directory: Path, seed: int) -> Optional[Path]:
        """从候选目录中选择指定随机种子的最高迭代 checkpoint。"""
        real_marker = "-seed-%d" % seed
        dry_marker = "seed_%d" % seed
        matches = [path for path in CheckpointManager.list_checkpoints(directory)
                   if any(part.endswith(real_marker) or part == dry_marker
                          for part in path.parent.parts)]
        return matches[-1] if matches else None

    def _evaluation_protocol(self, task_dir: Path) -> Dict[str, Any]:
        """读取当前运行冻结的种子及样本要求，旧运行仅接受已保存的种子集合。"""
        path = task_dir / 'evaluation_protocol.json'
        if path.is_file():
            protocol = read_json(path)
            state = read_json(task_dir / 'state.json')
            if not isinstance(protocol, dict) or protocol.get('run_id') != state.get('context', {}).get('run_id'):
                raise RuntimeError('恢复被拒绝：评估协议 run_id 不一致')
        else:
            summary_path = task_dir / 'summary.json'
            summary = read_json(summary_path) if summary_path.is_file() else {}
            seeds = summary.get('evaluation_seeds')
            if not seeds:
                raise RuntimeError('恢复被拒绝：旧任务缺少固定评估种子集合')
            protocol = {'seeds': seeds, 'dry_run': bool(summary.get('dry_run')),
                        'rollouts_per_seed': self.settings.rollouts_per_seed,
                        'counterfactual_enabled': self.settings.counterfactual_enabled}
        seeds = protocol.get('seeds')
        if (not isinstance(seeds, list) or not seeds or
                any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds) or
                type(protocol.get('dry_run')) is not bool or type(protocol.get('counterfactual_enabled')) is not bool or
                type(protocol.get('rollouts_per_seed')) is not int or protocol['rollouts_per_seed'] <= 0):
            raise RuntimeError('恢复被拒绝：固定评估协议无效')
        return protocol

    @staticmethod
    def _save_checkpoint_set(task_dir: Path, selected: Dict[str, Any], dry_run: bool) -> None:
        """保存每个种子的精确 checkpoint 路径和摘要，禁止恢复时隐式选择其他策略。"""
        state = read_json(task_dir / 'state.json')
        directory = selected['dir'].resolve()
        seeds = list(selected.get('checkpoint_seeds', []))
        checkpoints = list(selected.get('checkpoints', []))
        protocol_path = task_dir / 'evaluation_protocol.json'
        original = read_json(protocol_path).get('seeds') if protocol_path.is_file() else read_json(task_dir / 'summary.json').get('evaluation_seeds')
        if not seeds or seeds != original or len(checkpoints) != len(seeds):
            raise RuntimeError('禁止保存不完整或改变种子集合的 checkpoint 清单')
        entries = []
        for seed, checkpoint in zip(selected['checkpoint_seeds'], selected['checkpoints']):
            relative = checkpoint.resolve().relative_to(directory)
            entries.append({'seed': seed, 'path': str(relative),
                            'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
        write_json(directory / 'checkpoint_set.json', {
            'run_id': state.get('context', {}).get('run_id'), 'dry_run': dry_run,
            'config_hash': selected['manifest'].config_hash,
            'reward_version': selected['plan'].version, 'entries': entries})

    @staticmethod
    def _persist_training_execution(task_dir: Path, state: PersistentStateMachine,
                                    budget: BudgetTracker, **updates: Any) -> Dict[str, Any]:
        """在启动前原子记录预算与操作状态，崩溃后不能退款或重放不确定训练。"""
        path = task_dir / 'training_execution.json'
        saved = read_json(path) if path.is_file() else {}
        run_id = state.record.context.get('run_id')
        if saved and saved.get('run_id') != run_id:
            raise RuntimeError('训练执行记录 run_id 不一致')
        saved.update(updates)
        saved.update(run_id=run_id, budget=budget.dict(), updated_at=utc_now())
        write_json(path, saved)
        return saved

    def _reserve_training_operation(self, task_dir: Path, state: PersistentStateMachine,
                                    budget: BudgetTracker, operation_id: str, iterations: int,
                                    **updates: Any) -> None:
        """先扣减并落盘训练额度；已有操作不重复启动，包括结果不确定的操作。"""
        path = task_dir / 'training_execution.json'
        saved = read_json(path) if path.is_file() else {}
        operations = dict(saved.get('operations', {}))
        if operation_id in operations:
            raise RuntimeError('训练操作已经登记，禁止重复执行：' + operation_id)
        budget.consume_iterations(iterations)
        operations[operation_id] = {'status': 'reserved', 'iterations': iterations}
        self._persist_training_execution(task_dir, state, budget, operations=operations,
                                         phase='training', **updates)

    def _complete_training_operation(self, task_dir: Path, state: PersistentStateMachine,
                                     budget: BudgetTracker, operation_id: str) -> None:
        """原子标记子进程完成，保留全部已用预算和操作审计记录。"""
        saved = read_json(task_dir / 'training_execution.json')
        saved['operations'][operation_id]['status'] = 'completed'
        self._persist_training_execution(task_dir, state, budget, operations=saved['operations'])

    @staticmethod
    def _load_training_execution(task_dir: Path, state: PersistentStateMachine,
                                 dry_run: bool) -> Dict[str, Any]:
        """核对执行快照归属，并拒绝恢复仍存活的同一训练子进程。"""
        path = task_dir / 'training_execution.json'
        if not path.is_file():
            return {}
        saved = read_json(path)
        if (saved.get('run_id') != state.record.context.get('run_id') or
                type(saved.get('dry_run')) is not bool or saved['dry_run'] != dry_run):
            raise RuntimeError('恢复被拒绝：训练执行记录的运行或模式不一致')
        raw_budget = saved.get('budget', {})
        names = ('max_iterations', 'used_iterations', 'max_revisions', 'used_revisions')
        if not isinstance(raw_budget, dict) or any(type(raw_budget.get(name)) is not int for name in names):
            raise RuntimeError('恢复被拒绝：训练执行预算类型无效')
        budget = BudgetTracker.parse_obj(raw_budget)
        if (min(budget.used_iterations, budget.used_revisions, budget.max_iterations, budget.max_revisions) < 0 or
                budget.used_iterations > budget.max_iterations or budget.used_revisions > budget.max_revisions):
            raise RuntimeError('恢复被拒绝：训练执行预算无效')
        if saved.get('phase') not in ('compiled', 'training', 'trained', 'blocked'):
            raise RuntimeError('恢复被拒绝：训练执行阶段无效')
        operations = saved.get('operations', {})
        if (not isinstance(operations, dict) or any(
                not isinstance(item, dict) or item.get('status') not in ('reserved', 'completed') or
                type(item.get('iterations')) is not int or item['iterations'] < 0
                for item in operations.values()) or
                sum(item['iterations'] for item in operations.values()) > budget.used_iterations):
            raise RuntimeError('恢复被拒绝：训练操作与已扣预算不一致')
        experiment = str(saved.get('experiment_id', ''))
        if not experiment or not experiment.replace('-', '').replace('_', '').isalnum():
            raise RuntimeError('恢复被拒绝：训练执行候选标识无效')
        if saved['phase'] == 'training' and not any(
                operation.startswith(experiment + ':') for operation in operations):
            raise RuntimeError('恢复被拒绝：训练阶段缺少对应候选的已登记操作键')
        directory = task_dir / 'candidates' / experiment
        identity_path = directory / 'process_identity.json'
        pid_path = directory / 'pid'
        if identity_path.is_file():
            identity = read_json(identity_path)
            pid = int(identity.get('pid', 0))
            current = process_identity(pid)
            if current is not None and all(current.get(key) == identity.get(key)
                                           for key in ('pid', 'pgid', 'start_ticks')):
                raise RuntimeError('恢复被拒绝：已登记的训练子进程仍在运行')
        elif saved.get('phase') in ('training', 'blocked') and pid_path.is_file():
            # 旧 PID 没有启动时间，不能用于发送信号，也不能假定原进程已退出。
            import os
            pid = int(pid_path.read_text().strip())
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError('恢复被拒绝：旧训练 PID 仍存在，需先确认进程归属')
        return saved

    def _restore_selected_candidate(self, task_dir: Path, experiment_id: str) -> Dict[str, Any]:
        """从持久化候选目录恢复闭环所需的计划、manifest 和多种子 checkpoint。"""
        directory = self.store.candidate_dir(task_dir.name, experiment_id)
        if not directory.is_dir():
            raise FileNotFoundError("恢复候选不存在：%s" % experiment_id)
        plan = RewardPlan.parse_obj(read_json(directory / "reward_plan.json"))
        manifest = ExperimentManifest.parse_obj(read_json(directory / "manifest.json"))
        metadata = read_json(directory / "compile_metadata.json")
        if plan.version != manifest.reward_version:
            raise RuntimeError("恢复被拒绝：奖励版本与 manifest 不一致")
        if plan.task_id != task_dir.name or manifest.task_id != task_dir.name:
            raise RuntimeError('恢复被拒绝：候选的任务来源不一致')
        config_path = directory / 'config.yaml'
        actual_hash = hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path.is_file() else None
        if str(metadata.get("config_hash")) != manifest.config_hash or actual_hash != manifest.config_hash:
            raise RuntimeError("恢复被拒绝：配置哈希与 manifest 不一致")
        protocol = self._evaluation_protocol(task_dir)
        expected_result = 'dry_run_completed' if protocol['dry_run'] else 'completed'
        if manifest.training_result != expected_result:
            raise RuntimeError('恢复被拒绝：候选训练未完成或 Mock/真实模式不一致')
        seeds = list(protocol['seeds'])
        checkpoints: List[Path] = []
        index_path = directory / 'checkpoint_set.json'
        entries = {}
        if index_path.is_file():
            index = read_json(index_path)
            current_run = read_json(task_dir / 'state.json').get('context', {}).get('run_id')
            if (not isinstance(index, dict) or index.get('run_id') != current_run or index.get('config_hash') != manifest.config_hash or
                    index.get('reward_version') != plan.version or
                    type(index.get('dry_run')) is not bool or index['dry_run'] != protocol['dry_run']):
                raise RuntimeError('恢复被拒绝：checkpoint 来源与当前运行不一致')
            raw_entries = index.get('entries')
            if (not isinstance(raw_entries, list) or any(not isinstance(item, dict) or
                    type(item.get('seed')) is not int or not isinstance(item.get('path'), str) or
                    not isinstance(item.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', item['sha256'])
                    for item in raw_entries)):
                raise RuntimeError('恢复被拒绝：checkpoint 条目格式无效')
            if any(Path(item['path']).is_absolute() or '..' in Path(item['path']).parts or
                   not CheckpointManager.PATTERN.fullmatch(Path(item['path']).name)
                   for item in raw_entries):
                raise RuntimeError('恢复被拒绝：checkpoint 路径越界或格式无效')
            entries = {item['seed']: item for item in raw_entries}
            if set(entries) != set(seeds) or len(index.get('entries', [])) != len(seeds):
                raise RuntimeError('恢复被拒绝：checkpoint 种子集合不完整')
        for seed in seeds:
            entry = entries.get(seed)
            checkpoint = (directory / entry['path']) if entry else self._checkpoint_for_seed(directory, seed)
            if checkpoint is None or not checkpoint.is_file():
                raise FileNotFoundError('恢复被拒绝：缺少种子 %s 的 checkpoint' % seed)
            try:
                checkpoint.resolve().relative_to(directory.resolve())
            except ValueError as exc:
                raise RuntimeError('恢复被拒绝：checkpoint 路径越界') from exc
            if entry and hashlib.sha256(checkpoint.read_bytes()).hexdigest() != entry.get('sha256'):
                raise RuntimeError('恢复被拒绝：checkpoint 摘要不一致')
            if entry and not any(part == 'seed_%d' % seed or part.endswith('-seed-%d' % seed)
                                 for part in checkpoint.parent.parts):
                raise RuntimeError('恢复被拒绝：checkpoint 路径的种子来源不一致')
            checkpoints.append(checkpoint)
        if not entries and manifest.checkpoint:
            if (directory / manifest.checkpoint).resolve() != checkpoints[0].resolve():
                raise RuntimeError('恢复被拒绝：旧运行 checkpoint 与 manifest 记录不一致')
        selected = {
            "id": experiment_id, "dir": directory, "plan": plan, "manifest": manifest,
            "metadata": metadata,
            "checkpoints": checkpoints, "checkpoint_seeds": seeds, "checkpoint": checkpoints[0],
        }
        if manifest.parent_experiment_id:
            parent_dir = self.store.candidate_dir(task_dir.name, manifest.parent_experiment_id)
            parent_checkpoints = [self._checkpoint_for_seed(parent_dir, seed) for seed in seeds]
            # 保留原种子槽位，绝不能过滤后让另一种子的策略顶替缺失父策略。
            selected["parent_checkpoints"] = parent_checkpoints
        return selected

    def _recover_interrupted_finalization(
            self, task_dir: Path, state: PersistentStateMachine,
            dry_run: bool) -> Optional[Dict[str, Any]]:
        """把已记录阻塞原因但终态写入中断的运行安全收束为人工复核。"""
        decision_states = {
            AgentState.CONTINUE_TRAINING, AgentState.REVISE_REWARD,
            AgentState.REVISE_CURRICULUM, AgentState.ROLLBACK, AgentState.RESTART,
        }
        execution = self._load_training_execution(task_dir, state, dry_run)
        if execution and state.record.state in decision_states | {AgentState.FULL_TRAINING}:
            task = TaskSpec.parse_obj(read_json(task_dir / 'task_spec.json'))
            budget = BudgetTracker.parse_obj(execution['budget'])
            self._apply_persisted_admission_budget(task_dir, task, state, budget, dry_run)
            candidate = str(execution['experiment_id'])
            manifest_path = task_dir / 'candidates' / candidate / 'manifest.json'
            manifest = read_json(manifest_path) if manifest_path.is_file() else {}
            complete = (manifest.get('training_result') in ('completed', 'dry_run_completed') and
                        (manifest_path.parent / 'checkpoint_set.json').is_file())
            if not complete:
                candidate = str(execution.get('parent_experiment') or '')
            if not candidate:
                state.transition(AgentState.HUMAN_REVIEW, {
                    'reason': '初始训练中断且没有完整多种子策略；已用预算保留，不重放训练操作',
                    'training_budget': budget.dict()})
                return {'task_id': task_dir.name, 'state': 'HUMAN_REVIEW',
                        'reason': state.record.context['reason'],
                        'used_iterations': budget.used_iterations,
                        'used_revisions': budget.used_revisions, 'recoverable': False}
            selected = self._restore_selected_candidate(task_dir, candidate)
            records = read_json(task_dir / 'loop_history.json') if (task_dir / 'loop_history.json').is_file() else []
            round_index = int(execution.get('round_index', 0)) + 1
            round_index = max(1, round_index)
            write_json(task_dir / 'loop_status.json', {
                'round': round_index, 'experiment_id': candidate,
                'used_iterations': budget.used_iterations,
                'remaining_iterations': max(0, budget.max_iterations - budget.used_iterations),
                'used_revisions': budget.used_revisions,
                'remaining_revisions': max(0, budget.max_revisions - budget.used_revisions)})
            reason = ('训练中断后已恢复完整策略，评估可继续' if complete else
                      '修订训练中断或失败；保留父策略和已扣预算，不重放不确定训练操作')
            if budget.exhausted:
                reason += '；' + self._budget_exhaustion_reason(selected, budget, dry_run)
            return self._finalize_loop(task, task_dir, state, selected, {}, budget, records,
                                       AgentState.HUMAN_REVIEW, reason, dry_run)
        blocking_path = task_dir / "loop_blocking_error.txt"
        status_path = task_dir / "loop_status.json"
        history_path = task_dir / "loop_history.json"
        if (state.record.state not in decision_states or not blocking_path.is_file() or
                not status_path.is_file() or not history_path.is_file()):
            return None
        loop_status = read_json(status_path)
        loop_records = read_json(history_path)
        if not loop_records:
            return None
        experiment_id = str(
            loop_status.get("experiment_id") or loop_records[-1].get("experiment_id") or "")
        if not experiment_id:
            return None
        task = TaskSpec.parse_obj(read_json(task_dir / "task_spec.json"))
        selected = self._restore_selected_candidate(task_dir, experiment_id)
        round_index = max(1, int(loop_status.get("round", len(loop_records))))
        round_root = selected["dir"] / "rollouts" / ("round_%02d" % round_index)
        diagnosis_files = sorted(
            round_root.glob("*/diagnosis.json"), key=lambda path: path.stat().st_mtime)
        if not diagnosis_files:
            return None
        rollout_dir = diagnosis_files[-1].parent
        used_iterations = int(loop_status.get("used_iterations", 0))
        remaining_iterations = max(0, int(loop_status.get("remaining_iterations", 0)))
        max_revisions = min(
            task.training_budget.max_reward_revisions, self.settings.max_reward_revisions)
        # 修订预算在进入 _train_revision 前扣减；决策状态已经写入说明该次
        # 修订确实被消费，只是后续预算检查或训练启动尚未完成。
        used_revisions = min(
            max_revisions, int(loop_status.get("used_revisions", 0)) + 1)
        budget = BudgetTracker(
            max_iterations=used_iterations + remaining_iterations,
            used_iterations=used_iterations,
            max_revisions=max_revisions,
            used_revisions=used_revisions,
        )
        outcome = {
            "rollout_dir": rollout_dir,
            "clean": rollout_dir / "contact_sheet_clean.png",
            "annotated": rollout_dir / "contact_sheet_annotated.png",
        }
        self._apply_persisted_admission_budget(task_dir, task, state, budget, dry_run)
        error = blocking_path.read_text(encoding="utf-8").strip() or "未知安全阻塞"
        return self._finalize_loop(
            task, task_dir, state, selected, outcome, budget, loop_records,
            AgentState.HUMAN_REVIEW, "闭环修订无法安全执行：%s" % error, dry_run)

    def resume(self, task_id: str, dry_run: bool = False) -> Dict[str, Any]:
        """对同一任务加跨进程排他锁，避免恢复与新训练覆盖彼此。"""
        with FileLock(self.store.task_dir(task_id) / '.run.lock', blocking=False):
            return self._resume_locked(task_id, dry_run)

    def _resume_locked(self, task_id: str, dry_run: bool = False) -> Dict[str, Any]:
        """从人工审核时保存的候选、预算和 rollout 恢复自动闭环，不重复初始训练。"""
        task_dir = self.store.task_dir(task_id)
        state = PersistentStateMachine(task_dir / "state.json")
        if state.record.state == AgentState.COMPLETED:
            summary = read_json(task_dir / "summary.json")
            self._cleanup_completed_videos(task_dir, summary)
            return summary
        recovered = self._recover_interrupted_finalization(task_dir, state, dry_run)
        if recovered is not None:
            return recovered
        if state.record.state != AgentState.HUMAN_REVIEW:
            raise RuntimeError("只有 HUMAN_REVIEW 状态允许恢复闭环，当前状态：%s" %
                               state.record.state.value)
        summary_path = task_dir / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError("该任务尚未产生可恢复的训练候选，请重新下发任务")
        summary = read_json(summary_path)
        if bool(summary.get("dry_run", False)) != bool(dry_run):
            raise RuntimeError("恢复模式必须与原运行一致，不能把 Mock checkpoint 用于真实训练")
        experiment_id = summary.get("selected_experiment")
        if not experiment_id:
            raise RuntimeError("人工审核发生在训练前，当前没有可恢复 checkpoint")
        task = TaskSpec.parse_obj(read_json(task_dir / "task_spec.json"))
        execution = self._load_training_execution(task_dir, state, dry_run)
        if execution:
            execution_dir = task_dir / 'candidates' / str(execution['experiment_id'])
            manifest_path = execution_dir / 'manifest.json'
            if manifest_path.is_file() and (execution_dir / 'checkpoint_set.json').is_file():
                manifest = read_json(manifest_path)
                if manifest.get('training_result') in ('completed', 'dry_run_completed'):
                    experiment_id = execution['experiment_id']
        selected = self._restore_selected_candidate(task_dir, str(experiment_id))
        loop_status = read_json(task_dir / "loop_status.json") \
            if (task_dir / "loop_status.json").is_file() else {}
        used_iterations = int(summary.get("used_iterations", 0))
        remaining_iterations = int(summary.get("remaining_iterations", 0))
        used_revisions = int(summary.get("used_revisions", 0))
        # 旧版本在编译修订前就扣减预算。若阻塞发生后既没有谱系子节点，也没有
        # 新候选 manifest，说明修订从未真正生成，应以评估阶段快照恢复预算。
        blocking_path = task_dir / "loop_blocking_error.txt"
        lineage_path = task_dir / "lineage.json"
        lineage = read_json(lineage_path) if lineage_path.is_file() else {"edges": []}
        compiled_child = any(
            edge.get("parent") == selected["id"] and
            (self.store.candidate_dir(task_id, str(edge.get("child"))) / "manifest.json").is_file()
            for edge in lineage.get("edges", []))
        if blocking_path.is_file() and not compiled_child:
            used_revisions = int(loop_status.get("used_revisions", used_revisions))
        max_revisions = int(summary.get("max_revisions", task.training_budget.max_reward_revisions))
        budget = BudgetTracker(
            max_iterations=used_iterations + remaining_iterations,
            used_iterations=used_iterations,
            max_revisions=max_revisions,
            used_revisions=used_revisions,
        )
        if execution:
            saved_budget = BudgetTracker.parse_obj(execution['budget'])
            budget.used_iterations = max(budget.used_iterations, saved_budget.used_iterations)
            budget.used_revisions = max(budget.used_revisions, saved_budget.used_revisions)
            budget.max_iterations = min(budget.max_iterations, saved_budget.max_iterations)
            budget.max_revisions = min(budget.max_revisions, saved_budget.max_revisions)
        self._apply_persisted_admission_budget(task_dir, task, state, budget, dry_run)
        # 所有冻结预算与策略证据均核对后再短路；不得因恢复而重复昂贵评估或扣费。
        if budget.exhausted:
            return self._stop_exhausted_resume(task_dir, state, selected, budget, summary, dry_run)
        intent_path = task_dir / 'task_intent.json'
        if intent_path.is_file():
            intent = TaskIntentSpec.parse_obj(read_json(intent_path))
        else:
            # 旧运行不重设计意图：仅从已冻结的用户原文恢复最小输入，再核对原合同。
            intent = TaskIntentSpec(original_instruction=task.original_instruction, robot=task.robot,
                action_name=task.task_name, normalized_goal=task.original_instruction)
        semantics = TaskSemanticValidator.assess_task(intent, task)
        if semantics.status != 'CONSISTENT':
            raise RuntimeError('恢复被拒绝：原动作语义需要澄清或与验收合同冲突；%s。'
                               '请明确动作、旋转轴与接触方式后建立新实验，不能改写旧合同恢复。%s' % (
                                   '；'.join(semantics.issues), semantics.question or ''))
        loop_records = read_json(task_dir / "loop_history.json") \
            if (task_dir / "loop_history.json").is_file() else []
        start_round = max(1, int(loop_status.get("round", len(loop_records) + 1)))
        if execution and experiment_id != summary.get('selected_experiment'):
            start_round = max(start_round, int(execution.get('round_index', 0)) + 1)
        elif execution and execution.get('parent_experiment') and execution.get('phase') in ('training', 'blocked', 'compiled'):
            start_round = max(start_round, int(execution.get('round_index', 0)) + 1)
        # 旧阻塞文件只描述上一次已收束的人工复核。恢复后先清理，避免后续若被
        # 意外中断，恢复逻辑把陈旧原因误判为本轮错误。
        blocking_path = task_dir / "loop_blocking_error.txt"
        if blocking_path.is_file():
            blocking_path.unlink()
        state.transition(AgentState.ROLLOUT_COLLECTING, {
            "reason": "正在从已有 checkpoint 恢复自动闭环", "loop_round": start_round,
            "reward_version": selected["plan"].version,
        })
        return self._evaluate(
            task, task_dir, state, selected, dry_run, budget,
            start_round=start_round, existing_records=loop_records, reuse_first_rollout=True)

    @staticmethod
    def _budget_exhaustion_reason(selected: Dict[str, Any], budget: BudgetTracker,
                                  dry_run: bool) -> str:
        """区分已完成的策略训练与未启动的下一轮，不暗示任务已经通过验收。"""
        version = selected['plan'].version
        training_kind = '模拟训练' if dry_run else '奖励训练'
        return ('自动闭环预算耗尽；已完成 v%s %s（不代表验收通过）；'
                'PPO 训练迭代 %s/%s，奖励修订 %s/%s；'
                '本次下一轮未启动训练；已保留已有评估、失败分析与建议；'
                '恢复不会重新调用 Provider，仍可单独评估已有 checkpoint') % (
                    version, training_kind, budget.used_iterations, budget.max_iterations,
                    budget.used_revisions, budget.max_revisions)

    def _stop_exhausted_resume(self, task_dir: Path, state: PersistentStateMachine,
                               selected: Dict[str, Any], budget: BudgetTracker,
                               summary: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
        """耗尽的恢复仅刷新预算提示，保留历史、训练快照和验收结果且不请求模型。"""
        result = dict(summary)
        result.setdefault('reason_before_budget_gate', summary.get('reason'))
        if result.get('selected_experiment') != selected['id']:
            # 若训练中断后发现完整子策略，不能把父策略 final 文件冒充该子策略的产物。
            result.update({
                'selected_experiment': selected['id'],
                'checkpoint': str(Path(selected['checkpoint']).relative_to(task_dir)),
                'config': str((selected['dir'] / 'config.yaml').relative_to(task_dir)),
                'rollout': None, 'last_decision': None,
                'evaluation_seeds': list(selected.get('checkpoint_seeds', [])),
            })
        result.update(budget.summary_fields())
        result.update({
            'state': AgentState.HUMAN_REVIEW.value, 'result': 'human_review',
            'reason': self._budget_exhaustion_reason(selected, budget, dry_run),
            'reward_version': selected['plan'].version,
            'resume_allowed': False, 'next_training_started': False,
        })
        state.record.context.update({'reason': result['reason'], 'training_budget': budget.dict()})
        state.record.updated_at = utc_now()
        state._save()
        ReportBuilder().build(task_dir.name, result, task_dir)
        state.events.emit('resume_budget_exhausted', budget.summary_fields())
        return result

    def _apply_persisted_admission_budget(self, task_dir: Path, task: TaskSpec,
                                          state: PersistentStateMachine,
                                          budget: BudgetTracker, dry_run: bool) -> None:
        """恢复时取持久化准入、任务和当前配置的最小上限，保留所有已用额度。"""
        self._validate_acceptance_contract(task_dir, task, state)
        if min(budget.used_iterations, budget.used_revisions, budget.max_iterations, budget.max_revisions) < 0:
            raise RuntimeError('恢复被拒绝：持久化预算不能为负数')
        compatibility_path = task_dir / 'environment_compatibility.json'
        manifest_path = task_dir / 'environment_manifest.json'
        if compatibility_path.is_file() and manifest_path.is_file():
            current = compatibility_fingerprint(self.settings.training_root,
                EnvironmentInspector(self.settings.training_root).inspect(task.robot).dict())
            if current != read_json(compatibility_path):
                raise RuntimeError('恢复被拒绝：机器人环境或奖励实现兼容指纹已改变')
        iteration_cap = min(task.training_budget.max_total_iterations,
                            self.settings.max_total_iterations)
        revision_cap = min(task.training_budget.max_reward_revisions,
                           self.settings.max_reward_revisions)
        path = task_dir / "training_admission.json"
        if path.is_file():
            admission = TrainingAdmissionReport.parse_obj(read_json(path))
            if admission.run_id != state.record.context.get("run_id"):
                raise RuntimeError("恢复被拒绝：训练准入 run_id 与当前运行不一致")
            if not admission.allowed or (admission.decision == "DRY_RUN_ONLY" and not dry_run):
                raise RuntimeError("恢复被拒绝：当前准入记录不允许该训练模式")
            iteration_cap = min(iteration_cap, admission.max_iterations)
            revision_cap = min(revision_cap, admission.max_revisions)
            if admission.decision == "ALLOW_BUDGETED_EXPLORATION":
                iteration_cap = min(iteration_cap, self.settings.exploration_max_iterations)
                revision_cap = min(revision_cap, self.settings.exploration_max_revisions)
                budget.reserve_for_revisions = True
        budget.max_iterations = min(budget.max_iterations, iteration_cap)
        budget.max_revisions = min(budget.max_revisions, revision_cap)
        if budget.used_iterations > budget.max_iterations or budget.used_revisions > budget.max_revisions:
            raise RuntimeError('恢复被拒绝：已用预算超过当前允许上限，不能退款或扩大上限')

    @staticmethod
    def _task_acceptance_identity(task: TaskSpec) -> Dict[str, Any]:
        """返回会改变动作含义或验收范围的稳定任务字段。"""
        return {
            "task_id": task.task_id,
            "task_name": task.task_name,
            "original_instruction": task.original_instruction,
            "normalized_description": task.normalized_description,
            "initial_state": task.initial_state,
            "required_behaviors": [item.dict() for item in task.required_behaviors],
            "forbidden_behaviors": [item.dict() for item in task.forbidden_behaviors],
            "phases": [item.dict() for item in task.phases],
            "required_observations": list(task.required_observations),
            "required_sensors": list(task.required_sensors),
            "visual_evaluation_requirements": list(task.visual_evaluation_requirements),
            "velocity_frame": task.velocity_frame,
        }

    @staticmethod
    def _validate_acceptance_contract(task_dir: Path, task: TaskSpec,
                                      state: PersistentStateMachine) -> None:
        """核对运行身份及固定验收指标，防止恢复或奖励修订时悄悄降低验收标准。"""
        path = task_dir / "acceptance_contract.json"
        if not path.is_file():
            raise RuntimeError("恢复被拒绝：旧任务缺少验收合同，请重新下发任务")
        contract = read_json(path)
        expected = {
            "run_id": state.record.context.get("run_id"), "robot": task.robot,
            "success_metrics": [item.dict() for item in task.success_metrics],
            "safety_constraints": [item.dict() for item in task.safety_constraints],
        }
        if any(contract.get(key) != value for key, value in expected.items()):
            raise RuntimeError("验收合同不一致：改变目标/阈值应重新建立任务，不能复用旧运行")
        identity = dict(contract.get("task_identity") or {})
        identity.setdefault("velocity_frame", "body")
        identity["phases"] = [dict(p, scope=p.get("scope", "execution"))
                              for p in identity.get("phases", [])]
        if identity != TrainingOrchestrator._task_acceptance_identity(task):
            raise RuntimeError("验收合同不一致：改变任务动作或行为要求应重新建立任务")

    def _run(self, instruction: str, robot: str, dry_run: bool, stop_after_plan: bool) -> Dict[str, Any]:
        """锁住完整任务运行，保护状态、验收合同和最终结果。"""
        task_dir = self.store.task_dir(self._task_id(instruction, robot))
        with FileLock(task_dir / '.run.lock', blocking=False):
            return self._run_locked(instruction, robot, dry_run, stop_after_plan)

    def _run_locked(self, instruction: str, robot: str, dry_run: bool, stop_after_plan: bool) -> Dict[str, Any]:
        """执行受限子流程并返回结构化结果。"""
        if robot not in self.settings.allowed_robots:
            raise ValueError("unsupported robot: %s" % robot)
        task_id = self._task_id(instruction, robot)
        run_id = 'run-' + uuid.uuid4().hex[:12]
        task_dir = self.store.initialize_task(task_id)
        self._clear_stale_run_outputs(task_dir)
        state = PersistentStateMachine(task_dir / "state.json")
        state.start_new_run(run_id)
        write_json(task_dir / 'evaluation_protocol.json', {
            'run_id': run_id, 'dry_run': dry_run, 'seeds': list(self.settings.evaluation_seeds),
            'rollouts_per_seed': self.settings.rollouts_per_seed,
            'counterfactual_enabled': self.settings.counterfactual_enabled})
        if self.settings.memory_enabled:
            try:
                self.memory.save_procedural(self.procedural_memory.snapshot())
                self._write_working_memory(task_dir, state)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print("[记忆] 程序或工作记忆初始化失败，不阻塞训练：%s" % exc, flush=True)
        atomic_write_text(task_dir / "task_request.txt", instruction.rstrip() + "\n")
        manifest = self.inspect_environment(robot, self.settings.artifacts_path / "environment_manifest.json")
        write_json(task_dir / "environment_manifest.json", manifest)
        write_json(task_dir / "environment_compatibility.json",
                   compatibility_fingerprint(self.settings.training_root, manifest.dict()))
        state.transition(AgentState.ENVIRONMENT_INSPECTED)

        state.transition(AgentState.TASK_UNDERSTANDING, {"agent_role": "task_planner"})
        try:
            if hasattr(self.provider, "understand_task"):
                intent = self._provider_call(
                    task_dir, "task_planner", "understand_task",
                    lambda: self.provider.understand_task(instruction, robot))
            else:
                intent = TaskIntentSpec(
                    original_instruction=instruction, robot=robot,
                    action_name="custom_motion", normalized_goal=instruction,
                    required_behaviors=[instruction],
                    forbidden_behaviors=["身体触地", "关节或力矩超限"],
                    retrieval_keywords=[robot, instruction],
                )
        except ProviderError as exc:
            return self._block_initial_provider_failure(task_dir, state, "task_planner", exc)
        # 用户输入和上位机选择是任务身份的唯一来源，模型不得改写机器人或原始指令。
        intent.original_instruction = instruction
        intent.robot = robot
        write_json(task_dir / "task_intent.json", intent)
        semantic_block = self._check_task_semantics(task_dir, state, intent)
        if semantic_block is not None:
            return semantic_block
        capability_block = self._check_action_capabilities(task_dir, state, intent, manifest.dict())
        if capability_block is not None:
            return capability_block
        self._write_working_memory(task_dir, state)

        state.transition(AgentState.TASK_FEASIBILITY_CHECK, {
            "agent_role": "task_feasibility_pipeline"},
            operation_id="feasibility-start:" + run_id)

        def generate_prototype(task_intent):
            """调用语义动作阶段 Provider 并记录外部推理耗时。"""
            if not hasattr(self.provider, "generate_motion_prototype"):
                raise RuntimeError("当前 Provider 未实现动作阶段原型生成")
            return self._provider_call(
                task_dir, "motion_prototype", "generate_motion_prototype",
                lambda: self.provider.generate_motion_prototype(task_intent))

        def record_feasibility_stage(event):
            """将可行性子阶段写入状态机及当前任务的 JSONL 事件流。"""
            stage = str(event.get("stage", "unknown"))
            phase_states = {
                "motion_prototype_generation": AgentState.MOTION_PROTOTYPE_GENERATING,
                "static_motion_validation": AgentState.STATIC_MOTION_VALIDATION,
                "dynamic_motion_validation": AgentState.DYNAMIC_MOTION_VALIDATION,
            }
            next_state = phase_states.get(stage)
            if next_state is not None and event.get("status") == "started":
                state.transition(next_state, {
                    "feasibility_stage": stage,
                    "motion_type": event.get("motion_type"),
                }, operation_id="feasibility-phase:%s:%s" % (run_id, stage))
            state.events.emit("feasibility_stage_transition", {
                "task_id": task_id, "run_id": run_id,
                "operation_id": "feasibility:%s:%s" % (run_id, stage),
                **event,
            })

        feasibility = (FeasibilityPipeline.mocked(
            self.settings.training_root, prototype_provider=generate_prototype,
            stage_callback=record_feasibility_stage)
            if dry_run else FeasibilityPipeline(
                self.settings.training_root, prototype_provider=generate_prototype,
                stage_callback=record_feasibility_stage))
        compact_manifest = self.context_builder.compact_manifest(manifest.dict())
        feasibility_report = feasibility.assess(intent, compact_manifest)
        # 某些旧适配器未回传约束协议，使用同一确定性编译器补全，不改变物理结论。
        if not feasibility_report.motion_constraint_spec:
            try:
                feasibility_report.motion_constraint_spec = MotionConstraintCompiler.compile(intent).dict()
            except (TypeError, ValueError):
                pass
        policy = TrainingAdmissionPolicy()
        admission = policy.assess(feasibility_report, self.settings, dry_run=dry_run)
        simulation = feasibility_report.simulation_report or feasibility_report.physics_report
        # 没有目标 rollout 时，独立验证默认姿态环境；该结果只支持基础设施就绪。
        if (not dry_run and self.settings.feasibility_admission_mode == "budgeted_exploration"
                and admission.decision == "NEEDS_REVIEW"
                and admission.capability_status in ("SUPPORTED", "CAPABILITY_SUPPORTED")
                and feasibility_report.evidence_mode == "REAL"
                and feasibility_report.motion_type not in (None, "UNKNOWN")
                and feasibility_report.status not in (
                    FeasibilityStatus.MODEL_UNAVAILABLE, FeasibilityStatus.NEEDS_CLARIFICATION,
                    FeasibilityStatus.UNSUPPORTED)
                and policy.needs_environment_health(feasibility_report)):
            try:
                health = check_environment_health(self.settings.training_root, robot)
            except Exception as exc:
                health = {"backend": "isaacgym", "validated": False, "success": None,
                          "reason": "环境健康检查异常：%s" % str(exc)[:400]}
            write_json(task_dir / "environment_health.json", health)
            state.events.emit("environment_health_checked", {"run_id": run_id, **health})
            admission = policy.assess(feasibility_report, self.settings,
                                      environment_health=health)
        admission.run_id = run_id
        feasibility_report.training_admission = admission.dict()
        write_json(task_dir / "training_admission.json", admission)
        state.events.emit("training_admission_decided", admission.dict())
        write_json(task_dir / "feasibility_report.json", feasibility_report)
        state.events.emit("task_feasibility_completed", {
            "status": feasibility_report.status.value,
            "backend": feasibility_report.backend,
            "evidence_mode": feasibility_report.evidence_mode,
            "validation_level": feasibility_report.validation_level,
            "converted_model": feasibility_report.converted_model,
            "required_capabilities": feasibility_report.required_capabilities,
            "missing_requirements": feasibility_report.missing_requirements,
        })
        print("[可行性预检] %s（验证等级：%s）" % (
            feasibility_report.status.value, feasibility_report.validation_level), flush=True)
        print("[训练准入] %s：%s（总预算 %s 次迭代）" % (
            admission.decision, admission.reason, admission.max_iterations), flush=True)
        if not admission.allowed:
            is_unsupported = admission.decision == "REJECT"
            terminal = AgentState.FAILED if is_unsupported else AgentState.HUMAN_REVIEW
            details = feasibility_report.missing_requirements + feasibility_report.risks
            detail_text = "；证据/待处理项：" + "；".join(details[:5]) if details else ""
            reason = admission.reason + detail_text
            report = {
                "task_id": task_id, "state": terminal.value, "reason": reason,
                "feasibility_status": feasibility_report.status.value,
                "validation_level": feasibility_report.validation_level,
                "feasibility_report": "feasibility_report.json",
                "training_admission": admission.dict(),
                "missing_requirements": feasibility_report.missing_requirements,
                "risks": feasibility_report.risks,
                "recommendations": feasibility_report.recommendations,
            }
            write_json(task_dir / "blocking_report.json", report)
            state.transition(terminal, {"reason": reason,
                                       "feasibility_status": feasibility_report.status.value},
                            operation_id="feasibility-result:" + run_id)
            self._write_working_memory(task_dir, state)
            return report

        if self.settings.rag_enabled:
            state.transition(AgentState.RAG_RETRIEVING, {"rag_purpose": "task_design"})
        retrieval_query = "\n".join(
            [instruction, "机器人：%s" % robot, "动作：%s" % intent.action_name] +
            intent.retrieval_keywords)
        rag_context = self._retrieve_experience(
            retrieval_query + "\n需要设计任务、奖励和课程",
            "task_design", robot, exclude_task_id=task_id)
        write_json(task_dir / "rag" / "design_retrieval.json", rag_context)
        memory_context = self._retrieve_memory(
            retrieval_query, robot, exclude_task_id=task_id)
        write_json(task_dir / "memory" / "design_retrieval.json", memory_context)

        state.transition(AgentState.CONTEXT_BUILDING, {"agent_role": "context_builder"})
        context = self.context_builder.build(
            intent, manifest.dict(), rag_context, memory_context)
        context['current_task_identity'] = {'task_id': task_id, 'robot': robot,
                                            'original_instruction': instruction}
        context["training_admission"] = admission.dict()
        context["motion_constraints"] = feasibility_report.motion_constraint_spec
        context["robot_capability"] = {
            "model": feasibility_report.robot_model,
            "capability_checks": feasibility_report.capability_report,
            "source": "environment_manifest.json + feasibility_report.json",
        }
        context["probe_evidence"] = {
            "validation_level": feasibility_report.validation_level,
            "probe_status": admission.probe_status,
            "violations": simulation.get("violations", []),
            "risks": feasibility_report.risks[:6],
        }
        write_json(task_dir / "context_snapshot.json", context)

        state.transition(AgentState.PROMPT_COMPILING, {"agent_role": "prompt_compiler"})
        compiled_prompt = self.prompt_compiler.compile(
            intent, context, self.settings.num_reward_candidates, task_id=task_id)
        atomic_write_text(task_dir / "prompts" / "reward_design_prompt.md",
                          compiled_prompt["prompt"] + "\n")
        write_json(task_dir / "prompts" / "reward_design_prompt.json", {
            "version": compiled_prompt["version"], "sha256": compiled_prompt["sha256"]})
        self._write_working_memory(task_dir, state)

        augmented_capabilities = manifest.dict()
        augmented_capabilities["retrieved_experience"] = rag_context
        augmented_capabilities["long_term_memory"] = memory_context
        augmented_capabilities["task_intent"] = intent.dict()
        augmented_capabilities['current_task_identity'] = context['current_task_identity']
        augmented_capabilities["training_admission"] = admission.dict()
        augmented_capabilities["motion_constraints"] = feasibility_report.motion_constraint_spec
        augmented_capabilities["robot_capability"] = context["robot_capability"]
        augmented_capabilities["probe_evidence"] = context["probe_evidence"]
        design_request = {
            "instruction": instruction, "robot": robot,
            "task_intent": intent.dict(), "context": context,
            "prompt": {"version": compiled_prompt["version"],
                       "sha256": compiled_prompt["sha256"]},
            "provider_roles": {
                "mode": "mock" if dry_run else self.settings.provider,
                "task_planner": "mock" if dry_run else self.settings.task_planner_provider,
                "reward_designer": "mock" if dry_run else self.settings.reward_designer_provider,
                "visual_critic": "mock" if dry_run else self.settings.visual_critic_provider,
                "diagnosis": "mock" if dry_run else self.settings.diagnosis_provider,
            },
        }
        write_json(task_dir / "design_request.json", design_request)
        state.transition(AgentState.REWARD_DESIGNING, {"agent_role": "reward_designer"})
        try:
            if hasattr(self.provider, "design_task_bundle"):
                design = self._provider_call(
                    task_dir, "reward_designer", "design_task_bundle",
                    lambda: self.provider.design_task_bundle(compiled_prompt["prompt"]))
            else:
                design = self._provider_call(
                    task_dir, "reward_designer", "design_task_and_rewards",
                    lambda: self.provider.design_task_and_rewards(
                        instruction, robot, augmented_capabilities))
        except ProviderError as exc:
            return self._block_initial_provider_failure(task_dir, state, "reward_designer", exc)
        write_json(task_dir / "design_response.json", design)
        write_json(task_dir / "design_response_attempt_00.json", design)
        task = TaskSpec.parse_obj(design["task_spec"])
        write_json(task_dir / 'task_identity_normalization.json', {
            'provided': {'task_id': task.task_id, 'robot': task.robot,
                         'original_instruction': task.original_instruction},
            'effective': context['current_task_identity'], 'source': 'user_request',
            'reward_plan_task_ids': [plan.get('task_id') for plan in design['reward_plans']]})
        task.task_id = task_id
        task.robot = robot
        task.original_instruction = instruction
        task.training_budget.max_total_iterations = min(
            task.training_budget.max_total_iterations, admission.max_iterations)
        task.training_budget.max_reward_revisions = min(
            task.training_budget.max_reward_revisions, admission.max_revisions)
        task_adjustments = self._normalize_task_for_instruction(task)
        metric_mappings = normalize_task_metrics(task, manifest.evaluation_metrics)
        write_json(task_dir / "task_normalization.json", {
            "adjustments": task_adjustments, "metric_mappings": metric_mappings})
        unsupported, derivable = self.inspector.validate_task(task, manifest)
        write_json(task_dir / "task_spec.json", task)
        state.transition(AgentState.TASK_DESIGNED, {"unsupported": unsupported, "derivable": derivable})
        semantic_block = self._check_task_semantics(task_dir, state, intent, task)
        if semantic_block is not None:
            return semantic_block
        contract_block = self._check_task_contract(task_dir, state, task)
        if contract_block is not None:
            return contract_block
        if unsupported:
            report = {"task_id": task_id, "state": AgentState.HUMAN_REVIEW.value,
                      "unsupported_requirements": unsupported,
                      "reason": "任务所需物理量或验收指标尚不支持：%s" % "、".join(unsupported)}
            write_json(task_dir / "blocking_report.json", report)
            state.transition(AgentState.HUMAN_REVIEW)
            self._write_working_memory(task_dir, state)
            return report

        # 验收合同绑定当前运行，后续奖励修订复用同一 TaskSpec，不能借探索放行降低阈值。
        write_json(task_dir / "acceptance_contract.json", {
            "run_id": run_id, "robot": robot, "instruction": instruction,
            "task_identity": self._task_acceptance_identity(task),
            "success_metrics": [item.dict() for item in task.success_metrics],
            "safety_constraints": [item.dict() for item in task.safety_constraints],
            "source": "task_spec.json", "admission_decision": admission.decision,
        })

        plans = [RewardPlan.parse_obj(item) for item in design["reward_plans"]]
        if not plans:
            raise ValueError("provider returned no reward candidates")
        plans = plans[:self.settings.num_reward_candidates]
        normalization_records = []
        for plan in plans:
            provided_task_id = plan.task_id
            plan.task_id = task_id
            adjustments = self._normalize_plan_for_task(task, plan)
            normalization_records.append({"version": plan.version, "adjustments": adjustments,
                                          'provided_task_id': provided_task_id, 'effective_task_id': task_id})
        write_json(task_dir / "plan_normalization.json", normalization_records)

        state.transition(AgentState.REWARD_REVIEWING, {"agent_role": "reward_reviewer"})
        review = self.reward_reviewer.review(
            intent, task, plans, dict(manifest.dict(), require_objective_links=not dry_run,
                     max_abs_reward_weight=self.settings.max_abs_reward_weight), design.get("reward_hacking_risks", []))
        from ..rewards.quality_gate import capability_gaps
        local_gaps = capability_gaps(task, manifest.dict())
        write_json(task_dir / 'reward_capability_gaps.json', {'gaps': local_gaps,
            'requires_local_implementation': bool(local_gaps)})
        for attempt in range(self.settings.reward_design_review_retries + 1):
            write_json(task_dir / ("reward_review_attempt_%02d.json" % attempt), review)
            if review.approved or local_gaps or attempt == self.settings.reward_design_review_retries:
                break
            try:
                plans, patch_audit = self._repair_reward_candidates(
                    task, task_dir, plans, review, manifest, task_dir / 'reward_repairs', attempt + 1,
                    require_objective_links=not dry_run)
                JsonlEventRecorder(task_dir / 'events.jsonl').emit('reward_field_repair', patch_audit)
                write_json(task_dir / ("design_response_attempt_%02d.json" % (attempt + 1)), dict(
                    design, task_spec=task.dict(), reward_plans=[plan.dict() for plan in plans]))
                # 已规范化来源由补丁白名单保护；不能再调用可能改写无关项的规范化。
                review = self.reward_reviewer.review(
                    intent, task, plans, dict(manifest.dict(), require_objective_links=not dry_run,
                     max_abs_reward_weight=self.settings.max_abs_reward_weight), design.get("reward_hacking_risks", []))
            except (ProviderError, ValueError, KeyError) as exc:
                atomic_write_text(task_dir / 'reward_repairs' / ('error_%02d.txt' % (attempt + 1)), str(exc) + '\n')
                review.conflicts.append("奖励局部修复失败：%s" % exc)
        if not review.approved and not local_gaps:
            from ..rewards.front_flip_baseline import BASELINE_VERSION, build_front_flip_baseline
            baseline = build_front_flip_baseline(task, manifest.dict(),
                version=max((plan.version for plan in plans), default=0) + 1)
            if baseline is not None:
                write_json(task_dir / 'reward_review_before_baseline.json', review)
                write_json(task_dir / 'local_reward_baseline.json', {
                    'source': BASELINE_VERSION, 'run_id': run_id,
                    'reason': '模型候选全部被拒绝；提交独立本地基线，保持冻结任务合同与原有审核',
                    'task_identity': self._task_acceptance_identity(task), 'reward_plan': baseline.dict()})
                plans.append(baseline)
                review = self.reward_reviewer.review(intent, task, plans,
                    dict(manifest.dict(), require_objective_links=True,
                         max_abs_reward_weight=self.settings.max_abs_reward_weight), design.get('reward_hacking_risks', []))
                write_json(task_dir / 'reward_review_with_baseline.json', review)
                JsonlEventRecorder(task_dir / 'events.jsonl').emit('local_reward_baseline', {
                    'source': BASELINE_VERSION, 'run_id': run_id, 'candidate_index': len(plans),
                    'reward_version': baseline.version, 'approved': len(plans) in review.passed_candidate_indexes})
                print('[奖励基线] %s：%s；原有验收、硬约束和预算继续执行' % (
                    BASELINE_VERSION, '审查通过' if review.approved else '审查未通过'), flush=True)
        # 保留实际用于审查/编译的设计，但强制绑定原始 TaskSpec。
        write_json(task_dir / "design_response.json", dict(
            design, task_spec=task.dict(), reward_plans=[plan.dict() for plan in plans]))
        write_json(task_dir / "reward_review.json", review)
        if not review.approved:
            issues = sorted({issue for candidate in review.candidate_results
                             for issue in candidate.conflicts + candidate.omissions})
            details = ('需补齐本地实现：' + '、'.join(local_gaps) if local_gaps else
                       '；'.join(issues[:3] or (review.conflicts + review.omissions)[:3]))
            report = {
                "task_id": task_id,
                "state": AgentState.HUMAN_REVIEW.value,
                "reason": "奖励审查未通过，已禁止启动训练；" + details,
                "capability_gaps": local_gaps,
                "omissions": review.omissions,
                "conflicts": review.conflicts,
                "rejected_candidate_indexes": review.rejected_candidate_indexes,
            }
            write_json(task_dir / "blocking_report.json", report)
            state.transition(AgentState.HUMAN_REVIEW, {"reason": report["reason"]})
            self._write_working_memory(task_dir, state)
            return report
        reviewed_plans = [(index, plans[index - 1])
                          for index in review.passed_candidate_indexes]
        state.transition(AgentState.REWARD_CANDIDATES_CREATED, {
            "candidate_count": len(reviewed_plans),
            "rejected_candidate_indexes": review.rejected_candidate_indexes,
        })
        compiler = RewardCompiler(manifest, self.settings.max_abs_reward_weight)
        lineage = LineageGraph()
        candidate_data: List[Dict[str, Any]] = []
        git_commit = self._git_commit()
        for index, plan in reviewed_plans:
            # 同一句自然语言任务会得到稳定 task_id，但每次运行必须使用独立候选目录，
            # 否则旧 checkpoint 可能被误选为本次冒烟训练结果。
            experiment_id = "%s-candidate-%02d-v%02d" % (run_id, index, plan.version)
            directory = self.store.candidate_dir(task_id, experiment_id)
            directory.mkdir(parents=True, exist_ok=True)
            for name in ("metrics", "checkpoints", "rollouts", "prompts", "responses"):
                (directory / name).mkdir(exist_ok=True)
            try:
                self._validate_plan_metric_coverage(task, plan)
                if not dry_run:
                    from ..rewards.quality_gate import audit_objectives, require_objectives
                    write_json(directory / 'reward_quality_preflight.json', audit_objectives(task, plan))
                    require_objectives(task, plan)
                metadata = compiler.compile(plan, directory, task=task)
            except RewardValidationError as exc:
                atomic_write_text(directory / "validation_error.txt", str(exc) + "\n")
                continue
            command = self.controller.adapter.training_command(robot, directory / "config.yaml",
                                                               self.settings.smoke_iterations, index,
                                                               experiment_id, num_envs=64)
            exp_manifest = ExperimentManifest(
                experiment_id=experiment_id, task_id=task_id, git_commit=git_commit,
                config_hash=metadata["config_hash"], reward_version=plan.version, seed=index,
                robot=robot, training_command=command,
                provider_status="mock" if dry_run else self.settings.provider,
                training_result="compiled")
            write_json(directory / "manifest.json", exp_manifest)
            lineage.add(experiment_id, None, plan.version, metadata["config_hash"])
            candidate_data.append({"id": experiment_id, "dir": directory, "plan": plan,
                                   "manifest": exp_manifest, "metadata": metadata})
        if not candidate_data:
            state.transition(AgentState.FAILED, {"reason": "all reward plans failed validation"})
            raise RuntimeError("所有奖励计划均未通过安全校验；请检查各候选的 validation_error.txt")
        lineage.save(task_dir / "lineage.json")
        required_metrics = [item.name for item in task.success_metrics if item.required]
        safety_metrics = [item.name for item in task.safety_constraints if item.required]
        metrics_ready = (bool(required_metrics) and bool(safety_metrics) and
                         set(required_metrics + safety_metrics) <= set(manifest.evaluation_metrics))
        physical_training_ready = training_ready(
            feasibility_report.validation_level,
            [item["metadata"] for item in candidate_data], required_metrics)
        initial_cost = len(candidate_data) * max(
            self.settings.smoke_iterations, self.settings.screening_iterations)
        reserved_rounds = 1 + (task.training_budget.max_reward_revisions if not dry_run else 0)
        minimum_training_iterations = initial_cost + len(
            self.settings.evaluation_seeds) * reserved_rounds
        budget_ready = task.training_budget.max_total_iterations >= minimum_training_iterations
        readiness = {
            "ready": admission.allowed and metrics_ready and budget_ready,
            "admission_decision": admission.decision,
            "validation_level": (ValidationLevel.TRAINING_READY.value
                                 if physical_training_ready and metrics_ready and budget_ready
                                 else feasibility_report.validation_level),
            "physics_validated": admission.decision == "ALLOW_TRAINING",
            "reward_config_count": len(candidate_data),
            "required_evaluation_metrics": required_metrics,
            "required_safety_metrics": safety_metrics,
            "max_iterations": task.training_budget.max_total_iterations,
            "max_revisions": task.training_budget.max_reward_revisions,
            "minimum_training_iterations": minimum_training_iterations,
            "scope": "simulation_only",
            "reason": ("准入、奖励编译、验收指标与预算均已就绪" if metrics_ready and budget_ready
                       else "训练前置条件不完整：需注册成功/安全指标，并有足够候选筛选和多种子预算"),
        }
        feasibility_report.training_readiness = readiness
        write_json(task_dir / "feasibility_report.json", feasibility_report)
        write_json(task_dir / "training_readiness.json", readiness)
        state.events.emit("training_readiness_checked", readiness)
        state.transition(AgentState.CONFIGS_COMPILED, {
            "training_readiness": readiness or {"required": False,
                                                "reason": "静态姿态任务不申请动态 TRAINING_READY 等级"},
        })
        if not readiness["ready"]:
            state.transition(AgentState.HUMAN_REVIEW, {"reason": readiness["reason"]})
            self._write_working_memory(task_dir, state)
            return {"task_id": task_id, "state": AgentState.HUMAN_REVIEW.value,
                    "reason": readiness["reason"],
                    "training_readiness": "training_readiness.json"}
        state.transition(AgentState.VALIDATED, {
            "training_readiness": readiness or {"required": False},
        })
        if stop_after_plan:
            summary = {"task_id": task_id, "state": AgentState.VALIDATED.value,
                       "run_id": run_id, "candidates": [item["id"] for item in candidate_data],
                       "dry_run": True}
            write_json(task_dir / "summary.json", summary)
            self._write_working_memory(task_dir, state)
            return summary

        # 模型可以提出更小的任务预算，但不能突破本地配置的资源上限。
        budget = BudgetTracker(
            max_iterations=min(task.training_budget.max_total_iterations, self.settings.max_total_iterations),
            max_revisions=min(task.training_budget.max_reward_revisions, self.settings.max_reward_revisions),
            reserve_for_revisions=not dry_run or admission.decision == "ALLOW_BUDGETED_EXPLORATION",
        )
        initial_screening_cost = len(candidate_data) * max(
            self.settings.smoke_iterations, self.settings.screening_iterations)
        minimum_training_iterations = initial_screening_cost + len(
            self.settings.evaluation_seeds) * (1 + (
                budget.max_revisions if budget.reserve_for_revisions else 0))
        if minimum_training_iterations > budget.max_iterations:
            report = {
                "task_id": task_id, "state": AgentState.HUMAN_REVIEW.value,
                "reason": "任务训练预算不足以完成候选筛选和每个评估种子的至少一次训练",
                "required_minimum_iterations": minimum_training_iterations,
                "available_iterations": budget.max_iterations,
            }
            write_json(task_dir / "blocking_report.json", report)
            state.transition(AgentState.HUMAN_REVIEW, {"reason": report["reason"]})
            self._write_working_memory(task_dir, state, budget=budget)
            return report
        if dry_run:
            selected = self._dry_train(task, task_dir, state, candidate_data, budget)
        else:
            selected = self._real_train(task, task_dir, state, candidate_data, budget)
        return self._evaluate(task, task_dir, state, selected, dry_run, budget)

    def _dry_train(self, task: TaskSpec, task_dir: Path, state: PersistentStateMachine,
                   candidates: List[Dict[str, Any]], budget: BudgetTracker) -> Dict[str, Any]:
        """离线模拟候选筛选和多种子完整训练阶段。"""
        state.transition(AgentState.SMOKE_TRAINING)
        for index, item in enumerate(candidates):
            self._reserve_training_operation(task_dir, state, budget, item['id'] + ':smoke',
                self.settings.smoke_iterations, dry_run=True, experiment_id=item['id'], parent_experiment=None)
            metrics = {"success_rate": 0.75 + index * 0.05, "task_score": 0.8 - index * 0.02,
                       "ppo_stability": 0.9, "energy": 0.1 + index * 0.02,
                       "reward_hacking_score": 0.0, "hard_constraints_passed": 1.0}
            write_json(item["dir"] / "metrics" / "smoke.json", metrics)
            atomic_write_text(item["dir"] / "stdout.log", "dry-run smoke training completed\n")
            atomic_write_text(item["dir"] / "stderr.log", "")
            self._complete_training_operation(task_dir, state, budget, item['id'] + ':smoke')
        state.transition(AgentState.CANDIDATE_SCREENING)
        screening_increment = max(0, self.settings.screening_iterations - self.settings.smoke_iterations)
        for index, item in enumerate(candidates):
            self._reserve_training_operation(task_dir, state, budget, item['id'] + ':screening',
                screening_increment, dry_run=True, experiment_id=item['id'], parent_experiment=None)
            screening = {"iteration": self.settings.screening_iterations,
                         "hard_constraints_passed": True, "task_score": 0.82 - index * 0.04,
                         "success_rate": 0.8 - index * 0.03, "ppo_stability": 0.92 - index * 0.02,
                         "energy": 0.12 + index * 0.02, "reward_hacking_score": 0.0}
            screening["composite_score"] = (0.4 * screening["task_score"] + 0.3 * screening["success_rate"] +
                                             0.2 * screening["ppo_stability"] - 0.05 * screening["energy"] -
                                             0.05 * screening["reward_hacking_score"])
            write_json(item["dir"] / "metrics" / "screening.json", screening)
            item["screening_score"] = screening["composite_score"]
            self._complete_training_operation(task_dir, state, budget, item['id'] + ':screening')
        selected = self._select_safe_candidate(candidates)
        screening_path = selected["dir"] / "metrics" / "screening.json"
        selected_screening = read_json(screening_path)
        selected_screening["selected"] = True
        write_json(screening_path, selected_screening)
        state.transition(AgentState.FULL_TRAINING)
        checkpoints = []
        per_seed = budget.per_seed_allocation(
            self.settings.full_iterations, len(self.settings.evaluation_seeds))
        if per_seed <= 0:
            raise RuntimeError("initial full-training iteration budget exhausted")
        for seed in self.settings.evaluation_seeds:
            operation = selected['id'] + ':full:seed:%d' % seed
            self._reserve_training_operation(task_dir, state, budget, operation, per_seed,
                dry_run=True, experiment_id=selected['id'], parent_experiment=None, round_index=0)
            checkpoint = selected["dir"] / "checkpoints" / ("seed_%d" % seed) / ("model_%d.pt" % per_seed)
            atomic_write_text(checkpoint, "dry-run checkpoint placeholder; never deploy to hardware\n")
            checkpoints.append(checkpoint)
            self._complete_training_operation(task_dir, state, budget, operation)
        selected["checkpoints"] = checkpoints
        selected["checkpoint_seeds"] = list(self.settings.evaluation_seeds)
        selected["checkpoint"] = checkpoints[0]
        selected["manifest"].training_result = "dry_run_completed"
        selected["manifest"].iteration = per_seed
        selected["manifest"].checkpoint = str(checkpoints[0].relative_to(selected["dir"]))
        write_json(selected["dir"] / "manifest.json", selected["manifest"])
        self._save_checkpoint_set(task_dir, selected, True)
        self._persist_training_execution(task_dir, state, budget, phase='trained')
        return selected

    def _pilot_reward_quality(self, task: TaskSpec, item: Dict[str, Any], checkpoint: Path,
                              stage: str) -> bool:
        """短训练后检查真实奖励触发与同轨迹物理证据，不要求已完成最终动作。"""
        from ..rewards.quality_gate import audit_rollout
        directory = item['dir'] / 'reward_quality' / stage
        result = self.controller.run_evaluation_rollouts(
            item['id'], task.robot, item['dir'] / 'config.yaml', checkpoint,
            directory, seed=item['manifest'].seed, fps=self.settings.video_fps)
        if result.exit_code != 0:
            write_json(directory / 'audit.json', {'passed': False, 'errors': ['奖励审计 rollout 失败']})
            return False
        media = self._media_for_rollout(directory)
        try:
            trajectory = pd.read_parquet(media['trajectory'])
            rewards = pd.read_parquet(media['rewards'])
            collector = TrajectoryMetrics()
            metrics = (collector.compute_pilot(trajectory)
                       if classify_forward_flip(task.original_instruction) == ForwardFlipKind.AERIAL
                       else collector.compute(trajectory))
            report = audit_rollout(task, item['plan'], trajectory, rewards, metrics,
                                   metadata=read_json(media['metadata']), require_contact_metadata=True)
            if metrics.get('pilot_reset_count', 0) > 0:
                report['warnings'].append('早期策略发生原生重置，进度按独立 episode 计算；不代表动作完成')
        except (OSError, ValueError, KeyError) as exc:
            write_json(directory / 'audit.json', {'passed': False, 'errors': ['奖励审计证据不可用：' + str(exc)]})
            return False
        previous = item.get('pilot_quality')
        if previous:
            from ..rewards.quality_gate import compare_pilots
            report['trend'] = compare_pilots(task, previous, report)
            if report['trend']['suspected_reward_hacking']:
                report['passed'] = False
                report['errors'].append('奖励增长而目标指标倒退，候选暂停；须重新设计而非追加长训练')
        item['pilot_quality'] = report
        write_json(directory / 'audit.json', report)
        return report['passed']

    def _real_train(self, task: TaskSpec, task_dir: Path, state: PersistentStateMachine,
                    candidates: List[Dict[str, Any]], budget: BudgetTracker) -> Dict[str, Any]:
        """调用实际 Unitree 入口完成候选筛选和多种子训练。"""
        state.transition(AgentState.SMOKE_TRAINING)
        successful = []
        for item in candidates:
            operation = item['id'] + ':smoke'
            self._reserve_training_operation(task_dir, state, budget, operation,
                self.settings.smoke_iterations, dry_run=False, experiment_id=item['id'], parent_experiment=None)
            item["manifest"].started_at = utc_now()
            result = self.controller.run_smoke_training(item["id"], task.robot, item["dir"] / "config.yaml",
                                                        self.settings.smoke_iterations, item["manifest"].seed,
                                                        item["dir"])
            item["manifest"].ended_at = utc_now()
            item["manifest"].training_result = "completed" if result.exit_code == 0 else "failed"
            item["manifest"].failure_reason = "timeout" if result.timed_out else (None if result.exit_code == 0 else "process_exit_%d" % result.exit_code)
            write_json(item["dir"] / "manifest.json", item["manifest"])
            self._complete_training_operation(task_dir, state, budget, operation)
            CheckpointManager.prune(
                item["dir"], keep_per_run=self.settings.checkpoints_per_run)
            if result.exit_code == 0:
                checkpoint = CheckpointManager.latest(item['dir'])
                if checkpoint is not None and self._pilot_reward_quality(task, item, checkpoint, 'smoke'):
                    successful.append(item)
        if not successful:
            reason = '所有候选冒烟训练或奖励完整性审计失败；检查 reward_quality/smoke/audit.json 与 stderr.log'
            write_json(task_dir / 'blocking_report.json', {'state': 'HUMAN_REVIEW', 'reason': reason,
                'used_iterations': budget.used_iterations, 'candidates': [item['id'] for item in candidates]})
            state.transition(AgentState.HUMAN_REVIEW, {'reason': reason})
            raise RuntimeError(reason)
        state.transition(AgentState.CANDIDATE_SCREENING)
        screening_increment = max(0, self.settings.screening_iterations - self.settings.smoke_iterations)
        screened = []
        for item in successful:
            checkpoint = CheckpointManager.latest(item["dir"])
            if checkpoint is None:
                continue
            operation = item['id'] + ':screening'
            self._reserve_training_operation(task_dir, state, budget, operation,
                screening_increment, dry_run=False, experiment_id=item['id'], parent_experiment=None)
            result = self.controller.continue_training(item["id"], task.robot, item["dir"] / "config.yaml",
                                                       screening_increment, item["manifest"].seed,
                                                       item["dir"], checkpoint)
            self._complete_training_operation(task_dir, state, budget, operation)
            if result.exit_code != 0:
                continue
            CheckpointManager.prune(
                item["dir"], keep_per_run=self.settings.checkpoints_per_run)
            checkpoint = CheckpointManager.latest(item['dir'])
            if checkpoint is None or not self._pilot_reward_quality(task, item, checkpoint, 'screening'):
                continue
            latest = TensorBoardReader().latest(item["dir"])
            ppo = PPOCollector().collect(item["dir"])
            task_score, proxy_terms = self._task_reward_proxy(task, latest)
            finite_metrics = bool(latest) and all(math.isfinite(float(value)) for value in latest.values())
            finite_ppo = all(math.isfinite(float(value)) for value in
                             (ppo.value_loss, ppo.surrogate_loss, ppo.learning_rate, ppo.mean_noise_std))
            safety = finite_metrics and finite_ppo
            stability = 1.0 / (1.0 + max(0.0, ppo.value_loss)) if finite_ppo else 0.0
            score = 0.65 * task_score + 0.35 * stability if safety else None
            screening = {"iteration": self.settings.screening_iterations,
                         "task_proxy_terms": proxy_terms,
                         "hard_constraints_passed": safety, "task_proxy": task_score,
                         "ppo_stability": stability, "composite_score": score,
                         "note": "必须通过数值健康门槛；最终完成仍需确定性 rollout 和视觉验收"}
            write_json(item["dir"] / "metrics" / "screening.json", screening)
            screened.append(item)
        if not screened:
            reason = '所有候选筛选训练或奖励趋势审计失败；检查 reward_quality/screening/audit.json'
            write_json(task_dir / 'blocking_report.json', {'state': 'HUMAN_REVIEW', 'reason': reason,
                'used_iterations': budget.used_iterations, 'candidates': [item['id'] for item in successful]})
            state.transition(AgentState.HUMAN_REVIEW, {'reason': reason})
            raise RuntimeError(reason)
        try:
            selected = self._select_safe_candidate(screened)
        except RuntimeError:
            state.transition(AgentState.FAILED, {"reason": "all candidate hard constraints failed"})
            raise
        selected_screening = read_json(selected["dir"] / "metrics" / "screening.json")
        selected_screening["selected"] = True
        write_json(selected["dir"] / "metrics" / "screening.json", selected_screening)
        state.transition(AgentState.FULL_TRAINING)
        validation_checkpoints = []
        per_seed = budget.per_seed_allocation(
            self.settings.full_iterations, len(self.settings.evaluation_seeds))
        if per_seed <= 0:
            raise RuntimeError("initial full-training iteration budget exhausted")
        for seed in self.settings.evaluation_seeds:
            remaining = per_seed
            operation = selected['id'] + ':full:seed:%d' % seed
            self._reserve_training_operation(task_dir, state, budget, operation, remaining,
                dry_run=False, experiment_id=selected['id'], parent_experiment=None, round_index=0)
            before = set(CheckpointManager.list_checkpoints(selected["dir"]))
            result = self.controller.run_full_training(selected["id"] + "-seed-%d" % seed, task.robot,
                                                       selected["dir"] / "config.yaml", remaining, seed,
                                                       selected["dir"])
            if result.exit_code != 0:
                state.transition(AgentState.FAILED, {"reason": "full training failed", "seed": seed})
                raise RuntimeError("full random-initialized training failed for seed %d" % seed)
            new_checkpoints = [path for path in CheckpointManager.list_checkpoints(selected["dir"]) if path not in before]
            checkpoint = self._checkpoint_for_seed(selected['dir'], seed)
            if checkpoint not in new_checkpoints:
                checkpoint = None
            if checkpoint is None:
                state.transition(AgentState.FAILED, {
                    "reason": "full training produced no new seed checkpoint", "seed": seed})
                raise RuntimeError("full training produced no checkpoint for seed %d" % seed)
            CheckpointManager.prune(
                checkpoint.parent, keep_per_run=self.settings.checkpoints_per_run,
                protected=[checkpoint])
            validation_checkpoints.append(checkpoint)
            self._complete_training_operation(task_dir, state, budget, operation)
        selected["checkpoints"] = validation_checkpoints
        selected["checkpoint_seeds"] = list(self.settings.evaluation_seeds)
        selected["checkpoint"] = validation_checkpoints[0]
        selected["manifest"].training_result = "completed"
        selected["manifest"].iteration = per_seed
        selected["manifest"].checkpoint = str(validation_checkpoints[0].relative_to(selected["dir"]))
        write_json(selected["dir"] / "manifest.json", selected["manifest"])
        self._save_checkpoint_set(task_dir, selected, False)
        self._persist_training_execution(task_dir, state, budget, phase='trained')
        return selected

    @staticmethod
    def _task_reward_proxy(task: TaskSpec, latest: Dict[str, float]) -> tuple:
        """按动作选择原始奖励代理；代理只用于筛选，不替代最终物理验收。"""
        text = (task.task_name + ' ' + task.original_instruction).lower()
        if classify_forward_flip(task.original_instruction) == ForwardFlipKind.AERIAL:
            terms = ['takeoff_velocity', 'airborne_duration', 'pitch_rotation',
                     'net_pitch_progress', 'front_flip_recovery']
        elif '侧翻' in text or 'side_flip' in text:
            terms = ['takeoff_velocity', 'airborne_duration', 'roll_rotation', 'landing_recovery']
        elif is_front_leg_support_text((task.original_instruction,)):
            terms = ['front_leg_stand', 'front_leg_walk']
        elif '后腿' in text or 'rear_leg' in text:
            terms = ['rear_leg_stand', 'rear_leg_walk']
        elif task.task_name == 'jump' or '跳' in text:
            terms = ['jump_height', 'landing_stability']
        elif '转' in text or 'turn' in text:
            terms = ['tracking_ang_vel']
        else:
            terms = ['tracking_lin_vel']
        values = [float(latest['Episode/rew_raw_mean_' + name]) for name in terms
                  if 'Episode/rew_raw_mean_' + name in latest]
        return (float(statistics.mean(values)) if values else 0.0), terms

    @staticmethod
    def _rollout_score(task: TaskSpec, metrics: Dict[str, float]) -> float:
        """按任务进度加分并对碰撞、跌倒和姿态偏差扣分，用于选择中位 rollout。"""
        scores = []
        for threshold in task.success_metrics:
            value = metrics.get(threshold.name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                scores.append(0.0)
            elif threshold.operator in ('>=', '>'):
                scores.append(min(1.0, max(0.0, value / max(abs(threshold.value), 1e-6))))
            elif threshold.operator in ('<=', '<'):
                scores.append(1.0 / (1.0 + max(0.0, value - threshold.value) / max(abs(threshold.value), 1e-6)))
            else:
                scores.append(float(value == threshold.value))
        progress = statistics.mean(scores) if scores else 0.0
        def penalty(name: str) -> float:
            """缺失安全证据视为最差样本，避免 NaN 分数破坏视觉样本排序。"""
            value = metrics.get(name)
            return max(0.0, float(value)) if isinstance(value, (int, float)) and math.isfinite(value) else 1e6

        return progress - 0.1 * penalty("forbidden_collisions") - 2.0 * penalty("fall_rate")

    @staticmethod
    def _aggregate_rollout_metrics(task: TaskSpec,
                                   records: List[Dict[str, Any]]) -> Dict[str, float]:
        """跨全部种子和 rollout 保守聚合指标，防止单个偶然好样本触发完成。"""
        if not records:
            return {}
        thresholds = {item.name: item for item in
                      list(task.success_metrics) + list(task.safety_constraints)}
        names = sorted({name for record in records for name in record["metrics"]})
        aggregate: Dict[str, float] = {}
        implicit_upper_bounds = {
            "nan_count", "joint_limit_violations", "torque_limit_violations",
            "forbidden_collisions", "abnormal_terminations", "fall_rate",
        }
        for name in names:
            raw_values = [record["metrics"].get(name) for record in records]
            values = [float(value) if isinstance(value, (int, float)) else float("nan")
                      for value in raw_values]
            if not all(math.isfinite(value) for value in values):
                aggregate[name] = float("nan")
                continue
            threshold = thresholds.get(name)
            if name in implicit_upper_bounds or (threshold and threshold.operator in ("<", "<=")):
                aggregate[name] = max(values)
            elif threshold and threshold.operator in (">", ">="):
                aggregate[name] = min(values)
            elif threshold and threshold.operator == "==":
                aggregate[name] = max(values, key=lambda value: abs(value - threshold.value))
            else:
                aggregate[name] = float(statistics.median(values))
        aggregate["evaluated_rollout_count"] = float(len(records))
        return aggregate

    @staticmethod
    def _media_for_rollout(rollout_dir: Path) -> Dict[str, Path]:
        """返回一个既有 rollout 的标准媒体与数据文件映射。"""
        media = {name: rollout_dir / (name + ".mp4") for name in ("front", "side", "overview")}
        media.update({"trajectory": rollout_dir / "trajectory.parquet",
                      "rewards": rollout_dir / "rewards.parquet",
                      "metadata": rollout_dir / "metadata.json"})
        return media

    @staticmethod
    def _counterfactual_applicable(task: TaskSpec) -> bool:
        """仅为要求响应多种命令的任务启用零命令反事实探针。"""
        names = {item.name for item in task.success_metrics}
        command_metrics = bool(names & {
            "tracking_error", "walking_speed_tracking", "rear_leg_walk_velocity_tracking",
            "front_leg_walk_velocity_tracking", "yaw_tracking_error",
        })
        if not command_metrics:
            return False
        # “以 1m/s 倒着走”这类任务训练的是固定动作策略；零命令停车既未训练，
        # 也不是用户验收目标，不能作为硬约束。只有用户明确要求按命令切换、停止
        # 或覆盖多种速度时，才检查 command=0 的条件控制能力。
        if TrainingOrchestrator._explicit_locomotion_command(task) is not None:
            descriptions = "\n".join(
                [task.original_instruction, task.normalized_description] +
                [item.description for item in task.required_behaviors])
            conditional_tokens = (
                "响应指令", "跟随指令", "根据指令", "不同速度", "速度切换",
                "零速度", "静止", "停下", "停止", "command-conditioned",
            )
            return any(token in descriptions for token in conditional_tokens)
        return True

    def _counterfactual_summary(self, rollout_dir: Path) -> Dict[str, Any]:
        """计算零命令下的漂移速度和安全结果。"""
        frame = pd.read_parquet(rollout_dir / "trajectory.parquet")
        speed = (frame["base_vx"].astype(float) ** 2 + frame["base_vy"].astype(float) ** 2) ** 0.5
        finite = bool(len(speed)) and all(math.isfinite(float(value)) for value in speed)
        max_speed = float(speed.max()) if finite else float("inf")
        fall_rate = float(frame["fall"].astype(float).mean()) if "fall" in frame else 1.0
        safety = TrajectoryMetrics().compute(frame)
        mandatory = ('nan_count', 'joint_limit_violations', 'torque_limit_violations',
                     'forbidden_collisions', 'abnormal_terminations')
        safe = all(isinstance(safety.get(name), (int, float)) and
                   math.isfinite(safety[name]) and safety[name] == 0 for name in mandatory)
        passed = finite and safe and max_speed <= self.settings.counterfactual_zero_command_speed_max and fall_rate == 0.0
        return {"scenario": "zero_command", "passed": passed,
                "max_base_speed": max_speed, "fall_rate": fall_rate,
                "speed_threshold": self.settings.counterfactual_zero_command_speed_max}

    def _cached_round_rollout(self, round_root: Path, task: Optional[TaskSpec] = None,
                              selected: Optional[Dict[str, Any]] = None) -> Optional[tuple]:
        """读取已完整采集的轮次，供 Provider 故障恢复时跳过昂贵的重复仿真。"""
        metrics_path = round_root / "rollout_metrics.json"
        if not metrics_path.is_file():
            return None
        payload = read_json(metrics_path)
        records = list(payload.get("records", []))
        protocol = self._evaluation_protocol(round_root.parents[3])
        seeds = list(protocol['seeds'])
        if protocol['dry_run']:
            if len(records) != 1:
                return None
        elif ({seed: sum(item.get('seed') == seed for item in records) for seed in seeds} !=
                {seed: protocol['rollouts_per_seed'] for seed in seeds} or
                any(item.get('seed') not in seeds for item in records)):
            return None
        names = [str(item.get('rollout', '')) for item in records]
        if len(set(names)) != len(names) or any(not name or Path(name).name != name for name in names):
            return None
        if task is not None and protocol['counterfactual_enabled'] and self._counterfactual_applicable(task):
            if not self._counterfactual_evidence(round_root, seeds, bool(protocol['dry_run']))['passed']:
                return None
        for item in records:
            item_dir = round_root / str(item.get("rollout"))
            item["dir"] = item_dir
            item["media"] = self._media_for_rollout(item_dir)
            if not item["media"]["trajectory"].is_file():
                return None
            metadata_path = item['media']['metadata']
            if not metadata_path.is_file():
                return None
            metadata = read_json(metadata_path)
            if not protocol['dry_run'] and metadata.get('seed') != item.get('seed'):
                return None
            if selected is not None and not protocol['dry_run']:
                checkpoint = dict(zip(selected['checkpoint_seeds'], selected['checkpoints'])).get(item['seed'])
                if checkpoint is None or metadata.get('checkpoint') != checkpoint.name:
                    return None
                digest = item.get('checkpoint_sha256')
                if digest and digest != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
                    return None
            # 可复用原始轨迹，不能复用旧验收口径计算的派生指标。
            item["metrics"] = TrajectoryMetrics().compute(pd.read_parquet(item["media"]["trajectory"]))
            if task is not None:
                item["score"] = self._rollout_score(task, item["metrics"])
        representative_name = payload.get("representative_rollout")
        if not representative_name and records:
            representative_name = records[len(records) // 2].get("rollout")
        if not representative_name:
            return None
        rollout_dir = round_root / str(representative_name)
        media = self._media_for_rollout(rollout_dir)
        if not all(media[name].is_file() for name in ("front", "side", "overview", "trajectory")):
            return None
        return rollout_dir, media, records

    def _collect_round_rollout(self, task: TaskSpec, selected: Dict[str, Any], dry_run: bool,
                               round_index: int, state: PersistentStateMachine,
                               reuse_existing: bool = False) -> tuple:
        """为一个闭环轮次采集多种子 rollout，并返回代表性媒体及全部数值记录。"""
        state.transition(AgentState.ROLLOUT_COLLECTING, {"loop_round": round_index})
        round_root = selected["dir"] / "rollouts" / ("round_%02d" % round_index)
        if reuse_existing:
            cached = self._cached_round_rollout(round_root, task, selected)
            if cached is not None:
                return cached
        if dry_run:
            rollout_dir = round_root / "rollout_001"
            media = DryRunRolloutRecorder().record(
                rollout_dir, self.settings.video_fps, seed=selected["manifest"].seed,
                checkpoint=selected["checkpoint"].name)
            metrics = TrajectoryMetrics().compute(pd.read_parquet(media["trajectory"]))
            records = [{"rollout": "rollout_001", "seed": selected["manifest"].seed,
                        "score": self._rollout_score(task, metrics), "dir": rollout_dir,
                        "media": media, "metrics": metrics}]
            write_json(round_root / "rollout_metrics.json", {
                "records": [{"rollout": "rollout_001", "seed": selected["manifest"].seed,
                             "score": records[0]["score"], "metrics": metrics}],
                "aggregate": self._aggregate_rollout_metrics(task, records),
                "representative_rollout": "rollout_001"})
            write_json(round_root / "counterfactual_evaluation.json", {
                "applicable": self._counterfactual_applicable(task), "passed": True,
                "dry_run": True, "scenarios": []})
            return rollout_dir, media, records
        rollout_records = []
        protocol = self._evaluation_protocol(task_dir=selected['dir'].parents[1])
        checkpoints = selected.get("checkpoints", [selected["checkpoint"]])
        seeds = selected.get("checkpoint_seeds", protocol['seeds'])
        if list(seeds) != list(protocol['seeds']) or len(checkpoints) != len(seeds):
            raise RuntimeError('固定评估种子与 checkpoint 集合不完整')
        rollout_index = 0
        for checkpoint, seed in zip(checkpoints, seeds):
            for _ in range(protocol['rollouts_per_seed']):
                rollout_index += 1
                current_dir = round_root / ("rollout_%03d" % rollout_index)
                result = self.controller.run_evaluation_rollouts(
                    selected["id"], task.robot, selected["dir"] / "config.yaml", checkpoint,
                    current_dir, seed=seed, fps=self.settings.video_fps)
                if result.exit_code != 0:
                    state.transition(AgentState.FAILED,
                                     {"reason": "rollout process failed", "rollout": rollout_index})
                    raise RuntimeError("real evaluation rollout %d failed" % rollout_index)
                current_media = self._media_for_rollout(current_dir)
                metrics = TrajectoryMetrics().compute(pd.read_parquet(current_media["trajectory"]))
                rollout_records.append({
                    "score": self._rollout_score(task, metrics), "dir": current_dir,
                    "media": current_media, "seed": seed, "metrics": metrics,
                    'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                })
        if not rollout_records:
            raise RuntimeError("evaluation produced no rollout")
        counterfactuals = []
        if protocol['counterfactual_enabled'] and self._counterfactual_applicable(task):
            for checkpoint, seed in zip(checkpoints, seeds):
                current_dir = round_root / ("counterfactual_zero_seed_%d" % seed)
                result = self.controller.run_evaluation_rollouts(
                    selected["id"] + "-counterfactual", task.robot,
                    selected["dir"] / "config.yaml", checkpoint, current_dir,
                    seed=seed, fps=self.settings.video_fps,
                    command_override={"x": 0.0, "y": 0.0, "yaw": 0.0})
                if result.exit_code != 0:
                    raise RuntimeError("counterfactual rollout failed for seed %d" % seed)
                counterfactuals.append(dict(self._counterfactual_summary(current_dir), seed=seed))
        write_json(round_root / "counterfactual_evaluation.json", {
            "applicable": bool(counterfactuals),
            "passed": all(item["passed"] for item in counterfactuals),
            "dry_run": False, "scenarios": counterfactuals,
        })
        rollout_records.sort(key=lambda item: item["score"])
        representative = rollout_records[len(rollout_records) // 2]
        serializable = [{"rollout": item["dir"].name, "seed": item["seed"],
                         "score": item["score"], "metrics": item["metrics"],
                         'checkpoint_sha256': item['checkpoint_sha256']}
                        for item in rollout_records]
        write_json(round_root / "rollout_metrics.json", {
            "records": serializable,
            "aggregate": self._aggregate_rollout_metrics(task, rollout_records),
            "representative_rollout": representative["dir"].name,
        })
        return representative["dir"], representative["media"], rollout_records

    def _counterfactual_evidence(self, round_root: Path, seeds: List[int], dry_run: bool) -> Dict[str, Any]:
        """验证反事实每个种子的有限安全证据，缺文件或缺样本一律不通过。"""
        path = round_root / 'counterfactual_evaluation.json'
        failed = {'passed': False, 'scenarios': [], 'reason': '反事实证据缺失或不完整'}
        if not path.is_file():
            return failed
        try:
            evidence = read_json(path)
        except (OSError, ValueError):
            return failed
        if not isinstance(evidence, dict):
            return failed
        if dry_run:
            return evidence if evidence.get('dry_run') is True and evidence.get('passed') is True else failed
        scenarios = evidence.get('scenarios')
        if (evidence.get('dry_run') is not False or evidence.get('applicable') is not True or
                evidence.get('passed') is not True or not isinstance(scenarios, list) or
                len(scenarios) != len(seeds) or any(not isinstance(item, dict) for item in scenarios) or
                {item.get('seed') for item in scenarios} != set(seeds)):
            return failed
        for item in scenarios:
            speed, fall = item.get('max_base_speed'), item.get('fall_rate')
            if (item.get('passed') is not True or item.get('scenario') != 'zero_command' or
                    any(type(value) not in (int, float) or not math.isfinite(value) for value in (speed, fall)) or
                    speed < 0 or speed > self.settings.counterfactual_zero_command_speed_max or fall != 0):
                return failed
        return evidence

    @staticmethod
    def _reward_evidence(media: Dict[str, Path]) -> Dict[str, float]:
        """从代表性 rollout 的逐项奖励表提取紧凑均值，供诊断模型修订奖励。"""
        path = media.get("rewards")
        if path is None or not path.is_file():
            return {}
        frame = pd.read_parquet(path)
        return {name: float(frame[name].astype(float).mean()) for name in frame.columns
                if name.startswith("raw_") or name.startswith("weighted_")}

    def _diagnosis_capabilities(self, robot: str) -> Dict[str, Any]:
        """构造诊断阶段所需的紧凑能力清单，避免模型提出不可执行奖励。"""
        manifest = self.inspector.inspect(robot)
        return {
            "project": manifest.project,
            "robot": manifest.robot,
            "registered_rewards": [
                {
                    "name": item.name,
                    "implementation": item.implementation,
                    "parameters": item.parameters,
                    "default_weight": item.default_weight,
                    "sign": item.sign,
                    "dependencies": item.dependencies,
                    "supported_phases": item.supported_phases,
                }
                for item in manifest.rewards
            ],
            "terminations": manifest.terminations,
            "command_space": manifest.command_space,
            "evaluation_metrics": manifest.evaluation_metrics,
        }

    def _evaluate_round(self, task: TaskSpec, task_dir: Path, state: PersistentStateMachine,
                        selected: Dict[str, Any], dry_run: bool, budget: BudgetTracker,
                        round_index: int, loop_records: List[Dict[str, Any]],
                        reuse_existing_rollout: bool = False) -> Dict[str, Any]:
        """执行一轮 rollout、视觉评论、数值验收和结构化诊断。"""
        rollout_dir, media, rollout_records = self._collect_round_rollout(
            task, selected, dry_run, round_index, state, reuse_existing_rollout)
        evaluation_key = hashlib.sha256(("acceptance-v2:" + task.json(sort_keys=True)).encode()).hexdigest()
        state.transition(AgentState.VISUAL_EVALUATING, {"loop_round": round_index})
        visual_records = ConservativeVisualAggregator.select(rollout_records)
        visual_reports = []
        rollout_names = []
        representative_artifacts = None
        try:
            for item in visual_records:
                item_dir = item["dir"]
                item_media = item["media"]
                item_artifacts = VisualEvaluationPipeline().build(task, item_dir, item_media)
                if item_dir == rollout_dir:
                    representative_artifacts = item_artifacts
                cached_report = item_dir / "visual_report_individual.json"
                cache_key_path = item_dir / "visual_evaluation_key.json"
                if (cached_report.is_file() and cache_key_path.is_file() and
                        read_json(cache_key_path).get("key") == evaluation_key):
                    report = VisualBehaviorReport.parse_obj(read_json(cached_report))
                else:
                    report = self._provider_call(
                        task_dir, "visual_critic", "critique_visual_behavior",
                        lambda: self.provider.critique_visual_behavior(
                            task, item_artifacts.visual_files))
                    write_json(cached_report, report)
                    write_json(cache_key_path, {"key": evaluation_key})
                    atomic_write_text(item_dir / "visual_raw_response.txt",
                                      report.json(indent=2, ensure_ascii=False) + "\n")
                visual_reports.append(report)
                rollout_names.append(item_dir.name)
        except ProviderError as exc:
            error_detail = str(exc).strip()[-2000:] or exc.__class__.__name__
            atomic_write_text(rollout_dir / "visual_provider_error.txt", error_detail + "\n")
            artifacts = representative_artifacts or VisualEvaluationPipeline().build(task, rollout_dir, media)
            return {"provider_error": error_detail, "provider_stage": "visual", "rollout_dir": rollout_dir,
                    "clean": artifacts.clean_sheet, "annotated": artifacts.annotated_sheet}
        artifacts = representative_artifacts or VisualEvaluationPipeline().build(task, rollout_dir, media)
        visual = ConservativeVisualAggregator.aggregate(visual_reports, rollout_names)
        write_json(rollout_dir / "visual_report.json", visual)
        write_json(rollout_dir / "visual_ensemble.json", {
            "selection": rollout_names,
            "aggregation": visual.dict(),
            "individual_reports": [item.dict() for item in visual_reports],
        })
        state.transition(AgentState.NUMERIC_EVALUATING, {"loop_round": round_index})
        physical = self._aggregate_rollout_metrics(task, rollout_records)
        protocol = self._evaluation_protocol(task_dir)
        required = protocol['counterfactual_enabled'] and self._counterfactual_applicable(task)
        physical['counterfactual_applicable'] = required
        physical['counterfactual_passed'] = not required
        if required:
            counterfactual = self._counterfactual_evidence(rollout_dir.parent, protocol['seeds'], dry_run)
            physical["counterfactual_observed_passed"] = counterfactual.get('passed') is True
            # 旧实验可能缓存了按旧规则执行的零命令测试。保留观测值用于审计，
            # 但固定动作任务不再让未要求的停车能力阻塞联合验收。
            physical["counterfactual_passed"] = (
                counterfactual.get('passed') is True)
            speeds = [item.get("max_base_speed") for item in counterfactual.get("scenarios", [])
                      if isinstance(item.get("max_base_speed"), (int, float))]
            physical["zero_command_drift"] = max(speeds) if speeds else float('nan')
        physical.setdefault("tracking_error", float("nan"))
        physical.setdefault("fall_rate", float("nan"))
        physical.setdefault("joint_limit_violations", float('nan'))
        physical.setdefault("torque_limit_violations", float('nan'))
        physical.setdefault("forbidden_collisions", float("nan"))
        physical.setdefault("abnormal_terminations", float("nan"))
        write_json(rollout_dir / "numeric_summary.json", physical)
        evaluation = DeterministicEvaluator().evaluate(task, physical, visual)
        quality_reports = []
        if not dry_run:
            from ..rewards.quality_gate import audit_rollout
            for record in rollout_records:
                try:
                    report = audit_rollout(task, selected['plan'],
                        pd.read_parquet(record['media']['trajectory']),
                        pd.read_parquet(record['media']['rewards']), record['metrics'],
                        metadata=read_json(record['media']['metadata']), require_contact_metadata=True)
                except (OSError, ValueError, KeyError) as exc:
                    report = {'passed': False, 'errors': ['奖励审计证据不可用：' + str(exc)]}
                write_json(record['dir'] / 'reward_quality_audit.json', report)
                quality_reports.append(report)
            if any(not r['passed'] for r in quality_reports):
                evaluation.completed = False
                evaluation.hard_constraints_passed = False
                evaluation.violations.append('reward_signal_integrity')
        write_json(rollout_dir / "evaluation.json", evaluation)
        state.transition(AgentState.DIAGNOSING, {"loop_round": round_index})
        ppo = PPOCollector().collect(selected["dir"])
        payload = {
            "task": task.dict(), "reward_plan": selected["plan"].dict(),
            "capabilities": self._diagnosis_capabilities(task.robot),
            "visual": visual.dict(), "numeric": physical, "evaluation": evaluation.dict(),
            "reward_evidence": self._reward_evidence(media), "ppo": ppo.dict(),
            "reward_quality_audits": quality_reports,
            "pilot_quality_audits": [read_json(p) for p in sorted(
                (selected['dir'] / 'reward_quality').glob('*/audit.json'))],
            "loop_history": loop_records,
            "budget": {"used_iterations": budget.used_iterations,
                       "remaining_iterations": budget.max_iterations - budget.used_iterations,
                       "used_revisions": budget.used_revisions,
                       "remaining_revisions": budget.max_revisions - budget.used_revisions},
        }
        retrieval_query = "%s\n失败约束：%s\n证据冲突：%s\n视觉摘要：%s" % (
            task.original_instruction, ", ".join(evaluation.violations) or "无",
            ", ".join(evaluation.conflicts) or "无", visual.summary)
        rag_context = self._retrieve_experience(
            retrieval_query, "training_diagnosis", task.robot,
            exclude_task_id=task.task_id)
        write_json(rollout_dir / "rag_diagnosis_context.json", rag_context)
        payload["retrieved_experience"] = rag_context
        memory_context = self._retrieve_memory(
            retrieval_query, task.robot, exclude_task_id=task.task_id)
        write_json(rollout_dir / "memory_diagnosis_context.json", memory_context)
        payload["long_term_memory"] = memory_context
        try:
            diagnosis_path = rollout_dir / "diagnosis.json"
            diagnosis_key_path = rollout_dir / "diagnosis_evaluation_key.json"
            if (diagnosis_path.is_file() and diagnosis_key_path.is_file() and
                    read_json(diagnosis_key_path).get("key") == evaluation_key):
                diagnosis = TrainingDiagnosis.parse_obj(read_json(diagnosis_path))
            else:
                diagnosis = self._provider_call(
                    task_dir, "diagnosis", "diagnose_training",
                    lambda: self.provider.diagnose_training(payload))
        except ProviderError as exc:
            error_detail = str(exc).strip()[-2000:] or exc.__class__.__name__
            atomic_write_text(rollout_dir / "diagnosis_provider_error.txt", error_detail + "\n")
            return {"provider_error": error_detail, "provider_stage": "diagnosis", "rollout_dir": rollout_dir,
                    "clean": artifacts.clean_sheet, "annotated": artifacts.annotated_sheet}
        if evaluation.completed:
            diagnosis.decision = "complete"
        elif diagnosis.decision == "complete":
            diagnosis.decision = "continue"
        if diagnosis.decision == "revise_reward" and not diagnosis.reward_changes:
            diagnosis.decision = "continue"
        if diagnosis.decision == "revise_curriculum" and not diagnosis.curriculum_changes:
            diagnosis.decision = "continue"
        posture_metric = None
        if self._is_front_leg_support_task(task):
            posture_metric = physical.get("front_leg_stand_duration")
        elif "后腿" in task.original_instruction:
            posture_metric = physical.get("rear_leg_stand_duration")
        if (round_index >= 2 and posture_metric is not None and posture_metric < 0.2 and
                diagnosis.decision in ("continue", "revise_reward", "revise_curriculum")):
            # 连续两轮几乎没有目标姿态，说明当前策略已陷入四足局部最优；继续加载同一
            # optimizer/checkpoint 只会强化错误行为，应保留新奖励但重新初始化策略。
            diagnosis.checkpoint_strategy = "restart_from_scratch"
            diagnosis.expected_effects.append("从随机初始化重新探索，摆脱持续四足支撑局部最优")
        write_json(rollout_dir / "diagnosis.json", diagnosis)
        write_json(diagnosis_key_path, {"key": evaluation_key})
        write_json(rollout_dir / "decision.json", {
            "decision": diagnosis.decision, "checkpoint_strategy": diagnosis.checkpoint_strategy,
            "reward_changes": [item.dict() for item in diagnosis.reward_changes],
            "curriculum_changes": [item.dict() for item in diagnosis.curriculum_changes],
        })
        return {"rollout_dir": rollout_dir, "media": media, "visual": visual,
                "physical": physical, "evaluation": evaluation, "diagnosis": diagnosis,
                "clean": artifacts.clean_sheet, "annotated": artifacts.annotated_sheet}

    @staticmethod
    def _append_lineage(task_dir: Path, experiment_id: str, parent_id: str,
                        reward_version: int, config_hash: str) -> None:
        """把闭环产生的新奖励版本追加到现有实验谱系。"""
        path = task_dir / "lineage.json"
        lineage = read_json(path) if path.is_file() else {"nodes": [], "edges": []}
        lineage["nodes"].append({"experiment_id": experiment_id, "reward_version": reward_version,
                                 "config_hash": config_hash, "result": "pending"})
        lineage["edges"].append({"parent": parent_id, "child": experiment_id})
        write_json(path, lineage)

    def _redesign_failed_rewards(self, task: TaskSpec, task_dir: Path, selected: Dict[str, Any],
                                 diagnosis: TrainingDiagnosis, round_index: int, manifest: Any) -> Any:
        """保存失败分析，再请求独立完整设计；审核失败有限重试，不改变验收合同。"""
        from ..rewards.quality_gate import METRIC_SIGNALS
        root = task_dir / 'reward_redesign' / ('round_%02d' % round_index)
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / 'failure_analysis.json', diagnosis)
        round_root = selected['dir'] / 'rollouts' / ('round_%02d' % round_index)
        def load(path: Path) -> Any:
            return read_json(path) if path.is_file() else None
        from ..rewards.quality_gate import capability_gaps
        gaps = capability_gaps(task, manifest.dict())
        if gaps:
            write_json(root / 'capability_gaps.json', {'gaps': gaps})
            raise RewardValidationError('需要补齐本地奖励实现，不能靠重复设计解决：' + '、'.join(gaps))
        payload = {
            'immutable_task': task.dict(), 'previous_plan': selected['plan'].dict(),
            'LOCAL_METRIC_REWARD_SIGNALS': {k: sorted(v) for k, v in METRIC_SIGNALS.items()},
            'failure_analysis': diagnosis.dict(), 'environment_manifest': manifest.dict(),
            'remaining_budget': load(task_dir / 'loop_status.json'),
            'rollout_metrics': load(round_root / 'rollout_metrics.json'),
            'visual_reports': [read_json(p) for p in sorted(round_root.glob('rollout_*/visual_report.json'))],
            'reward_evidence': {p.parent.name: self._reward_evidence({'rewards': p})
                                for p in sorted(round_root.glob('rollout_*/rewards.parquet'))},
            'evaluations': [read_json(p) for p in sorted(round_root.glob('rollout_*/evaluation.json'))],
            'reward_quality_audits': [read_json(p) for p in sorted(
                round_root.glob('rollout_*/reward_quality_audit.json'))],
            'ppo': PPOCollector().collect(selected['dir']).dict(),
            'effective_reward_scales': load(selected['dir'] / 'effective_reward_scales.json'),
        }
        if not hasattr(self.provider, 'redesign_rewards'):
            raise RewardValidationError('当前奖励 Provider 不支持失败后完整重设计')
        intent_path = task_dir / 'task_intent.json'
        intent = TaskIntentSpec.parse_obj(read_json(intent_path)) if intent_path.is_file() else TaskIntentSpec(
            original_instruction=task.original_instruction, robot=task.robot,
            action_name=task.task_name, normalized_goal=task.normalized_description)
        revised, review = None, None
        for attempt in range(self.settings.reward_design_review_retries + 1):
            try:
                if attempt == 0:
                    write_json(root / 'request_00.json', payload)
                    raw = self._provider_call(task_dir, 'reward_designer', 'redesign_rewards',
                        lambda: self.provider.redesign_rewards(payload))
                    write_json(root / 'response_00.json', raw)
                    revised = RewardPlan.parse_obj(raw.dict() if hasattr(raw, 'dict') else raw)
                    revised.task_id = task.task_id
                    revised.parent_version = selected['plan'].version
                    revised.version = selected['plan'].version + 1
                else:
                    if revised is None or review is None:
                        raise RewardValidationError('重设计未保留可定位的提案，禁止重新生成完整计划')
                    updated, _ = self._repair_reward_candidates(task, task_dir, [revised], review, manifest,
                        root / 'field_repairs', attempt, extra_context=payload)
                    revised = updated[0]
                write_json(root / ('proposal_%02d.json' % attempt), revised)
                review = self.reward_reviewer.review(intent, task, [revised],
                    dict(manifest.dict(), require_objective_links=True,
                         max_abs_reward_weight=self.settings.max_abs_reward_weight),
                    ['失败后的新奖励必须重新验证收益、动作形态及安全约束'])
                write_json(root / ('review_%02d.json' % attempt), review)
                if not review.approved:
                    raise RewardValidationError('重设计审核未通过（验收指标及执行配置）：' + '；'.join(review.conflicts + review.omissions))
                self._validate_plan_metric_coverage(task, revised)
                from ..rewards.validator import RewardPlanValidator
                RewardPlanValidator(manifest.rewards, self.settings.max_abs_reward_weight).validate(revised)
                write_json(root / 'approved_reward_plan.json', revised)
                return revised, ['GPT 基于失败分析提出新奖励方案；审核问题仅按本地字段白名单修复']
            except (ProviderError, ValueError, KeyError) as exc:
                payload['validation_feedback'] = str(exc)
                atomic_write_text(root / ('validation_error_%02d.txt' % attempt), str(exc) + '\n')
                if attempt == self.settings.reward_design_review_retries:
                    raise RewardValidationError(str(exc)) from exc
        raise RewardValidationError('重设计未产生可执行计划')

    def _compile_revision(self, task: TaskSpec, task_dir: Path, state: PersistentStateMachine,
                          selected: Dict[str, Any], diagnosis: TrainingDiagnosis,
                          round_index: int, dry_run: bool) -> Dict[str, Any]:
        """执行诊断修改、任务专用规范化和安全编译，生成新的候选版本目录。"""
        self._validate_acceptance_contract(task_dir, task, state)
        manifest = self.inspector.inspect(task.robot)
        if self.settings.failure_reward_redesign and not dry_run:
            revised, audit = self._redesign_failed_rewards(task, task_dir, selected, diagnosis, round_index, manifest)
        elif diagnosis.reward_changes or diagnosis.curriculum_changes:
            reviser = RewardPlanReviser(manifest.rewards, self.settings.max_abs_reward_weight)
            revised, audit = reviser.revise(selected["plan"], diagnosis)
        else:
            revised = selected["plan"].copy(deep=True)
            audit = ["奖励和课程保持不变，仅按诊断追加训练"]
        if not (self.settings.failure_reward_redesign and not dry_run):
            audit.extend(self._normalize_plan_for_task(task, revised))
        self._validate_plan_metric_coverage(task, revised)
        experiment_id = "%s-revision-%02d-v%02d" % (selected["id"], round_index, revised.version)
        directory = self.store.candidate_dir(task.task_id, experiment_id)
        directory.mkdir(parents=True, exist_ok=False)
        for name in ("metrics", "checkpoints", "rollouts", "prompts", "responses"):
            (directory / name).mkdir()
        metadata = RewardCompiler(manifest, self.settings.max_abs_reward_weight).compile(
            revised, directory, task=task)
        write_json(directory / "revision_audit.json", {
            "parent_experiment": selected["id"], "diagnosis": diagnosis.dict(), "adjustments": audit})
        command = self.controller.adapter.training_command(
            task.robot, directory / "config.yaml", self.settings.mid_iterations,
            selected["manifest"].seed, experiment_id)
        manifest = ExperimentManifest(
            experiment_id=experiment_id, parent_experiment_id=selected["id"], task_id=task.task_id,
            git_commit=self._git_commit(), config_hash=metadata["config_hash"],
            reward_version=revised.version, seed=selected["manifest"].seed, robot=task.robot,
            training_command=command,
            provider_status="mock" if dry_run else self.settings.provider,
            training_result="compiled")
        write_json(directory / "manifest.json", manifest)
        self._append_lineage(task_dir, experiment_id, selected["id"], revised.version, metadata["config_hash"])
        return {"id": experiment_id, "dir": directory, "plan": revised,
                "manifest": manifest, "metadata": metadata}

    def _train_revision(self, task: TaskSpec, task_dir: Path, state: PersistentStateMachine,
                        selected: Dict[str, Any], diagnosis: TrainingDiagnosis, round_index: int,
                        dry_run: bool, budget: BudgetTracker) -> Dict[str, Any]:
        """按诊断的 checkpoint 策略对新奖励版本执行多种子续训或重新训练。"""
        revised = self._compile_revision(task, task_dir, state, selected, diagnosis, round_index, dry_run)
        # 只有安全校验与配置编译均完成后，才算真正消费了一次修订机会。
        budget.consume_revision()
        self._persist_training_execution(task_dir, state, budget, phase='compiled',
            dry_run=dry_run, experiment_id=revised['id'], parent_experiment=selected['id'],
            round_index=round_index)
        decision_states = {
            "continue": AgentState.CONTINUE_TRAINING, "revise_reward": AgentState.REVISE_REWARD,
            "revise_curriculum": AgentState.REVISE_CURRICULUM, "restart": AgentState.RESTART,
            "rollback": AgentState.ROLLBACK,
        }
        state.transition(decision_states.get(diagnosis.decision, AgentState.REVISE_REWARD),
                         {"loop_round": round_index, "reward_version": revised["plan"].version,
                          "decision": diagnosis.decision})
        parent_checkpoints = selected.get("checkpoints", [selected["checkpoint"]])
        seeds = selected.get("checkpoint_seeds", self.settings.evaluation_seeds[:len(parent_checkpoints)])
        if not seeds:
            seeds = [selected["manifest"].seed]
            parent_checkpoints = [selected["checkpoint"]]
        requested = self.settings.full_iterations if (
            diagnosis.decision == "restart" or diagnosis.checkpoint_strategy == "restart_from_scratch") \
            else self.settings.mid_iterations
        per_seed = budget.per_seed_allocation(requested, len(seeds))
        if per_seed <= 0:
            raise RuntimeError("training iteration budget exhausted")
        state.transition(AgentState.FULL_TRAINING,
                         {"loop_round": round_index, "revision_iterations_per_seed": per_seed})
        checkpoints = []
        used_parent_checkpoints = selected.get("parent_checkpoints", parent_checkpoints) \
            if diagnosis.checkpoint_strategy == "continue_from_parent" else parent_checkpoints
        restart_requested = diagnosis.decision == 'restart' or diagnosis.checkpoint_strategy == 'restart_from_scratch'
        if not restart_requested and (len(used_parent_checkpoints) != len(seeds) or any(
                checkpoint is None or not checkpoint.is_file() for checkpoint in used_parent_checkpoints)):
            raise RuntimeError('修订被拒绝：对应种子的父 checkpoint 不完整')
        for index, seed in enumerate(seeds):
            operation = revised['id'] + ':revision:seed:%d' % seed
            self._reserve_training_operation(task_dir, state, budget, operation, per_seed)
            if dry_run:
                checkpoint = revised["dir"] / "checkpoints" / ("seed_%d" % seed) / ("model_%d.pt" % per_seed)
                atomic_write_text(checkpoint, "dry-run revised checkpoint placeholder; never deploy to hardware\n")
            else:
                run_name = revised["id"] + "-seed-%d" % seed
                restart = diagnosis.decision == "restart" or diagnosis.checkpoint_strategy == "restart_from_scratch"
                before = set(CheckpointManager.list_checkpoints(revised["dir"]))
                if restart:
                    result = self.controller.run_full_training(
                        run_name, task.robot, revised["dir"] / "config.yaml", per_seed, seed, revised["dir"])
                else:
                    parent = used_parent_checkpoints[index]
                    result = self.controller.continue_training(
                        run_name, task.robot, revised["dir"] / "config.yaml", per_seed, seed,
                        revised["dir"], parent)
                if result.exit_code != 0:
                    raise RuntimeError("revision training failed for seed %d" % seed)
                new_checkpoints = [path for path in CheckpointManager.list_checkpoints(revised["dir"])
                                   if path not in before]
                checkpoint = self._checkpoint_for_seed(revised['dir'], seed)
                if checkpoint not in new_checkpoints:
                    checkpoint = None
                if checkpoint is None:
                    raise RuntimeError("revision training produced no checkpoint for seed %d" % seed)
                CheckpointManager.prune(
                    checkpoint.parent, keep_per_run=self.settings.checkpoints_per_run,
                    protected=[checkpoint])
            checkpoints.append(checkpoint)
            self._complete_training_operation(task_dir, state, budget, operation)
        revised["checkpoints"] = checkpoints
        revised["checkpoint_seeds"] = list(seeds)
        revised["checkpoint"] = checkpoints[0]
        revised["parent_checkpoints"] = list(parent_checkpoints)
        revised["manifest"].training_result = "dry_run_completed" if dry_run else "completed"
        revised["manifest"].iteration = per_seed
        revised["manifest"].checkpoint = str(checkpoints[0].relative_to(revised["dir"]))
        write_json(revised["dir"] / "manifest.json", revised["manifest"])
        self._save_checkpoint_set(task_dir, revised, dry_run)
        self._persist_training_execution(task_dir, state, budget, phase='trained')
        return revised

    def _finalize_loop(self, task: TaskSpec, task_dir: Path, state: PersistentStateMachine,
                       selected: Dict[str, Any], outcome: Dict[str, Any], budget: BudgetTracker,
                       loop_records: List[Dict[str, Any]], final_state: AgentState,
                       reason: str, dry_run: bool) -> Dict[str, Any]:
        """复制最终可追溯产物、保存闭环摘要并进入终态。"""
        completed = final_state == AgentState.COMPLETED
        curatable = final_state in (AgentState.COMPLETED, AgentState.FAILED)
        if curatable and self.settings.memory_enabled:
            state.transition(AgentState.MEMORY_CURATING, {
                "reason": "正在验证是否可晋升为长期情景记忆",
                "loop_rounds": len(loop_records),
                "used_revisions": budget.used_revisions,
            })
        else:
            state.transition(final_state, {"reason": reason, "loop_rounds": len(loop_records),
                                           "used_revisions": budget.used_revisions})
        lineage_path = task_dir / "lineage.json"
        if lineage_path.is_file():
            lineage = read_json(lineage_path)
            for node in lineage.get("nodes", []):
                if node.get("experiment_id") == selected["id"]:
                    node["result"] = final_state.value.lower()
            write_json(lineage_path, lineage)
        final_dir = task_dir / "final"
        final_dir.mkdir(exist_ok=True)
        sources = [(selected["checkpoint"], "checkpoint.pt"),
                   (selected["dir"] / "config.yaml", "config.yaml"),
                   (selected["dir"] / "reward_plan.json", "reward_plan.json")]
        if outcome.get("clean"):
            sources.append((outcome["clean"], "contact_sheet_clean.png"))
        if outcome.get("annotated"):
            sources.append((outcome["annotated"], "contact_sheet_annotated.png"))
        for source, name in sources:
            if Path(source).is_file():
                shutil.copy2(source, final_dir / name)
        summary = {
            "task_id": task.task_id, "state": final_state.value,
            "result": "completed" if completed else "failed" if final_state == AgentState.FAILED else "human_review",
            "reason": reason, "selected_experiment": selected["id"],
            "checkpoint": "final/checkpoint.pt", "config": "final/config.yaml",
            "rollout": str(outcome["rollout_dir"].relative_to(task_dir)) if outcome.get("rollout_dir") else None,
            **budget.summary_fields(),
            "resume_allowed": final_state == AgentState.HUMAN_REVIEW and not budget.exhausted,
            "next_training_started": False,
            "loop_rounds": len(loop_records), "reward_version": selected["plan"].version,
            "evaluation_seeds": list(selected.get("checkpoint_seeds", [])),
            "last_decision": loop_records[-1].get("decision") if loop_records else None,
            "dry_run": dry_run,
        }
        write_json(task_dir / "loop_history.json", loop_records)
        # 先落盘摘要，证据构建器只读取任务目录的真实文件，不依赖易变的进程内对象。
        write_json(task_dir / "summary.json", summary)
        ReportBuilder().build(task.task_id, summary, task_dir)

        promotion: Dict[str, Any] = {
            "promoted": False,
            "reason": (
                "dry-run 使用模拟证据，不得进入生产长期记忆" if dry_run else
                "只有通过联合验收且证据完整的实验才能进入长期记忆"
            ),
        }
        if curatable and self.settings.memory_enabled:
            try:
                intent_path = task_dir / "task_intent.json"
                if intent_path.is_file():
                    intent = TaskIntentSpec.parse_obj(read_json(intent_path))
                else:
                    intent = TaskIntentSpec(
                        original_instruction=task.original_instruction,
                        robot=task.robot,
                        action_name=task.task_name,
                        normalized_goal=task.normalized_description,
                        required_behaviors=[item.description for item in task.required_behaviors],
                        forbidden_behaviors=[item.description for item in task.forbidden_behaviors],
                        retrieval_keywords=[task.robot, task.task_name],
                    )
                experience_root = task_dir / "memory" / "reward_experience" / str(selected["id"])
                experience_root.mkdir(parents=True, exist_ok=True)
                snapshot = dict(summary)
                snapshot["provider_error"] = bool(outcome.get("provider_error"))
                write_json(experience_root / "training_result_snapshot.json", snapshot)
                evidence = self.reward_experience_evidence.build(task_dir, selected, summary, outcome)
                write_json(experience_root / "evidence.json", evidence)
                eligibility = self.reward_experience_eligibility.check(evidence)
                write_json(experience_root / "eligibility.json", eligibility)
                state.events.emit("reward_experience_eligibility", {
                    "task_id": task.task_id, "eligible": eligibility.eligible,
                    "outcome": eligibility.outcome, "reason": eligibility.reason,
                })
                experience = None
                if not eligibility.eligible:
                    promotion["reason"] = eligibility.reason
                    print("[经验] 未进入总结阶段：%s" % eligibility.reason, flush=True)
                    record = None
                else:
                    try:
                        experience_agent = RewardExperienceAgent(
                            lambda payload: self._provider_call(
                                task_dir, "reward_experience", "summarize_reward_experience",
                                lambda: self.provider.summarize_reward_experience(payload)))
                        experience = experience_agent.create(evidence, eligibility)
                        self.reward_experience_validator.validate(experience, evidence, task_dir)
                        write_json(experience_root / "experience.json", experience)
                        write_json(experience_root / "validation.json", {
                            "valid": True, "reason": "证据 ID、奖励差异、任务和因果措辞检查通过"})
                        state.events.emit("reward_experience_validated", {
                            "task_id": task.task_id, "experience_id": experience.experience_id,
                            "outcome": experience.outcome, "confidence": experience.confidence,
                        })
                    except Exception as exc:
                        experience = None
                        reason_text = "%s: %s" % (exc.__class__.__name__, str(exc)[:1200])
                        write_json(experience_root / "validation.json", {
                            "valid": False, "reason": reason_text})
                        state.events.emit("reward_experience_rejected", {
                            "task_id": task.task_id, "error_type": exc.__class__.__name__,
                        })
                        promotion["reason"] = "Reward Experience 生成或验证失败，未写入长期记忆：%s" % reason_text
                        print("[经验] 总结或校验失败，不晋升情景记忆：%s" % reason_text, flush=True)
                    record = (self.memory_curator.curate(
                        intent, task, selected["plan"], summary, outcome, selected, task_dir,
                        require_multi_seed=self.settings.memory_require_multi_seed,
                        experience=experience,
                        require_validated_experience=True,
                        memory_root=self.memory.root) if experience is not None else None)
                if record is not None:
                    memory_path = self.memory.promote(record)
                    state.events.emit("episodic_memory_promoted", {
                        "task_id": task.task_id, "memory_id": record.memory_id,
                        "experience_id": (record.reward_experience or {}).get("experience_id"),
                    })
                    semantic = self.memory_consolidator.consolidate(
                        self.memory, record, self.settings.memory_min_semantic_support)
                    maintenance = self.memory.maintain(
                        self.settings.memory_max_records,
                        self.settings.memory_max_age_days,
                        self.settings.memory_min_confidence,
                    )
                    promotion = {
                        "promoted": True,
                        "memory_id": record.memory_id,
                        "path": relative_display(memory_path, self.settings.agent_root),
                        "outcome": record.outcome,
                        "gate": self.memory_curator.last_reason,
                        "semantic": semantic.dict() if semantic is not None else None,
                        "maintenance": maintenance,
                    }
                    print("[记忆] 已晋升长期经验 %s" % record.memory_id, flush=True)
                else:
                    if experience is not None:
                        promotion["reason"] = self.memory_curator.last_reason
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                promotion = {"promoted": False, "reason": "记忆整理失败：%s" % exc}
                print("[记忆] 整理失败，但不改变训练完成判定：%s" % exc, flush=True)
        elif self.settings.memory_enabled:
            # 人工复核和 Provider 故障也生成 INCONCLUSIVE 证据记录，但绝不调用总结模型。
            try:
                experiment_id = str(selected.get("id", "unknown"))
                experience_root = task_dir / "memory" / "reward_experience" / experiment_id
                experience_root.mkdir(parents=True, exist_ok=True)
                snapshot = dict(summary)
                snapshot["provider_error"] = bool(outcome.get("provider_error"))
                write_json(experience_root / "training_result_snapshot.json", snapshot)
                evidence = self.reward_experience_evidence.build(task_dir, selected, summary, outcome)
                eligibility = self.reward_experience_eligibility.check(evidence)
                write_json(experience_root / "evidence.json", evidence)
                write_json(experience_root / "eligibility.json", eligibility)
                promotion["reason"] = eligibility.reason
                state.events.emit("reward_experience_eligibility", {
                    "task_id": task.task_id, "eligible": False,
                    "outcome": eligibility.outcome, "reason": eligibility.reason,
                })
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                promotion["reason"] = "未能构建 INCONCLUSIVE 证据包：%s" % exc
        write_json(task_dir / "memory" / "promotion.json", promotion)
        if self.settings.rag_enabled:
            try:
                self.knowledge.refresh(force=True)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print("[RAG] 最终实验写入经验索引失败：%s" % exc, flush=True)
        if curatable and self.settings.memory_enabled:
            state.transition(final_state, {"reason": reason, "memory": promotion})
        summary["memory"] = promotion
        write_json(task_dir / "summary.json", summary)
        self._write_working_memory(
            task_dir, state, budget=budget, selected=selected,
            outcome=outcome, loop_round=len(loop_records))
        if completed:
            self._cleanup_completed_videos(task_dir, summary)
        return summary

    def _cleanup_completed_videos(self, task_dir: Path, summary: Dict[str, Any]) -> None:
        """在经验和终态保存后清理视频；清理失败不推翻已完成的训练。"""
        try:
            result = ExperimentCleaner(task_dir.parent).cleanup_completed_videos(task_dir.name)
        except (OSError, ValueError) as exc:
            result = {"status": "failed", "error": str(exc)}
        # 单独保存每次清理结果，保留首次删除清单供审计及重复恢复。
        audit_path = task_dir / "video_cleanup.json"
        if audit_path.is_file():
            previous = read_json(audit_path)
            if result.get("status") == "completed" and not result.get("video_files") and previous.get("status") == "completed":
                result = previous
            else:
                result["removed"] = sorted(set(previous.get("removed", []) + result.get("removed", [])))
                result["video_files"] = len(result["removed"])
                result["reclaim_bytes"] = previous.get("reclaim_bytes", 0) + result.get("reclaim_bytes", 0)
        write_json(audit_path, result)
        summary["video_cleanup"] = result
        write_json(task_dir / "summary.json", summary)

    def _evaluate(self, task: TaskSpec, task_dir: Path, state: PersistentStateMachine,
                  selected: Dict[str, Any], dry_run: bool, budget: BudgetTracker,
                  start_round: int = 1, existing_records: Optional[List[Dict[str, Any]]] = None,
                  reuse_first_rollout: bool = False) -> Dict[str, Any]:
        """循环执行评估、诊断、奖励修订和再训练，直到通过或达到真实阻塞条件。"""
        selected.setdefault("checkpoint_seeds", self.settings.evaluation_seeds[:len(
            selected.get("checkpoints", [selected["checkpoint"]]))])
        loop_records: List[Dict[str, Any]] = list(existing_records or [])
        round_index = start_round
        first_iteration = True
        while True:
            self._write_working_memory(
                task_dir, state, budget=budget, selected=selected,
                loop_round=round_index)
            write_json(task_dir / "loop_status.json", {
                "updated_at": utc_now(),
                "round": round_index, "reward_version": selected["plan"].version,
                "experiment_id": selected["id"], "used_iterations": budget.used_iterations,
                "remaining_iterations": budget.max_iterations - budget.used_iterations,
                "used_revisions": budget.used_revisions,
                "remaining_revisions": budget.max_revisions - budget.used_revisions,
            })
            outcome = self._evaluate_round(
                task, task_dir, state, selected, dry_run, budget, round_index, loop_records,
                reuse_existing_rollout=reuse_first_rollout and first_iteration)
            self._write_working_memory(
                task_dir, state, budget=budget, selected=selected,
                outcome=outcome, loop_round=round_index)
            first_iteration = False
            if outcome.get("provider_error"):
                stage_name = "视觉评估" if outcome.get("provider_stage") == "visual" else "训练诊断"
                reason = "%s Provider 失败：%s" % (stage_name, outcome["provider_error"])
                return self._finalize_loop(task, task_dir, state, selected, outcome, budget,
                                           loop_records, AgentState.HUMAN_REVIEW, reason, dry_run)
            diagnosis = outcome["diagnosis"]
            evaluation = outcome["evaluation"]
            if not evaluation.completed:
                failure_root = task_dir / 'failure_analysis'
                failure_root.mkdir(exist_ok=True)
                write_json(failure_root / ('round_%02d.json' % round_index), diagnosis)
                # 当前策略失败不证明任务不可实现；有预算则进入完整重设计。
                if (self.settings.failure_reward_redesign and not dry_run and
                        diagnosis.decision == 'failed' and
                        budget.used_revisions < budget.max_revisions and
                        budget.used_iterations < budget.max_iterations):
                    diagnosis.decision = 'restart'
                    diagnosis.checkpoint_strategy = 'restart_from_scratch'
            record = {
                "round": round_index, "experiment_id": selected["id"],
                "reward_version": selected["plan"].version, "decision": diagnosis.decision,
                "checkpoint_strategy": diagnosis.checkpoint_strategy,
                "completed": evaluation.completed,
                "hard_constraints_passed": evaluation.hard_constraints_passed,
                "task_metrics_passed": evaluation.task_metrics_passed,
                "visual_alignment_passed": evaluation.visual_alignment_passed,
                "violations": evaluation.violations, "conflicts": evaluation.conflicts,
                "used_iterations": budget.used_iterations,
            }
            loop_records.append(record)
            write_json(task_dir / "loop_history.json", loop_records)
            write_json(task_dir / "loop_status.json", {
                "updated_at": utc_now(), "round": round_index,
                "reward_version": selected["plan"].version,
                "experiment_id": selected["id"], "phase": "evaluated",
                "decision": diagnosis.decision,
                "used_iterations": budget.used_iterations,
                "remaining_iterations": budget.max_iterations - budget.used_iterations,
                "used_revisions": budget.used_revisions,
                "remaining_revisions": budget.max_revisions - budget.used_revisions,
            })
            if evaluation.completed and diagnosis.decision == "complete":
                return self._finalize_loop(task, task_dir, state, selected, outcome, budget,
                                           loop_records, AgentState.COMPLETED,
                                           "视觉、任务指标和硬约束全部通过", dry_run)
            if (self.settings.failure_reward_redesign and not dry_run and
                    budget.exhausted):
                return self._finalize_loop(task, task_dir, state, selected, outcome, budget,
                    loop_records, AgentState.HUMAN_REVIEW,
                    self._budget_exhaustion_reason(selected, budget, dry_run), dry_run)
            if diagnosis.decision == "failed":
                return self._finalize_loop(task, task_dir, state, selected, outcome, budget,
                                           loop_records, AgentState.FAILED, "诊断判定任务不可恢复", dry_run)
            if diagnosis.decision == "human_review":
                return self._finalize_loop(task, task_dir, state, selected, outcome, budget,
                                           loop_records, AgentState.HUMAN_REVIEW,
                                           "诊断发现需要用户决定的真实歧义或证据冲突", dry_run)
            if budget.exhausted:
                return self._finalize_loop(task, task_dir, state, selected, outcome, budget,
                                           loop_records, AgentState.HUMAN_REVIEW,
                                           self._budget_exhaustion_reason(selected, budget, dry_run) + '；' +
                                           "任务指标通过=%s，视觉通过=%s，硬约束通过=%s；待处理=%s" % (
                                               evaluation.task_metrics_passed, evaluation.visual_alignment_passed,
                                               evaluation.hard_constraints_passed, diagnosis.decision), dry_run)
            try:
                selected = self._train_revision(
                    task, task_dir, state, selected, diagnosis, round_index, dry_run, budget)
            except (RuntimeError, RewardValidationError, ValueError) as exc:
                atomic_write_text(task_dir / "loop_blocking_error.txt", str(exc) + "\n")
                if (task_dir / 'training_execution.json').is_file():
                    self._persist_training_execution(task_dir, state, budget, phase='blocked')
                return self._finalize_loop(task, task_dir, state, selected, outcome, budget,
                                           loop_records, AgentState.HUMAN_REVIEW,
                                           "闭环修订无法安全执行：%s" % exc, dry_run)
            round_index += 1
