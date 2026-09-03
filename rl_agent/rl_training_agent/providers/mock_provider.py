from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Type, TypeVar

from pydantic import BaseModel

from ..schemas.decisions import DiagnosisItem, EvidenceItem, TrainingDiagnosis
from ..schemas.experiments import ConversationHandle, ProviderHealth
from ..schemas.task import (
    BehaviorRequirement, ForbiddenBehavior, MetricThreshold, TaskPhase, TaskSpec, TrainingBudget,
)
from ..schemas.visual import VisualBehaviorReport, VisualPhaseResult

ModelT = TypeVar("ModelT", bound=BaseModel)


class MockLLMReasoningProvider:
    """仅在显式选择时使用的确定性离线 Provider。"""

    def __init__(self, num_candidates: int = 3):
        """初始化 MockLLMReasoningProvider 实例及其运行依赖。"""
        self.num_candidates = num_candidates

    def doctor(self) -> ProviderHealth:
        """检查运行环境、外部依赖和服务健康状态。"""
        return ProviderHealth(available=True, opencli_available=False, extension_connected=False,
                              chatgpt_logged_in=False, image_upload_supported=True,
                              details=["deterministic mock provider"])

    def open_or_bind(self) -> None:
        """打开或绑定配置指定的 ChatGPT 浏览器会话。"""
        return None

    def new_conversation(self, title_hint: str) -> ConversationHandle:
        """创建新会话并返回可持久化的会话句柄。"""
        return ConversationHandle(conversation_id="mock-" + hashlib.sha1(title_hint.encode()).hexdigest()[:8],
                                  title_hint=title_hint)

    def send_text(self, prompt: str, conversation: ConversationHandle) -> str:
        """向指定网页会话发送文本并返回最新助手回复。"""
        return json.dumps({"mock": True, "conversation": conversation.conversation_id})

    def send_with_files(self, prompt: str, files: List[Path], conversation: ConversationHandle) -> str:
        """上传本地文件、发送提示词并返回最新助手回复。"""
        return json.dumps({"mock": True, "files": [path.name for path in files]})

    def parse_json_response(self, raw_response: str, schema: Type[ModelT]) -> ModelT:
        """提取并校验网页回复中的结构化 JSON。"""
        return schema.parse_obj(json.loads(raw_response))

    def design_task_and_rewards(self, instruction: str, robot: str, capabilities: Dict[str, Any]) -> Dict[str, Any]:
        """依据任务描述和环境能力生成任务规格与奖励候选。"""
        is_jump = "跳" in instruction or "jump" in instruction.lower()
        digest = hashlib.sha1((robot + instruction).encode("utf-8")).hexdigest()[:10]
        if is_jump:
            phases = [
                TaskPhase(name="stable_start", description="stable standing posture"),
                TaskPhase(name="takeoff", description="four feet leave the ground nearly together"),
                TaskPhase(name="flight", description="body remains level while airborne"),
                TaskPhase(name="landing", description="land and recover without falling"),
            ]
            behaviors = [
                BehaviorRequirement(name="simultaneous_takeoff", description="all four feet take off", phase="takeoff"),
                BehaviorRequirement(name="level_body", description="small roll and pitch", phase="flight"),
                BehaviorRequirement(name="stable_landing", description="recover standing", phase="landing"),
            ]
            observations = ["base_pos", "base_lin_vel", "rpy", "feet_contacts", "contact_forces"]
            metrics = [
                MetricThreshold(name="jump_height", operator=">=", value=0.12, unit="m"),
                MetricThreshold(name="takeoff_time_spread", operator="<=", value=0.08, unit="s"),
                MetricThreshold(name="landing_pitch_abs", operator="<=", value=0.35, unit="rad"),
            ]
        else:
            phases = [TaskPhase(name="start", description="stable start"),
                      TaskPhase(name="locomotion", description="track commanded forward velocity"),
                      TaskPhase(name="finish", description="remain upright")]
            behaviors = [BehaviorRequirement(name="forward_progress", description="walk steadily forward", phase="locomotion"),
                         BehaviorRequirement(name="stable_body", description="keep body level", phase="all")]
            observations = ["base_lin_vel", "base_ang_vel", "projected_gravity", "commands"]
            metrics = [MetricThreshold(name="tracking_error", operator="<=", value=0.25, unit="m/s"),
                       MetricThreshold(name="fall_rate", operator="<=", value=0.05, unit="ratio")]
        task = TaskSpec(
            task_id="task-" + digest, robot=robot, task_name="jump" if is_jump else "forward_walk",
            original_instruction=instruction, normalized_description=instruction, initial_state="default standing pose",
            required_behaviors=behaviors,
            forbidden_behaviors=[ForbiddenBehavior(name="fall", description="base or forbidden link collision"),
                                 ForbiddenBehavior(name="unsafe_joint_motion", description="joint or torque limit violation")],
            phases=phases, required_observations=observations, required_sensors=["proprioception", "contact"],
            success_metrics=metrics,
            safety_constraints=[MetricThreshold(name="nan_count", operator="==", value=0),
                                MetricThreshold(name="joint_limit_violations", operator="==", value=0)],
            training_budget=TrainingBudget(),
            visual_evaluation_requirements=["natural motion", "correct phase order", "no visible jitter"],
        )
        registry_names = {item["name"] for item in capabilities.get("rewards", [])}
        wanted = (["jump_height", "feet_synchrony", "landing_stability", "horizontal_drift"] if is_jump else
                  ["tracking_lin_vel", "tracking_ang_vel"]) + ["orientation", "torques", "action_rate", "collision"]
        available = [name for name in wanted if name in registry_names]
        plans: List[Dict[str, Any]] = []
        emphases = ["task_completion", "stability", "curriculum"]
        for index in range(self.num_candidates):
            terms = []
            for name in available:
                base = {"tracking_lin_vel": 1.0, "tracking_ang_vel": 0.5, "orientation": -0.5,
                        "torques": -0.0002, "action_rate": -0.01, "collision": -1.0,
                        "jump_height": 3.0, "feet_synchrony": 0.5, "landing_stability": 1.5,
                        "horizontal_drift": -1.0}[name]
                factor = 1.2 if (index == 0 and name.startswith("tracking")) else 1.0
                factor = 1.5 if (index == 1 and name in ("orientation", "collision")) else factor
                terms.append({
                    "name": name, "implementation": "registry:" + name,
                    "purpose": "existing environment reward: " + name, "weight": base * factor,
                    "parameters": {}, "active_phases": ["all"], "normalization": "environment_dt",
                    "expected_training_trend": "improve", "dependencies": [],
                    "failure_modes_addressed": [], "reward_hacking_risks": ["proxy optimization"],
                })
            plans.append({
                "task_id": task.task_id, "version": index + 1, "parent_version": None,
                "design_rationale": ["offline deterministic %s candidate" % emphases[index % 3]],
                "terms": terms,
                "terminations": [{"name": "base_contact", "condition": "termination_contact", "enabled": True}],
                "curriculum": ([{"name": "easy_commands", "start_iteration": 0, "end_iteration": 300,
                                 "parameter_changes": {"command_scale": 0.5}}] if index == 2 else []),
                "success_metrics": [item.dict() for item in task.success_metrics], "known_conflicts": [],
                "expected_learning_stages": ["stability", "task progress", "refinement"],
            })
        return {"task_spec": task.dict(), "reward_plans": plans,
                "reward_hacking_risks": ["standing still may exploit stability rewards"],
                "termination_suggestions": ["forbidden body contact", "orientation limit"]}

    def design_visual_evaluation(self, task: TaskSpec) -> Dict[str, Any]:
        """为任务生成视觉评估输入与事件设计。"""
        return {"events": [phase.name for phase in task.phases], "cameras": ["front", "side", "overview"]}

    def critique_visual_behavior(self, task: TaskSpec, files: List[Path]) -> VisualBehaviorReport:
        """基于视觉材料生成不受奖励数值锚定的行为评论。"""
        frames = [0, 10, 20]
        return VisualBehaviorReport(visual_success=True, alignment_score=0.9, confidence=0.85,
                                    summary="Mock rollout follows the requested phase order.",
                                    phase_results=[VisualPhaseResult(phase=phase.name, score=1.0,
                                                                     evidence_frames=frames[:1]) for phase in task.phases],
                                    evidence_frames=frames)

    def diagnose_training(self, payload: Dict[str, Any]) -> TrainingDiagnosis:
        """融合视觉、物理和 PPO 证据生成训练诊断。"""
        return TrainingDiagnosis(
            diagnosis=[DiagnosisItem(category="completion", finding="mock safety, task and visual checks passed", severity="info")],
            evidence=[EvidenceItem(source="dry_run", metric="completed", value=True,
                                   interpretation="all deterministic gates passed")],
            decision="complete", confidence=0.95, expected_effects=["dry-run workflow completion"], risks=[],
            checkpoint_strategy="continue_from_current")

    def close(self) -> None:
        """释放 Provider 持有或绑定的浏览器资源。"""
        return None
