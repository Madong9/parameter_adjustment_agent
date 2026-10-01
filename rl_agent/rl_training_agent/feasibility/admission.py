"""依据真实能力、探针证据和本地预算决定仿真训练准入。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .schema import CompleteFeasibilityReport


class TrainingAdmissionReport(BaseModel):
    """准入结论与物理等级独立保存；准入从不代表动作已成功。"""

    policy_version: str = "training-admission-v1"
    decision: Literal["ALLOW_TRAINING", "ALLOW_BUDGETED_EXPLORATION",
                      "DRY_RUN_ONLY", "NEEDS_REVIEW", "REJECT"]
    capability_status: str
    probe_status: str
    scope: Literal["simulation_only"] = "simulation_only"
    reason: str
    max_iterations: int = 0
    max_revisions: int = 0
    unresolved_requirements: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    environment_health: Dict[str, Any] = Field(default_factory=dict)
    run_id: str = ""

    @property
    def allowed(self) -> bool:
        """仅明确准入或演练结论允许继续编排流程。"""
        return self.decision in (
            "ALLOW_TRAINING", "ALLOW_BUDGETED_EXPLORATION", "DRY_RUN_ONLY")


class TrainingAdmissionPolicy:
    """根据本地检查放行有界探索，不用一次候选失败证明目标不可能。"""

    @staticmethod
    def probe_status(report: CompleteFeasibilityReport) -> str:
        """把现有候选结果归类；物理报告本身保持原样。"""
        simulation = report.simulation_report or report.physics_report
        if report.evidence_mode != "REAL" or "mock" in str(simulation.get("backend", "")).lower():
            return "MOCK_OR_MIXED"
        if simulation.get("backend") == "isaacgym" and simulation.get("validated") is True:
            if simulation.get("success") is True:
                return "VALIDATED" if report.status.value == "PHYSICS_VALIDATED" else "PARTIAL_VALIDATION"
            violations = simulation.get("violations", [])
            if any(str(item).lower().startswith("non_finite_") for item in violations):
                return "RUNTIME_INVALID"
            if violations and all("tracking" in str(item) for item in violations):
                return "REFERENCE_TRACKING_FAILED"
            return "CANDIDATE_ROLLOUT_FAILED"
        if simulation.get("status") == "UNAVAILABLE":
            return "BACKEND_UNAVAILABLE"
        return "OPTIMIZATION_INCONCLUSIVE"

    @classmethod
    def needs_environment_health(cls, report: CompleteFeasibilityReport) -> bool:
        """指出目标证据不能独立证明仿真基础设施健康的情形。"""
        return cls.probe_status(report) in (
            "RUNTIME_INVALID", "BACKEND_UNAVAILABLE", "OPTIMIZATION_INCONCLUSIVE")

    def assess(self, report: CompleteFeasibilityReport, settings: Any,
               dry_run: bool = False,
               environment_health: Optional[Dict[str, Any]] = None) -> TrainingAdmissionReport:
        """核对能力、真实模型、仿真证据及模式，并给出不可自动增大的预算。"""
        capability = report.capability_report or report.capability_check
        raw_status = capability.get("status", "UNKNOWN")
        capability_status = getattr(raw_status, "value", str(raw_status))
        health = dict(environment_health or {})
        probe = self.probe_status(report)
        missing = list(capability.get("missing_requirements", []))
        result = TrainingAdmissionReport(
            decision="NEEDS_REVIEW", capability_status=capability_status,
            probe_status=probe, reason="准入证据不足", environment_health=health,
            unresolved_requirements=missing,
            evidence_ids=["feasibility_report.json#/capability_report",
                          "feasibility_report.json#/simulation_report"],
            limitations=["准入只允许仿真训练；不构成物理通过、策略验收或真机部署许可",
                         "候选失败和优化未收敛均不证明任务整体不可行"],
        )
        if dry_run and report.validation_level == "MOCK_VALIDATED":
            result.decision = "DRY_RUN_ONLY"
            result.reason = "Mock 仅放行离线流程演练"
            result.max_iterations = settings.max_total_iterations
            result.max_revisions = settings.max_reward_revisions
            return result
        if capability_status == "UNSUPPORTED":
            result.decision = "REJECT"
            result.reason = "当前机器人/环境能力清单明确缺少目标所需能力"
            return result
        if report.evidence_mode != "REAL" or probe == "MOCK_OR_MIXED":
            result.reason = "Mock 或混合证据不能放行真实训练"
            return result
        required_checks = {"robot_model", "joints_and_actuators", "evaluation_metrics"}
        checks = {item.get("name"): item for item in capability.get("checks", [])}
        if (capability_status not in ("SUPPORTED", "CAPABILITY_SUPPORTED") or missing or
                any(name not in checks for name in required_checks) or
                any(item.get("status") not in ("SUPPORTED", "CAPABILITY_SUPPORTED")
                    for item in checks.values())):
            result.reason = "机器人、观测、命令或评价指标尚未全部确认"
            return result
        if report.status.value in ("MODEL_UNAVAILABLE", "NEEDS_CLARIFICATION", "UNSUPPORTED"):
            result.reason = "需先补全模型、明确任务或解决能力缺失"
            return result
        if not report.motion_constraint_spec or report.motion_type in (None, "UNKNOWN"):
            result.reason = "动作约束未生成或目标动作类型尚未明确"
            return result
        if report.robot_model.get("model_status") != "AVAILABLE":
            result.reason = "当前机器人模型来源不可确认"
            return result
        simulation = report.simulation_report or report.physics_report
        runtime_observed = (probe != "RUNTIME_INVALID" and
                            simulation.get("backend") == "isaacgym" and
                            simulation.get("validated") is True and
                            simulation.get("success") in (True, False) and
                            float(simulation.get("duration") or 0) > 0)
        baseline_ok = (health.get("backend") == "isaacgym" and
                       health.get("validated") is True and health.get("success") is True and
                       float(health.get("duration") or 0) > 0)
        if not (runtime_observed or baseline_ok):
            result.reason = "缺少当前机器人真实 Isaac Gym 运行证据，需先修复仿真基础设施"
            return result
        if baseline_ok:
            result.evidence_ids.append("environment_health.json")
        physics_pass = (
            report.status.value == "PHYSICS_VALIDATED" and runtime_observed and
            simulation.get("success") is True and
            report.validation_level in ("STATIC_PHYSICS_VALIDATED", "DYNAMIC_PHYSICS_VALIDATED"))
        if physics_pass:
            result.decision = "ALLOW_TRAINING"
            result.reason = "能力及目标物理探针通过，可进入奖励审查和仿真训练"
            result.max_iterations = settings.max_total_iterations
            result.max_revisions = settings.max_reward_revisions
        elif probe == "RUNTIME_INVALID" and not baseline_ok:
            result.reason = "目标物理探针出现非有限数值，需先通过独立 Isaac Gym 环境健康检查"
        elif settings.feasibility_admission_mode == "strict":
            result.reason = "strict 模式要求目标探针通过，当前候选证据不满足"
        else:
            result.decision = "ALLOW_BUDGETED_EXPLORATION"
            result.reason = "能力与仿真环境已确认；保留探针失败/未知结论，允许预算内探索训练"
            result.max_iterations = min(settings.max_total_iterations, settings.exploration_max_iterations)
            result.max_revisions = min(settings.max_reward_revisions, settings.exploration_max_revisions)
            result.limitations.extend(report.risks[:6])
        return result


def check_environment_health(training_root: Any, robot: str) -> Dict[str, Any]:
    """在隔离 worker 中检查默认姿态，原生崩溃或超时不能杀死训练协调器。"""
    from .simulation.schema import SimulationReport

    marker = "__DYNAMIC_FEASIBILITY_REPORT__"
    request = {"mode": "environment_health", "training_root": str(training_root), "robot": robot}
    try:
        result = subprocess.run(
            [sys.executable, "-m", "rl_training_agent.feasibility.isaacgym_worker"],
            input=json.dumps(request), text=True, capture_output=True, check=False,
            cwd=str(Path(__file__).resolve().parents[2]), timeout=60)
        if result.returncode != 0:
            raise RuntimeError("环境检查 worker 退出码 %s" % result.returncode)
        lines = [line[len(marker):] for line in result.stdout.splitlines() if line.startswith(marker)]
        if not lines:
            raise ValueError("环境检查 worker 未返回报告")
        return SimulationReport.parse_raw(lines[-1]).dict()
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return {"backend": "isaacgym", "validated": False, "success": None,
                "status": "UNAVAILABLE", "reason": str(exc)[:400]}


def _check_environment_health_in_process(training_root: Any, robot: str) -> Dict[str, Any]:
    """复用 PPO 默认姿态做独立短时环境检查；不给目标动作增加通过证据。"""
    from .robot_models.loader import RobotModelLoader
    from .simulation.isaacgym_validator import IsaacGymFeasibilityValidator
    from .motion_prototype.schema import MotionPhase, MotionPrototype
    from .ik.schema import IKResult

    model = RobotModelLoader(training_root).load_robot_model(robot)
    if model.model_status != "AVAILABLE":
        return {"backend": "not_run", "validated": False, "success": None,
                "reason": model.model_error or "机器人模型不可用"}
    prototype = MotionPrototype(
        robot=robot, action="environment_health_default_pose",
        source="ppo_default_configuration",
        phases=[MotionPhase(name="default_pose", duration_seconds=1.0,
                            body_goal={"feet": "support"})])
    target = IKResult(status="CONFIGURATION", success=True, backend="ppo_config",
                      joint_positions=model.default_joint_positions)
    return IsaacGymFeasibilityValidator(training_root).validate(
        model.runtime_dict(), prototype, target).dict()
