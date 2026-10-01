"""串联机器人能力、真实运动目标、IK 和 Unitree Isaac Gym 物理预检。"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..schemas.agent_workflow import TaskIntentSpec
from .capability_checker import CapabilityChecker
from .balance.schema import BalanceMotionPlan
from .balance.validator import IsaacGymBalanceValidator
from .ik.mock import MockIKSolver
from .ik.pinocchio_solver import PinocchioIKSolver
from .ik.schema import IKResult
from .motion_prototype.dynamic_generator import DynamicMotionPrototypeGenerator
from .motion_prototype.generator import MotionPrototypeGenerator
from .motion_prototype.schema import (DynamicMotionPrototype, ManipulationPrototype,
                                      MotionPrototype, MotionType)
from .motion_prototype.trajectory_generator import TrajectoryPrototypeGenerator
from .motion_prototype.target_builder import RobotMotionTargetBuilder
from .motion_constraints import MotionConstraintCompiler
from .planning import DeterministicMotionPlannerRegistry
from .whole_body import WholeBodyMotionSolver
from .robot_models.loader import RobotModelLoader
from .robot_models.schema import RobotModel
from .schema import (CompleteFeasibilityReport, DynamicFeasibilityReport,
                     FeasibilityCheck, FeasibilityLevel, FeasibilityStatus,
                     StageStatus, ValidationLevel, ValidationStageReport)
from .validation.candidate_search import StaticPoseCandidateGenerator
from .validation.static_validator import StaticPoseValidator
from .simulation.isaacgym_validator import IsaacGymFeasibilityValidator
from .simulation.isaacgym_dynamic_validator import (IsaacGymDynamicValidator,
                                                    MockIsaacGymDynamicValidator)
from .simulation.mujoco_validator import MockMuJoCoValidator
from .simulation.schema import SimulationReport


StageCallback = Callable[[Dict[str, Any]], None]


class FeasibilityPipeline:
    """在 PPO 前执行确定性能力检查、Pinocchio IK 和 Unitree Isaac Gym 短时预检。"""

    def __init__(self, training_root: Path, capability_checker: Optional[CapabilityChecker] = None,
                 ik_solver: Optional[Any] = None, simulation_validator: Optional[Any] = None,
                 prototype_provider: Optional[Callable[[TaskIntentSpec], Any]] = None,
                 robot_model_loader: Optional[RobotModelLoader] = None,
                 stage_callback: Optional[StageCallback] = None,
                 dynamic_validator: Optional[Any] = None):
        """创建真实物理预检服务并允许测试注入明确标记的 adapter。"""
        self.training_root = Path(training_root).resolve()
        self.capability_checker = capability_checker or CapabilityChecker(self.training_root)
        self.robot_model_loader = robot_model_loader or RobotModelLoader(self.training_root)
        self._dependency_fallback_reason = None
        isaac_validator = IsaacGymFeasibilityValidator(self.training_root)
        isaac_available = isaac_validator.is_available()
        pinocchio_available = importlib.util.find_spec("pinocchio") is not None
        self.ik_solver = ik_solver or (PinocchioIKSolver() if pinocchio_available else MockIKSolver())
        self.simulation_validator = simulation_validator or (
            isaac_validator if isaac_available else MockMuJoCoValidator())
        if dynamic_validator is not None:
            self.dynamic_validator = dynamic_validator
        elif isaac_available and not (
                "mock" in str(getattr(self.simulation_validator, "backend", "")).lower()):
            self.dynamic_validator = IsaacGymDynamicValidator(
                self.training_root, ik_solver=self.ik_solver)
        else:
            self.dynamic_validator = MockIsaacGymDynamicValidator(self.training_root)
        if not pinocchio_available:
            self._dependency_fallback_reason = "Pinocchio 不可用；静态姿态 IK 将不能形成真实物理验证"
        if not isaac_available:
            self._dependency_fallback_reason = "%sUnitree Isaac Gym 环境不可用" % (
                (self._dependency_fallback_reason + "；")
                if self._dependency_fallback_reason else "")
        self.prototype_provider = prototype_provider
        self.stage_callback = stage_callback

    @classmethod
    def mocked(cls, training_root: Path, prototype_provider: Optional[Callable] = None,
               stage_callback: Optional[StageCallback] = None):
        """构造仅供测试或明确 dry-run 使用的 Mock pipeline。"""
        return cls(training_root, ik_solver=MockIKSolver(),
                   simulation_validator=MockMuJoCoValidator(),
                   dynamic_validator=MockIsaacGymDynamicValidator(training_root),
                   prototype_provider=prototype_provider, stage_callback=stage_callback)

    def _emit_stage(self, stage: str, status: str, **details: Any) -> None:
        """把每个 Feasibility 子阶段写入可选的统一 JSONL 事件回调。"""
        if self.stage_callback is not None:
            self.stage_callback({"stage": stage, "status": status, **details})

    def assess(self, intent: TaskIntentSpec, manifest: Any) -> CompleteFeasibilityReport:
        """执行模型发现、能力检查、目标生成、IK、仿真和真实等级聚合。"""
        assessment = self.capability_checker.assess(intent, manifest)
        self._emit_stage("capability_check", "completed", result_status=assessment.status.value)
        use_mjcf = str(getattr(self.simulation_validator, "backend", "")).lower() == "mujoco"
        model_descriptor = self.robot_model_loader.load_robot_model(
            intent.robot, require_mjcf=use_mjcf)
        model = model_descriptor.runtime_dict()
        model["_allow_mock_targets"] = (
            "mock" in str(getattr(self.ik_solver, "backend", "")).lower() or
            "mock" in str(getattr(self.simulation_validator, "backend", "")).lower())
        capability_payload = assessment.dict()
        checks = list(assessment.checks)
        risks = list(assessment.risks)
        recommendations: List[str] = []
        motion_type = DynamicMotionPrototypeGenerator.classify(intent)
        try:
            constraint_spec = MotionConstraintCompiler.compile(intent, motion_type)
            planning_report = DeterministicMotionPlannerRegistry().plan(
                constraint_spec, intent, model_descriptor)
            whole_body_report = WholeBodyMotionSolver().assess(
                constraint_spec, planning_report,
                str(getattr(self.ik_solver, "backend", "unavailable")), model_descriptor,
                physics_backend=str(getattr(
                    self.simulation_validator if motion_type in (
                        MotionType.STATIC_POSE, MotionType.BALANCE)
                    else self.dynamic_validator, "backend", "unavailable")))
            capability_payload["_motion_constraint_spec"] = constraint_spec.dict()
            capability_payload["_planning_report"] = planning_report.dict()
            capability_payload["_whole_body_report"] = whole_body_report.dict()
            self._emit_stage("motion_constraint_compilation", "completed",
                             motion_type=motion_type.value,
                             planning_status=planning_report.status,
                             whole_body_status=whole_body_report.status)
        except (TypeError, ValueError) as exc:
            capability_payload["_motion_constraint_error"] = str(exc)[:400]
        if self._dependency_fallback_reason and motion_type in (MotionType.STATIC_POSE,
                                                                 MotionType.BALANCE):
            risks.append(self._dependency_fallback_reason)
            recommendations.append("安装并配置 Pinocchio 与 Unitree Isaac Gym 环境后重新执行真实物理预检。")
        if assessment.status in (FeasibilityStatus.UNSUPPORTED, FeasibilityStatus.NEEDS_CLARIFICATION):
            recommendations.append("先解决能力缺失或任务歧义；当前结论不允许进入 Reward Designer。")
            self._emit_stage("pipeline", "blocked", reason="capability_check_failed")
            return self._report(
                intent, status=assessment.status, validation_level="CAPABILITY_ONLY",
                evidence_mode="REAL", assessment=capability_payload, model=model_descriptor,
                checks=checks, risks=risks, recommendations=recommendations,
                motion_type=motion_type.value,
            )
        mock_mode = (
            "mock" in str(getattr(self.dynamic_validator, "backend", "")).lower()
            if motion_type not in (MotionType.STATIC_POSE, MotionType.BALANCE)
            else self._uses_mock_backend())
        if model_descriptor.model_status != "AVAILABLE" and not mock_mode:
            reason = model_descriptor.model_error or "机器人物理模型不可用"
            checks.append(FeasibilityCheck(
                name="robot_model", status=FeasibilityStatus.CONDITIONAL,
                summary="真实机器人 URDF/配置不可用，不能执行 IK 或 Isaac Gym rollout",
                evidence=[reason]))
            risks.append(reason)
            recommendations.append("修复 URDF/机器人配置后重新执行物理预检。Go2 不依赖 URDF 派生 MJCF。")
            self._emit_stage("robot_model", "MODEL_UNAVAILABLE", reason=reason)
            return self._report(
                intent, status=FeasibilityStatus.MODEL_UNAVAILABLE,
                validation_level="CAPABILITY_ONLY", evidence_mode="REAL",
                assessment=capability_payload, model=model_descriptor,
                checks=checks, risks=risks, recommendations=recommendations,
                motion_type=motion_type.value,
            )
        if model_descriptor.model_status != "AVAILABLE" and mock_mode:
            reason = model_descriptor.model_error or "机器人模型不可用"
            checks.append(FeasibilityCheck(
                name="robot_model", status=FeasibilityStatus.CONDITIONAL,
                summary="机器人 URDF 不可用；仅允许继续 Mock 流程演练",
                evidence=[reason, "validation_level=MOCK_VALIDATED only"],
            ))
            risks.append("Mock 流程未加载/验证真实机器人模型：%s" % reason)
        model = model_descriptor.runtime_dict()
        model["_allow_mock_targets"] = (
            "mock" in str(getattr(self.ik_solver, "backend", "")).lower() or
            "mock" in str(getattr(self.simulation_validator, "backend", "")).lower())

        if motion_type == MotionType.MANIPULATION:
            raw_manipulation = intent.constraints.get("manipulation_prototype")
            manipulation_payload = None
            schema_error = ""
            if raw_manipulation is not None:
                try:
                    manipulation_payload = ManipulationPrototype.parse_obj(raw_manipulation).dict()
                    if manipulation_payload["robot"] != intent.robot:
                        raise ValueError("prototype robot does not match selected robot")
                except Exception as exc:
                    schema_error = str(exc)[:300]
            reason = ("当前没有连接真实机械臂/末端执行器模型、末端 IK、碰撞与工作空间 validator；"
                      "只有能力清单不能证明操作目标可达")
            if schema_error:
                reason += "；操作目标 Schema 无效：" + schema_error
            risks.append(reason)
            recommendations.append("接入对应机器人的真实末端执行器模型、Pinocchio frame IK、碰撞几何和环境验证后再评估。")
            self._emit_stage("manipulation_validation", "inconclusive", reason=reason)
            return self._report(
                intent, status=FeasibilityStatus.CONDITIONAL,
                validation_level=ValidationLevel.CAPABILITY_ONLY.value,
                evidence_mode="REAL", assessment=capability_payload,
                model=model_descriptor, checks=checks, risks=risks,
                recommendations=recommendations, motion_type=motion_type.value,
                motion_report={
                    "status": "CAPABILITY_ONLY" if not schema_error else "INCONCLUSIVE",
                    "manipulation_prototype": manipulation_payload,
                    "schema_error": schema_error or None,
                    "reason": reason,
                    "limitations": [
                        "目标 Schema 表示末端笛卡尔意图，不会自动求 joint angle。",
                        "目前未实现机械臂 IK、碰撞、抓取接触和操作物体动力学验证。",
                    ],
                },
                kinematic_report={"status": "NOT_RUN", "backend": "end_effector_ik_unavailable",
                                  "success": None, "reason": reason},
                simulation_report={"status": "NOT_RUN", "backend": "not_run",
                                   "reason": reason},
                simulation_result={"status": "NOT_RUN", "backend": "not_run",
                                   "validated": False},
            )

        if motion_type in (MotionType.JUMP, MotionType.ACROBATIC):
            try:
                trajectory = TrajectoryPrototypeGenerator.generate(intent, motion_type)
            except (TypeError, ValueError) as exc:
                reason = "跳跃/特技阶段骨架无效：%s" % str(exc)[:300]
                trajectory_payload = {"status": "INCONCLUSIVE", "reason": reason}
            else:
                reason = ("已有阶段模板，但尚无经过验证的跳跃/空翻轨迹、起落目标和专用物理验证器"
                          if motion_type == MotionType.JUMP else
                          "空翻/特技需要起跳、飞行旋转、落地和恢复的确定性轨迹及专用验证器")
                trajectory_payload = {
                    "status": "INCONCLUSIVE", "prototype": trajectory.dict(),
                    "reason": reason, "backend": "local_phase_template",
                }
            risks.append(reason)
            recommendations.append("当前不运行步态探针；先实现并验证该动作的阶段轨迹与 Isaac Gym 专用 validator。")
            self._emit_stage("motion_prototype_generation", "completed",
                             motion_type=motion_type.value, source="local_phase_template")
            return self._report(
                intent, status=FeasibilityStatus.CONDITIONAL,
                validation_level=ValidationLevel.CAPABILITY_ONLY.value,
                evidence_mode="REAL", assessment=capability_payload,
                model=model_descriptor, checks=checks, risks=risks,
                recommendations=recommendations, motion_type=motion_type.value,
                motion_report={"status": "INCONCLUSIVE", "trajectory": trajectory_payload},
                trajectory_report=trajectory_payload,
            )

        if motion_type == MotionType.BALANCE and self._special_balance_request(intent):
            if self._single_leg_request(intent):
                return self._assess_balance_candidates(
                    intent, assessment, model_descriptor, checks, risks, recommendations)
            return self._assess_balance_plan(
                intent, assessment, model_descriptor, checks, risks, recommendations, capability_payload)

        if motion_type not in (MotionType.STATIC_POSE, MotionType.BALANCE):
            self._emit_stage("motion_prototype_generation", "started",
                             motion_type=motion_type.value)
            try:
                dynamic_prototype = DynamicMotionPrototypeGenerator.generate(intent)
            except (TypeError, ValueError) as exc:
                reason = "低维动态原型生成失败：%s" % str(exc)[:300]
                risks.append(reason)
                recommendations.append("修正任务时长/速度等结构化意图后重新执行预检。")
                self._emit_stage("motion_prototype_generation", "failed", reason=reason)
                return self._report(
                    intent, status=FeasibilityStatus.CONDITIONAL,
                    validation_level=ValidationLevel.CAPABILITY_ONLY.value,
                    evidence_mode="REAL", assessment=capability_payload,
                    model=model_descriptor, checks=checks, risks=risks,
                    recommendations=recommendations, motion_type=motion_type.value,
                    motion_report={"status": "INVALID", "reason": reason},
                )
            self._emit_stage("motion_prototype_generation", "completed",
                             motion_type=motion_type.value,
                             source=dynamic_prototype.source)
            return self._assess_dynamic(
                intent, assessment, model_descriptor, dynamic_prototype,
                checks, risks, recommendations)

        proposal = None
        provider_failed = False
        self._emit_stage("motion_prototype_generation", "started",
                         motion_type=motion_type.value)
        try:
            proposal = self.prototype_provider(intent) if self.prototype_provider else None
            prototype = MotionPrototypeGenerator.generate(intent, proposal)
        except Exception as exc:
            provider_failed = True
            prototype = MotionPrototypeGenerator.deterministic(intent)
            prototype.notes.append("语义阶段 Provider 失败，已回退规则原型：%s" % str(exc)[:300])
            risks.append("动作语义使用本地规则回退，不能作为完整用户意图的物理验证")
        risks.extend("动作原型说明：" + str(note) for note in prototype.notes)
        if prototype.source == "deterministic_fallback":
            provider_failed = True
            risks.append("当前动作原型为确定性回退；物理验证范围受限于规则生成的目标")
        self._emit_stage("motion_prototype_generation", "completed",
                         motion_type=motion_type.value, source=prototype.source)
        self._emit_stage("motion_prototype", "completed", source=prototype.source)
        self._emit_stage("static_motion_validation", "started",
                         motion_type=motion_type.value)
        motion_scope_issues = self._unsupported_motion_scope(intent, prototype, model)
        if motion_scope_issues:
            risks.extend(motion_scope_issues)
            recommendations.append(
                "当前 Isaac Gym 预检只按 PPO 关节控制器短时执行给定姿态目标，不验证步态、跳跃或目标之间的"
                "完整动态轨迹；需提供确定性的时变目标/控制器，或由后续 PPO 训练和评估验证。")
        try:
            targets = RobotMotionTargetBuilder.build(prototype, model)
            motion_scope_issues = sorted(set(
                motion_scope_issues + self._unsupported_motion_scope(intent, prototype, model)))
            targets_truncated = len(targets) > 8
            if targets_truncated:
                risks.append("动作目标超过 8 个；为限制预检预算，仅验证前 8 个")
                targets = targets[:8]
            if not targets:
                raise ValueError("MotionPrototype 没有可物理验证的 RobotMotionTarget")
        except Exception as exc:
            reason = "RobotMotionTarget 无法生成：%s" % str(exc)[:400]
            checks.append(FeasibilityCheck(
                name="robot_motion_target", status=FeasibilityStatus.CONDITIONAL,
                summary=reason, evidence=["motion target generation failed"]))
            risks.append(reason)
            recommendations.append("提供模型坐标系下的脚端笛卡尔目标，或实现该动作的本地目标规划器。")
            self._emit_stage("motion_target", "unavailable", reason=reason)
            return self._report(
                intent, status=FeasibilityStatus.CONDITIONAL,
                validation_level="CAPABILITY_ONLY",
                evidence_mode=self._evidence_mode([], [], prototype),
                assessment=capability_payload, model=model_descriptor,
                checks=checks, risks=risks, recommendations=recommendations,
                prototype=prototype, motion_report={"status": "UNAVAILABLE", "reason": reason},
            )
        self._emit_stage("motion_target", "completed", target_count=len(targets))

        ik_results: List[Dict[str, Any]] = []
        for phase_name, target in targets:
            try:
                result = self.ik_solver.solve_ik(
                    model, target, model_descriptor.default_joint_positions)
            except TypeError:
                # 旧第三方适配器仍可通过兼容的两参数接口调用。
                result = self.ik_solver.solve_ik(model, target)
            except Exception as exc:
                result = IKResult(status="FAILED", success=False,
                                  backend=getattr(self.ik_solver, "backend", "unknown"),
                                  reason="IK adapter exception: %s" % str(exc)[:300])
            ik_results.append({"phase": phase_name, "target": target, "result": result})
        ik_result = self._combine_ik_results(ik_results)
        ik_payload = ik_result.dict()
        ik_payload["target_results"] = [{
            "phase": item["phase"], "target": item["target"],
            "result": item["result"].dict(),
        } for item in ik_results]
        static_candidate_reports = []
        for item in ik_results:
            static_candidate_reports.append({
                "phase": item["phase"],
                **StaticPoseValidator().assess(
                    model, item.get("target"), item["result"].joint_positions,
                    self_collision_checked=item["result"].self_collision_checked),
            })
        static_stage_statuses = [
            check.get("status") for candidate in static_candidate_reports
            for check in candidate.values() if isinstance(check, dict) and "stage" in check
        ]
        static_dynamics_payload = {
            "status": ("FAILED" if "FAILED" in static_stage_statuses else
                       "PASSED" if static_stage_statuses and
                       all(value == StageStatus.PASSED.value for value in static_stage_statuses)
                       else "INCONCLUSIVE"),
            "candidate_count": len(static_candidate_reports),
            "candidate_results": static_candidate_reports,
            "note": "缺失真实力矩/摩擦/碰撞证据的子项保持 UNKNOWN，不会伪装成通过。",
            "limitations": [
                "Go2 PPO 配置未启用自碰撞，Pinocchio 当前没有报告自碰撞几何检查。",
                "RNEA torque 仅为固定根重力补偿估算，不包含足底接触力分配。",
                "当前静态候选没有独立接触力样本/摩擦标定，摩擦锥子项未知。",
            ],
        }
        for candidate in static_candidate_reports:
            for component in ("joint_limits", "self_collision",
                             "center_of_mass_support_polygon", "torque", "friction"):
                check = candidate.get(component)
                if isinstance(check, dict):
                    check_status = check.get("status", "UNKNOWN")
                    if hasattr(check_status, "value"):
                        check_status = check_status.value
                    self._emit_stage("static_check:" + component,
                                     str(check_status).lower(),
                                     phase=candidate.get("phase"),
                                     backend=check.get("backend"),
                                     reason=check.get("reason", ""),
                                     metrics=check.get("metrics", {}))
        checks.append(FeasibilityCheck(
            name="pinocchio_ik",
            status=(FeasibilityStatus.CAPABILITY_SUPPORTED if ik_result.success is True else
                    FeasibilityStatus.PHYSICS_FAILED if ik_result.success is False and
                    ik_result.backend == "pinocchio" else FeasibilityStatus.CONDITIONAL),
            summary=ik_result.reason,
            evidence=["backend=" + ik_result.backend, "status=" + ik_result.status,
                      "residual=" + str(ik_result.residual)],
        ))
        self._emit_stage("pinocchio_ik", ik_result.status, backend=ik_result.backend,
                         success=ik_result.success)

        simulation_results: List[Dict[str, Any]] = []
        try:
            if hasattr(self.simulation_validator, "validate_targets"):
                result = self.simulation_validator.validate_targets(model, prototype, ik_results)
                simulation_results.append({"phase": "short_rollout", "result": result})
            else:
                for item in ik_results:
                    result = self.simulation_validator.validate(model, prototype, item["result"])
                    simulation_results.append({"phase": item["phase"], "result": result})
        except Exception as exc:
            result = SimulationReport(
                status="UNAVAILABLE", success=None,
                backend=getattr(self.simulation_validator, "backend", "unknown"),
                validated=False, reason="simulation adapter exception: %s" % str(exc)[:300])
            simulation_results.append({"phase": "short_rollout", "result": result})
        simulation_result = self._combine_simulation_results(simulation_results)
        checks.append(FeasibilityCheck(
            name=("isaacgym_rollout" if simulation_result.backend == "isaacgym" else
                  "mujoco_rollout" if simulation_result.backend == "mujoco" else
                  "mock_rollout"),
            status=(FeasibilityStatus.CAPABILITY_SUPPORTED if simulation_result.success is True else
                    FeasibilityStatus.PHYSICS_FAILED if simulation_result.success is False and
                    simulation_result.backend in ("isaacgym", "mujoco") and
                    simulation_result.validated
                    else FeasibilityStatus.CONDITIONAL),
            summary=simulation_result.reason,
            evidence=["backend=" + simulation_result.backend,
                      "status=" + simulation_result.status,
                      "validated=" + str(simulation_result.validated)],
        ))
        self._emit_stage("physics_rollout", simulation_result.status,
                         backend=simulation_result.backend,
                         validated=simulation_result.validated,
                         success=simulation_result.success)

        evidence_mode = self._evidence_mode(
            [item["result"] for item in ik_results],
            [item["result"] for item in simulation_results], prototype)
        all_real_success = (
            ik_results and simulation_results and
            all(item["result"].success is True and item["result"].backend == "pinocchio"
                for item in ik_results) and
            all(item["result"].success is True and
                item["result"].backend in ("isaacgym", "mujoco") and
                item["result"].validated and
                (item["result"].backend == "isaacgym" or item["result"].self_collision_checked)
                for item in simulation_results) and
            evidence_mode == "REAL" and not provider_failed and not targets_truncated and
            not motion_scope_issues
        )
        any_real_physics_failure = (
            any(item["result"].backend == "pinocchio" and item["result"].success is False
                for item in ik_results) or
            any(item["result"].backend in ("isaacgym", "mujoco") and
                item["result"].validated and
                item["result"].success is False for item in simulation_results)
        )
        mock_physics_only = bool(ik_results and simulation_results) and all(
            "mock" in str(item["result"].backend).lower()
            for item in ik_results + simulation_results)
        if all_real_success:
            status = FeasibilityStatus.PHYSICS_VALIDATED
            validation_level = ValidationLevel.STATIC_PHYSICS_VALIDATED.value
            recommendations.append("已对列出的目标完成 Pinocchio IK 和 Unitree Isaac Gym PPO 环境短时 rollout；这不是 PPO 策略成功保证。")
        elif any_real_physics_failure and not motion_scope_issues:
            status = FeasibilityStatus.PHYSICS_FAILED
            validation_level = ValidationLevel.PHYSICS_FAILED.value
            recommendations.append("真实 IK 或 Isaac Gym 检测到不可达/安全违规；已阻止进入 Reward Designer。")
        elif motion_scope_issues and not mock_physics_only:
            status = FeasibilityStatus.CONDITIONAL
            validation_level = "CAPABILITY_ONLY"
            recommendations.append("短时物理预检没有覆盖用户要求的完整动作语义；不放行奖励设计/PPO，也不把姿态通过等同动作成功。")
        elif evidence_mode in ("MOCK", "MIXED") and all(
                item["result"].success is True for item in ik_results + simulation_results):
            status = FeasibilityStatus.CAPABILITY_SUPPORTED
            validation_level = "MOCK_VALIDATED"
            recommendations.append("Mock 适配器返回成功；该结果不是物理验证，不允许进入生产 Reward Designer。")
        else:
            status = (FeasibilityStatus.CONDITIONAL if assessment.status in
                      (FeasibilityStatus.CONDITIONAL, FeasibilityStatus.UNKNOWN) or
                      provider_failed or not targets else FeasibilityStatus.CAPABILITY_SUPPORTED)
            validation_level = "CAPABILITY_ONLY"
            recommendations.append("当前仅有能力证据或物理后端未完成验证；不能进入生产 Reward Designer。")
        risks.extend(item["result"].reason for item in ik_results
                     if item["result"].status in ("UNAVAILABLE", "SKIPPED", "FAILED"))
        risks.extend(item["result"].reason for item in simulation_results
                     if item["result"].status in ("UNAVAILABLE", "SKIPPED", "FAILED"))
        motion_payload = prototype.dict()
        self._emit_stage("feasibility", status.value, validation_level=validation_level,
                         evidence_mode=evidence_mode)
        return self._report(
            intent, status=status, validation_level=validation_level,
            evidence_mode=evidence_mode, assessment=capability_payload,
            model=model_descriptor, checks=checks, risks=risks,
            recommendations=recommendations, prototype=prototype,
            motion_report={"status": "TARGETS_BUILT", "targets": [
                {"phase": name, "target": target} for name, target in targets],
                "validation_scope": "复用 Unitree PPO 控制器，在同一 Isaac Gym 环境中短时插值执行目标关节姿态；不使用或训练策略",
                "unsupported_scope": motion_scope_issues},
            motion_type=motion_type.value,
            ik_report=ik_payload, ik_result=ik_payload,
            simulation_report=simulation_result.dict(),
            simulation_result=simulation_result.dict(),
            static_dynamics_report=static_dynamics_payload,
            kinematic_report={"status": ik_result.status, "backend": ik_result.backend,
                              "success": ik_result.success, "residual": ik_result.residual,
                              "joint_positions": ik_result.joint_positions,
                              "violations": ik_result.violations},
        )

    def _assess_balance_plan(self, intent: TaskIntentSpec, assessment: Any,
                             model_descriptor: RobotModel,
                             checks: List[FeasibilityCheck], risks: List[str],
                             recommendations: List[str],
                             pipeline_payload: Dict[str, Any]) -> CompleteFeasibilityReport:
        """验证六项平衡规划结果，并仅在真实 Isaac Gym 成功时给出物理通过。"""
        capability_payload = dict(pipeline_payload)
        planning_payload = dict(capability_payload.get("_planning_report", {}))
        try:
            plan = BalanceMotionPlan.parse_obj(planning_payload.get("balance_plan", {}))
        except Exception as exc:
            reason = "平衡规划报告缺失或 Schema 无效：%s" % str(exc)[:300]
            risks.append(reason)
            recommendations.append("重新运行质心、接触、基座、足端、浮动基座 IK 和逆动力学规划。")
            return self._report(
                intent, status=FeasibilityStatus.CONDITIONAL,
                validation_level=ValidationLevel.CAPABILITY_ONLY.value,
                evidence_mode="REAL", assessment=capability_payload,
                model=model_descriptor, checks=checks, risks=risks,
                recommendations=recommendations, motion_type=MotionType.BALANCE.value,
                motion_report={"status": "INCONCLUSIVE", "reason": reason},
            )

        checks.append(FeasibilityCheck(
            name="deterministic_balance_planning",
            status=(FeasibilityStatus.CAPABILITY_SUPPORTED
                    if plan.status == "READY_FOR_PHYSICS"
                    else FeasibilityStatus.CONDITIONAL),
            summary=plan.reason,
            evidence=["backend=" + plan.backend,
                      "samples=" + str(len(plan.samples)),
                      "solvers=" + ",".join(plan.available_solvers)] + list(plan.violations),
        ))
        self._emit_stage(
            "balance_motion_planning", plan.status, backend=plan.backend,
            sample_count=len(plan.samples), violations=list(plan.violations),
            metrics=dict(plan.metrics))
        if plan.status != "READY_FOR_PHYSICS":
            reason = plan.reason or "六项平衡数值规划存在未满足约束"
            risks.extend(list(plan.violations) or [reason])
            recommendations.append(
                "根据 contact_force_qp、基座动力学或 IK 残差调整动作约束；数值规划未通过时不会启动 PPO。")
            return self._report(
                intent, status=FeasibilityStatus.CONDITIONAL,
                validation_level=ValidationLevel.CAPABILITY_ONLY.value,
                evidence_mode="REAL", assessment=capability_payload,
                model=model_descriptor, checks=checks, risks=risks,
                recommendations=recommendations, motion_type=MotionType.BALANCE.value,
                motion_report={"status": plan.status, "balance_plan": plan.dict()},
                static_dynamics_report={"status": "INCONCLUSIVE",
                                        "balance_plan": plan.dict()},
            )

        self._emit_stage("balance_physics_validation", "started",
                         backend=getattr(self.simulation_validator, "backend", "unknown"))
        result = IsaacGymBalanceValidator(self.simulation_validator).validate(
            model_descriptor, plan)
        real_backend = result.backend == "isaacgym" and result.validated
        if real_backend and result.success is True:
            status = FeasibilityStatus.PHYSICS_VALIDATED
            validation_level = ValidationLevel.STATIC_PHYSICS_VALIDATED.value
            evidence_mode = "REAL"
            recommendations.append(
                "六项数值规划及 Unitree Isaac Gym 短时执行均通过；这只表示训练前可尝试，不表示策略已学会。")
        elif real_backend and result.success is False:
            status = FeasibilityStatus.PHYSICS_FAILED
            validation_level = ValidationLevel.PHYSICS_FAILED.value
            evidence_mode = "REAL"
            risks.extend(result.violations or [result.reason])
            recommendations.append("Isaac Gym 发现跌倒、接触、关节或力矩违规；当前禁止进入 Reward Designer。")
        elif "mock" in result.backend.lower() and result.success is True:
            status = FeasibilityStatus.CAPABILITY_SUPPORTED
            validation_level = ValidationLevel.MOCK_VALIDATED.value
            evidence_mode = "MOCK"
            recommendations.append("Mock 只验证接口，不允许作为真实物理放行依据。")
        else:
            status = FeasibilityStatus.CONDITIONAL
            validation_level = ValidationLevel.CAPABILITY_ONLY.value
            evidence_mode = "REAL"
            risks.append(result.reason or "平衡 Isaac Gym rollout 未形成物理证据")
            recommendations.append("恢复 Unitree Isaac Gym 后端后重新执行平衡短时 rollout。")
        checks.append(FeasibilityCheck(
            name="isaacgym_balance_rollout",
            status=(FeasibilityStatus.CAPABILITY_SUPPORTED if result.success is True else
                    FeasibilityStatus.PHYSICS_FAILED if real_backend and result.success is False
                    else FeasibilityStatus.CONDITIONAL),
            summary=result.reason,
            evidence=["backend=" + result.backend,
                      "validated=" + str(result.validated),
                      "status=" + result.status],
        ))
        self._emit_stage(
            "balance_physics_validation", result.status, backend=result.backend,
            validated=result.validated, success=result.success,
            metrics=dict(result.metrics), violations=list(result.violations))
        return self._report(
            intent, status=status, validation_level=validation_level,
            evidence_mode=evidence_mode, assessment=capability_payload,
            model=model_descriptor, checks=checks, risks=risks,
            recommendations=recommendations, motion_type=MotionType.BALANCE.value,
            motion_report={"status": plan.status, "balance_plan": plan.dict(),
                           "validation_scope": "六项数值规划→PPO关节控制器→Unitree Isaac Gym短时rollout"},
            kinematic_report={"status": plan.status, "backend": plan.backend,
                              "success": plan.status == "READY_FOR_PHYSICS",
                              "metrics": dict(plan.metrics)},
            simulation_report=result.dict(), simulation_result=result.dict(),
            static_dynamics_report={"status": result.status,
                                    "balance_plan": plan.dict(),
                                    "simulation": result.dict()},
        )

    def _assess_dynamic(self, intent: TaskIntentSpec, assessment: Any,
                        model_descriptor: RobotModel, prototype: DynamicMotionPrototype,
                        checks: List[FeasibilityCheck], risks: List[str],
                        recommendations: List[str]) -> CompleteFeasibilityReport:
        """对动态类别执行真实 Isaac Gym 探针，未知类别保持条件状态。"""
        capability_payload = assessment.dict()
        if prototype.motion_type != MotionType.LOCOMOTION:
            reason = ("动作类别未知，不能安全生成运动轨迹" if prototype.motion_type == MotionType.UNKNOWN
                      else "Go2 当前没有该动作类型的专用动态物理验证器：%s" %
                      prototype.motion_type.value)
            risks.append(reason)
            recommendations.append("补充明确动作语义或为该 MotionType 实现确定性物理验证器。")
            self._emit_stage("dynamic_motion_validation", "blocked",
                             motion_type=prototype.motion_type.value, reason=reason)
            return self._report(
                intent, status=FeasibilityStatus.CONDITIONAL,
                validation_level=ValidationLevel.CAPABILITY_ONLY.value,
                evidence_mode="REAL", assessment=capability_payload,
                model=model_descriptor, checks=checks, risks=risks,
                recommendations=recommendations, prototype=prototype,
                motion_type=prototype.motion_type.value,
                motion_report={"status": "NOT_RUN", "trajectory": prototype.dict()},
                dynamic_report=DynamicFeasibilityReport(
                    task=intent.normalized_goal, motion_type=prototype.motion_type.value,
                    backend="not_run", validation_level=ValidationLevel.CAPABILITY_ONLY.value,
                    duration=prototype.duration,
                    gait=prototype.gait.dict() if prototype.gait is not None else None,
                    trajectory=prototype.dict(), limitations=[reason], reason=reason).dict(),
            )

        self._emit_stage("dynamic_motion_validation", "started",
                         motion_type=prototype.motion_type.value,
                         duration=prototype.duration)
        try:
            result = self.dynamic_validator.validate(model_descriptor.runtime_dict(), prototype)
        except Exception as exc:
            result = SimulationReport(
                status="UNAVAILABLE", success=None,
                backend=getattr(self.dynamic_validator, "backend", "unknown"),
                validated=False, validation_level=ValidationLevel.CAPABILITY_ONLY.value,
                reason="动态验证 adapter 异常：%s" % str(exc)[:400],
            )
        backend_is_real = result.backend == "isaacgym" and result.validated
        if (backend_is_real and result.success is True and
                result.validation_level == ValidationLevel.DYNAMIC_PHYSICS_VALIDATED.value and
                result.metrics.get("foot_trajectory_consumed") is True and
                result.metrics.get("joint_reference_trajectory_generated") is True):
            status = FeasibilityStatus.PHYSICS_VALIDATED
            validation_level = ValidationLevel.DYNAMIC_PHYSICS_VALIDATED.value
            evidence_mode = "REAL"
            recommendations.append(
                "真实 Go2 Isaac Gym 短时足端轨迹/Pinocchio IK 验证通过；这不代表策略已学会动作，"
                "完整目标仍需 PPO 训练和数值/视觉验收。")
        elif backend_is_real and result.success is True:
            status = FeasibilityStatus.CONDITIONAL
            validation_level = ValidationLevel.CAPABILITY_ONLY.value
            evidence_mode = "REAL"
            risks.append("动态 validator 缺少完整的 FootTrajectory→IK→JointReference 证据")
            recommendations.append("只有真实执行了完整足端参考轨迹后才可报告动态物理通过。")
        elif backend_is_real and result.success is False:
            status = FeasibilityStatus.PHYSICS_FAILED
            validation_level = ValidationLevel.PHYSICS_FAILED.value
            evidence_mode = "REAL"
            risks.append(result.reason)
            recommendations.append("真实动态 rollout 未通过安全/跟踪验收，已阻止 Reward Designer 和 PPO。")
        elif "mock" in str(result.backend).lower():
            status = FeasibilityStatus.CAPABILITY_SUPPORTED
            validation_level = ValidationLevel.MOCK_VALIDATED.value
            evidence_mode = "MOCK"
            risks.append("动态后端为 Mock；没有物理验证证据")
            recommendations.append("Mock 结果只能检查接口；不能作为动态动作可行性结论。")
        else:
            status = FeasibilityStatus.CONDITIONAL
            validation_level = ValidationLevel.CAPABILITY_ONLY.value
            evidence_mode = "REAL"
            risks.append(result.reason or "动态物理验证未能完成")
            recommendations.append("修复 Isaac Gym/动作映射或验证器支持范围后重新检查。")

        checks.append(FeasibilityCheck(
            name="isaacgym_dynamic_rollout",
            status=(FeasibilityStatus.CAPABILITY_SUPPORTED if status == FeasibilityStatus.PHYSICS_VALIDATED
                    else FeasibilityStatus.PHYSICS_FAILED if status == FeasibilityStatus.PHYSICS_FAILED
                    else FeasibilityStatus.CONDITIONAL),
            summary=result.reason,
            evidence=["backend=" + result.backend,
                      "validation_level=" + result.validation_level,
                      "validated=" + str(result.validated)] + result.violations,
        ))
        self._emit_stage("dynamic_motion_validation", result.status,
                         backend=result.backend, validated=result.validated,
                         success=result.success, validation_level=validation_level,
                         foot_trajectory_consumed=bool(
                             result.metrics.get("foot_trajectory_consumed")),
                         ik_samples=result.metrics.get("ik_samples", 0))
        trajectory_consumed = bool(result.metrics.get("foot_trajectory_consumed"))
        limitations = [
            "本预检使用逐时刻 Pinocchio 足端 IK 关节位置参考和 Unitree position-PD 环境；不创建或训练 PPO policy。",
            "结论仅适用于本次 Go2 URDF、Unitree Isaac Gym 配置、步态参数和短时 rollout。",
            "该 IK-derived 开环参考不是闭环 locomotion policy；真实长时稳定性、扰动恢复及任务成功仍需训练后评估。",
            "当前已检查关节位置限位与 q(t) 有限差分速度限位；加速度/力矩跟踪能力仍由仿真 rollout 观察，未作为完整轨迹优化约束。",
        ] + list(prototype.notes)
        dynamic_report = DynamicFeasibilityReport(
            task=intent.normalized_goal, motion_type=prototype.motion_type.value,
            backend=result.backend, success=result.success,
            duration=prototype.duration,
            gait=prototype.gait.dict() if prototype.gait is not None else None,
            validation_level=validation_level, trajectory=prototype.dict(),
            metrics=dict(result.metrics), violations=list(result.violations),
            limitations=limitations, reason=result.reason,
        )
        motion_report = {
            "status": result.status, "trajectory": prototype.dict(),
            "validation_scope": "Unitree Go2 同一 Isaac Gym asset/PD/PhysX 中执行 Pinocchio 足端轨迹关节参考；非 PPO policy",
            "limitations": limitations,
        }
        trajectory_report = {
            "status": ("COMPILED" if trajectory_consumed else "INCONCLUSIVE"),
            "foot_trajectory": (prototype.foot_trajectory.dict()
                                 if prototype.foot_trajectory is not None else None),
            "backend": result.metrics.get("reference_source", "local_semantic_prototype"),
            "consumed_by_isaacgym_validator": trajectory_consumed,
            "ik_samples": result.metrics.get("ik_samples", 0),
            "joint_reference_samples": result.metrics.get("joint_reference_samples", 0),
            "reason": result.reason,
            "limitations": ["短时关节位置参考；不等价于已经学会或能长期执行该动作。"],
        }
        self._emit_stage("feasibility", status.value,
                         validation_level=validation_level, evidence_mode=evidence_mode)
        return self._report(
            intent, status=status, validation_level=validation_level,
            evidence_mode=evidence_mode, assessment=capability_payload,
            model=model_descriptor, checks=checks, risks=risks,
            recommendations=recommendations, prototype=prototype,
            motion_type=prototype.motion_type.value,
            motion_report=motion_report, dynamic_report=dynamic_report.dict(),
            simulation_report=result.dict(), simulation_result=result.dict(),
            trajectory_report=trajectory_report,
            physics_report={"status": result.status, "backend": result.backend,
                            "validated": result.validated, "success": result.success,
                            "validation_level": result.validation_level,
                            "metrics": dict(result.metrics), "violations": list(result.violations),
                            "reason": result.reason},
        )

    def _report(self, intent: TaskIntentSpec, status: FeasibilityStatus,
                validation_level: str, evidence_mode: str, assessment: Dict[str, Any],
                model: RobotModel, checks: List[FeasibilityCheck], risks: List[str],
                recommendations: List[str], prototype: Optional[Any] = None,
                motion_type: Optional[str] = None,
                motion_report: Optional[Dict[str, Any]] = None,
                dynamic_report: Optional[Dict[str, Any]] = None,
                ik_report: Optional[Dict[str, Any]] = None,
                ik_result: Optional[Dict[str, Any]] = None,
                kinematic_report: Optional[Dict[str, Any]] = None,
                static_dynamics_report: Optional[Dict[str, Any]] = None,
                trajectory_report: Optional[Dict[str, Any]] = None,
                physics_report: Optional[Dict[str, Any]] = None,
                simulation_report: Optional[Dict[str, Any]] = None,
                simulation_result: Optional[Dict[str, Any]] = None) -> CompleteFeasibilityReport:
        """以同一等级同时填充旧字段和新的物理验证报告字段。"""
        ik_data = kinematic_report or ik_report or {}
        static_data = static_dynamics_report or {}
        trajectory_data = trajectory_report or {}
        physics_data = physics_report or simulation_report or {}
        constraint_data = dict(assessment.get("_motion_constraint_spec", {}))
        planning_data = dict(assessment.get("_planning_report", {}))
        whole_body_data = dict(assessment.get("_whole_body_report", {}))
        assessment_public = {key: value for key, value in assessment.items()
                             if not str(key).startswith("_")}
        capability_status = str(assessment.get("status", FeasibilityStatus.UNKNOWN.value))
        ik_success = ik_data.get("success")
        ik_backend = str(ik_data.get("backend", "not_run"))
        physical_backend = str(physics_data.get("backend", "not_run"))
        physical_validated = bool(physics_data.get("validated")) and physical_backend in (
            "isaacgym", "mujoco")

        def stage(name: str, raw_status: Any, reason: str = "", backend: str = "not_run",
                  metrics: Optional[Dict[str, Any]] = None,
                  evidence: Optional[List[str]] = None) -> ValidationStageReport:
            """把旧字段状态映射为统一、不会把未知升格为通过的子阶段报告。"""
            raw = raw_status.value if hasattr(raw_status, "value") else str(raw_status)
            mapping = {
                "PASSED": StageStatus.PASSED, "SOLVED": StageStatus.PASSED,
                "SUPPORTED": StageStatus.PASSED, "CAPABILITY_SUPPORTED": StageStatus.PASSED,
                "PHYSICS_VALIDATED": StageStatus.PASSED, "DYNAMIC_PHYSICS_VALIDATED": StageStatus.PASSED,
                "STATIC_PHYSICS_VALIDATED": StageStatus.PASSED, "FAILED": StageStatus.FAILED,
                "GENERATED": StageStatus.PASSED, "READY_FOR_SOLVER": StageStatus.PASSED,
                "READY_FOR_PHYSICS": StageStatus.PASSED,
                "PHYSICS_FAILED": StageStatus.FAILED, "UNSUPPORTED": StageStatus.FAILED,
                "CONDITIONAL": StageStatus.CONDITIONAL, "MODEL_UNAVAILABLE": StageStatus.CONDITIONAL,
                "UNKNOWN": StageStatus.UNKNOWN, "INCONCLUSIVE": StageStatus.INCONCLUSIVE,
                "UNAVAILABLE": StageStatus.INCONCLUSIVE, "SKIPPED": StageStatus.NOT_RUN,
                "NOT_RUN": StageStatus.NOT_RUN, "MOCK_VALIDATED": StageStatus.CONDITIONAL,
            }
            try:
                normalized = FeasibilityStatus(raw).value
                mapped = mapping.get(normalized, mapping.get(raw, StageStatus.UNKNOWN))
            except ValueError:
                mapped = mapping.get(raw, StageStatus.UNKNOWN)
            return ValidationStageReport(
                stage=name, status=mapped, reason=reason, backend=backend,
                metrics=metrics or {}, evidence=evidence or [],
            )

        capability_stage = stage(
            "capability", capability_status,
            reason="基于本地机器人/环境清单执行能力检查",
            backend="deterministic_registry",
            metrics={"missing_requirements": assessment.get("missing_requirements", [])},
            evidence=[str(value) for check in assessment.get("checks", [])
                      for value in (check.get("evidence", []) if isinstance(check, dict) else [])],
        )
        kinematic_stage = stage(
            "kinematic", "SOLVED" if ik_success is True else
            "FAILED" if ik_success is False and ik_backend == "pinocchio" else "INCONCLUSIVE",
            reason=str(ik_data.get("reason", "IK 证据未提供")), backend=ik_backend,
            metrics={key: ik_data.get(key) for key in ("residual", "iterations", "violations")
                     if key in ik_data},
            evidence=["IK backend=" + ik_backend] if ik_backend != "not_run" else [],
        )
        static_status = static_data.get("status", "NOT_RUN")
        static_stage = stage(
            "static_dynamics", static_status,
            reason=str(static_data.get("note", static_data.get("reason", "静态动力学阶段未执行"))),
            backend="deterministic_geometry" if static_status != "NOT_RUN" else "not_run",
            metrics={key: value for key, value in static_data.items()
                     if key not in ("candidate_results", "note", "reason")},
            evidence=["static candidate results supplied"] if static_data else [],
        )
        trajectory_stage = stage(
            "trajectory", trajectory_data.get("status", "NOT_RUN"),
            reason=str(trajectory_data.get("reason", "轨迹验证器未执行")),
            backend=str(trajectory_data.get("backend", "not_run")),
            metrics={key: value for key, value in trajectory_data.items()
                     if key not in ("prototype", "reason", "limitations")},
        )
        physics_stage = stage(
            "physics", physics_data.get("validation_level",
                                        physics_data.get("status", "NOT_RUN")),
            reason=str(physics_data.get("reason", "真实物理 rollout 未执行")),
            backend=physical_backend,
            metrics=dict(physics_data.get("metrics", {})),
            evidence=["backend=%s" % physical_backend] if physical_backend != "not_run" else [],
        )
        stage_reports = [
            ValidationStageReport(stage="language_understanding", status=StageStatus.PASSED,
                                  reason="TaskIntentSpec 已通过结构化 Schema 校验",
                                  backend="TaskIntentSpec"),
            stage("motion_planning", planning_data.get("status", "INCONCLUSIVE"),
                  reason=("确定性规划器已按 MotionConstraintSpec 分派" if planning_data else
                          str(assessment.get("_motion_constraint_error",
                                             "MotionConstraintSpec 未生成"))),
                  backend=("deterministic_planner_registry" if planning_data else "not_run"),
                  metrics={
                      "missing_planners": planning_data.get("missing_planners", []),
                      "solver_requirements": planning_data.get("solver_requirements", []),
                  }),
            stage("whole_body_solve", whole_body_data.get("status", "INCONCLUSIVE"),
                  reason=("已核对动作所需全身求解后端" if whole_body_data else
                          "全身求解能力未核对"),
                  backend=str(whole_body_data.get("backend", "not_run")),
                  metrics={
                      "available_solvers": whole_body_data.get("available_solvers", []),
                      "missing_solvers": whole_body_data.get("missing_solvers", []),
                  }),
            capability_stage, kinematic_stage, static_stage, trajectory_stage, physics_stage,
        ]
        if status in (FeasibilityStatus.PHYSICS_VALIDATED, FeasibilityStatus.PHYSICS_FAILED) \
                and physical_validated:
            evidence_level = FeasibilityLevel.LEVEL_5_PHYSICS
            confidence = 0.9 if status == FeasibilityStatus.PHYSICS_VALIDATED else 0.85
        elif trajectory_stage.status == StageStatus.PASSED:
            evidence_level, confidence = FeasibilityLevel.LEVEL_4_TRAJECTORY, 0.7
        elif ik_success is True and ik_backend == "pinocchio":
            evidence_level, confidence = FeasibilityLevel.LEVEL_2_KINEMATIC, 0.55
        elif assessment:
            evidence_level, confidence = FeasibilityLevel.LEVEL_1_CAPABILITY, 0.35
        else:
            evidence_level, confidence = FeasibilityLevel.LEVEL_0_LANGUAGE, 0.15
        evidence_items = sorted({str(item) for check in checks for item in check.evidence})
        limitations = []
        for source in (motion_report or {}, trajectory_data, physics_data):
            limitations.extend(str(item) for item in source.get("limitations", [])
                               if item)
        limitations.extend(str(item) for item in risks if item)
        return CompleteFeasibilityReport(
            status=status, backend=(simulation_report or {}).get("backend"),
            task=intent.normalized_goal,
            action_type=motion_type,
            confidence=confidence,
            confidence_basis="仅表示当前结论的证据覆盖度；不是机器人动作成功率或 PPO 学习概率",
            feasibility_level=evidence_level,
            stage_reports=stage_reports,
            kinematic_report=ik_data,
            static_dynamics_report=static_data,
            trajectory_report=trajectory_data,
            physics_report=physics_data,
            evidence=evidence_items,
            limitations=sorted(set(limitations)),
            recommended_next_step=(recommendations[0] if recommendations else "人工复核当前证据"),
            required_capabilities=assessment_public.get("required_capabilities", []),
            checks=checks, missing_requirements=assessment_public.get("missing_requirements", []),
            risks=sorted(set(str(item) for item in risks if item)),
            recommendations=recommendations,
            capability_check=assessment_public, capability_report=assessment_public,
            motion_prototype=prototype.dict() if prototype is not None else None,
            motion_type=motion_type,
            motion_report=motion_report or {},
            motion_constraint_spec=constraint_data,
            planning_report=planning_data,
            whole_body_report=whole_body_data,
            dynamic_report=dynamic_report,
            ik_report=ik_report or {}, ik_result=ik_result,
            simulation_report=simulation_report or {},
            simulation_result=simulation_result,
            robot_model=model.public_dict(), converted_model=model.converted_model,
            evidence_mode=evidence_mode, validation_level=validation_level,
        )

    @staticmethod
    def _task_text(intent: TaskIntentSpec) -> str:
        """合并任务原文和标准目标以确定是否属于特殊平衡动作。"""
        return " ".join((intent.original_instruction, intent.action_name,
                         intent.normalized_goal)).lower()

    @classmethod
    def _special_balance_request(cls, intent: TaskIntentSpec) -> bool:
        """识别当前姿态构造器尚不能确定性生成的单腿、倒立/handstand目标。"""
        text = cls._task_text(intent)
        return any(term in text for term in (
            "单腿", "单足", "one-leg", "single-leg", "倒立", "handstand", "inverted",
            "前腿站立", "前足站立", "后腿站立", "后足站立", "front_leg_stand",
            "hind_leg_stand", "rear_leg_stand", "front_leg_walk", "hind_leg_walk"))

    @classmethod
    def _single_leg_request(cls, intent: TaskIntentSpec) -> bool:
        """识别需要枚举四种单足支撑侧的平衡目标。"""
        text = cls._task_text(intent)
        return any(term in text for term in ("单腿", "单足", "one-leg", "single-leg"))

    def _assess_balance_candidates(
            self, intent: TaskIntentSpec, assessment: Any, model: RobotModel,
            checks: List[FeasibilityCheck], risks: List[str],
            recommendations: List[str]) -> CompleteFeasibilityReport:
        """对有限的平衡候选运行 IK 与静态必要条件，不冒充平衡物理验证。"""
        has_real_ik = str(getattr(self.ik_solver, "backend", "")).lower() == "pinocchio"
        text = self._task_text(intent)
        if self._single_leg_request(intent):
            candidate_family = "single_leg"
            generator = StaticPoseCandidateGenerator.single_leg_candidates
            generator_arguments = ()
        elif any(term in text for term in ("前腿", "前足", "front_leg")):
            candidate_family = "front_support"
            generator = StaticPoseCandidateGenerator.leg_pair_candidates
            generator_arguments = ("front",)
        elif any(term in text for term in ("后腿", "后足", "hind_leg", "rear_leg")):
            candidate_family = "hind_support"
            generator = StaticPoseCandidateGenerator.leg_pair_candidates
            generator_arguments = ("hind",)
        else:
            candidate_family = "unsupported_pose"
            generator = None
            generator_arguments = ()
        try:
            candidates = (generator(model.runtime_dict(), *generator_arguments)
                          if generator is not None else [])
        except Exception as exc:
            candidates = []
            candidate_generation_error = str(exc)[:250]
        else:
            candidate_generation_error = ""
        candidate_results: List[Dict[str, Any]] = []
        if has_real_ik:
            for candidate in candidates:
                try:
                    ik = self.ik_solver.solve_ik(
                        model.runtime_dict(), candidate.target, model.default_joint_positions)
                except Exception as exc:
                    ik = IKResult(
                        status="FAILED", success=False, backend="pinocchio",
                        reason="候选 IK adapter 异常：%s" % str(exc)[:250])
                static = StaticPoseValidator().assess(
                    model.runtime_dict(), candidate.target, ik.joint_positions,
                    self_collision_checked=ik.self_collision_checked)
                candidate_results.append({
                    "candidate": candidate.dict(), "ik": ik.dict(),
                    "static_dynamics": static,
                })
                self._emit_stage(
                    "static_candidate", "evaluated", candidate_id=candidate.candidate_id,
                    ik_status=ik.status, ik_success=ik.success,
                    static_status=static.get("status", "INCONCLUSIVE"))
        ik_successes = sum(1 for item in candidate_results
                           if item["ik"].get("success") is True)
        feasible = sum(1 for item in candidate_results
                       if item["static_dynamics"].get("status") == "PASSED")
        if not candidates:
            search_status = "SEARCH_FAILED"
            reason = ("机器人模型无法生成候选；搜索失败不代表动作理论上不可能" +
                      (("：" + candidate_generation_error) if candidate_generation_error else ""))
        elif not has_real_ik:
            search_status = "NOT_RUN"
            reason = "当前 IK backend 不是 Pinocchio；Mock 结果不用于候选可行性结论"
        elif ik_successes == 0:
            search_status = "SEARCH_FAILED"
            reason = ("%s 个 %s 候选均未通过 Pinocchio IK；有限搜索失败不代表理论上不可能" %
                      (len(candidates), candidate_family))
        else:
            search_status = "INCONCLUSIVE"
            reason = ("存在 IK 可达候选，但支撑 CoM/接触、碰撞和力矩证据不完整；"
                      "尚不能判断静态平衡或动态保持能力")
        candidate_report = {
            "status": "INCONCLUSIVE", "search_status": search_status,
            "total_candidates": len(candidates),
            "evaluated_candidates": len(candidate_results),
            "ik_success_count": ik_successes, "feasible_candidates": feasible,
            "candidate_ids": [item.candidate_id for item in candidates],
            "reason": reason,
            "interpretation": "有限候选搜索结果不能证明动作理论上可行或不可能",
        }
        checks.append(FeasibilityCheck(
            name="balance_candidate_search", status=FeasibilityStatus.CONDITIONAL,
            summary=reason,
            evidence=["Go2 nominal FK candidate targets", "candidate_family=" + candidate_family,
                      "Pinocchio IK" if has_real_ik else "IK backend not run",
                      "support contact wrench unavailable"],
        ))
        risks.append(reason)
        recommendations.append(
            "平衡目标仍需浮动基座/接触力约束下的支撑稳定性分析和 Isaac Gym rollout；"
            "当前不会自动进入奖励设计。")
        self._emit_stage("static_candidate_search", search_status,
                         candidate_count=len(candidates), evaluated=len(candidate_results),
                         ik_success_count=ik_successes, feasible_candidates=feasible,
                         reason=reason)
        ik_payload = {
            "status": "SOLVED" if ik_successes else
                      "FAILED" if candidates and has_real_ik else "NOT_RUN",
            "success": True if ik_successes else False if candidates and has_real_ik else None,
            "backend": getattr(self.ik_solver, "backend", "unknown"),
            "candidate_results": candidate_results, "reason": reason,
        }
        static_payload = {
            "status": "INCONCLUSIVE", "candidate_search": candidate_report,
            "candidate_results": candidate_results,
            "limitations": [
                "固定基座 Pinocchio IK 仅检查脚端目标的运动学可达性。",
                "候选支撑足几何不足以提供完整接触力/摩擦证据；动态平衡未验证。",
                "有限候选搜索失败不构成机器人能力不可能的证明。",
            ],
        }
        return self._report(
            intent, status=FeasibilityStatus.CONDITIONAL,
            validation_level=ValidationLevel.CAPABILITY_ONLY.value,
            evidence_mode="REAL" if has_real_ik else "MOCK",
            assessment=assessment.dict(), model=model, checks=checks, risks=risks,
            recommendations=recommendations, motion_type=MotionType.BALANCE.value,
            motion_report={"status": search_status, "candidate_search": candidate_report},
            ik_report=ik_payload, ik_result=ik_payload, kinematic_report=ik_payload,
            static_dynamics_report=static_payload,
            simulation_report={"status": "NOT_RUN", "backend": "not_run",
                               "reason": "没有单足平衡专用物理 validator"},
            simulation_result={"status": "NOT_RUN", "backend": "not_run",
                               "validated": False},
        )

    @staticmethod
    def _combine_ik_results(results: Sequence[Dict[str, Any]]) -> IKResult:
        """保守合并各阶段目标 IK，失败优先、未知不伪装为成功。"""
        values = [item["result"] for item in results]
        success = (False if any(item.success is False for item in values) else
                   True if values and all(item.success is True for item in values) else None)
        status = "FAILED" if success is False else "SOLVED" if success is True else (
            "UNAVAILABLE" if any(item.status == "UNAVAILABLE" for item in values) else "SKIPPED")
        residuals = [item.residual for item in values if item.residual is not None]
        iterations = max((item.iterations for item in values), default=0)
        joint_positions = values[-1].joint_positions if values else {}
        backend = values[0].backend if values and all(
            item.backend == values[0].backend for item in values) else "mixed"
        return IKResult(
            status=status, success=success, backend=backend,
            joint_positions=joint_positions,
            violations=sorted({issue for item in values for issue in item.violations}),
            iterations=iterations, residual=max(residuals) if residuals else None,
            error=max(residuals) if residuals else None,
            self_collision_checked=all(item.self_collision_checked for item in values),
            base_height=values[-1].base_height if values else None,
            base_orientation_xyzw=values[-1].base_orientation_xyzw if values else [0, 0, 0, 1],
            duration_seconds=sum(item.duration_seconds for item in values),
            reason="；".join("%s: %s" % (entry["phase"], entry["result"].reason)
                            for entry in results),
        )

    @staticmethod
    def _combine_simulation_results(results: Sequence[Dict[str, Any]]) -> SimulationReport:
        """合并所有真实/Mock rollout，保留最差物理值及各目标原始报告。"""
        values = [item["result"] for item in results]
        success = (False if any(item.success is False for item in values) else
                   True if values and all(item.success is True for item in values) else None)
        status = "FAILED" if success is False else "PASSED" if success is True else (
            "UNAVAILABLE" if any(item.status == "UNAVAILABLE" for item in values) else "SKIPPED")
        backends = {item.backend for item in values}
        backend = next(iter(backends)) if len(backends) == 1 else "mixed"
        metrics: Dict[str, Any] = {}
        reducers = {"max_roll": max, "max_pitch": max,
                    "max_joint_limit_excess": max, "min_height": min,
                    "max_actuator_tracking_error": max, "max_torque_limit_ratio": max}
        metric_keys = {key for item in values for key in item.metrics
                       if key != "target_results"}
        for key in metric_keys:
            present = [(item, item.metrics[key]) for item in values if key in item.metrics]
            raw_values = [value for _, value in present]
            if key in reducers:
                metrics[key] = reducers[key](raw_values)
            elif key in ("steps", "target_count", "forbidden_body_contact_samples"):
                metrics[key] = sum(raw_values)
            elif key in ("contact_sample_fraction", "foot_contact_sample_fraction"):
                weights = [int(item.metrics.get("steps", 1)) for item, _ in present]
                metrics[key] = (sum(float(value) * weight for value, weight in
                                    zip(raw_values, weights)) / max(1, sum(weights)))
            elif all(value == raw_values[0] for value in raw_values):
                metrics[key] = raw_values[0]
            elif len(values) == 1:
                metrics[key] = raw_values[0]
        if "target_count" not in metrics:
            metrics["target_count"] = len(values)
        metrics["simulation_phase_count"] = len(values)
        metrics["phase_results"] = [{
            "phase": entry["phase"],
            "success": entry["result"].success,
            "validation_level": entry["result"].validation_level,
            "violations": entry["result"].violations,
        } for entry in results]
        violations = sorted({issue for item in values for issue in item.violations})
        if values and any(item.validation_level == "PHYSICS_FAILED" for item in values):
            merged_validation_level = "PHYSICS_FAILED"
        elif values and all(item.validation_level == ValidationLevel.DYNAMIC_PHYSICS_VALIDATED.value
                            for item in values):
            merged_validation_level = ValidationLevel.DYNAMIC_PHYSICS_VALIDATED.value
        elif values and all(item.validation_level == ValidationLevel.STATIC_PHYSICS_VALIDATED.value
                            for item in values):
            merged_validation_level = ValidationLevel.STATIC_PHYSICS_VALIDATED.value
        elif values and all(item.validation_level == "PHYSICS_VALIDATED" for item in values):
            merged_validation_level = ValidationLevel.STATIC_PHYSICS_VALIDATED.value
        elif values and all(item.validation_level == "PHYSICS_ROLLOUT" for item in values):
            merged_validation_level = "PHYSICS_ROLLOUT"
        elif values and all(item.validation_level == "MOCK_VALIDATED" for item in values):
            merged_validation_level = "MOCK_VALIDATED"
        else:
            merged_validation_level = "CAPABILITY_ONLY"
        return SimulationReport(
            status=status, success=success, backend=backend,
            validated=bool(values) and all(item.validated for item in values),
            validation_level=merged_validation_level,
            converted_model=bool(values) and all(item.converted_model for item in values),
            model_source=values[0].model_source if values else "",
            duration=sum(item.duration for item in values),
            fall=True if any(item.fall is True for item in values) else
                False if values and all(item.fall is False for item in values) else None,
            self_collision_checked=bool(values) and all(item.self_collision_checked for item in values),
            violations=violations, metrics=metrics,
            reason="；".join("%s: %s" % (entry["phase"], entry["result"].reason)
                            for entry in results),
        )

    @staticmethod
    def _evidence_mode(ik_results: Sequence[IKResult], simulation_results: Sequence[SimulationReport],
                       prototype: Optional[MotionPrototype] = None) -> str:
        """依实际 adapter 和原型来源标明真实、Mock 或混合证据。"""
        backends = [item.backend for item in ik_results] + [item.backend for item in simulation_results]
        if prototype is not None and prototype.source == "mock":
            backends.append("mock-motion")
        if backends and all(item.lower() in ("mock", "mock-motion") for item in backends):
            return "MOCK"
        if any("mock" in item.lower() for item in backends):
            return "MIXED"
        return "REAL"

    def _uses_mock_backend(self) -> bool:
        """判断当前流水线是否至少注入了一个显式 Mock adapter。"""
        return any("mock" in str(getattr(adapter, "backend", "")).lower()
                   for adapter in (self.ik_solver, self.simulation_validator))

    @staticmethod
    def _unsupported_motion_scope(intent: TaskIntentSpec, prototype: MotionPrototype,
                                  robot_model: Dict[str, Any]) -> List[str]:
        """拒绝把静态姿态保持冒充为行走、跳跃或完整多足姿态的物理验证。"""
        text = " ".join((intent.original_instruction, intent.action_name,
                         intent.normalized_goal)).lower()
        issues: List[str] = []
        dynamic_terms = ("走", "行走", "跑", "倒退", "后退", "walk", "run", "locomotion",
                         "跳", "jump", "gait", "步态")
        if (any(term in text for term in dynamic_terms) or
                (intent.target_velocity is not None and abs(float(intent.target_velocity)) > 1.0e-6)):
            issues.append("目标包含动态步态/移动/跳跃；短时 Isaac Gym 姿态预检不是动态动作控制器")

        allowed_goals = {
            ("torso", "upright"), ("feet", "support"),
            ("front_feet", "support"), ("front_feet", "lift"),
            ("hind_feet", "support"), ("hind_feet", "lift"),
        }
        foot_frames = set(str(item) for item in robot_model.get("foot_frames", []))
        for phase in prototype.phases:
            for key, value in phase.body_goal.items():
                pair = (str(key).lower(), str(value).lower())
                if pair not in allowed_goals:
                    issues.append("阶段 %s 的语义目标 %s=%s 没有对应的确定性物理验证器" %
                                  (phase.name, key, value))
            for target in phase.robot_targets:
                missing = sorted(foot_frames - set(target.feet))
                if missing:
                    issues.append("阶段 %s 的 RobotMotionTarget 缺少脚端目标：%s" %
                                  (phase.name, ", ".join(missing)))
                orientation = [float(value) for value in target.base_orientation_xyzw]
                norm = sum(value * value for value in orientation) ** 0.5
                if (norm <= 1.0e-12 or abs(orientation[0] / norm) > 1.0e-6 or
                        abs(orientation[1] / norm) > 1.0e-6 or
                        abs(orientation[2] / norm) > 1.0e-6 or
                        abs(abs(orientation[3] / norm) - 1.0) > 1.0e-6):
                    issues.append("阶段 %s 需要非零基座旋转；当前脚端 IK 未求解浮动基座姿态" % phase.name)
            if phase.target_poses:
                issues.append("阶段 %s 使用旧式单末端 TargetPose；没有完整的多足姿态目标" % phase.name)

        explicit_heights = [float(target.base_height)
                            for phase in prototype.phases for target in phase.robot_targets]
        if explicit_heights and max(explicit_heights) - min(explicit_heights) > 1.0e-4:
            issues.append("目标要求基座高度随阶段变化；当前短时关节目标 rollout 不控制浮动基座轨迹")

        body_goals = {(str(key).lower(), str(value).lower())
                      for phase in prototype.phases for key, value in phase.body_goal.items()}
        hind_task = any(term in text for term in ("后腿", "hind_leg", "rear_leg"))
        front_task = any(term in text for term in ("前腿", "front_leg"))
        if hind_task:
            if ("front_feet", "lift") not in body_goals:
                issues.append("后腿站立目标未明确描述抬起前足")
            if ("hind_feet", "support") not in body_goals:
                issues.append("后腿站立目标未明确描述后足支撑")
        if front_task:
            if ("front_feet", "lift") not in body_goals:
                issues.append("前腿站立目标未明确描述抬起前足")
            if ("hind_feet", "support") not in body_goals:
                issues.append("前腿站立目标未明确描述后足支撑")
        if any(term in text for term in ("站立", "站着", "stand", "balance")) and not any(
                value == "support" for _, value in body_goals):
            issues.append("静态站立/平衡任务没有声明支撑足，物理预检无法确认支撑约束")
        return sorted(set(issues))
