from __future__ import annotations

import hashlib
import json
import math
import statistics
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..environment.inspector import EnvironmentInspector
from ..environment.metric_registry import normalize_task_metrics
from ..environment.project_adapter import UnitreeProjectAdapter
from ..evaluation.deterministic import DeterministicEvaluator
from ..metrics.report_builder import ReportBuilder
from ..metrics.ppo_collector import PPOCollector
from ..metrics.tensorboard_reader import TensorBoardReader
from ..metrics.trajectory_metrics import TrajectoryMetrics
from ..providers.base import LLMReasoningProvider
from ..providers.errors import ProviderError
from ..providers.mock_provider import MockLLMReasoningProvider
from ..providers.opencli_chatgpt import OpenCLIChatGPTWebProvider
from ..rewards.compiler import RewardCompiler
from ..rewards.reviser import RewardPlanReviser
from ..rewards.validator import RewardValidationError
from ..schemas.decisions import TrainingDiagnosis
from ..schemas.experiments import ExperimentManifest
from ..schemas.rewards import RewardPlan, RewardTerm
from ..schemas.task import TaskSpec
from ..settings import Settings
from ..storage.experiment_store import ExperimentStore
from ..storage.lineage import LineageGraph
from ..training.checkpoint_manager import CheckpointManager
from ..training.controller import TrainingController
from ..utils.io import atomic_write_text, read_json, utc_now, write_json
from ..visual.evaluation_pipeline import VisualEvaluationPipeline
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
        adapter = UnitreeProjectAdapter(settings.training_root, settings.agent_root, settings.experiments_path)
        self.controller = TrainingController(adapter, timeout_seconds=settings.training_timeout_seconds)

    @staticmethod
    def provider_for(settings: Settings, name: str, record_dir: Optional[Path] = None) -> LLMReasoningProvider:
        """创建生产网页 Provider 或显式测试 Provider。"""
        if name == "mock":
            return MockLLMReasoningProvider(settings.num_reward_candidates)
        if name != "opencli":
            raise ValueError("production provider must be 'opencli'; tests may select 'mock'")
        return OpenCLIChatGPTWebProvider(record_dir=record_dir)

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
        instruction = task.original_instruction
        mentions_front = "前腿" in instruction or "前脚" in instruction or "前足" in instruction
        support_action = "站立" in instruction or "走路" in instruction or "行走" in instruction
        explicitly_lifted = "抬起前腿" in instruction or "前腿离地" in instruction or "前足离地" in instruction
        return mentions_front and support_action and not explicitly_lifted

    @staticmethod
    def _normalize_task_for_instruction(task: TaskSpec) -> List[str]:
        """纠正自然语言中“用前腿站立”被误解为“抬起前腿”的任务语义。"""
        adjustments: List[str] = []
        if not TrainingOrchestrator._is_front_leg_support_task(task):
            return adjustments
        for behavior in task.required_behaviors:
            if behavior.name == "front_leg_lifted_posture" or "前腿离地" in behavior.description:
                behavior.name = "front_leg_support_posture"
                behavior.description = "保持两个前足支撑、两个后足离地，并维持目标高度、俯仰角和横滚稳定。"
                adjustments.append("纠正前腿站立语义：前足支撑、后足离地")
        requirement = "必须确认前足支撑、后足离地，不能把前腿站立误判为抬起前腿。"
        if requirement not in task.visual_evaluation_requirements:
            task.visual_evaluation_requirements.append(requirement)
        return adjustments

    @staticmethod
    def _normalize_plan_for_task(task: TaskSpec, plan: RewardPlan) -> List[str]:
        """继承任务验收指标，并移除后腿任务中会诱导爬行的冲突奖励。"""
        adjustments: List[str] = []
        existing_metrics = {item.name for item in plan.success_metrics}
        for metric in task.success_metrics:
            if metric.required and metric.name not in existing_metrics:
                plan.success_metrics.append(metric.copy(deep=True))
                adjustments.append("补充任务必选验收指标：%s" % metric.name)
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
            conflicting = {"tracking_lin_vel", "orientation", "landing_stability"}
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
                if "stand" in token or "站" in stage.name:
                    stage.parameter_changes["command_scale"] = 0.0
                if "walk" in token or "行走" in stage.name:
                    value = stage.parameter_changes.get("lin_vel_x")
                    if value is None:
                        stage.parameter_changes["lin_vel_x"] = [0.25, 0.35]
                        adjustments.append("为前腿行走阶段补充非零前向命令")
                    elif isinstance(value, (int, float)) and abs(float(value)) <= 0.2:
                        stage.parameter_changes["lin_vel_x"] = (
                            [-0.35, -0.25] if float(value) < 0 else [0.25, 0.35])
                        adjustments.append("将前腿行走命令移出 Unitree 速度死区")
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
        return adjustments

    def inspect_environment(self, robot: str, output: Optional[Path] = None):
        """检查训练环境并保存能力清单。"""
        destination = output or self.settings.artifacts_path / "environment_manifest.json"
        return self.inspector.write(destination, robot)

    def _git_commit(self) -> str:
        """读取训练项目当前 Git 提交标识。"""
        result = subprocess.run(["git", "-C", str(self.settings.training_root), "rev-parse", "HEAD"],
                                text=True, capture_output=True, timeout=10, check=False)
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    @staticmethod
    def _clear_stale_run_outputs(task_dir: Path) -> None:
        """开始同一任务的新运行时移除会误导界面的旧终态摘要，保留候选和 checkpoint。"""
        for name in ("blocking_report.json", "summary.json", "report.md", "loop_status.json",
                     "loop_history.json", "loop_blocking_error.txt"):
            path = task_dir / name
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
        marker = "-seed-%d" % seed
        matches = [path for path in CheckpointManager.list_checkpoints(directory)
                   if marker in str(path.parent)]
        return matches[-1] if matches else None

    def _restore_selected_candidate(self, task_dir: Path, experiment_id: str) -> Dict[str, Any]:
        """从持久化候选目录恢复闭环所需的计划、manifest 和多种子 checkpoint。"""
        directory = self.store.candidate_dir(task_dir.name, experiment_id)
        if not directory.is_dir():
            raise FileNotFoundError("恢复候选不存在：%s" % experiment_id)
        plan = RewardPlan.parse_obj(read_json(directory / "reward_plan.json"))
        manifest = ExperimentManifest.parse_obj(read_json(directory / "manifest.json"))
        checkpoints: List[Path] = []
        seeds: List[int] = []
        for seed in self.settings.evaluation_seeds:
            checkpoint = self._checkpoint_for_seed(directory, seed)
            if checkpoint is not None:
                checkpoints.append(checkpoint)
                seeds.append(seed)
        if not checkpoints and manifest.checkpoint:
            checkpoint = directory / manifest.checkpoint
            if checkpoint.is_file():
                checkpoints = [checkpoint]
                seeds = [manifest.seed]
        if not checkpoints:
            raise FileNotFoundError("恢复候选没有可用 checkpoint：%s" % experiment_id)
        selected = {
            "id": experiment_id, "dir": directory, "plan": plan, "manifest": manifest,
            "metadata": read_json(directory / "compile_metadata.json"),
            "checkpoints": checkpoints, "checkpoint_seeds": seeds, "checkpoint": checkpoints[0],
        }
        if manifest.parent_experiment_id:
            parent_dir = self.store.candidate_dir(task_dir.name, manifest.parent_experiment_id)
            parent_checkpoints = [self._checkpoint_for_seed(parent_dir, seed) for seed in seeds]
            selected["parent_checkpoints"] = [path for path in parent_checkpoints if path is not None]
        return selected

    def resume(self, task_id: str, dry_run: bool = False) -> Dict[str, Any]:
        """从人工审核时保存的候选、预算和 rollout 恢复自动闭环，不重复初始训练。"""
        task_dir = self.store.task_dir(task_id)
        state = PersistentStateMachine(task_dir / "state.json")
        if state.record.state == AgentState.COMPLETED:
            return read_json(task_dir / "summary.json")
        summary_path = task_dir / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError("该任务尚未产生可恢复的训练候选，请重新下发任务")
        summary = read_json(summary_path)
        experiment_id = summary.get("selected_experiment")
        if not experiment_id:
            raise RuntimeError("人工审核发生在训练前，当前没有可恢复 checkpoint")
        task = TaskSpec.parse_obj(read_json(task_dir / "task_spec.json"))
        selected = self._restore_selected_candidate(task_dir, str(experiment_id))
        used_iterations = int(summary.get("used_iterations", 0))
        remaining_iterations = int(summary.get("remaining_iterations", 0))
        used_revisions = int(summary.get("used_revisions", 0))
        max_revisions = int(summary.get("max_revisions", task.training_budget.max_reward_revisions))
        budget = BudgetTracker(
            max_iterations=used_iterations + remaining_iterations,
            used_iterations=used_iterations,
            max_revisions=max_revisions,
            used_revisions=used_revisions,
        )
        loop_records = read_json(task_dir / "loop_history.json") \
            if (task_dir / "loop_history.json").is_file() else []
        loop_status = read_json(task_dir / "loop_status.json") \
            if (task_dir / "loop_status.json").is_file() else {}
        start_round = max(1, int(loop_status.get("round", len(loop_records) + 1)))
        state.transition(AgentState.ROLLOUT_COLLECTING, {
            "reason": "正在从已有 checkpoint 恢复自动闭环", "loop_round": start_round,
            "reward_version": selected["plan"].version,
        })
        return self._evaluate(
            task, task_dir, state, selected, dry_run, budget,
            start_round=start_round, existing_records=loop_records, reuse_first_rollout=True)

    def _run(self, instruction: str, robot: str, dry_run: bool, stop_after_plan: bool) -> Dict[str, Any]:
        """执行受限子流程并返回结构化结果。"""
        if robot not in self.settings.allowed_robots:
            raise ValueError("unsupported robot: %s" % robot)
        task_id = self._task_id(instruction, robot)
        run_id = "run-" + hashlib.sha1(utc_now().encode("utf-8")).hexdigest()[:8]
        task_dir = self.store.initialize_task(task_id)
        self._clear_stale_run_outputs(task_dir)
        state = PersistentStateMachine(task_dir / "state.json")
        atomic_write_text(task_dir / "task_request.txt", instruction.rstrip() + "\n")
        manifest = self.inspect_environment(robot, self.settings.artifacts_path / "environment_manifest.json")
        write_json(task_dir / "environment_manifest.json", manifest)
        state.transition(AgentState.ENVIRONMENT_INSPECTED)

        design_request = {"instruction": instruction, "robot": robot, "capabilities": manifest.dict()}
        write_json(task_dir / "design_request.json", design_request)
        design = self.provider.design_task_and_rewards(instruction, robot, manifest.dict())
        write_json(task_dir / "design_response.json", design)
        task = TaskSpec.parse_obj(design["task_spec"])
        task.task_id = task_id
        task.robot = robot
        task_adjustments = self._normalize_task_for_instruction(task)
        metric_mappings = normalize_task_metrics(task, manifest.evaluation_metrics)
        write_json(task_dir / "task_normalization.json", {
            "adjustments": task_adjustments, "metric_mappings": metric_mappings})
        unsupported, derivable = self.inspector.validate_task(task, manifest)
        write_json(task_dir / "task_spec.json", task)
        state.transition(AgentState.TASK_DESIGNED, {"unsupported": unsupported, "derivable": derivable})
        if unsupported:
            report = {"task_id": task_id, "state": AgentState.HUMAN_REVIEW.value,
                      "unsupported_requirements": unsupported,
                      "reason": "required physical quantities are neither available nor derivable"}
            write_json(task_dir / "blocking_report.json", report)
            state.transition(AgentState.HUMAN_REVIEW)
            return report

        plans = [RewardPlan.parse_obj(item) for item in design["reward_plans"]]
        if not plans:
            raise ValueError("provider returned no reward candidates")
        plans = plans[:self.settings.num_reward_candidates]
        normalization_records = []
        for plan in plans:
            plan.task_id = task_id
            adjustments = self._normalize_plan_for_task(task, plan)
            normalization_records.append({"version": plan.version, "adjustments": adjustments})
        write_json(task_dir / "plan_normalization.json", normalization_records)
        state.transition(AgentState.REWARD_CANDIDATES_CREATED, {"candidate_count": len(plans)})
        compiler = RewardCompiler(manifest, self.settings.max_abs_reward_weight)
        lineage = LineageGraph()
        candidate_data: List[Dict[str, Any]] = []
        git_commit = self._git_commit()
        for index, plan in enumerate(plans, start=1):
            # 同一句自然语言任务会得到稳定 task_id，但每次运行必须使用独立候选目录，
            # 否则旧 checkpoint 可能被误选为本次冒烟训练结果。
            experiment_id = "%s-candidate-%02d-v%02d" % (run_id, index, plan.version)
            directory = self.store.candidate_dir(task_id, experiment_id)
            directory.mkdir(parents=True, exist_ok=True)
            for name in ("metrics", "checkpoints", "rollouts", "prompts", "responses"):
                (directory / name).mkdir(exist_ok=True)
            try:
                self._validate_plan_metric_coverage(task, plan)
                metadata = compiler.compile(plan, directory)
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
                provider_status="mock" if dry_run else "opencli", training_result="compiled")
            write_json(directory / "manifest.json", exp_manifest)
            lineage.add(experiment_id, None, plan.version, metadata["config_hash"])
            candidate_data.append({"id": experiment_id, "dir": directory, "plan": plan,
                                   "manifest": exp_manifest, "metadata": metadata})
        if not candidate_data:
            state.transition(AgentState.FAILED, {"reason": "all reward plans failed validation"})
            raise RuntimeError("所有奖励计划均未通过安全校验；请检查各候选的 validation_error.txt")
        lineage.save(task_dir / "lineage.json")
        state.transition(AgentState.CONFIGS_COMPILED)
        state.transition(AgentState.VALIDATED)
        if stop_after_plan:
            summary = {"task_id": task_id, "state": AgentState.VALIDATED.value,
                       "run_id": run_id, "candidates": [item["id"] for item in candidate_data],
                       "dry_run": True}
            write_json(task_dir / "summary.json", summary)
            return summary

        # 模型可以提出更小的任务预算，但不能突破本地配置的资源上限。
        budget = BudgetTracker(
            max_iterations=min(task.training_budget.max_total_iterations, self.settings.max_total_iterations),
            max_revisions=min(task.training_budget.max_reward_revisions, self.settings.max_reward_revisions),
        )
        initial_screening_cost = len(candidate_data) * max(
            self.settings.smoke_iterations, self.settings.screening_iterations)
        if initial_screening_cost + len(self.settings.evaluation_seeds) > budget.max_iterations:
            report = {
                "task_id": task_id, "state": AgentState.HUMAN_REVIEW.value,
                "reason": "任务训练预算不足以完成候选筛选和每个评估种子的至少一次训练",
                "required_minimum_iterations": initial_screening_cost + len(self.settings.evaluation_seeds),
                "available_iterations": budget.max_iterations,
            }
            write_json(task_dir / "blocking_report.json", report)
            state.transition(AgentState.HUMAN_REVIEW, {"reason": report["reason"]})
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
            budget.consume_iterations(self.settings.smoke_iterations)
            metrics = {"success_rate": 0.75 + index * 0.05, "task_score": 0.8 - index * 0.02,
                       "ppo_stability": 0.9, "energy": 0.1 + index * 0.02,
                       "reward_hacking_score": 0.0, "hard_constraints_passed": 1.0}
            write_json(item["dir"] / "metrics" / "smoke.json", metrics)
            atomic_write_text(item["dir"] / "stdout.log", "dry-run smoke training completed\n")
            atomic_write_text(item["dir"] / "stderr.log", "")
        state.transition(AgentState.CANDIDATE_SCREENING)
        screening_increment = max(0, self.settings.screening_iterations - self.settings.smoke_iterations)
        for index, item in enumerate(candidates):
            budget.consume_iterations(screening_increment)
            screening = {"iteration": self.settings.screening_iterations,
                         "hard_constraints_passed": True, "task_score": 0.82 - index * 0.04,
                         "success_rate": 0.8 - index * 0.03, "ppo_stability": 0.92 - index * 0.02,
                         "energy": 0.12 + index * 0.02, "reward_hacking_score": 0.0}
            screening["composite_score"] = (0.4 * screening["task_score"] + 0.3 * screening["success_rate"] +
                                             0.2 * screening["ppo_stability"] - 0.05 * screening["energy"] -
                                             0.05 * screening["reward_hacking_score"])
            write_json(item["dir"] / "metrics" / "screening.json", screening)
            item["screening_score"] = screening["composite_score"]
        selected = self._select_safe_candidate(candidates)
        screening_path = selected["dir"] / "metrics" / "screening.json"
        selected_screening = read_json(screening_path)
        selected_screening["selected"] = True
        write_json(screening_path, selected_screening)
        state.transition(AgentState.FULL_TRAINING)
        checkpoints = []
        per_seed = min(
            self.settings.full_iterations,
            (budget.max_iterations - budget.used_iterations) // len(self.settings.evaluation_seeds),
        )
        if per_seed <= 0:
            raise RuntimeError("initial full-training iteration budget exhausted")
        for seed in self.settings.evaluation_seeds:
            budget.consume_iterations(per_seed)
            checkpoint = selected["dir"] / "checkpoints" / ("seed_%d" % seed) / ("model_%d.pt" % per_seed)
            atomic_write_text(checkpoint, "dry-run checkpoint placeholder; never deploy to hardware\n")
            checkpoints.append(checkpoint)
        selected["checkpoints"] = checkpoints
        selected["checkpoint_seeds"] = list(self.settings.evaluation_seeds)
        selected["checkpoint"] = checkpoints[0]
        selected["manifest"].training_result = "dry_run_completed"
        selected["manifest"].iteration = per_seed
        selected["manifest"].checkpoint = str(checkpoints[0].relative_to(selected["dir"]))
        write_json(selected["dir"] / "manifest.json", selected["manifest"])
        return selected

    def _real_train(self, task: TaskSpec, task_dir: Path, state: PersistentStateMachine,
                    candidates: List[Dict[str, Any]], budget: BudgetTracker) -> Dict[str, Any]:
        """调用实际 Unitree 入口完成候选筛选和多种子训练。"""
        state.transition(AgentState.SMOKE_TRAINING)
        successful = []
        for item in candidates:
            budget.consume_iterations(self.settings.smoke_iterations)
            item["manifest"].started_at = utc_now()
            result = self.controller.run_smoke_training(item["id"], task.robot, item["dir"] / "config.yaml",
                                                        self.settings.smoke_iterations, item["manifest"].seed,
                                                        item["dir"])
            item["manifest"].ended_at = utc_now()
            item["manifest"].training_result = "completed" if result.exit_code == 0 else "failed"
            item["manifest"].failure_reason = "timeout" if result.timed_out else (None if result.exit_code == 0 else "process_exit_%d" % result.exit_code)
            write_json(item["dir"] / "manifest.json", item["manifest"])
            if result.exit_code == 0:
                successful.append(item)
        if not successful:
            state.transition(AgentState.FAILED, {"reason": "all smoke trainings failed"})
            raise RuntimeError("all reward candidates failed smoke training; inspect candidate stderr.log files")
        state.transition(AgentState.CANDIDATE_SCREENING)
        screening_increment = max(0, self.settings.screening_iterations - self.settings.smoke_iterations)
        screened = []
        for item in successful:
            checkpoint = CheckpointManager.latest(item["dir"])
            if checkpoint is None:
                continue
            budget.consume_iterations(screening_increment)
            result = self.controller.continue_training(item["id"], task.robot, item["dir"] / "config.yaml",
                                                       screening_increment, item["manifest"].seed,
                                                       item["dir"], checkpoint)
            if result.exit_code != 0:
                continue
            latest = TensorBoardReader().latest(item["dir"])
            ppo = PPOCollector().collect(item["dir"])
            if task.task_name == "jump":
                task_score = latest.get("Episode/rew_raw_mean_jump_height", 0.0)
            else:
                task_score = latest.get("Episode/rew_raw_mean_tracking_lin_vel", 0.0)
            finite_metrics = bool(latest) and all(math.isfinite(float(value)) for value in latest.values())
            finite_ppo = all(math.isfinite(float(value)) for value in
                             (ppo.value_loss, ppo.surrogate_loss, ppo.learning_rate, ppo.mean_noise_std))
            safety = finite_metrics and finite_ppo
            stability = 1.0 / (1.0 + max(0.0, ppo.value_loss)) if finite_ppo else 0.0
            score = 0.65 * task_score + 0.35 * stability if safety else None
            screening = {"iteration": self.settings.screening_iterations,
                         "hard_constraints_passed": safety, "task_proxy": task_score,
                         "ppo_stability": stability, "composite_score": score,
                         "note": "必须通过数值健康门槛；最终完成仍需确定性 rollout 和视觉验收"}
            write_json(item["dir"] / "metrics" / "screening.json", screening)
            screened.append(item)
        if not screened:
            state.transition(AgentState.FAILED, {"reason": "all candidate screenings failed"})
            raise RuntimeError("all candidate screenings failed")
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
        per_seed = min(
            self.settings.full_iterations,
            (budget.max_iterations - budget.used_iterations) // len(self.settings.evaluation_seeds),
        )
        if per_seed <= 0:
            raise RuntimeError("initial full-training iteration budget exhausted")
        for seed in self.settings.evaluation_seeds:
            remaining = per_seed
            budget.consume_iterations(remaining)
            before = set(CheckpointManager.list_checkpoints(selected["dir"]))
            result = self.controller.run_full_training(selected["id"] + "-seed-%d" % seed, task.robot,
                                                       selected["dir"] / "config.yaml", remaining, seed,
                                                       selected["dir"])
            if result.exit_code != 0:
                state.transition(AgentState.FAILED, {"reason": "full training failed", "seed": seed})
                raise RuntimeError("full random-initialized training failed for seed %d" % seed)
            new_checkpoints = [path for path in CheckpointManager.list_checkpoints(selected["dir"]) if path not in before]
            checkpoint = new_checkpoints[-1] if new_checkpoints else CheckpointManager.latest(selected["dir"])
            if checkpoint is None:
                raise RuntimeError("full training produced no checkpoint for seed %d" % seed)
            validation_checkpoints.append(checkpoint)
        selected["checkpoints"] = validation_checkpoints
        selected["checkpoint_seeds"] = list(self.settings.evaluation_seeds)
        selected["checkpoint"] = validation_checkpoints[0]
        selected["manifest"].training_result = "completed"
        selected["manifest"].iteration = per_seed
        selected["manifest"].checkpoint = str(validation_checkpoints[0].relative_to(selected["dir"]))
        write_json(selected["dir"] / "manifest.json", selected["manifest"])
        return selected

    @staticmethod
    def _rollout_score(task: TaskSpec, metrics: Dict[str, float]) -> float:
        """按任务进度加分并对碰撞、跌倒和姿态偏差扣分，用于选择中位 rollout。"""
        if task.task_name == "jump":
            progress = metrics.get("jump_height", 0.0)
        else:
            progress = max(metrics.get("walking_speed_tracking", 0.0),
                           metrics.get("tracking_lin_vel", 0.0))
        return (progress - 0.1 * metrics.get("forbidden_collisions", 0.0) -
                2.0 * metrics.get("fall_rate", 0.0) - 0.1 * metrics.get("max_abs_roll", 0.0))

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
            values = [float(record["metrics"].get(name, float("nan"))) for record in records]
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

    def _cached_round_rollout(self, round_root: Path) -> Optional[tuple]:
        """读取已完整采集的轮次，供 Provider 故障恢复时跳过昂贵的重复仿真。"""
        metrics_path = round_root / "rollout_metrics.json"
        if not metrics_path.is_file():
            return None
        payload = read_json(metrics_path)
        records = list(payload.get("records", []))
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
            cached = self._cached_round_rollout(round_root)
            if cached is not None:
                return cached
        if dry_run:
            rollout_dir = round_root / "rollout_001"
            media = DryRunRolloutRecorder().record(
                rollout_dir, self.settings.video_fps, seed=selected["manifest"].seed,
                checkpoint=selected["checkpoint"].name)
            metrics = TrajectoryMetrics().compute(pd.read_parquet(media["trajectory"]))
            records = [{"rollout": "rollout_001", "seed": selected["manifest"].seed,
                        "metrics": metrics}]
            write_json(round_root / "rollout_metrics.json", {
                "records": records, "aggregate": self._aggregate_rollout_metrics(task, records)})
            return rollout_dir, media, records
        rollout_records = []
        checkpoints = selected.get("checkpoints", [selected["checkpoint"]])
        seeds = selected.get("checkpoint_seeds", self.settings.evaluation_seeds[:len(checkpoints)])
        rollout_index = 0
        for checkpoint, seed in zip(checkpoints, seeds):
            for _ in range(self.settings.rollouts_per_seed):
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
                })
        if not rollout_records:
            raise RuntimeError("evaluation produced no rollout")
        rollout_records.sort(key=lambda item: item["score"])
        representative = rollout_records[len(rollout_records) // 2]
        serializable = [{"rollout": item["dir"].name, "seed": item["seed"],
                         "score": item["score"], "metrics": item["metrics"]}
                        for item in rollout_records]
        write_json(round_root / "rollout_metrics.json", {
            "records": serializable,
            "aggregate": self._aggregate_rollout_metrics(task, rollout_records),
            "representative_rollout": representative["dir"].name,
        })
        return representative["dir"], representative["media"], rollout_records

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
        artifacts = VisualEvaluationPipeline().build(task, rollout_dir, media)
        state.transition(AgentState.VISUAL_EVALUATING, {"loop_round": round_index})
        try:
            visual = self.provider.critique_visual_behavior(task, artifacts.visual_files)
        except ProviderError as exc:
            error_detail = str(exc).strip()[-2000:] or exc.__class__.__name__
            atomic_write_text(rollout_dir / "visual_provider_error.txt", error_detail + "\n")
            return {"provider_error": error_detail, "provider_stage": "visual", "rollout_dir": rollout_dir,
                    "clean": artifacts.clean_sheet, "annotated": artifacts.annotated_sheet}
        write_json(rollout_dir / "visual_report.json", visual)
        atomic_write_text(rollout_dir / "visual_raw_response.txt",
                          visual.json(indent=2, ensure_ascii=False) + "\n")
        state.transition(AgentState.NUMERIC_EVALUATING, {"loop_round": round_index})
        physical = self._aggregate_rollout_metrics(task, rollout_records)
        if task.task_name == "jump":
            physical.update({"takeoff_time_spread": 0.0, "landing_pitch_abs": 0.03})
        physical.setdefault("tracking_error", float("nan"))
        physical.setdefault("fall_rate", float("nan"))
        physical.setdefault("joint_limit_violations", 0.0)
        physical.setdefault("torque_limit_violations", 0.0)
        physical.setdefault("forbidden_collisions", float("nan"))
        physical.setdefault("abnormal_terminations", float("nan"))
        write_json(rollout_dir / "numeric_summary.json", physical)
        evaluation = DeterministicEvaluator().evaluate(task, physical, visual)
        write_json(rollout_dir / "evaluation.json", evaluation)
        state.transition(AgentState.DIAGNOSING, {"loop_round": round_index})
        ppo = PPOCollector().collect(selected["dir"])
        payload = {
            "task": task.dict(), "reward_plan": selected["plan"].dict(),
            "capabilities": self._diagnosis_capabilities(task.robot),
            "visual": visual.dict(), "numeric": physical, "evaluation": evaluation.dict(),
            "reward_evidence": self._reward_evidence(media), "ppo": ppo.dict(),
            "loop_history": loop_records,
            "budget": {"used_iterations": budget.used_iterations,
                       "remaining_iterations": budget.max_iterations - budget.used_iterations,
                       "used_revisions": budget.used_revisions,
                       "remaining_revisions": budget.max_revisions - budget.used_revisions},
        }
        try:
            diagnosis = self.provider.diagnose_training(payload)
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

    def _compile_revision(self, task: TaskSpec, task_dir: Path, state: PersistentStateMachine,
                          selected: Dict[str, Any], diagnosis: TrainingDiagnosis,
                          round_index: int, dry_run: bool) -> Dict[str, Any]:
        """执行诊断修改、任务专用规范化和安全编译，生成新的候选版本目录。"""
        manifest = self.inspector.inspect(task.robot)
        if diagnosis.reward_changes or diagnosis.curriculum_changes:
            reviser = RewardPlanReviser(manifest.rewards, self.settings.max_abs_reward_weight)
            revised, audit = reviser.revise(selected["plan"], diagnosis)
        else:
            revised = selected["plan"].copy(deep=True)
            audit = ["奖励和课程保持不变，仅按诊断追加训练"]
        audit.extend(self._normalize_plan_for_task(task, revised))
        self._validate_plan_metric_coverage(task, revised)
        experiment_id = "%s-revision-%02d-v%02d" % (selected["id"], round_index, revised.version)
        directory = self.store.candidate_dir(task.task_id, experiment_id)
        directory.mkdir(parents=True, exist_ok=False)
        for name in ("metrics", "checkpoints", "rollouts", "prompts", "responses"):
            (directory / name).mkdir()
        metadata = RewardCompiler(manifest, self.settings.max_abs_reward_weight).compile(revised, directory)
        write_json(directory / "revision_audit.json", {
            "parent_experiment": selected["id"], "diagnosis": diagnosis.dict(), "adjustments": audit})
        command = self.controller.adapter.training_command(
            task.robot, directory / "config.yaml", self.settings.mid_iterations,
            selected["manifest"].seed, experiment_id)
        manifest = ExperimentManifest(
            experiment_id=experiment_id, parent_experiment_id=selected["id"], task_id=task.task_id,
            git_commit=self._git_commit(), config_hash=metadata["config_hash"],
            reward_version=revised.version, seed=selected["manifest"].seed, robot=task.robot,
            training_command=command, provider_status="mock" if dry_run else "opencli",
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
        remaining = budget.max_iterations - budget.used_iterations
        requested = self.settings.full_iterations if (
            diagnosis.decision == "restart" or diagnosis.checkpoint_strategy == "restart_from_scratch") \
            else self.settings.mid_iterations
        per_seed = min(requested, remaining // len(seeds))
        if per_seed <= 0:
            raise RuntimeError("training iteration budget exhausted")
        state.transition(AgentState.FULL_TRAINING,
                         {"loop_round": round_index, "revision_iterations_per_seed": per_seed})
        checkpoints = []
        used_parent_checkpoints = selected.get("parent_checkpoints", parent_checkpoints) \
            if diagnosis.checkpoint_strategy == "continue_from_parent" else parent_checkpoints
        for index, seed in enumerate(seeds):
            budget.consume_iterations(per_seed)
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
                    parent = used_parent_checkpoints[min(index, len(used_parent_checkpoints) - 1)]
                    result = self.controller.continue_training(
                        run_name, task.robot, revised["dir"] / "config.yaml", per_seed, seed,
                        revised["dir"], parent)
                if result.exit_code != 0:
                    raise RuntimeError("revision training failed for seed %d" % seed)
                new_checkpoints = [path for path in CheckpointManager.list_checkpoints(revised["dir"])
                                   if path not in before]
                checkpoint = new_checkpoints[-1] if new_checkpoints else None
                if checkpoint is None:
                    raise RuntimeError("revision training produced no checkpoint for seed %d" % seed)
            checkpoints.append(checkpoint)
        revised["checkpoints"] = checkpoints
        revised["checkpoint_seeds"] = list(seeds)
        revised["checkpoint"] = checkpoints[0]
        revised["parent_checkpoints"] = list(parent_checkpoints)
        revised["manifest"].training_result = "dry_run_completed" if dry_run else "completed"
        revised["manifest"].iteration = per_seed
        revised["manifest"].checkpoint = str(checkpoints[0].relative_to(revised["dir"]))
        write_json(revised["dir"] / "manifest.json", revised["manifest"])
        return revised

    def _finalize_loop(self, task: TaskSpec, task_dir: Path, state: PersistentStateMachine,
                       selected: Dict[str, Any], outcome: Dict[str, Any], budget: BudgetTracker,
                       loop_records: List[Dict[str, Any]], final_state: AgentState,
                       reason: str, dry_run: bool) -> Dict[str, Any]:
        """复制最终可追溯产物、保存闭环摘要并进入终态。"""
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
        completed = final_state == AgentState.COMPLETED
        summary = {
            "task_id": task.task_id, "state": final_state.value,
            "result": "completed" if completed else "failed" if final_state == AgentState.FAILED else "human_review",
            "reason": reason, "selected_experiment": selected["id"],
            "checkpoint": "final/checkpoint.pt", "config": "final/config.yaml",
            "rollout": str(outcome["rollout_dir"].relative_to(task_dir)) if outcome.get("rollout_dir") else None,
            "used_iterations": budget.used_iterations, "remaining_iterations": budget.max_iterations - budget.used_iterations,
            "used_revisions": budget.used_revisions, "max_revisions": budget.max_revisions,
            "loop_rounds": len(loop_records), "reward_version": selected["plan"].version,
            "last_decision": loop_records[-1].get("decision") if loop_records else None,
            "dry_run": dry_run,
        }
        write_json(task_dir / "loop_history.json", loop_records)
        ReportBuilder().build(task.task_id, summary, task_dir)
        return summary

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
            first_iteration = False
            if outcome.get("provider_error"):
                stage_name = "视觉评估" if outcome.get("provider_stage") == "visual" else "训练诊断"
                reason = "%s Provider 失败：%s" % (stage_name, outcome["provider_error"])
                return self._finalize_loop(task, task_dir, state, selected, outcome, budget,
                                           loop_records, AgentState.HUMAN_REVIEW, reason, dry_run)
            diagnosis = outcome["diagnosis"]
            evaluation = outcome["evaluation"]
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
            if diagnosis.decision == "failed":
                return self._finalize_loop(task, task_dir, state, selected, outcome, budget,
                                           loop_records, AgentState.FAILED, "诊断判定任务不可恢复", dry_run)
            if diagnosis.decision == "human_review":
                return self._finalize_loop(task, task_dir, state, selected, outcome, budget,
                                           loop_records, AgentState.HUMAN_REVIEW,
                                           "诊断发现需要用户决定的真实歧义或证据冲突", dry_run)
            if budget.used_revisions >= budget.max_revisions or budget.used_iterations >= budget.max_iterations:
                return self._finalize_loop(task, task_dir, state, selected, outcome, budget,
                                           loop_records, AgentState.HUMAN_REVIEW,
                                           "自动闭环预算耗尽，目标仍未通过验收", dry_run)
            try:
                budget.consume_revision()
                selected = self._train_revision(
                    task, task_dir, state, selected, diagnosis, round_index, dry_run, budget)
            except (RuntimeError, RewardValidationError, ValueError) as exc:
                atomic_write_text(task_dir / "loop_blocking_error.txt", str(exc) + "\n")
                return self._finalize_loop(task, task_dir, state, selected, outcome, budget,
                                           loop_records, AgentState.HUMAN_REVIEW,
                                           "闭环修订无法安全执行：%s" % exc, dry_run)
            round_index += 1
