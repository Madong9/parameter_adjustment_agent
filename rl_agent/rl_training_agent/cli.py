from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from pydantic import BaseModel

from .environment.inspector import EnvironmentInspector
from .benchmark import BenchmarkRunner
from .environment.project_adapter import UnitreeProjectAdapter
from .maintenance.experiment_cleanup import ExperimentCleaner
from .memory.store import LongTermMemoryStore
from .orchestration.orchestrator import TrainingOrchestrator
from .providers.bailian_glm import BailianGLMProvider
from .providers.opencli_chatgpt import OpenCLIChatGPTWebProvider
from .providers.errors import ProviderError
from .providers.registry import ProviderRegistry
from .rag.knowledge_base import TrainingKnowledgeBase
from .schemas.task import TaskSpec
from .settings import load_settings
from .storage.migrations import ArtifactMigrator
from .utils.io import atomic_write_text, read_json, write_json
from .utils.paths import ensure_within
from .visual.contact_sheet import ContactSheetBuilder
from .visual.frame_sampler import FrameSampler
from .visual.evaluation_pipeline import VisualEvaluationPipeline
from .visual.video_metadata import read_video_metadata


class IntegrationReply(BaseModel):
    image_visible: bool
    message: str


def _json_print(value: Any) -> None:
    """以可读的 UTF-8 JSON 格式输出结果。"""
    if hasattr(value, "dict"):
        value = value.dict()
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def doctor() -> Dict[str, Any]:
    """检查运行环境、外部依赖和服务健康状态。"""
    settings = load_settings()
    checks: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_supported": sys.version_info[:2] >= (3, 8),
        "agent_root_writable": os.access(str(settings.agent_root), os.W_OK),
        "training_project": settings.training_root.is_dir(),
        "training_entry": (settings.training_root / "legged_gym" / "scripts" / "train.py").is_file(),
        "evaluation_entry": (settings.training_root / "legged_gym" / "scripts" / "play.py").is_file(),
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "isaacgym_installed": importlib.util.find_spec("isaacgym") is not None,
        "cuda_available": False,
        "experiment_root_writable": False,
    }
    try:
        import torch
        checks["torch_version"] = torch.__version__
        checks["cuda_available"] = torch.cuda.is_available()
    except Exception as exc:
        checks["torch_error"] = str(exc)
    settings.experiments_path.mkdir(parents=True, exist_ok=True)
    checks["experiment_root_writable"] = os.access(str(settings.experiments_path), os.W_OK)
    # 生产网页推理采用 ChatGPT -> 豆包主备组合。ChatGPT 不可用但豆包
    # 健康时不应把整套系统误报为不可生产运行。
    checks["opencli"] = ProviderRegistry().create(
        "opencli-doubao", settings, settings.artifacts_path / "provider_doctor").doctor().dict()
    checks["bailian"] = BailianGLMProvider().health()
    upload_failure = settings.artifacts_path / "opencli_test" / "upload_failure.txt"
    validated_response = settings.artifacts_path / "opencli_test" / "validated_response.json"
    if upload_failure.exists():
        checks["opencli"]["image_upload_supported"] = False
        checks["opencli"]["details"].append("latest real upload probe failed; rerun opencli-test after extension recovery")
    checks["opencli_real_text_probe_recorded"] = validated_response.exists()
    checks["healthy"] = all(checks[key] for key in
                            ("python_supported", "agent_root_writable", "training_project", "training_entry",
                             "evaluation_entry", "torch_installed", "isaacgym_installed", "cuda_available",
                             "experiment_root_writable"))
    bailian_required = settings.provider == "multi-agent"
    checks["production_ready"] = (
        checks["healthy"] and checks["opencli"]["available"] and
        checks["opencli"]["image_upload_supported"] and
        (checks["bailian"]["available"] or not bailian_required)
    )
    return checks


def _provider(settings, name: str, record_dir: Optional[Path] = None):
    """根据显式名称创建网页或模拟推理 Provider。"""
    return TrainingOrchestrator.provider_for(settings, name, record_dir)


def build_parser() -> argparse.ArgumentParser:
    """构造 Agent 顶层命令行解析器。"""
    parser = argparse.ArgumentParser(prog="python -m rl_training_agent")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    inspect = sub.add_parser("inspect-env")
    inspect.add_argument("--robot", default="go2")
    sub.add_parser("rag-index")
    rag_query = sub.add_parser("rag-query")
    rag_query.add_argument("--query", required=True)
    rag_query.add_argument("--robot")
    rag_query.add_argument("--top-k", type=int)
    sub.add_parser("memory-stats")
    cleanup = sub.add_parser("cleanup-experiments")
    cleanup.add_argument("--task-id")
    cleanup.add_argument("--apply", action="store_true")
    cleanup.add_argument("--videos-only", action="store_true")
    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--suite", default="config/benchmarks.yaml")
    benchmark.add_argument("--case")
    provider_choices = ["multi-agent", "opencli-doubao", "opencli", "doubao", "mock"]
    visual_provider_choices = ["opencli-doubao", "opencli", "doubao", "mock"]
    benchmark.add_argument("--provider", choices=provider_choices, default="multi-agent")
    benchmark.add_argument("--dry-run", action="store_true")
    migrate = sub.add_parser("migrate-artifacts")
    migrate.add_argument("--apply", action="store_true")
    memory_query = sub.add_parser("memory-query")
    memory_query.add_argument("--query", required=True)
    memory_query.add_argument("--robot", default="go2")
    memory_query.add_argument("--top-k", type=int)
    for name in ("plan", "train"):
        command = sub.add_parser(name)
        command.add_argument("--task", required=True)
        command.add_argument("--robot", default="go2")
        command.add_argument(
            "--provider", choices=provider_choices, default="multi-agent")
        if name == "train":
            command.add_argument("--dry-run", action="store_true")
    resume = sub.add_parser("resume")
    resume.add_argument("--task-id", required=True)
    resume.add_argument(
        "--provider", choices=provider_choices, default="multi-agent")
    resume.add_argument("--dry-run", action="store_true")
    for name in ("status", "report"):
        command = sub.add_parser(name)
        command.add_argument("--task-id", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--task-id", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--provider", choices=visual_provider_choices, default="opencli-doubao")
    play = sub.add_parser("play")
    play.add_argument("--task-id", required=True)
    play.add_argument("--checkpoint", required=True)
    play.add_argument("--seed", default=1, type=int)
    play.add_argument("--num-envs", default=1, type=int)
    feasibility_view = sub.add_parser("feasibility-view")
    feasibility_view.add_argument("--task", required=True)
    feasibility_view.add_argument("--robot", default="go2")
    feasibility_view.add_argument("--seed", default=1, type=int)
    feasibility_view.add_argument("--max-seconds", default=5.0, type=float)
    sub.add_parser("opencli-test")
    ui = sub.add_parser("ui")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", default=8765, type=int)
    ui.add_argument("--no-browser", action="store_true")
    sub.add_parser("desktop")
    sub.add_parser("desktop-tk")
    audit = sub.add_parser("visual-audit")
    audit.add_argument("--task-id", required=True)
    audit.add_argument("--rollout", default="final/evaluation_rollout")
    audit.add_argument("--provider", choices=visual_provider_choices, default="opencli-doubao")
    visual = sub.add_parser("visual-test")
    visual.add_argument("--video", required=True)
    visual.add_argument("--task", required=True)
    visual.add_argument("--provider", choices=visual_provider_choices, default="opencli-doubao")
    return parser


def main(argv=None) -> int:
    """解析命令行参数并执行对应的 Agent 工作流。"""
    args = build_parser().parse_args(argv)
    settings = load_settings()
    if args.command == "ui":
        from .web_ui import serve
        return serve(args.host, args.port, not args.no_browser)
    if args.command == "desktop":
        from .desktop_launcher import serve_desktop
        return serve_desktop()
    if args.command == "desktop-tk":
        from .desktop_ui import serve_tk_desktop
        return serve_tk_desktop()
    if args.command == "doctor":
        result = doctor()
        _json_print(result)
        return 0 if result["production_ready"] else 1
    if args.command == "inspect-env":
        path = settings.artifacts_path / "environment_manifest.json"
        result = EnvironmentInspector(settings.training_root).write(path, args.robot)
        _json_print({"output": "artifacts/environment_manifest.json", "variables": len(result.reward_variables),
                     "rewards": len(result.rewards), "robots": result.robots})
        return 0
    if args.command in ("rag-index", "rag-query"):
        knowledge = TrainingKnowledgeBase.from_settings(settings)
        stats = knowledge.refresh(force=args.command == "rag-index")
        if args.command == "rag-index":
            _json_print(stats)
            return 0
        result = knowledge.retrieve(
            args.query, "manual_query", top_k=args.top_k or settings.rag_top_k,
            robot=args.robot)
        _json_print({"index": stats, "result": result.prompt_payload()})
        return 0
    if args.command in ("memory-stats", "memory-query"):
        memory = LongTermMemoryStore(
            settings.memory_path, settings.agent_root, settings.memory_max_context_chars)
        if args.command == "memory-stats":
            _json_print(memory.stats())
            return 0
        _json_print(memory.retrieve(
            args.query, args.robot, top_k=args.top_k or settings.memory_top_k))
        return 0
    if args.command == "cleanup-experiments":
        result = ExperimentCleaner(settings.experiments_path).run(
            task_id=args.task_id,
            keep_per_run=settings.checkpoints_per_run,
            apply=args.apply,
            cleanup_checkpoints=not args.videos_only,
            cleanup_videos=True,
        )
        _json_print(result)
        return 0
    if args.command == "benchmark":
        if args.dry_run and args.provider != "mock":
            raise ValueError("benchmark --dry-run must use --provider mock")
        suite = Path(args.suite)
        if not suite.is_absolute():
            suite = settings.agent_root / suite
        result = BenchmarkRunner(settings, suite).run(
            args.provider, dry_run=args.dry_run, case_id=args.case)
        _json_print(result)
        return 0
    if args.command == "migrate-artifacts":
        result = ArtifactMigrator(
            settings.experiments_path, settings.memory_path).migrate(dry_run=not args.apply)
        _json_print(result)
        return 0
    if args.command in ("plan", "train"):
        planned_task_id = TrainingOrchestrator._task_id(args.task, args.robot)
        provider = _provider(settings, args.provider,
                             settings.experiments_path / planned_task_id / "provider_records")
        try:
            orchestrator = TrainingOrchestrator(settings, provider)
            if args.command == "plan":
                result = orchestrator.plan(args.task, args.robot)
            else:
                if args.dry_run and args.provider != "mock":
                    raise ValueError("--dry-run must use --provider mock to avoid accidental webpage or GPU work")
                result = orchestrator.train(args.task, args.robot, args.dry_run)
            _json_print(result)
            return 0
        except ProviderError as exc:
            print("[Agent] 推理服务不可用：{}".format(exc), file=sys.stderr)
            return 2
        finally:
            provider.close()
    if args.command in ("status", "report"):
        task_dir = settings.experiments_path / args.task_id
        ensure_within(task_dir, settings.experiments_path)
        state = read_json(task_dir / "state.json")
        summary = read_json(task_dir / "summary.json") if (task_dir / "summary.json").exists() else {}
        _json_print({"state": state, "summary": summary,
                     "report": str((task_dir / "report.md").relative_to(settings.agent_root)) if (task_dir / "report.md").exists() else None})
        return 0
    if args.command == "resume":
        task_dir = ensure_within(settings.experiments_path / args.task_id, settings.experiments_path)
        state = read_json(task_dir / "state.json")
        if state["state"] == "COMPLETED":
            _json_print(read_json(task_dir / "summary.json"))
            return 0
        provider = _provider(settings, args.provider, task_dir / "provider_records")
        try:
            result = TrainingOrchestrator(settings, provider).resume(args.task_id, args.dry_run)
            _json_print(result)
            return 0
        except ProviderError as exc:
            print("[Agent] 推理服务不可用：{}".format(exc), file=sys.stderr)
            return 2
        finally:
            provider.close()
    if args.command == "evaluate":
        task_dir = ensure_within(settings.experiments_path / args.task_id, settings.experiments_path)
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = settings.agent_root / checkpoint
        checkpoint = ensure_within(checkpoint, task_dir)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        task = read_json(task_dir / "task_spec.json")
        config_candidates = [parent / "config.yaml" for parent in checkpoint.parents if (parent / "config.yaml").is_file()]
        if not config_candidates:
            config_candidates = list((task_dir / "candidates").glob("*/config.yaml"))
        if not config_candidates:
            raise FileNotFoundError("no compiled config.yaml found for checkpoint")
        rollout_dir = task_dir / "final" / "evaluation_rollout"
        adapter = TrainingOrchestrator(settings, _provider(settings, args.provider)).controller
        result = adapter.run_evaluation_rollouts("manual-evaluation", task["robot"], config_candidates[0], checkpoint,
                                                 rollout_dir, seed=1, fps=settings.video_fps)
        _json_print({"task_id": args.task_id, "checkpoint": str(checkpoint.relative_to(settings.agent_root)),
                     "exit_code": result.exit_code,
                     "rollout": str(rollout_dir.relative_to(settings.agent_root))})
        return 0 if result.exit_code == 0 else 1
    if args.command == "play":
        task_dir = ensure_within(settings.experiments_path / args.task_id, settings.experiments_path)
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = settings.agent_root / checkpoint
        checkpoint = ensure_within(checkpoint, task_dir)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        task = read_json(task_dir / "task_spec.json")
        config_candidates = [parent / "config.yaml" for parent in checkpoint.parents
                             if (parent / "config.yaml").is_file()]
        if not config_candidates:
            raise FileNotFoundError("checkpoint 所属候选目录中没有 config.yaml")
        adapter = UnitreeProjectAdapter(settings.training_root, settings.agent_root,
                                        settings.experiments_path)
        command = adapter.play_command(task["robot"], config_candidates[0], checkpoint,
                                       seed=args.seed, num_envs=args.num_envs)
        return subprocess.call(command, cwd=str(settings.training_root), shell=False)
    if args.command == "feasibility-view":
        # 该入口绕过 LLM，只把用户文字编译为受限运动原型，并使用正式物理链路播放。
        from .feasibility.ik.pinocchio_solver import PinocchioIKSolver
        from .feasibility.motion_prototype.dynamic_generator import (
            DynamicMotionPrototypeGenerator,
        )
        from .feasibility.motion_prototype.schema import MotionType
        from .feasibility.robot_models.loader import RobotModelLoader
        from .feasibility.simulation.isaacgym_dynamic_validator import (
            IsaacGymDynamicValidator,
        )
        from .schemas.agent_workflow import TaskIntentSpec

        if not 0.1 <= float(args.max_seconds) <= 30.0:
            raise ValueError("--max-seconds 必须位于 0.1 到 30 秒之间")
        intent = TaskIntentSpec(
            original_instruction=args.task,
            robot=args.robot,
            action_name=args.task,
            normalized_goal=args.task,
        )
        prototype = DynamicMotionPrototypeGenerator.generate(intent)
        if prototype.motion_type != MotionType.LOCOMOTION:
            raise ValueError(
                "当前 Viewer 入口只支持前进/后退 LOCOMOTION；该任务被识别为 %s" %
                prototype.motion_type.value)
        if prototype.duration > float(args.max_seconds) + 1.0e-9:
            raise ValueError(
                "动作原型时长 %.2f 秒超过 --max-seconds %.2f 秒；请提高显式上限" %
                (prototype.duration, float(args.max_seconds)))
        model = RobotModelLoader(settings.training_root).load_robot_model(args.robot)
        if model.model_status != "AVAILABLE":
            raise RuntimeError(model.model_error or "机器人物理模型不可用")
        _json_print({
            "阶段": "运动原型已生成，即将打开 Isaac Gym Viewer",
            "提示": "这是训练前开环物理预检，不是已训练策略；按 Esc 或关闭窗口可退出。",
            "motion_prototype": prototype,
        })
        max_steps = max(5, int(float(args.max_seconds) / 0.01) + 1)
        report = IsaacGymDynamicValidator(
            settings.training_root, seed=args.seed,
            max_seconds=float(args.max_seconds), max_steps=max_steps,
            ik_solver=PinocchioIKSolver(), visualize=True,
        ).validate(model, prototype)
        _json_print({"阶段": "可视化预检结束", "report": report})
        return 0 if report.success is True else 1
    if args.command == "opencli-test":
        output = settings.artifacts_path / "opencli_test"
        output.mkdir(parents=True, exist_ok=True)
        image_path = output / "upload_test.png"
        image = np.full((160, 240, 3), 255, dtype=np.uint8)
        cv2.circle(image, (120, 80), 35, (0, 0, 255), -1)
        cv2.imwrite(str(image_path), image)
        provider = OpenCLIChatGPTWebProvider(record_dir=output / "records")
        try:
            health = provider.doctor()
            if not health.available:
                _json_print({"skipped": True, "health": health})
                return 0
            provider.open_or_bind()
            conversation = provider.new_conversation("rl-agent-image-upload-test")
            upload_error = None
            try:
                raw = provider.send_with_files(
                    'Inspect the attached generated test image. Return only JSON: {"image_visible": true/false, "message": "short description"}.',
                    [image_path], conversation)
                failure_path = output / "upload_failure.txt"
                if failure_path.exists():
                    failure_path.unlink()
            except ProviderError as exc:
                upload_error = str(exc)
                (output / "upload_failure.txt").write_text(upload_error + "\n", encoding="utf-8")
                raw = provider.send_text(
                    'OpenCLI image upload was unavailable. Return only JSON: {"image_visible": false, "message": "text channel verified"}.',
                    conversation)
            parsed = provider.parse_json_response(raw, IntegrationReply)
            write_json(output / "validated_response.json", parsed)
            _json_print({"skipped": False, "text_validated": True,
                         "image_upload_validated": upload_error is None,
                         "upload_error": upload_error, "validated": parsed})
            return 0
        finally:
            provider.close()
    if args.command == "visual-audit":
        task_dir = ensure_within(settings.experiments_path / args.task_id, settings.experiments_path)
        rollout_dir = ensure_within(task_dir / args.rollout, task_dir)
        task = TaskSpec.parse_obj(read_json(task_dir / "task_spec.json"))
        artifacts = VisualEvaluationPipeline().build(task, rollout_dir)
        provider = _provider(settings, args.provider, rollout_dir / "visual_provider_records")
        try:
            report = provider.critique_visual_behavior(task, artifacts.visual_files)
            report_path = rollout_dir / "visual_report_enhanced.json"
            write_json(report_path, report)
            atomic_write_text(rollout_dir / "visual_raw_response_enhanced.txt",
                              report.json(indent=2, ensure_ascii=False) + "\n")
            _json_print({
                "task_id": args.task_id,
                "rollout": str(rollout_dir.relative_to(settings.agent_root)),
                "visual_success": report.visual_success,
                "confidence": report.confidence,
                "evidence_findings": [item.dict() for item in report.evidence_findings],
                "uncertain_items": report.uncertain_items,
                "report": str(report_path.relative_to(settings.agent_root)),
            })
            return 0
        except ProviderError as exc:
            print("[Agent] 增强视觉评估失败：{}".format(exc), file=sys.stderr)
            return 2
        finally:
            provider.close()
    if args.command == "visual-test":
        video = Path(args.video).resolve()
        metadata = read_video_metadata(video)
        frames = FrameSampler.decode(video, FrameSampler.uniform_indices(metadata.frame_count, 12))
        output = settings.artifacts_path / "visual_test"
        sheet = ContactSheetBuilder().build(frames, output / "contact_sheet_clean.png")
        _json_print({"task": args.task, "metadata": metadata, "contact_sheet": str(sheet.relative_to(settings.agent_root))})
        return 0
    return 2
