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
from .environment.project_adapter import UnitreeProjectAdapter
from .orchestration.orchestrator import TrainingOrchestrator
from .providers.opencli_chatgpt import OpenCLIChatGPTWebProvider
from .providers.errors import ProviderError
from .schemas.task import TaskSpec
from .settings import load_settings
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
    checks["opencli"] = OpenCLIChatGPTWebProvider().doctor().dict()
    upload_failure = settings.artifacts_path / "opencli_test" / "upload_failure.txt"
    validated_response = settings.artifacts_path / "opencli_test" / "validated_response.json"
    if upload_failure.exists():
        checks["opencli"]["image_upload_supported"] = False
        checks["opencli"]["details"].append("latest real upload probe failed; rerun opencli-test after extension recovery")
    checks["opencli_real_text_probe_recorded"] = validated_response.exists()
    checks["healthy"] = all(checks[key] for key in
                            ("python_supported", "agent_root_writable", "training_project", "training_entry",
                             "evaluation_entry", "torch_installed", "isaacgym_installed", "experiment_root_writable"))
    checks["production_ready"] = (checks["healthy"] and checks["opencli"]["available"] and
                                  checks["opencli"]["image_upload_supported"])
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
    for name in ("plan", "train"):
        command = sub.add_parser(name)
        command.add_argument("--task", required=True)
        command.add_argument("--robot", default="go2")
        command.add_argument("--provider", choices=["opencli", "mock"], default="opencli")
        if name == "train":
            command.add_argument("--dry-run", action="store_true")
    resume = sub.add_parser("resume")
    resume.add_argument("--task-id", required=True)
    resume.add_argument("--provider", choices=["opencli", "mock"], default="opencli")
    resume.add_argument("--dry-run", action="store_true")
    for name in ("status", "report"):
        command = sub.add_parser(name)
        command.add_argument("--task-id", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--task-id", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--provider", choices=["opencli", "mock"], default="opencli")
    play = sub.add_parser("play")
    play.add_argument("--task-id", required=True)
    play.add_argument("--checkpoint", required=True)
    play.add_argument("--seed", default=1, type=int)
    play.add_argument("--num-envs", default=1, type=int)
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
    audit.add_argument("--provider", choices=["opencli", "mock"], default="opencli")
    visual = sub.add_parser("visual-test")
    visual.add_argument("--video", required=True)
    visual.add_argument("--task", required=True)
    visual.add_argument("--provider", choices=["opencli", "mock"], default="opencli")
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
            print("[Agent] 网页推理服务不可用：{}".format(exc), file=sys.stderr)
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
            print("[Agent] 网页推理服务不可用：{}".format(exc), file=sys.stderr)
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
