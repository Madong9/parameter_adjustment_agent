"""规划或执行不破坏可恢复性与验收证据的实验产物清理。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..training.checkpoint_manager import CheckpointManager
from ..utils.io import read_json
from ..utils.paths import ensure_within


class ExperimentCleaner:
    """删除冗余视频和中间 checkpoint，同时保留最终策略与关键证据。"""

    def __init__(self, experiment_root: Path):
        """保存并规范化实验根目录。"""
        self.experiment_root = experiment_root.resolve()

    def _scope(self, task_id: Optional[str]) -> Path:
        """返回全部实验或单任务清理范围，并阻止路径逃逸。"""
        if not task_id:
            return self.experiment_root
        if not task_id.startswith("task-") or "/" in task_id or "\\" in task_id:
            raise ValueError("task_id 格式非法")
        return ensure_within(self.experiment_root / task_id, self.experiment_root)

    def cleanup_completed_videos(self, task_id: str) -> Dict[str, Any]:
        """联合验收完成且经验整理落盘后，删除该任务的视频，保留其余证据。"""
        scope = self._scope(task_id)
        if (self.experiment_root / task_id).is_symlink():
            raise ValueError("不能清理符号链接任务目录")
        summary = read_json(scope / "summary.json")
        state = read_json(scope / "state.json")
        if summary.get("state") != "COMPLETED" or state.get("state") != "COMPLETED":
            return {"status": "skipped", "reason": "任务尚未成功完成"}
        extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}
        removed = []
        errors = []
        reclaimed = 0
        for path in sorted(scope.rglob("*")):
            if path.suffix.lower() not in extensions or path.is_symlink() or not path.is_file():
                continue
            try:
                ensure_within(path, scope)
                size = path.stat().st_size
                path.unlink()
                reclaimed += size
                removed.append(path.relative_to(scope).as_posix())
            except (OSError, ValueError) as exc:
                errors.append({"path": path.relative_to(scope).as_posix(), "error": str(exc)})
        return {"status": "partial" if errors else "completed", "video_files": len(removed),
                "reclaim_bytes": reclaimed, "removed": removed, "errors": errors}

    @staticmethod
    def _selected_rollouts(metrics_path: Path) -> Set[str]:
        """从数值排序中还原最差、中央和最好三个视觉样本。"""
        try:
            payload = read_json(metrics_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return set()
        records = sorted(
            payload.get("records", []),
            key=lambda item: (float(item.get("score", 0.0)), str(item.get("seed", ""))),
        )
        if not records:
            return set()
        indices = {0, len(records) // 2, len(records) - 1}
        return {str(records[index].get("rollout")) for index in indices}

    @staticmethod
    def _protected_checkpoints(scope: Path) -> Set[Path]:
        """收集 manifest 明确引用的恢复 checkpoint。"""
        protected: Set[Path] = set()
        for manifest_path in scope.rglob("manifest.json"):
            try:
                checkpoint = read_json(manifest_path).get("checkpoint")
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if checkpoint:
                path = (manifest_path.parent / str(checkpoint)).resolve()
                if path.is_file():
                    protected.add(path)
        return protected

    def _checkpoint_candidates(self, scope: Path, keep_per_run: int) -> List[Path]:
        """找出各训练运行中除末次和显式恢复点外的 checkpoint。"""
        if keep_per_run <= 0:
            raise ValueError("keep_per_run must be positive")
        protected = self._protected_checkpoints(scope)
        candidates: List[Path] = []
        parents = {path.parent for path in scope.rglob("model_*.pt")}
        for parent in parents:
            paths = [path for path in CheckpointManager.list_checkpoints(parent)
                     if path.parent == parent]
            retained = {path.resolve() for path in paths[-keep_per_run:]}
            retained.update(protected)
            candidates.extend(path for path in paths if path.resolve() not in retained)
        return candidates

    def _video_candidates(self, scope: Path) -> List[Path]:
        """找出未入选视觉聚合的普通视频和仅用于数值反事实测试的视频。"""
        candidates: List[Path] = []
        for metrics_path in scope.glob("**/rollouts/round_*/rollout_metrics.json"):
            keep = self._selected_rollouts(metrics_path)
            if not keep:
                continue
            for rollout_dir in metrics_path.parent.glob("rollout_*"):
                if rollout_dir.is_dir() and rollout_dir.name not in keep:
                    candidates.extend(rollout_dir.glob("*.mp4"))
            for counterfactual_dir in metrics_path.parent.glob("counterfactual_*"):
                if counterfactual_dir.is_dir():
                    candidates.extend(counterfactual_dir.glob("*.mp4"))
        return candidates

    @staticmethod
    def _task_directories(scope: Path) -> List[Path]:
        """兼容全部实验根目录和单个 task 目录两种清理范围。"""
        if scope.name.startswith("task-"):
            return [scope]
        return [path for path in scope.glob("task-*") if path.is_dir()]

    def _obsolete_candidate_videos(self, scope: Path) -> List[Path]:
        """找出终态任务中当前选中候选之外的旧失败或淘汰候选视频。"""
        candidates: List[Path] = []
        for task_dir in self._task_directories(scope):
            summary_path = task_dir / "summary.json"
            if not summary_path.is_file():
                continue
            try:
                summary = read_json(summary_path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if summary.get("state") not in ("COMPLETED", "HUMAN_REVIEW", "FAILED"):
                continue
            selected = str(summary.get("selected_experiment") or "")
            if not selected:
                continue
            candidate_root = task_dir / "candidates"
            for candidate_dir in candidate_root.iterdir() if candidate_root.is_dir() else ():
                if candidate_dir.is_dir() and candidate_dir.name != selected:
                    candidates.extend(candidate_dir.rglob("*.mp4"))
        return candidates

    def run(self, task_id: Optional[str] = None, keep_per_run: int = 1,
            apply: bool = False, cleanup_checkpoints: bool = True,
            cleanup_videos: bool = True) -> Dict[str, Any]:
        """返回清理预览；仅在 apply 为真时删除已验证位于实验目录内的文件。"""
        if not cleanup_checkpoints and not cleanup_videos:
            raise ValueError("至少选择一种清理类型")
        scope = self._scope(task_id)
        if not scope.exists():
            raise FileNotFoundError(scope)
        checkpoints = self._checkpoint_candidates(scope, keep_per_run) \
            if cleanup_checkpoints else []
        videos = (self._video_candidates(scope) + self._obsolete_candidate_videos(scope)) \
            if cleanup_videos else []
        paths = sorted({path.resolve() for path in checkpoints + videos})
        for path in paths:
            ensure_within(path, self.experiment_root)
        bytes_to_reclaim = sum(path.stat().st_size for path in paths if path.is_file())
        if apply:
            for path in paths:
                if path.is_file():
                    path.unlink()
        return {
            "mode": "applied" if apply else "preview",
            "scope": task_id or "all",
            "cleanup_checkpoints": cleanup_checkpoints,
            "cleanup_videos": cleanup_videos,
            "checkpoint_files": len({path.resolve() for path in checkpoints}),
            "video_files": len({path.resolve() for path in videos}),
            "total_files": len(paths),
            "reclaim_bytes": bytes_to_reclaim,
            "reclaim_gib": round(bytes_to_reclaim / float(1024 ** 3), 3),
            "preserved": [
                "final/checkpoint.pt 与 manifest 引用的恢复点",
                "每个训练运行的最新 checkpoint",
                "最差、中央和最好视觉 rollout",
                "全部 rollout 数值汇总和 parquet 轨迹",
            ],
        }
