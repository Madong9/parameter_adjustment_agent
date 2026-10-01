"""实现固定环境、动作、种子和阈值的可复现真实训练基准协议。"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, validator

from .orchestration.orchestrator import TrainingOrchestrator
from .settings import Settings
from .utils.io import read_json, utc_now, write_json


class BenchmarkCase(BaseModel):
    """定义一项固定自然语言动作及其必须存在的验收指标。"""

    id: str
    instruction: str
    required_metrics: List[str] = Field(default_factory=list)
    thresholds: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class BenchmarkSuiteSpec(BaseModel):
    """定义可版本化的机器人动作基准套件。"""

    version: int
    suite: str
    robot: str
    environment_commit: str = "current"
    seeds: List[int]
    repetitions: int = 2
    cases: List[BenchmarkCase]

    @validator("seeds", "cases")
    def nonempty_lists(cls, value: List[Any]) -> List[Any]:
        """确保基准至少包含一个种子和一个动作。"""
        if not value:
            raise ValueError("benchmark lists must not be empty")
        return value


class BenchmarkRunner:
    """执行固定基准并将每次运行结果复制为不可变的汇总证据。"""

    def __init__(self, settings: Settings, suite_path: Path):
        """加载套件并保存运行配置。"""
        self.settings = settings
        self.suite_path = suite_path
        self.suite = BenchmarkSuiteSpec.parse_obj(
            yaml.safe_load(suite_path.read_text(encoding="utf-8")))

    def _run_id(self, dry_run: bool) -> str:
        """根据套件、时间和模式生成本次基准标识。"""
        value = "%s|%s|%s" % (self.suite.suite, utc_now(), dry_run)
        return "benchmark-" + hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]

    def _git_commit(self) -> str:
        """读取训练工程提交并验证套件指定的环境版本。"""
        result = subprocess.run(
            ["git", "-C", str(self.settings.training_root), "rev-parse", "HEAD"],
            text=True, capture_output=True, timeout=10, check=False)
        commit = result.stdout.strip() if result.returncode == 0 else "unknown"
        expected = self.suite.environment_commit
        if expected != "current" and not commit.startswith(expected):
            raise RuntimeError("benchmark environment commit mismatch: expected %s, got %s" % (
                expected, commit))
        return commit

    @staticmethod
    def _metric_names(task_dir: Path, rollout: Optional[str]) -> List[str]:
        """从最终联合评估中提取实际出现的指标名称。"""
        if not rollout:
            return []
        path = task_dir / rollout / "evaluation.json"
        if not path.is_file():
            return []
        return [str(item.get("name")) for item in read_json(path).get("metrics", [])]

    @staticmethod
    def _thresholds_match(task_dir: Path, expected: Dict[str, Dict[str, Any]]) -> bool:
        """确认模型生成的任务阈值与基准套件固定阈值完全一致。"""
        if not expected:
            return True
        task = read_json(task_dir / "task_spec.json")
        actual = {item["name"]: {"operator": item["operator"], "value": item["value"]}
                  for item in task.get("success_metrics", []) + task.get("safety_constraints", [])}
        return all(actual.get(name) == value for name, value in expected.items())

    def run(self, provider_name: str, dry_run: bool = False,
            case_id: Optional[str] = None) -> Dict[str, Any]:
        """顺序执行动作与重复次数；真实模式不会自动回退到模拟 Provider。"""
        run_id = self._run_id(dry_run)
        output_dir = self.settings.artifacts_path / "benchmarks" / run_id
        environment_commit = self._git_commit()
        settings = self.settings.copy(deep=True)
        settings.evaluation_seeds = list(self.suite.seeds)
        records: List[Dict[str, Any]] = []
        cases = [item for item in self.suite.cases if not case_id or item.id == case_id]
        if not cases:
            raise ValueError("benchmark case not found: %s" % case_id)
        for case in cases:
            for repetition in range(1, self.suite.repetitions + 1):
                task_id = TrainingOrchestrator._task_id(case.instruction, self.suite.robot)
                provider = TrainingOrchestrator.provider_for(
                    settings, provider_name,
                    settings.experiments_path / task_id / "provider_records")
                try:
                    result = TrainingOrchestrator(settings, provider).train(
                        case.instruction, self.suite.robot, dry_run=dry_run)
                finally:
                    provider.close()
                task_dir = settings.experiments_path / result["task_id"]
                metrics = self._metric_names(task_dir, result.get("rollout"))
                records.append({
                    "case_id": case.id, "repetition": repetition,
                    "task_id": result["task_id"], "state": result.get("state"),
                    "reward_version": result.get("reward_version"),
                    "loop_rounds": result.get("loop_rounds"),
                    "required_metrics": case.required_metrics,
                    "observed_metrics": metrics,
                    "metric_coverage": all(item in metrics for item in case.required_metrics),
                    "thresholds_match": self._thresholds_match(task_dir, case.thresholds),
                    "summary": result,
                })
                write_json(output_dir / "records" / ("%s-%02d.json" % (case.id, repetition)), records[-1])
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            grouped.setdefault(record["case_id"], []).append(record)
        reproducibility = {
            name: {
                "runs": len(items),
                "completed": sum(item["state"] == "COMPLETED" for item in items),
                "consistent_terminal_state": len({item["state"] for item in items}) == 1,
                "metric_coverage": all(item["metric_coverage"] for item in items),
                "thresholds_match": all(item["thresholds_match"] for item in items),
            }
            for name, items in grouped.items()
        }
        report = {
            "format_version": 1, "run_id": run_id, "suite": self.suite.dict(),
            "dry_run": dry_run, "provider": provider_name, "created_at": utc_now(),
            "environment_git_commit": environment_commit,
            "records": records, "reproducibility": reproducibility,
            "real_evidence": not dry_run,
        }
        write_json(output_dir / "report.json", report)
        return report
