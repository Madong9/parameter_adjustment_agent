from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any, Dict

from ..environment.capability_manifest import CapabilityManifest
from ..schemas.rewards import RewardPlan
from ..utils.io import atomic_write_text, sha256_text, utc_now, write_json
from .validator import RewardPlanValidator


class RewardCompiler:
    def __init__(self, manifest: CapabilityManifest, max_abs_weight: float = 100.0):
        """初始化 RewardCompiler 实例及其运行依赖。"""
        self.manifest = manifest
        self.validator = RewardPlanValidator(manifest.rewards, max_abs_weight)

    def compile(self, plan: RewardPlan, output_dir: Path) -> Dict[str, Any]:
        """校验奖励计划并生成独立配置、差异和哈希。"""
        self.validator.validate(plan)
        output_dir.mkdir(parents=True, exist_ok=True)
        defaults = {item.name: item.default_weight for item in self.manifest.rewards}
        scales = {term.name: term.weight for term in plan.terms}
        compiled = {
            "schema_version": 1, "generated_at": utc_now(), "task_id": plan.task_id,
            "reward_version": plan.version, "parent_version": plan.parent_version,
            "robot": self.manifest.robot,
            "rewards": {
                "scales": scales,
                # 保留阶段和参数元数据，真实训练包装器据此动态启停奖励并应用目标参数。
                "terms": [item.dict() for item in plan.terms],
            },
            "curriculum": [item.dict() for item in plan.curriculum],
            "terminations": [item.dict() for item in plan.terminations],
            "success_metrics": [item.dict() for item in plan.success_metrics],
        }
        canonical = json.dumps(compiled, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        config_path = output_dir / "config.yaml"
        atomic_write_text(config_path, canonical)
        before = json.dumps({"rewards": {"scales": defaults}}, indent=2, sort_keys=True).splitlines(True)
        after = canonical.splitlines(True)
        diff = "".join(difflib.unified_diff(before, after, fromfile="original-registry", tofile="config.yaml"))
        atomic_write_text(output_dir / "config.diff", diff)
        write_json(output_dir / "reward_plan.json", plan)
        metadata = {"config_hash": sha256_text(canonical), "config_path": "config.yaml",
                    "diff_path": "config.diff", "effective_terms": sorted(scales)}
        write_json(output_dir / "compile_metadata.json", metadata)
        return metadata
