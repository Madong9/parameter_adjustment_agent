"""为训练环境和奖励注册表生成确定性的兼容指纹，不使用奖励配置哈希代替环境版本。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def compatibility_fingerprint(training_root: Path, manifest: Dict[str, Any]) -> Dict[str, str]:
    """绑定机器人资产、环境源码与奖励接口；缺失源码也进入指纹，不读取密钥。"""
    robot = str(manifest.get("robot", ""))
    files = []
    for directory in (training_root / "legged_gym/envs/base",
                      training_root / "legged_gym/envs" / robot):
        files.extend(sorted(directory.rglob("*.py")))
    asset_root = training_root / "resources/robots" / robot
    files.extend(sorted(asset_root.rglob("*.urdf")))
    files.extend(sorted(asset_root.rglob("*.xml")))
    source_hashes = {str(path.relative_to(training_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in files if path.is_file()}
    environment = {"robot": robot, "source_hashes": source_hashes,
                   "observations": manifest.get("observations", []),
                   "terminations": manifest.get("terminations", []),
                   "commands": manifest.get("command_space", [])}
    registry = {"rewards": manifest.get("rewards", []),
                "metrics": manifest.get("evaluation_metrics", [])}
    for name in ("action_rewards.py", "config_runtime.py"):
        path = Path(__file__).resolve().parents[1] / "training" / name
        registry[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
    return {key: hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                           separators=(",", ":")).encode("utf-8")).hexdigest()
            for key, value in (("environment_version", environment), ("reward_registry_version", registry))}
