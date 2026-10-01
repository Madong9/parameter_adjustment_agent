"""隔离 Unitree Isaac Gym 动态预检，避免原生扩展故障杀死上位机训练进程。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

from .ik.pinocchio_solver import PinocchioIKSolver
from .robot_models.schema import RobotModel
from .simulation.isaacgym_dynamic_validator import IsaacGymDynamicValidator
from .simulation.schema import SimulationReport
from .motion_prototype.schema import DynamicMotionPrototype


REPORT_MARKER = "__DYNAMIC_FEASIBILITY_REPORT__"


def _build_request_report(request: Dict[str, Any]) -> SimulationReport:
    """从隔离 worker 的 JSON 请求重建模型/轨迹并执行一次真实短时验证。"""
    if request.get("mode") == "environment_health":
        from .admission import _check_environment_health_in_process
        return SimulationReport.parse_obj(_check_environment_health_in_process(
            Path(str(request["training_root"])), str(request["robot"])))
    raw_model = dict(request["robot_model"])
    descriptor_fields = {
        name: raw_model[name] for name in RobotModel.__fields__ if name in raw_model
    }
    descriptor = RobotModel(**descriptor_fields)
    raw_model["_descriptor"] = descriptor
    trajectory = DynamicMotionPrototype.parse_obj(request["trajectory"])
    solver_config = dict(request.get("ik_solver", {}))
    validator = IsaacGymDynamicValidator(
        Path(str(request["training_root"])),
        seed=int(request.get("seed", 1)),
        max_seconds=float(request.get("max_seconds", 5.0)),
        max_steps=int(request.get("max_steps", 250)),
        velocity_error_limit=float(request.get("velocity_error_limit", 0.35)),
        foot_slip_limit=float(request.get("foot_slip_limit", 0.25)),
        no_contact_limit_seconds=float(request.get("no_contact_limit_seconds", 0.8)),
        visualize=bool(request.get("visualize", False)),
        ik_solver=PinocchioIKSolver(**solver_config),
    )
    return validator._validate_in_process(raw_model, trajectory)


def main() -> int:
    """读取一条 JSON 请求，将结构化物理报告写到 stdout 标记行。"""
    try:
        request = json.load(sys.stdin)
        report = _build_request_report(request)
    except Exception as exc:
        report = SimulationReport(
            status="UNAVAILABLE", success=None, backend="isaacgym",
            validated=False, validation_level="CAPABILITY_ONLY",
            reason="Isaac Gym worker 初始化失败：%s" % str(exc)[:400],
        )
    sys.stdout.write(REPORT_MARKER + report.json(ensure_ascii=False) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
