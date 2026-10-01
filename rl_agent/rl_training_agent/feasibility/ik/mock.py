"""提供仅用于单元测试和离线演练的 IK 替身。"""
from __future__ import annotations

from typing import Any, Dict

from .schema import IKResult


class MockIKSolver:
    """返回明确标记为 mock 的成功结果，不代表真实机器人 IK 通过。"""

    backend = "mock"

    def solve_ik(self, robot_model: Any, target_pose: Dict[str, Any],
                 initial_configuration: Any = None) -> IKResult:
        """对非空语义目标返回固定关节样例。"""
        return IKResult(status="SOLVED", success=True, backend=self.backend,
                        joint_positions={"mock_joint": 0.0}, iterations=1,
                        residual=0.0, error=0.0,
                        base_height=(float(target_pose.get("base_height"))
                                     if target_pose.get("base_height") is not None else None),
                        base_orientation_xyzw=target_pose.get(
                            "base_orientation_xyzw", [0.0, 0.0, 0.0, 1.0]),
                        duration_seconds=float(target_pose.get("time_seconds", 0.0) or 0.0),
                        reason="测试替身占位构型；不代表物理可行")
