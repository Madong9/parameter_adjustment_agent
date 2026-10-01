"""执行静态姿态的关节限位、质心/支撑多边形和证据覆盖检查。"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..schema import StageStatus, ValidationStageReport


Point2 = Tuple[float, float]


class StaticPoseValidator:
    """组合可验证的静态几何检查；未知的碰撞/力学数据保持 UNKNOWN。"""

    @staticmethod
    def _convex_hull(points: Sequence[Point2]) -> List[Point2]:
        """使用单调链计算二维支撑点凸包，自动移除重复点。"""
        unique = sorted(set((float(x), float(y)) for x, y in points))
        if len(unique) <= 1:
            return unique

        def cross(origin: Point2, first: Point2, second: Point2) -> float:
            """计算二维叉积以判断转向和共线。"""
            return ((first[0] - origin[0]) * (second[1] - origin[1]) -
                    (first[1] - origin[1]) * (second[0] - origin[0]))

        lower: List[Point2] = []
        for point in unique:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 1.0e-12:
                lower.pop()
            lower.append(point)
        upper: List[Point2] = []
        for point in reversed(unique):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 1.0e-12:
                upper.pop()
            upper.append(point)
        return lower[:-1] + upper[:-1]

    @classmethod
    def support_polygon_margin(cls, center_of_mass_xy: Sequence[float],
                               support_points_xy: Sequence[Sequence[float]]) -> Dict[str, Any]:
        """计算质心投影到支撑凸包边界的有符号最小距离。"""
        if len(center_of_mass_xy) != 2 or not all(
                math.isfinite(float(value)) for value in center_of_mass_xy):
            return {"status": "UNKNOWN", "reason": "CoM 投影缺失或不是有限二维坐标"}
        try:
            points = [(float(point[0]), float(point[1])) for point in support_points_xy
                      if len(point) >= 2 and math.isfinite(float(point[0])) and
                      math.isfinite(float(point[1]))]
        except (TypeError, ValueError, IndexError):
            points = []
        hull = cls._convex_hull(points)
        if len(hull) < 3:
            return {
                "status": "UNKNOWN",
                "reason": "有效支撑点少于三个；点足/接触面几何不足以形成二维支撑多边形",
                "support_polygon": [list(point) for point in hull],
            }
        com = (float(center_of_mass_xy[0]), float(center_of_mass_xy[1]))
        signed_edge_distances: List[float] = []
        inside = True
        for index, start in enumerate(hull):
            end = hull[(index + 1) % len(hull)]
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            if length <= 1.0e-12:
                continue
            # 凸包为逆时针：左侧半平面为内部。
            signed = (dx * (com[1] - start[1]) - dy * (com[0] - start[0])) / length
            signed_edge_distances.append(signed)
            inside = inside and signed >= -1.0e-9
        margin = min(signed_edge_distances) if signed_edge_distances else float("nan")
        return {
            "status": "STATIC_STABLE" if inside else "STATIC_UNSTABLE",
            "center_of_mass_projection": [com[0], com[1]],
            "support_polygon": [list(point) for point in hull],
            "margin_m": margin,
            "inside_support_polygon": inside,
            "reason": "CoM 投影位于支撑凸包内" if inside else
                      "CoM 投影位于支撑凸包外；该几何检查不包含加速度和接触力",
        }

    @staticmethod
    def check_joint_limits(joint_positions: Dict[str, float],
                           joint_limits: Dict[str, Any]) -> ValidationStageReport:
        """将实际 IK 关节位置与模型位置范围逐关节比较。"""
        if not joint_positions:
            return ValidationStageReport(
                stage="joint_limits", status=StageStatus.UNKNOWN,
                reason="没有 IK 关节位置输入", backend="deterministic",
            )
        violations: List[str] = []
        unchecked: List[str] = []
        for name, raw_position in joint_positions.items():
            raw_limit = joint_limits.get(name)
            if raw_limit is None:
                unchecked.append(str(name))
                continue
            limit = raw_limit.dict() if hasattr(raw_limit, "dict") else dict(raw_limit)
            lower, upper = limit.get("lower"), limit.get("upper")
            position = float(raw_position)
            if not math.isfinite(position):
                violations.append("non_finite_joint_position:%s" % name)
            elif lower is None or upper is None:
                unchecked.append(str(name))
            elif position < float(lower) - 1.0e-8 or position > float(upper) + 1.0e-8:
                violations.append("joint_position_limit:%s" % name)
        status = (StageStatus.FAILED if violations else
                  StageStatus.CONDITIONAL if unchecked else StageStatus.PASSED)
        return ValidationStageReport(
            stage="joint_limits", status=status,
            reason=("关节位置超限" if violations else
                    "部分关节缺少有限位置限位" if unchecked else "所有 IK 关节位置均在模型限位内"),
            backend="deterministic", metrics={"checked_joint_count": len(joint_positions),
                                               "unchecked_joints": unchecked,
                                               "violations": violations},
            evidence=["RobotModel joint limits", "Pinocchio IK joint positions"],
        )

    @staticmethod
    def _rotate_xy(point: Sequence[float], quaternion_xyzw: Sequence[float]) -> Point2:
        """把根坐标系中的三维点按四元数旋转后投影到地面 XY。"""
        x, y, z = (float(value) for value in point[:3])
        qx, qy, qz, qw = (float(value) for value in quaternion_xyzw[:4])
        norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if norm <= 1.0e-12:
            raise ValueError("base orientation quaternion has zero norm")
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
        # q * [x,y,z] * conjugate(q)，展开成旋转矩阵前两行。
        rx = (1 - 2*(qy*qy + qz*qz))*x + 2*(qx*qy - qz*qw)*y + 2*(qx*qz + qy*qw)*z
        ry = 2*(qx*qy + qz*qw)*x + (1 - 2*(qx*qx + qz*qz))*y + 2*(qy*qz - qx*qw)*z
        return rx, ry

    @staticmethod
    def _pinocchio_com(robot_model: Dict[str, Any],
                       joint_positions: Dict[str, float]) -> Optional[List[float]]:
        """仅在真实 Pinocchio URDF 惯性模型和关节解都可用时计算 CoM。"""
        urdf_path = robot_model.get("urdf_path")
        if not urdf_path:
            return None
        try:
            import pinocchio as pin
            model = pin.buildModelFromUrdf(str(urdf_path))
            data = model.createData()
            configuration = pin.neutral(model)
            for name, position in joint_positions.items():
                joint_id = model.getJointId(str(name))
                if joint_id <= 0 or joint_id >= model.njoints:
                    continue
                joint = model.joints[joint_id]
                if joint.nq != 1:
                    continue
                configuration[joint.idx_q] = float(position)
            center = pin.centerOfMass(model, data, configuration)
            values = [float(value) for value in center.reshape(-1)[:3]]
            return values if len(values) == 3 and all(math.isfinite(item) for item in values) else None
        except (ImportError, OSError, RuntimeError, ValueError, AttributeError, TypeError):
            return None

    @staticmethod
    def _pinocchio_gravity_torques(robot_model: Dict[str, Any],
                                  joint_positions: Dict[str, float]) -> Optional[Dict[str, float]]:
        """通过 Pinocchio RNEA 估算固定根、零速度/加速度下的重力补偿力矩。"""
        urdf_path = robot_model.get("urdf_path")
        if not urdf_path:
            return None
        try:
            import numpy as np
            import pinocchio as pin
            model = pin.buildModelFromUrdf(str(urdf_path))
            data = model.createData()
            configuration = pin.neutral(model)
            for name, position in joint_positions.items():
                joint_id = model.getJointId(str(name))
                if joint_id <= 0 or joint_id >= model.njoints:
                    continue
                joint = model.joints[joint_id]
                if joint.nq == 1:
                    configuration[joint.idx_q] = float(position)
            zero_velocity = np.zeros(model.nv)
            torque = pin.rnea(model, data, configuration, zero_velocity, zero_velocity)
            result: Dict[str, float] = {}
            for name in joint_positions:
                joint_id = model.getJointId(str(name))
                if joint_id <= 0 or joint_id >= model.njoints:
                    continue
                joint = model.joints[joint_id]
                if joint.nv == 1:
                    result[str(name)] = float(torque[joint.idx_v])
            return result if result and all(math.isfinite(value) for value in result.values()) else None
        except (ImportError, OSError, RuntimeError, ValueError, AttributeError, TypeError):
            return None

    def assess(self, robot_model: Dict[str, Any], target: Optional[Dict[str, Any]],
               joint_positions: Dict[str, float], self_collision_checked: bool = False
               ) -> Dict[str, Any]:
        """生成静态姿态各确定性子项报告；缺失动力学数据会被明确标未知。"""
        limits = robot_model.get("limits", {})
        joint_stage = self.check_joint_limits(joint_positions, limits)
        collision_stage = ValidationStageReport(
            stage="self_collision", status=(StageStatus.PASSED if self_collision_checked
                                             else StageStatus.UNKNOWN),
            backend="pinocchio_geometry" if self_collision_checked else "not_available",
            reason="碰撞后端已报告检查" if self_collision_checked else
                   "当前 URDF/Pinocchio 路径没有报告经过验证的自碰撞检查",
        )
        target = target or {}
        feet = target.get("feet", {})
        com = self._pinocchio_com(robot_model, joint_positions)
        support_points: List[Point2] = []
        com_report: Dict[str, Any]
        if com is not None and isinstance(feet, dict) and len(feet) >= 3:
            parsed = [(str(name), [float(value) for value in pos[:3]])
                      for name, pos in feet.items() if isinstance(pos, (list, tuple)) and len(pos) == 3]
            if len(parsed) >= 3:
                lowest = min(item[1][2] for item in parsed)
                supporting = [item for item in parsed if item[1][2] <= lowest + 0.02]
                rotation = target.get("base_orientation_xyzw", [0.0, 0.0, 0.0, 1.0])
                support_points = [self._rotate_xy(position, rotation) for _, position in supporting]
                com_xy = self._rotate_xy(com, rotation)
                com_report = self.support_polygon_margin(com_xy, support_points)
                com_report.update({"backend": "pinocchio", "com_position_root_m": com,
                                   "support_frames": [name for name, _ in supporting],
                                   "support_point_count": len(supporting)})
            else:
                com_report = {"status": "UNKNOWN", "reason": "脚端支撑坐标缺失"}
        else:
            com_report = {"status": "UNKNOWN", "backend": "not_available",
                          "reason": "没有 Pinocchio CoM 惯性数据和至少三个候选脚端目标"}
        com_status = {"STATIC_STABLE": StageStatus.PASSED,
                      "STATIC_UNSTABLE": StageStatus.FAILED}.get(
                          com_report.get("status"), StageStatus.UNKNOWN)
        com_stage = ValidationStageReport(
            stage="center_of_mass_support_polygon", status=com_status,
            reason=str(com_report.get("reason", "CoM 支撑检查未完成")),
            backend=str(com_report.get("backend", "deterministic_geometry")),
            metrics={key: value for key, value in com_report.items()
                     if key not in ("reason", "backend")},
            evidence=["Pinocchio model inertias" if com is not None else
                      "CoM evidence unavailable", "RobotMotionTarget foot positions"],
        )
        gravity_torques = self._pinocchio_gravity_torques(robot_model, joint_positions)
        effort_limits = {}
        for name, raw_limit in limits.items():
            limit = raw_limit.dict() if hasattr(raw_limit, "dict") else dict(raw_limit)
            if limit.get("effort") is not None:
                effort_limits[str(name)] = float(limit["effort"])
        if gravity_torques and effort_limits:
            from .torque_validator import TorqueFeasibilityValidator
            gravity_result = TorqueFeasibilityValidator.validate(
                gravity_torques, effort_limits,
                backend="pinocchio_rnea_fixed_base_gravity_only")
            gravity_payload = gravity_result.dict()
            gravity_payload["stage"] = "static_torque"
            gravity_payload["status"] = StageStatus.CONDITIONAL
            gravity_payload["reason"] = (
                "Pinocchio RNEA 固定根/重力补偿估算已与 URDF effort 对比；缺少接触约束力分配，"
                "不能据此确认完整静态力矩可行性")
            gravity_payload["metrics"]["gravity_torque_estimate_by_joint"] = gravity_torques
            gravity_payload["metrics"]["contact_dynamics_included"] = False
            torque_stage = ValidationStageReport.parse_obj(gravity_payload)
        else:
            torque_stage = ValidationStageReport(
                stage="static_torque", status=StageStatus.UNKNOWN,
                backend="not_available",
                reason="Pinocchio gravity torque 或 actuator effort limit 不完整；未合成所需力矩",
            )
        friction_stage = ValidationStageReport(
            stage="friction_cone", status=StageStatus.UNKNOWN,
            backend="not_available",
            reason="当前目标 IK 不提供接触力和经标定摩擦系数",
        )
        return {
            "status": "FAILED" if any(stage.status == StageStatus.FAILED for stage in
                                       (joint_stage, collision_stage, com_stage,
                                        torque_stage, friction_stage)) else
                      "INCONCLUSIVE" if any(stage.status in (StageStatus.UNKNOWN,
                                                             StageStatus.CONDITIONAL)
                                            for stage in (joint_stage, collision_stage, com_stage,
                                                           torque_stage, friction_stage)) else "PASSED",
            "joint_limits": joint_stage.dict(),
            "self_collision": collision_stage.dict(),
            "center_of_mass_support_polygon": com_stage.dict(),
            "torque": torque_stage.dict(),
            "friction": friction_stage.dict(),
            "support_points_xy": [list(point) for point in support_points],
        }
