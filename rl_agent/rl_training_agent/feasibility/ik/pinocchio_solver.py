"""通过真实 Pinocchio URDF 模型执行多脚端阻尼最小二乘逆运动学。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schema import IKResult


class PinocchioIKSolver:
    """使用 Pinocchio 运动学、配置关节范围及可选碰撞几何求机器人构型。"""

    def __init__(self, max_iterations: int = 120, tolerance: float = 1.0e-3,
                 damping: float = 1.0e-2):
        """设置迭代、误差和阻尼界限，限制训练前预检计算成本。"""
        self.max_iterations = max(1, int(max_iterations))
        self.tolerance = max(1.0e-8, float(tolerance))
        self.damping = max(1.0e-8, float(damping))
        self.backend = "pinocchio"
        self._model_cache: Dict[str, Tuple[Any, Any]] = {}

    @staticmethod
    def _model_path(robot_model: Any) -> Path:
        """从统一模型对象或字典取得 URDF 来源路径。"""
        if isinstance(robot_model, dict):
            value = robot_model.get("urdf_path")
        else:
            value = getattr(robot_model, "urdf_path", None)
        return Path(str(value or "")).expanduser()

    @staticmethod
    def _normalize_targets(target_pose: Dict[str, Any]) -> List[Tuple[str, Any]]:
        """兼容旧单末端接口并提取多脚端笛卡尔位置目标。"""
        feet = target_pose.get("feet")
        if isinstance(feet, dict):
            return list(feet.items())
        frame = target_pose.get("frame")
        position = target_pose.get("position")
        return [(str(frame), position)] if frame and position is not None else []

    def solve_ik(self, robot_model: Any, target_pose: Dict[str, Any],
                 initial_configuration: Optional[Dict[str, float]] = None) -> IKResult:
        """对脚端目标进行真实 Pinocchio FK/Jacobian IK 并输出限位内关节配置。"""
        urdf = self._model_path(robot_model)
        if not urdf.is_file():
            return IKResult(status="UNAVAILABLE", success=None, backend=self.backend,
                            reason="机器人 URDF 不存在或未配置：%s" % urdf.name)
        targets = self._normalize_targets(target_pose)
        if not targets:
            return IKResult(status="SKIPPED", success=None, backend=self.backend,
                            reason="动作目标没有脚端笛卡尔坐标；不会自行猜测关节角")
        try:
            import numpy as np
            import pinocchio as pin
        except ImportError as exc:
            return IKResult(status="UNAVAILABLE", success=None, backend=self.backend,
                            reason="Pinocchio Python 依赖不可用：%s" % exc)
        try:
            cache_key = str(urdf.resolve())
            cached = self._model_cache.get(cache_key)
            if cached is None:
                model = pin.buildModelFromUrdf(str(urdf))
                data = model.createData()
                self._model_cache.clear()
                self._model_cache[cache_key] = (model, data)
            else:
                model, data = cached
            configuration = self._initial_configuration(pin, model, robot_model,
                                                        initial_configuration)
            frame_targets = []
            for frame_name, raw_value in targets:
                frame_id = model.getFrameId(str(frame_name))
                if frame_id >= model.nframes:
                    return IKResult(status="FAILED", success=False, backend=self.backend,
                                    reason="URDF 中不存在目标 frame：%s" % frame_name)
                position = raw_value.get("position") if isinstance(raw_value, dict) else raw_value
                desired = np.asarray(position, dtype=float).reshape(3)
                if not np.all(np.isfinite(desired)):
                    return IKResult(status="FAILED", success=False, backend=self.backend,
                                    reason="目标位置含非有限数值：%s" % frame_name)
                frame_targets.append((str(frame_name), frame_id, desired))
            duration = float(target_pose.get("time_seconds", 0.0) or 0.0)
            base_height = float(target_pose.get("base_height", 0.0) or 0.0)
            orientation = [float(item) for item in target_pose.get(
                "base_orientation_xyzw", [0.0, 0.0, 0.0, 1.0])]
            residual = float("inf")
            for iteration in range(1, self.max_iterations + 1):
                pin.forwardKinematics(model, data, configuration)
                pin.updateFramePlacements(model, data)
                errors = []
                jacobians = []
                for _, frame_id, desired in frame_targets:
                    current = data.oMf[frame_id].translation
                    errors.append(desired - current)
                    jacobian = pin.computeFrameJacobian(
                        model, data, configuration, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                    jacobians.append(jacobian[:3, :])
                error_vector = np.concatenate(errors)
                jacobian_stack = np.vstack(jacobians)
                residual = float(np.linalg.norm(error_vector) / max(1, len(frame_targets)) ** 0.5)
                if residual <= self.tolerance:
                    robot_joint_names = (robot_model.get("joint_names", [])
                                         if isinstance(robot_model, dict) else
                                         getattr(robot_model, "joint_names", []))
                    positions = self._joint_positions(model, configuration, robot_joint_names)
                    violations = self._position_limit_violations(
                        model, configuration, robot_joint_names)
                    if violations:
                        return IKResult(status="FAILED", success=False, backend=self.backend,
                                        joint_positions=positions, violations=violations,
                                        reason="求解结果违反 URDF joint limit", iterations=iteration,
                                        residual=residual, error=residual,
                                        base_height=base_height,
                                        base_orientation_xyzw=orientation,
                                        duration_seconds=duration)
                    return IKResult(
                        status="SOLVED", success=True, backend=self.backend,
                        joint_positions=positions, iterations=iteration,
                        residual=residual, error=residual,
                        base_height=base_height,
                        base_orientation_xyzw=orientation,
                        duration_seconds=duration,
                        reason="Pinocchio IK 收敛；动力学/接触约束由实际仿真 backend 后续复核",
                    )
                system = jacobian_stack.dot(jacobian_stack.T) + (
                    self.damping ** 2) * np.eye(jacobian_stack.shape[0])
                delta = jacobian_stack.T.dot(np.linalg.solve(system, error_vector))
                configuration = pin.integrate(model, configuration, delta)
                configuration = np.minimum(
                    np.maximum(configuration, model.lowerPositionLimit), model.upperPositionLimit)
            return IKResult(status="FAILED", success=False, backend=self.backend,
                            violations=["ik_not_converged"], iterations=self.max_iterations,
                            residual=residual, error=residual,
                            base_height=base_height, base_orientation_xyzw=orientation,
                            duration_seconds=duration,
                            reason="Pinocchio IK 未在迭代上限内收敛")
        except Exception as exc:
            return IKResult(status="FAILED", success=False, backend=self.backend,
                            reason="Pinocchio 模型、目标或 IK 求解失败：%s" % str(exc)[:500])

    @staticmethod
    def _initial_configuration(pin: Any, model: Any, robot_model: Any,
                               supplied: Optional[Dict[str, float]]) -> Any:
        """从调用方姿态或模型配置加载初始值并裁剪到 URDF 关节限位。"""
        import numpy as np

        configuration = pin.neutral(model)
        if supplied is not None:
            values = supplied
        elif isinstance(robot_model, dict):
            values = robot_model.get("default_joint_positions", {})
        else:
            values = getattr(robot_model, "default_joint_positions", {})
        for name, value in dict(values or {}).items():
            joint_id = model.getJointId(str(name))
            if joint_id <= 0 or joint_id >= model.njoints:
                continue
            joint = model.joints[joint_id]
            if joint.nq == 1:
                configuration[joint.idx_q] = float(value)
        return np.minimum(np.maximum(configuration, model.lowerPositionLimit),
                          model.upperPositionLimit)

    @staticmethod
    def _joint_positions(model: Any, configuration: Any,
                         joint_names: List[str]) -> Dict[str, float]:
        """只返回机器人描述符列出的单自由度关节，避免依赖 C++ names 容器绑定。"""
        values: Dict[str, float] = {}
        for name in joint_names:
            index = model.getJointId(str(name))
            if index <= 0 or index >= model.njoints:
                continue
            joint = model.joints[index]
            if joint.nq == 1:
                values[str(name)] = float(configuration[joint.idx_q])
        return values

    @staticmethod
    def _position_limit_violations(model: Any, configuration: Any,
                                   joint_names: List[str]) -> List[str]:
        """检查 RobotModel 指定关节的 Pinocchio URDF 位置限位。"""
        violations = []
        for name in joint_names:
            index = model.getJointId(str(name))
            if index <= 0 or index >= model.njoints:
                continue
            joint = model.joints[index]
            if joint.nq != 1:
                continue
            position = float(configuration[joint.idx_q])
            lower = float(model.lowerPositionLimit[joint.idx_q])
            upper = float(model.upperPositionLimit[joint.idx_q])
            if position < lower - 1.0e-8 or position > upper + 1.0e-8:
                violations.append("joint_limit:%s" % name)
        return violations
