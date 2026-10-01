"""实现前/后腿支撑动作的确定性全身数值规划。"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

from ..motion_constraints import MotionConstraintSpec
from .schema import BalanceMotionPlan, BalanceTrajectorySample


class BalanceMotionPlanner:
    """串联接触时序、质心/基座/足端轨迹、浮动基座 IK、接触力和逆动力学。"""

    LEG_ORDER = ("FL", "FR", "RL", "RR")

    def __init__(self, dt: float = 0.1, max_samples: int = 61,
                 ik_iterations: int = 100, ik_tolerance: float = 2.0e-2,
                 friction_coefficient: float = 0.8):
        """设置低成本规划预算和确定性数值阈值。"""
        self.dt = max(0.02, float(dt))
        self.max_samples = max(3, int(max_samples))
        self.ik_iterations = max(10, int(ik_iterations))
        self.ik_tolerance = max(1.0e-5, float(ik_tolerance))
        self.friction_coefficient = max(0.05, float(friction_coefficient))
        self.backend = "pinocchio_scipy_balance_planner"

    def plan(self, spec: MotionConstraintSpec, intent: Any,
             robot_model: Any) -> BalanceMotionPlan:
        """为双腿支撑平衡任务生成六项数值结果；不进行 PPO 训练。"""
        descriptor = (robot_model.get("_descriptor") if isinstance(robot_model, dict)
                      else robot_model)
        robot_name = str(getattr(descriptor, "robot_name", ""))
        if descriptor is None or not getattr(descriptor, "urdf_path", None):
            return self._failed(spec, robot_name or spec.robot, "机器人 URDF descriptor 不可用")
        support_legs = self._support_legs(spec)
        if len(support_legs) != 2 or set(support_legs) not in (
                {"FL", "FR"}, {"RL", "RR"}):
            return self._failed(
                spec, robot_name or spec.robot,
                "当前 BalancePlanner 只覆盖双前腿或双后腿支撑；单腿/对角支撑需独立动力学插件")
        lifted_legs = [leg for leg in self.LEG_ORDER if leg not in support_legs]
        moving = self._is_walking(intent)
        try:
            import numpy as np
            import pinocchio as pin
            from scipy.optimize import minimize
        except ImportError as exc:
            return self._failed(spec, robot_name or spec.robot,
                                "Pinocchio/NumPy/SciPy 依赖不可用：%s" % exc)
        try:
            model = pin.buildModelFromUrdf(
                str(descriptor.urdf_path), pin.JointModelFreeFlyer())
            data = model.createData()
            q = self._initial_configuration(pin, model, descriptor)
            nominal_feet = self._frame_positions(pin, model, data, q, descriptor.foot_frames)
            frame_for_leg = self._frame_for_leg(nominal_feet)
            if any(leg not in frame_for_leg for leg in self.LEG_ORDER):
                raise ValueError("URDF 足端 frame 无法映射到 FL/FR/RL/RR")
            pin.centerOfMass(model, data, q)
            initial_com = np.asarray(data.com[0], dtype=float).copy()
            times = self._sample_times(spec.duration, np)
            targets = [self._trajectory_target(
                float(time_value), spec.duration, support_legs, lifted_legs,
                moving, nominal_feet, frame_for_leg, initial_com, descriptor, np)
                for time_value in times]
            configurations = []
            ik_residuals = []
            for target in targets:
                q, residual = self._solve_floating_base_ik(
                    pin, np, model, data, q, target, frame_for_leg)
                if residual > self.ik_tolerance:
                    raise ValueError("浮动基座 IK 未收敛，residual=%.6f" % residual)
                configurations.append(q.copy())
                ik_residuals.append(float(residual))
            velocities, accelerations = self._differentiate_configurations(
                pin, model, configurations, times, np)
            mass = float(pin.computeTotalMass(model))
            samples: List[BalanceTrajectorySample] = []
            max_wrench_residual = max_base_residual = max_torque_ratio = 0.0
            force_failures = 0
            for index, (target, configuration) in enumerate(zip(targets, configurations)):
                pin.forwardKinematics(model, data, configuration, velocities[index], accelerations[index])
                pin.updateFramePlacements(model, data)
                pin.centerOfMass(model, data, configuration, velocities[index], accelerations[index])
                com = np.asarray(data.com[0], dtype=float).copy()
                com_acc = np.asarray(data.acom[0], dtype=float).copy()
                feet = self._frame_positions(
                    pin, model, data, configuration, descriptor.foot_frames)
                contacts = [leg for leg, active in target["contacts"].items() if active]
                forces, wrench_residual, force_ok = self._solve_contact_forces(
                    np, minimize, mass, com, com_acc, feet, frame_for_leg, contacts)
                if not force_ok:
                    force_failures += 1
                torques, base_residual, torque_ratio = self._inverse_dynamics(
                    pin, np, model, data, configuration, velocities[index],
                    accelerations[index], forces, frame_for_leg, descriptor)
                max_wrench_residual = max(max_wrench_residual, wrench_residual)
                max_base_residual = max(max_base_residual, base_residual)
                max_torque_ratio = max(max_torque_ratio, torque_ratio)
                samples.append(BalanceTrajectorySample(
                    time=float(times[index]), phase=target["phase"],
                    base_position=[float(value) for value in configuration[:3]],
                    base_orientation_xyzw=[float(value) for value in configuration[3:7]],
                    com_position=[float(value) for value in com],
                    feet={name: [float(value) for value in position]
                          for name, position in feet.items()},
                    contacts=dict(target["contacts"]),
                    contact_forces={leg: [float(value) for value in force]
                                    for leg, force in forces.items()},
                    joint_positions=self._joint_values(model, configuration, descriptor.joint_names),
                    joint_velocities=self._joint_velocity_values(
                        model, velocities[index], descriptor.joint_names),
                    joint_accelerations=self._joint_velocity_values(
                        model, accelerations[index], descriptor.joint_names),
                    joint_torques=torques, ik_residual=ik_residuals[index],
                    wrench_residual=wrench_residual,
                    base_dynamics_residual=base_residual,
                ))
            violations = []
            if force_failures:
                violations.append("contact_force_qp_infeasible_samples:%d" % force_failures)
            if max_torque_ratio > 1.0 + 1.0e-6:
                violations.append("inverse_dynamics_effort_limit_exceeded")
            if max_base_residual > 0.12:
                violations.append("floating_base_dynamics_residual_exceeded")
            status = "READY_FOR_PHYSICS" if not violations else "INCONCLUSIVE"
            solvers = [
                "contact_schedule", "com_trajectory", "base_pose_trajectory",
                "end_effector_trajectory", "floating_base_ik",
                "contact_force_optimization", "inverse_dynamics",
            ]
            return BalanceMotionPlan(
                status=status, robot=descriptor.robot_name, action=spec.action,
                duration=spec.duration, dt=self._effective_dt(times),
                support_legs=support_legs, lifted_legs=lifted_legs, moving=moving,
                samples=samples, available_solvers=solvers, violations=violations,
                limitations=[
                    "接触模型使用点足与固定摩擦系数，最终结论必须来自 Isaac Gym rollout。",
                    "当前未执行几何自碰撞检查；Isaac Gym 中的禁止接触作为最终安全证据。",
                    "数值规划是训练前参考，不是 policy，也不证明 PPO 已学会动作。",
                ],
                metrics={
                    "sample_count": float(len(samples)),
                    "max_ik_residual": max(ik_residuals) if ik_residuals else float("inf"),
                    "max_contact_wrench_residual": max_wrench_residual,
                    "max_base_dynamics_residual": max_base_residual,
                    "max_torque_ratio": max_torque_ratio,
                    "contact_force_failure_count": float(force_failures),
                },
                reason=("六项确定性数值规划完成，可进入专用 Isaac Gym rollout" if not violations
                        else "数值规划存在未满足约束，不能进入物理通过状态"),
            )
        except Exception as exc:
            return self._failed(spec, robot_name or spec.robot,
                                "平衡数值规划失败：%s" % str(exc)[:500],
                                support_legs=support_legs, lifted_legs=lifted_legs,
                                moving=moving)

    @staticmethod
    def _support_legs(spec: MotionConstraintSpec) -> List[str]:
        """从约束阶段提取明确支撑腿集合。"""
        for phase in reversed(spec.phases):
            contacts = [str(item).upper() for item in phase.active_contacts
                        if str(item).upper() in BalanceMotionPlanner.LEG_ORDER]
            if contacts:
                return contacts
        return []

    @staticmethod
    def _is_walking(intent: Any) -> bool:
        """从目标字段判断是否要求支撑腿交替前行。"""
        text = " ".join((str(getattr(intent, "original_instruction", "")),
                         str(getattr(intent, "action_name", "")),
                         str(getattr(intent, "normalized_goal", "")))).lower()
        return any(token in text for token in ("走", "行走", "前进", "倒退", "walk", "move"))

    def _sample_times(self, duration: float, np: Any) -> Any:
        """在最大样本预算内生成包含终点的均匀时间网格。"""
        count = min(self.max_samples, max(3, int(math.ceil(duration / self.dt)) + 1))
        return np.linspace(0.0, float(duration), count)

    @staticmethod
    def _effective_dt(times: Sequence[float]) -> float:
        """返回均匀时间网格的实际采样周期。"""
        return float(times[1] - times[0]) if len(times) > 1 else 0.1

    @staticmethod
    def _smoothstep(value: float) -> float:
        """使用零端点速度的三次插值。"""
        clipped = min(1.0, max(0.0, float(value)))
        return clipped * clipped * (3.0 - 2.0 * clipped)

    @staticmethod
    def _quaternion_from_pitch(pitch: float) -> List[float]:
        """返回绕 Y 轴旋转的 XYZW 四元数。"""
        return [0.0, math.sin(float(pitch) * 0.5), 0.0,
                math.cos(float(pitch) * 0.5)]

    def _trajectory_target(self, time_value: float, duration: float,
                           support_legs: List[str], lifted_legs: List[str], moving: bool,
                           nominal_feet: Dict[str, Sequence[float]],
                           frame_for_leg: Dict[str, str], initial_com: Any,
                           descriptor: Any, np: Any) -> Dict[str, Any]:
        """生成质心迁移、后腿抬起、接触切换、基座姿态和足端目标。"""
        phase_value = time_value / max(duration, 1.0e-9)
        transfer = self._smoothstep((phase_value - 0.10) / 0.30)
        lift = self._smoothstep((phase_value - 0.30) / 0.22)
        move_progress = self._smoothstep((phase_value - 0.52) / 0.38) if moving else 0.0
        support_center = np.mean([
            np.asarray(nominal_feet[frame_for_leg[leg]], dtype=float)[:2]
            for leg in support_legs], axis=0)
        desired_com_xy = ((1.0 - transfer) * np.asarray(initial_com[:2]) +
                          transfer * support_center)
        pitch_sign = 1.0 if set(support_legs) == {"FL", "FR"} else -1.0
        pitch = pitch_sign * 0.80 * transfer
        base_x = float(desired_com_xy[0] - initial_com[0])
        if moving:
            base_x += (0.06 if pitch_sign < 0.0 else -0.06) * move_progress
        base_position = [base_x, float(desired_com_xy[1] - initial_com[1]), 0.0]
        contacts = {leg: True for leg in self.LEG_ORDER}
        if phase_value >= 0.35:
            contacts = {leg: leg in support_legs for leg in self.LEG_ORDER}
        phase = "prepare" if phase_value < 0.10 else (
            "com_transfer" if phase_value < 0.30 else
            "unload_and_lift" if phase_value < 0.52 else
            "support_walk" if moving and phase_value < 0.90 else "balance_hold")
        feet = {name: list(map(float, position)) for name, position in nominal_feet.items()}
        for leg in lifted_legs:
            feet[frame_for_leg[leg]][2] += float(descriptor.nominal_lift_height) * lift
        if moving and phase_value >= 0.52:
            local = (phase_value - 0.52) / 0.38
            cycles = 2.0
            step_length = 0.03 if pitch_sign < 0.0 else -0.03
            for offset, leg in enumerate(support_legs):
                cycle_position = max(0.0, local * cycles - 0.5 * offset)
                step_index = int(math.floor(cycle_position))
                within = cycle_position - step_index
                landed_steps = step_index
                swing = within < 0.35 and cycle_position > 0.0
                if swing:
                    swing_progress = self._smoothstep(within / 0.35)
                    feet[frame_for_leg[leg]][0] += step_length * (landed_steps + swing_progress)
                    feet[frame_for_leg[leg]][2] += 0.035 * math.sin(math.pi * swing_progress)
                    contacts[leg] = False
                else:
                    feet[frame_for_leg[leg]][0] += step_length * landed_steps
                    contacts[leg] = True
            if not any(contacts[leg] for leg in support_legs):
                contacts[support_legs[0]] = True
        return {
            "time": time_value, "phase": phase, "feet": feet, "contacts": contacts,
            "base_position": base_position,
            "base_orientation_xyzw": self._quaternion_from_pitch(pitch),
            "com_target": [float(desired_com_xy[0]), float(desired_com_xy[1]),
                           float(initial_com[2])],
            "support_legs": list(support_legs), "lifted_legs": list(lifted_legs),
        }

    @staticmethod
    def _initial_configuration(pin: Any, model: Any, descriptor: Any) -> Any:
        """构造自由浮动基座模型的标称初始构型。"""
        import numpy as np
        configuration = pin.neutral(model)
        configuration[:3] = np.zeros(3)
        configuration[3:7] = np.asarray([0.0, 0.0, 0.0, 1.0])
        for name, value in descriptor.default_joint_positions.items():
            joint_id = model.getJointId(str(name))
            if 0 < joint_id < model.njoints and model.joints[joint_id].nq == 1:
                configuration[model.joints[joint_id].idx_q] = float(value)
        return configuration

    @staticmethod
    def _frame_positions(pin: Any, model: Any, data: Any, configuration: Any,
                         frame_names: Sequence[str]) -> Dict[str, List[float]]:
        """计算指定足端 frame 的世界坐标。"""
        pin.forwardKinematics(model, data, configuration)
        pin.updateFramePlacements(model, data)
        result = {}
        for name in frame_names:
            frame_id = model.getFrameId(str(name))
            if frame_id >= model.nframes:
                raise ValueError("URDF 缺少足端 frame：%s" % name)
            result[str(name)] = [float(value) for value in data.oMf[frame_id].translation]
        return result

    @staticmethod
    def _frame_for_leg(feet: Dict[str, Sequence[float]]) -> Dict[str, str]:
        """把 URDF 足端 frame 映射为标准腿名。"""
        result = {}
        for leg in BalanceMotionPlanner.LEG_ORDER:
            matches = [name for name in feet if str(name).upper().startswith(leg + "_")]
            if len(matches) == 1:
                result[leg] = matches[0]
        return result

    def _solve_floating_base_ik(self, pin: Any, np: Any, model: Any, data: Any,
                                initial: Any, target: Dict[str, Any],
                                frame_for_leg: Dict[str, str]) -> Tuple[Any, float]:
        """同时求解自由基座位姿和四足笛卡尔目标。"""
        configuration = initial.copy()
        desired_rotation = pin.Quaternion(np.asarray(
            target["base_orientation_xyzw"], dtype=float)).matrix()
        desired_base = np.asarray(target["base_position"], dtype=float)
        residual = float("inf")
        for _iteration in range(self.ik_iterations):
            pin.forwardKinematics(model, data, configuration)
            pin.updateFramePlacements(model, data)
            errors = []
            jacobians = []
            for leg in self.LEG_ORDER:
                frame_name = frame_for_leg[leg]
                frame_id = model.getFrameId(frame_name)
                desired = np.asarray(target["feet"][frame_name], dtype=float)
                current = np.asarray(data.oMf[frame_id].translation)
                jacobian = pin.computeFrameJacobian(
                    model, data, configuration, frame_id,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                # 已抬起的非支撑腿只约束离地高度，避免把语义目标过约束成不可达姿态。
                if leg in target.get("lifted_legs", []) and target["phase"] not in (
                        "prepare", "com_transfer"):
                    errors.append(np.asarray([desired[2] - current[2]]))
                    jacobians.append(jacobian[2:3, :])
                else:
                    errors.append(desired - current)
                    jacobians.append(jacobian[:3, :])
            base_translation_error = desired_base - np.asarray(configuration[:3])
            pin.centerOfMass(model, data, configuration)
            current_com = np.asarray(data.com[0], dtype=float)
            com_jacobian = pin.jacobianCenterOfMass(model, data, configuration)
            current_rotation = pin.Quaternion(np.asarray(configuration[3:7])).matrix()
            orientation_error = pin.log3(current_rotation.T.dot(desired_rotation))
            base_jacobian = np.zeros((6, model.nv))
            base_jacobian[:3, :3] = np.eye(3)
            base_jacobian[3:, 3:6] = np.eye(3)
            desired_com = np.asarray(target["com_target"], dtype=float)
            errors.extend([0.80 * (desired_com[:2] - current_com[:2]),
                           0.35 * base_translation_error[2:3],
                           0.65 * orientation_error])
            jacobians.extend([0.80 * com_jacobian[:2, :],
                              0.35 * base_jacobian[2:3, :],
                              0.65 * base_jacobian[3:, :]])
            error_vector = np.concatenate(errors)
            jacobian = np.vstack(jacobians)
            residual = float(np.linalg.norm(error_vector) / math.sqrt(len(errors)))
            if residual <= self.ik_tolerance:
                return configuration, residual
            damping = 2.0e-3
            system = jacobian.dot(jacobian.T) + damping * np.eye(jacobian.shape[0])
            delta = jacobian.T.dot(np.linalg.solve(system, error_vector))
            max_norm = float(np.linalg.norm(delta))
            if max_norm > 0.18:
                delta *= 0.18 / max_norm
            configuration = pin.integrate(model, configuration, delta)
            for joint_id in range(2, model.njoints):
                joint = model.joints[joint_id]
                if joint.nq == 1:
                    index = joint.idx_q
                    configuration[index] = min(float(model.upperPositionLimit[index]),
                                               max(float(model.lowerPositionLimit[index]),
                                                   float(configuration[index])))
        return configuration, residual

    @staticmethod
    def _differentiate_configurations(pin: Any, model: Any, configurations: Sequence[Any],
                                      times: Sequence[float], np: Any) -> Tuple[List[Any], List[Any]]:
        """以 Pinocchio configuration difference 计算速度和加速度参考。"""
        count = len(configurations)
        velocities = [np.zeros(model.nv) for _ in range(count)]
        for index in range(1, count):
            dt = max(1.0e-9, float(times[index] - times[index - 1]))
            velocities[index] = pin.difference(
                model, configurations[index - 1], configurations[index]) / dt
        if count > 1:
            velocities[0] = velocities[1].copy()
        accelerations = [np.zeros(model.nv) for _ in range(count)]
        for index in range(1, count):
            dt = max(1.0e-9, float(times[index] - times[index - 1]))
            accelerations[index] = (velocities[index] - velocities[index - 1]) / dt
        if count > 1:
            accelerations[0] = accelerations[1].copy()
        return velocities, accelerations

    def _solve_contact_forces(self, np: Any, minimize: Any, mass: float, com: Any,
                              com_acc: Any, feet: Dict[str, Sequence[float]],
                              frame_for_leg: Dict[str, str], contacts: List[str]
                              ) -> Tuple[Dict[str, Any], float, bool]:
        """用带摩擦锥约束的最小二乘 QP 分配接触力。"""
        if not contacts:
            return {}, float("inf"), False
        count = len(contacts)
        matrix = np.zeros((6, 3 * count))
        for index, leg in enumerate(contacts):
            matrix[:3, 3 * index:3 * index + 3] = np.eye(3)
            arm = np.asarray(feet[frame_for_leg[leg]], dtype=float) - np.asarray(com)
            matrix[3:, 3 * index:3 * index + 3] = np.asarray([
                [0.0, -arm[2], arm[1]],
                [arm[2], 0.0, -arm[0]],
                [-arm[1], arm[0], 0.0],
            ])
        gravity = np.asarray([0.0, 0.0, -9.81])
        target = np.concatenate([mass * (np.asarray(com_acc) - gravity), np.zeros(3)])
        initial = np.zeros(3 * count)
        initial[2::3] = mass * 9.81 / float(count)

        def objective(values: Any) -> float:
            """最小化归一化质心合力/力矩误差和接触力正则项。"""
            residual = matrix.dot(values) - target
            return float(residual.dot(residual) / max(1.0, (mass * 9.81) ** 2) +
                         1.0e-7 * values.dot(values))

        constraints = []
        for index in range(count):
            base = 3 * index
            constraints.extend([
                {"type": "ineq", "fun": lambda x, b=base: x[b + 2]},
                {"type": "ineq", "fun": lambda x, b=base: self.friction_coefficient * x[b + 2] - x[b]},
                {"type": "ineq", "fun": lambda x, b=base: self.friction_coefficient * x[b + 2] + x[b]},
                {"type": "ineq", "fun": lambda x, b=base: self.friction_coefficient * x[b + 2] - x[b + 1]},
                {"type": "ineq", "fun": lambda x, b=base: self.friction_coefficient * x[b + 2] + x[b + 1]},
            ])
        result = minimize(objective, initial, method="SLSQP", constraints=constraints,
                          options={"maxiter": 120, "ftol": 1.0e-10, "disp": False})
        values = np.asarray(result.x if result.x is not None else initial, dtype=float)
        residual = float(np.linalg.norm(matrix.dot(values) - target) /
                         max(1.0, mass * 9.81))
        forces = {leg: values[3 * index:3 * index + 3]
                  for index, leg in enumerate(contacts)}
        return forces, residual, bool(result.success and residual <= 0.08)

    @staticmethod
    def _inverse_dynamics(pin: Any, np: Any, model: Any, data: Any,
                          configuration: Any, velocity: Any, acceleration: Any,
                          forces: Dict[str, Any], frame_for_leg: Dict[str, str],
                          descriptor: Any) -> Tuple[Dict[str, float], float, float]:
        """用 Pinocchio RNEA 减去接触广义力并检查基座残差与关节力矩。"""
        generalized = np.asarray(pin.rnea(
            model, data, configuration, velocity, acceleration), dtype=float).copy()
        for leg, force in forces.items():
            frame_id = model.getFrameId(frame_for_leg[leg])
            jacobian = pin.computeFrameJacobian(
                model, data, configuration, frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            generalized -= jacobian[:3, :].T.dot(np.asarray(force, dtype=float))
        mass = max(1.0e-9, float(pin.computeTotalMass(model)))
        base_residual = float(np.linalg.norm(generalized[:6]) / (mass * 9.81))
        torques = {}
        max_ratio = 0.0
        for name in descriptor.joint_names:
            joint_id = model.getJointId(str(name))
            if not 0 < joint_id < model.njoints or model.joints[joint_id].nv != 1:
                continue
            value = float(generalized[model.joints[joint_id].idx_v])
            torques[str(name)] = value
            limit = descriptor.limits.get(str(name))
            effort = getattr(limit, "effort", None) if limit is not None else None
            if effort is not None and float(effort) > 0.0:
                max_ratio = max(max_ratio, abs(value) / float(effort))
        return torques, base_residual, max_ratio

    @staticmethod
    def _joint_values(model: Any, configuration: Any,
                      joint_names: Sequence[str]) -> Dict[str, float]:
        """从自由基座构型提取受控单自由度关节角。"""
        values = {}
        for name in joint_names:
            joint_id = model.getJointId(str(name))
            if 0 < joint_id < model.njoints and model.joints[joint_id].nq == 1:
                values[str(name)] = float(configuration[model.joints[joint_id].idx_q])
        return values

    @staticmethod
    def _joint_velocity_values(model: Any, vector: Any,
                               joint_names: Sequence[str]) -> Dict[str, float]:
        """从 Pinocchio 切空间向量提取受控关节速度或加速度。"""
        values = {}
        for name in joint_names:
            joint_id = model.getJointId(str(name))
            if 0 < joint_id < model.njoints and model.joints[joint_id].nv == 1:
                values[str(name)] = float(vector[model.joints[joint_id].idx_v])
        return values

    def _failed(self, spec: MotionConstraintSpec, robot: str, reason: str,
                support_legs: Sequence[str] = (), lifted_legs: Sequence[str] = (),
                moving: bool = False) -> BalanceMotionPlan:
        """构造不会被误认为物理通过的保守失败报告。"""
        return BalanceMotionPlan(
            status="INCONCLUSIVE", robot=robot or spec.robot, action=spec.action,
            duration=spec.duration, dt=self.dt, support_legs=list(support_legs),
            lifted_legs=list(lifted_legs), moving=moving,
            violations=["balance_planning_failed"], reason=reason,
            limitations=["没有完整数值解时禁止进入真实物理通过状态"],
        )
