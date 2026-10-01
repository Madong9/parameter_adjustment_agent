"""运行真实 MuJoCo MJCF 短时 rollout 并确定性检查物理约束。"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional

from ..ik.schema import IKResult
from ..motion_prototype.schema import MotionPrototype
from .schema import SimulationReport


class MujocoRolloutValidator:
    """从 IK 构型初始化真实动力学仿真，施加关节伺服并核验稳定、安全指标。"""

    def __init__(self, model_path: Optional[Path] = None, max_seconds: float = 5.0,
                 max_steps: int = 10000, tilt_limit_rad: float = 1.2,
                 tracking_error_limit_rad: float = 0.30):
        """设置 MJCF 文件、仿真时长、步数、姿态和 actuator 误差阈值。"""
        self.model_path = model_path
        self.max_seconds = max(0.01, float(max_seconds))
        self.max_steps = max(1, int(max_steps))
        self.tilt_limit_rad = float(tilt_limit_rad)
        self.tracking_error_limit_rad = float(tracking_error_limit_rad)
        self.backend = "mujoco"

    def _resolve_model(self, robot_model: Any) -> Optional[Path]:
        """从统一模型 descriptor 解析明确配置或转换后的 MJCF 路径。"""
        value = None
        if isinstance(robot_model, dict):
            value = robot_model.get("mujoco_xml") or robot_model.get("mjcf_path")
        else:
            value = getattr(robot_model, "mjcf_path", None)
        return Path(str(value)).expanduser() if value else self.model_path

    def validate(self, robot_model: Any, prototype: MotionPrototype,
                 ik_result: IKResult) -> SimulationReport:
        """运行有界物理 rollout，检查跌倒、姿态、关节、碰撞和目标跟踪。"""
        model_path = self._resolve_model(robot_model)
        if model_path is None or not model_path.is_file():
            return SimulationReport(status="UNAVAILABLE", success=None,
                                   reason="MODEL_UNAVAILABLE：没有可加载的显式 MJCF")
        if not ik_result.success or not ik_result.joint_positions:
            status = "UNAVAILABLE" if ik_result.status == "UNAVAILABLE" else "SKIPPED"
            return SimulationReport(status=status, success=None,
                                    reason="缺少真实 IK 关节构型，不能执行 MuJoCo 物理 rollout")
        try:
            import mujoco
            import numpy as np
        except ImportError as exc:
            return SimulationReport(status="UNAVAILABLE", success=None,
                                   reason="MuJoCo/NumPy 依赖不可用：%s" % exc)
        try:
            model = mujoco.MjModel.from_xml_path(str(model_path))
            data = mujoco.MjData(model)
            mujoco.mj_resetData(model, data)
            joint_targets = {str(name): float(value)
                             for name, value in ik_result.joint_positions.items()}
            joint_ids = self._joint_ids(mujoco, model)
            actuator_map = self._position_actuators(mujoco, model, joint_ids)
            missing = sorted(set(joint_targets) - set(actuator_map))
            if missing:
                return SimulationReport(status="UNAVAILABLE", success=None,
                                        reason="IK 关节没有 MuJoCo 位置 actuator：%s" %
                                        ", ".join(missing))
            for name, target in joint_targets.items():
                joint_id = joint_ids.get(name)
                if joint_id is None:
                    return SimulationReport(status="UNAVAILABLE", success=None,
                                            reason="MJCF 中缺少 IK 关节：%s" % name)
                qpos_address = int(model.jnt_qposadr[joint_id])
                qvel_address = int(model.jnt_dofadr[joint_id])
                data.qpos[qpos_address] = target
                data.qvel[qvel_address] = 0.0
                data.ctrl[actuator_map[name]] = target
            free_joint = next((index for index in range(model.njnt)
                               if int(model.jnt_type[index]) == int(mujoco.mjtJoint.mjJNT_FREE)), None)
            if free_joint is None:
                return SimulationReport(status="UNAVAILABLE", success=None,
                                        reason="MJCF 缺少 free-root joint，无法评估躯干稳定性")
            root_qpos = int(model.jnt_qposadr[free_joint])
            root_qvel = int(model.jnt_dofadr[free_joint])
            requested_height = ik_result.base_height
            if requested_height is not None:
                data.qpos[root_qpos + 2] = float(requested_height)
            orientation_xyzw = list(ik_result.base_orientation_xyzw)
            if len(orientation_xyzw) != 4:
                raise ValueError("IK 结果基座四元数格式错误")
            data.qpos[root_qpos + 3:root_qpos + 7] = [
                orientation_xyzw[3], orientation_xyzw[0],
                orientation_xyzw[1], orientation_xyzw[2],
            ]
            data.qvel[root_qvel:root_qvel + 6] = 0.0
            mujoco.mj_forward(model, data)
            initial_height = float(data.qpos[root_qpos + 2])
            min_height = initial_height
            max_roll = 0.0
            max_pitch = 0.0
            max_joint_limit_excess = 0.0
            max_tracking_error = 0.0
            contact_samples = 0
            foot_contact_samples = 0
            forbidden_body_contact_samples = 0
            self_collision_samples = 0
            self_collision_checked = True
            duration_limit = min(self.max_seconds, ik_result.duration_seconds or
                                 sum(item.duration_seconds for item in prototype.phases))
            steps = min(self.max_steps, max(1, int(math.ceil(duration_limit / model.opt.timestep))))
            violations = []
            fall = False
            for _ in range(steps):
                mujoco.mj_step(model, data)
                qpos = np.asarray(data.qpos)
                if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(data.qvel)):
                    violations.append("non_finite_state")
                    break
                height = float(qpos[root_qpos + 2])
                min_height = min(min_height, height)
                roll, pitch = self._roll_pitch(qpos[root_qpos + 3:root_qpos + 7])
                max_roll = max(max_roll, abs(roll))
                max_pitch = max(max_pitch, abs(pitch))
                contact_samples += int(data.ncon > 0)
                frame_has_foot_contact = False
                frame_has_forbidden_body_contact = False
                frame_has_self_contact = False
                for contact_index in range(data.ncon):
                    contact = data.contact[contact_index]
                    geom_ids = (int(contact.geom1), int(contact.geom2))
                    body_ids = [int(model.geom_bodyid[geom_id]) for geom_id in geom_ids]
                    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
                                  for body_id in body_ids]
                    geom_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
                                  for geom_id in geom_ids]
                    names = [value.lower() for value in body_names + geom_names]
                    world_contact = 0 in body_ids
                    if world_contact and any(any(token in name for token in ("foot", "toe"))
                                             for name in names):
                        frame_has_foot_contact = True
                    if world_contact and any(any(token in name for token in
                                                 ("base", "trunk", "torso", "pelvis", "abdomen"))
                                             for name in names):
                        frame_has_forbidden_body_contact = True
                    if not world_contact and body_ids[0] != body_ids[1]:
                        frame_has_self_contact = True
                foot_contact_samples += int(frame_has_foot_contact)
                forbidden_body_contact_samples += int(frame_has_forbidden_body_contact)
                self_collision_samples += int(frame_has_self_contact)
                if frame_has_forbidden_body_contact and "forbidden_body_contact" not in violations:
                    violations.append("forbidden_body_contact")
                if frame_has_self_contact and "self_collision" not in violations:
                    violations.append("self_collision")
                for joint_id in range(model.njnt):
                    if not model.jnt_limited[joint_id] or int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
                        continue
                    address = int(model.jnt_qposadr[joint_id])
                    lower, upper = model.jnt_range[joint_id]
                    value = float(qpos[address])
                    max_joint_limit_excess = max(max_joint_limit_excess,
                                                 float(lower - value), float(value - upper), 0.0)
                tracking_errors = []
                for name, target in joint_targets.items():
                    qpos_address = int(model.jnt_qposadr[joint_ids[name]])
                    tracking_errors.append(abs(float(qpos[qpos_address]) - target))
                max_tracking_error = max(max_tracking_error, max(tracking_errors or [0.0]))
                if height < max(0.12, initial_height * 0.5):
                    fall = True
                    violations.append("base_height_below_fall_threshold")
                    break
                if max_roll > self.tilt_limit_rad or max_pitch > self.tilt_limit_rad:
                    violations.append("body_tilt_limit_exceeded")
                    break
            if max_joint_limit_excess > 1.0e-4:
                violations.append("joint_limit_exceeded")
            if max_tracking_error > self.tracking_error_limit_rad:
                violations.append("actuator_tracking_error_exceeded")
            expects_foot_support = any(
                value in ("support", "alternating_contact")
                for phase in prototype.phases for value in phase.body_goal.values())
            if expects_foot_support and foot_contact_samples == 0:
                violations.append("required_foot_contact_missing")
            actual_duration = min(duration_limit, steps * model.opt.timestep)
            return SimulationReport(
                status="FAILED" if violations else "PASSED", success=not violations,
                backend="mujoco", validated=True, validation_level="PHYSICS_ROLLOUT",
                converted_model=bool(robot_model.get("converted_model", False))
                if isinstance(robot_model, dict) else False,
                model_source=str(robot_model.get("mjcf_source") or robot_model.get("mjcf_path", ""))
                if isinstance(robot_model, dict) else model_path.name,
                duration=actual_duration, fall=fall, violations=sorted(set(violations)),
                self_collision_checked=self_collision_checked,
                metrics={"max_roll": max_roll, "max_pitch": max_pitch,
                         "min_height": min_height, "initial_height": initial_height,
                         "max_joint_limit_excess": max_joint_limit_excess,
                         "max_actuator_tracking_error": max_tracking_error,
                         "contact_sample_fraction": contact_samples / float(max(1, steps)),
                         "foot_contact_sample_fraction": foot_contact_samples / float(max(1, steps)),
                         "forbidden_body_contact_samples": forbidden_body_contact_samples,
                         "self_collision_samples": self_collision_samples,
                         "steps": steps},
                reason="MuJoCo 真实物理 rollout 已完成" if not violations else
                "MuJoCo 真实物理 rollout 检测到约束违规",
            )
        except Exception as exc:
            return SimulationReport(status="FAILED", success=False, backend="mujoco",
                                    validated=False, reason="MuJoCo 模型或 rollout 失败：%s" %
                                    str(exc)[:600])

    @staticmethod
    def _joint_ids(mujoco: Any, model: Any) -> Dict[str, int]:
        """根据 MJCF 关节名称构造稳定索引映射。"""
        return {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index): index
            for index in range(model.njnt)
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        }

    @classmethod
    def _position_actuators(cls, mujoco: Any, model: Any,
                            joint_ids: Dict[str, int]) -> Dict[str, int]:
        """只接受直接绑定于对应关节的 position servo actuator。"""
        joint_by_id = {joint_id: name for name, joint_id in joint_ids.items()}
        actuators = {}
        for actuator_id in range(model.nu):
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            joint_name = joint_by_id.get(joint_id)
            if joint_name is None:
                continue
            is_position = (
                int(model.actuator_biastype[actuator_id]) == int(mujoco.mjtBias.mjBIAS_AFFINE)
                and model.actuator_gainprm[actuator_id, 0] > 0.0
                and model.actuator_biasprm[actuator_id, 1] < 0.0
            )
            if is_position:
                actuators[joint_name] = actuator_id
        return actuators

    @staticmethod
    def _roll_pitch(quaternion_wxyz: Any):
        """将 MuJoCo free-root 四元数转换为 roll/pitch 弧度值。"""
        w, x, y, z = [float(value) for value in quaternion_wxyz]
        sin_roll = 2.0 * (w * x + y * z)
        cos_roll = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sin_roll, cos_roll)
        sin_pitch = 2.0 * (w * y - z * x)
        pitch = math.copysign(math.pi / 2.0, sin_pitch) if abs(sin_pitch) >= 1.0 else math.asin(sin_pitch)
        return roll, pitch


class MockMuJoCoValidator:
    """提供仅用于测试的替身；结果绝不伪装成真实 MuJoCo rollout。"""

    def __init__(self, succeed: bool = True):
        """配置替身应返回的确定性成功或失败结果。"""
        self.succeed = succeed
        self.backend = "mock"

    def validate(self, robot_model: Any, prototype: MotionPrototype,
                 ik_result: IKResult) -> SimulationReport:
        """返回带 mock backend 标签的结果，不声称已运行物理引擎。"""
        if not ik_result.success:
            return SimulationReport(status="SKIPPED", success=None, backend="mock",
                                    validation_level="MOCK_VALIDATED",
                                    reason="Mock IK 未成功")
        violations = [] if self.succeed else ["mock_fall"]
        return SimulationReport(
            status="PASSED" if self.succeed else "FAILED",
            success=self.succeed, backend="mock", validated=False,
            validation_level="MOCK_VALIDATED",
            duration=sum(item.duration_seconds for item in prototype.phases),
            fall=not self.succeed, violations=violations,
            metrics={"max_pitch": 0.0, "min_height": 0.3},
            reason="Mock rollout；不是物理仿真证据",
        )
