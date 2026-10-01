"""实现 Go2 资产解析和基于标称姿态正运动学的脚端目标生成。"""
from __future__ import annotations

from typing import Any, Dict, List

from .schema import RobotModel


class Go2RobotModel:
    """从配置的 Go2 URDF/训练参数提取模型事实，不在代码中固定资产路径。"""

    @staticmethod
    def enrich(model: RobotModel, urdf_root: Any,
               config: Dict[str, Any]) -> RobotModel:
        """从 URDF 读取 Go2 可动关节及位置、速度、力矩范围。"""
        import xml.etree.ElementTree as ET

        root = ET.parse(str(urdf_root)).getroot()
        names: List[str] = []
        limits = {}
        for joint in root.findall("joint"):
            name = joint.get("name")
            if not name or joint.get("type") not in ("revolute", "continuous", "prismatic"):
                continue
            names.append(name)
            item = joint.find("limit")
            if item is not None:
                limits[name] = {
                    key: float(item.get(source)) if item.get(source) is not None else None
                    for key, source in (("lower", "lower"), ("upper", "upper"),
                                        ("velocity", "velocity"), ("effort", "effort"))
                }
        configured = set(model.default_joint_positions)
        if configured - set(names):
            model.model_status = "MODEL_UNAVAILABLE"
            model.model_error = "配置关节不在 URDF 中：%s" % ", ".join(sorted(configured - set(names)))
            return model
        model.joint_names = names
        model.limits = limits
        kp = float(config.get("position_kp", 20.0))
        kd = float(config.get("position_kd", 0.5))
        model.actuators = [{
            "name": "position_" + name,
            "joint": name,
            "control": "position",
            "kp": kp,
            "kd": kd,
        } for name in model.default_joint_positions]
        return model

    @staticmethod
    def nominal_feet_positions(model: RobotModel, body_goal: Dict[str, str]) -> Dict[str, List[float]]:
        """用 Pinocchio FK 从训练默认姿态计算脚端目标并应用语义抬脚偏移。"""
        import numpy as np
        import pinocchio as pin

        if model.urdf_path is None:
            raise ValueError("Go2 URDF path is unavailable")
        pin_model = pin.buildModelFromUrdf(str(model.urdf_path))
        data = pin_model.createData()
        configuration = pin.neutral(pin_model)
        for joint_name, target in model.default_joint_positions.items():
            joint_id = pin_model.getJointId(joint_name)
            if joint_id >= pin_model.njoints:
                raise ValueError("Pinocchio URDF has no configured joint %s" % joint_name)
            joint = pin_model.joints[joint_id]
            if joint.nq != 1:
                raise ValueError("Go2 expected single-DoF joint %s" % joint_name)
            configuration[joint.idx_q] = float(target)
        pin.forwardKinematics(pin_model, data, configuration)
        pin.updateFramePlacements(pin_model, data)
        lift_height = float(model.nominal_lift_height)
        lift_frames = set()
        for key, value in body_goal.items():
            normalized_key, normalized_value = key.lower(), value.lower()
            if "feet" in normalized_key and normalized_value == "lift":
                if "front" in normalized_key:
                    lift_frames.update(frame for frame in model.foot_frames
                                       if frame.startswith(("FL_", "FR_")))
                elif "hind" in normalized_key or "rear" in normalized_key:
                    lift_frames.update(frame for frame in model.foot_frames
                                       if frame.startswith(("RL_", "RR_")))
                else:
                    lift_frames.update(model.foot_frames)
        positions: Dict[str, List[float]] = {}
        for frame_name in model.foot_frames:
            frame_id = pin_model.getFrameId(frame_name)
            if frame_id >= pin_model.nframes:
                raise ValueError("Go2 URDF lacks configured foot frame %s" % frame_name)
            position = data.oMf[frame_id].translation.copy()
            if frame_name in lift_frames:
                position[2] += lift_height
            positions[frame_name] = [float(item) for item in np.asarray(position).reshape(3)]
        return positions
