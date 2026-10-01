"""定义机器人模型、关节限制和环境依赖能力报告。"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class JointLimit(BaseModel):
    """描述单个可控关节的 URDF 位置、速度和力矩限制。"""

    lower: Optional[float] = None
    upper: Optional[float] = None
    velocity: Optional[float] = None
    acceleration: Optional[float] = None
    effort: Optional[float] = None


class RobotModel(BaseModel):
    """提供供 IK 和仿真适配器共同使用的统一机器人模型视图。"""

    robot_name: str
    urdf_path: Optional[Path] = Field(default=None, exclude=True)
    mjcf_path: Optional[Path] = Field(default=None, exclude=True)
    urdf_source: str = ""
    mjcf_source: str = ""
    joint_names: List[str] = Field(default_factory=list)
    actuators: List[Dict[str, object]] = Field(default_factory=list)
    limits: Dict[str, JointLimit] = Field(default_factory=dict)
    default_joint_positions: Dict[str, float] = Field(default_factory=dict)
    base_initial_height: float = 0.0
    nominal_lift_height: float = 0.12
    foot_frames: List[str] = Field(default_factory=list)
    converted_model: bool = False
    conversion_source: Optional[str] = None
    source_sha256: Optional[str] = None
    model_status: str = "AVAILABLE"
    model_error: Optional[str] = None

    def runtime_dict(self) -> Dict[str, object]:
        """返回包含运行时解析路径的适配器输入字典。"""
        return {
            "robot_name": self.robot_name,
            "urdf_path": str(self.urdf_path) if self.urdf_path else None,
            "mujoco_xml": str(self.mjcf_path) if self.mjcf_path else None,
            "mjcf_path": str(self.mjcf_path) if self.mjcf_path else None,
            "mjcf_source": self.mjcf_source,
            "joint_names": list(self.joint_names),
            "actuators": [dict(item) for item in self.actuators],
            "limits": {name: (limit.dict() if hasattr(limit, "dict") else dict(limit))
                       for name, limit in self.limits.items()},
            "default_joint_positions": dict(self.default_joint_positions),
            "base_initial_height": self.base_initial_height,
            "nominal_lift_height": self.nominal_lift_height,
            "foot_frames": list(self.foot_frames),
            "converted_model": self.converted_model,
            "conversion_source": self.conversion_source,
            "source_sha256": self.source_sha256,
            "model_status": self.model_status,
            "model_error": self.model_error,
            "_descriptor": self,
        }

    def public_dict(self) -> Dict[str, object]:
        """返回不含主机绝对路径的可持久化机器人来源信息。"""
        values = self.dict()
        values["urdf_path"] = self.urdf_source
        values["mjcf_path"] = self.mjcf_source
        return values


class EnvironmentCapabilityReport(BaseModel):
    """记录当前解释器中的物理后端和 Go2 模型资产可用性。"""

    python_version: str
    mujoco_available: bool
    mujoco_version: Optional[str] = None
    pinocchio_available: bool
    pinocchio_version: Optional[str] = None
    go2_urdf: str = ""
    go2_mjcf: str = ""
    conversion_needed: bool = True
    conversion_available: bool = False
    isaacgym_available: bool = False
    unitree_env_available: bool = False
    notes: List[str] = Field(default_factory=list)
