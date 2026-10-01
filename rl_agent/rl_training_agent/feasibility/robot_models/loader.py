"""加载相对路径机器人资产并提供环境物理后端能力检查。"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ...utils.paths import ensure_within
from .converter import URDFConversionError, URDFToMJCFConverter
from .go2 import Go2RobotModel
from .schema import EnvironmentCapabilityReport, RobotModel


class RobotModelUnavailableError(RuntimeError):
    """表示机器人模型来源缺失、转换失败或物理模型无法编译。"""


class RobotModelLoader:
    """依据 YAML 注册表统一解析真实 URDF、MJCF 与机器人关节配置。"""

    DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "config" / "robot_models.yaml"

    def __init__(self, training_root: Path, config_path: Optional[Path] = None,
                 converter: Optional[URDFToMJCFConverter] = None):
        """设置训练目录、机器人配置文件及可替换 URDF 转换器。"""
        self.training_root = Path(training_root).resolve()
        self.config_path = Path(config_path or self.DEFAULT_CONFIG).resolve()
        self.agent_root = self.config_path.parent.parent.resolve()
        self.converter = converter or URDFToMJCFConverter()

    def _config(self) -> Dict[str, Any]:
        """读取并校验相对路径机器人模型注册表。"""
        if not self.config_path.is_file():
            raise RobotModelUnavailableError("机器人模型配置不存在：%s" % self.config_path.name)
        try:
            payload = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise RobotModelUnavailableError("机器人模型配置无法读取：%s" % exc)
        robots = payload.get("robots") if isinstance(payload, dict) else None
        if not isinstance(robots, dict):
            raise RobotModelUnavailableError("机器人模型配置缺少 robots 映射")
        return robots

    def load_robot_model(self, robot_name: str, require_mjcf: bool = False) -> RobotModel:
        """加载机器人 URDF/config；只有显式要求时才解析或生成 MuJoCo MJCF。"""
        configurations = self._config()
        config = configurations.get(str(robot_name).lower())
        if not isinstance(config, dict):
            return RobotModel(robot_name=robot_name, model_status="MODEL_UNAVAILABLE",
                              model_error="模型注册表未配置机器人 %s" % robot_name)
        urdf_relative = str(config.get("urdf", "")).strip()
        if not urdf_relative:
            return RobotModel(robot_name=robot_name, model_status="MODEL_UNAVAILABLE",
                              model_error="robot_models.yaml 未配置 URDF")
        try:
            urdf_path = ensure_within(self.training_root / urdf_relative, self.training_root)
        except ValueError as exc:
            return RobotModel(robot_name=robot_name, model_status="MODEL_UNAVAILABLE",
                              model_error="URDF 路径越界：%s" % exc)
        if not urdf_path.is_file():
            return RobotModel(robot_name=robot_name, urdf_source=urdf_relative,
                              model_status="MODEL_UNAVAILABLE", model_error="URDF 文件不存在")
        model = RobotModel(
            robot_name=str(robot_name).lower(), urdf_path=urdf_path,
            urdf_source=urdf_relative,
            default_joint_positions={str(name): float(value) for name, value in
                                     config.get("default_joint_positions", {}).items()},
            base_initial_height=float(config.get("base_initial_height", 0.0)),
            nominal_lift_height=float(config.get("nominal_lift_height", 0.12)),
            foot_frames=[str(item) for item in config.get("foot_frames", [])],
        )
        if model.robot_name == "go2":
            model = Go2RobotModel.enrich(model, urdf_path, config)
            for actuator in model.actuators:
                actuator["kp"] = float(config.get("position_kp", 20.0))
                actuator["kd"] = float(config.get("position_kd", 0.5))
        else:
            model = self._read_generic_joint_metadata(model, urdf_path)
        if model.model_status == "MODEL_UNAVAILABLE":
            return model

        # Isaac Gym 预检和 Pinocchio IK 直接使用 PPO 环境/URDF，不应触发派生 MJCF。
        if not require_mjcf:
            return model

        mjcf_relative = str(config.get("mjcf", "")).strip()
        if mjcf_relative:
            try:
                mjcf_path = ensure_within(self.training_root / mjcf_relative, self.training_root)
            except ValueError as exc:
                model.model_status = "MODEL_UNAVAILABLE"
                model.model_error = "MJCF 路径越界：%s" % exc
                return model
            if not mjcf_path.is_file():
                model.model_status = "MODEL_UNAVAILABLE"
                model.model_error = "配置的 MJCF 不存在：%s" % mjcf_relative
                return model
            model.mjcf_path = mjcf_path
            model.mjcf_source = mjcf_relative
        elif bool(config.get("convert_urdf_to_mjcf", False)):
            output_relative = str(config.get("generated_mjcf", "")).strip()
            if not output_relative:
                model.model_status = "MODEL_UNAVAILABLE"
                model.model_error = "开启 URDF 转换但未配置 generated_mjcf 输出位置"
                return model
            try:
                output_path = ensure_within(self.config_path.parent / output_relative, self.agent_root)
                metadata = self.converter.convert(urdf_path, output_path, config,
                                                  source_relative=urdf_relative)
            except (ValueError, OSError, URDFConversionError) as exc:
                model.model_status = "MODEL_UNAVAILABLE"
                model.model_error = "MODEL_UNAVAILABLE: %s" % str(exc)[:600]
                return model
            model.mjcf_path = output_path
            model.mjcf_source = output_relative
            model.converted_model = bool(metadata.get("converted_model"))
            model.conversion_source = urdf_relative
            model.source_sha256 = metadata.get("source_sha256")
        else:
            model.model_status = "MODEL_UNAVAILABLE"
            model.model_error = "没有配置 MJCF，也未启用 URDF 转换"
            return model
        try:
            import xml.etree.ElementTree as ET
            mjcf_root = ET.parse(str(model.mjcf_path)).getroot()
            if not model.actuators:
                model.actuators = [{
                    "name": item.get("name", ""),
                    "joint": item.get("joint", ""),
                    "control": item.tag,
                } for item in mjcf_root.findall("./actuator/*")]
        except (ET.ParseError, OSError):
            model.model_status = "MODEL_UNAVAILABLE"
            model.model_error = "MJCF actuator 配置无法解析"
        return model

    def inspect_environment(self) -> EnvironmentCapabilityReport:
        """输出 Python、Go2 资产与物理后端可用性，不触发文件生成。"""
        mujoco_available = importlib.util.find_spec("mujoco") is not None
        pinocchio_available = importlib.util.find_spec("pinocchio") is not None
        mujoco_version = self._package_version("mujoco") if mujoco_available else None
        pin_version = (self._package_version("pin") or self._package_version("pinocchio")
                       if pinocchio_available else None)
        configs = self._config()
        go2 = configs.get("go2", {}) if isinstance(configs.get("go2", {}), dict) else {}
        urdf_relative = str(go2.get("urdf", ""))
        urdf_path = self.training_root / urdf_relative if urdf_relative else None
        isaacgym_package = self.training_root / "isaacgym" / "python" / "isaacgym"
        unitree_env_package = self.training_root / "legged_gym" / "envs"
        explicit_mjcf = str(go2.get("mjcf", "")).strip()
        mjcf_path = self.training_root / explicit_mjcf if explicit_mjcf else None
        existing_generated = str(go2.get("generated_mjcf", "")).strip()
        generated_path = self.config_path.parent / existing_generated if existing_generated else None
        actual_mjcf = mjcf_path if mjcf_path and mjcf_path.is_file() else (
            generated_path if generated_path and generated_path.is_file() else None)
        return EnvironmentCapabilityReport(
            python_version=platform.python_version(),
            mujoco_available=mujoco_available,
            mujoco_version=mujoco_version,
            pinocchio_available=pinocchio_available,
            pinocchio_version=pin_version,
            go2_urdf=urdf_relative if urdf_path and urdf_path.is_file() else "",
            go2_mjcf=(str(actual_mjcf.resolve().relative_to(self.agent_root).as_posix())
                      if actual_mjcf and actual_mjcf.is_file() else ""),
            conversion_needed=not bool(actual_mjcf and actual_mjcf.is_file()),
            conversion_available=mujoco_available and bool(urdf_path and urdf_path.is_file()),
            isaacgym_available=isaacgym_package.is_dir(),
            unitree_env_available=unitree_env_package.is_dir(),
            notes=[
                "Pinocchio is required for real IK; absence must not produce a physical pass.",
                "Go2 真实预检复用 Unitree Isaac Gym PPO 环境；派生 MJCF 不作为 Go2 物理验收依据。",
            ],
        )

    @staticmethod
    def _package_version(name: str) -> Optional[str]:
        """读取已安装 Python 包版本；对不可查询的本地包返回空值。"""
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    @staticmethod
    def _read_generic_joint_metadata(model: RobotModel, urdf_path: Path) -> RobotModel:
        """从非 Go2 URDF 读取通用关节名字及限制。"""
        import xml.etree.ElementTree as ET

        try:
            root = ET.parse(str(urdf_path)).getroot()
            names = []
            limits = {}
            for joint in root.findall("joint"):
                name = joint.get("name")
                if not name or joint.get("type") not in ("revolute", "continuous", "prismatic"):
                    continue
                names.append(name)
                element = joint.find("limit")
                if element is not None:
                    limits[name] = {
                        key: float(element.get(key)) if element.get(key) is not None else None
                        for key in ("lower", "upper", "velocity", "effort")
                    }
            model.joint_names = names
            model.limits = limits
            return model
        except (ET.ParseError, OSError, TypeError, ValueError) as exc:
            model.model_status = "MODEL_UNAVAILABLE"
            model.model_error = "URDF joint 元数据解析失败：%s" % exc
            return model


def load_robot_model(robot_name: str, training_root: Path,
                     config_path: Optional[Path] = None,
                     require_mjcf: bool = False) -> RobotModel:
    """提供调用方约定的统一机器人模型加载函数。"""
    return RobotModelLoader(training_root, config_path).load_robot_model(
        robot_name, require_mjcf=require_mjcf)
