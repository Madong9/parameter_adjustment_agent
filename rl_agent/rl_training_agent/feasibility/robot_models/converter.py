"""把使用标准基本碰撞体的 URDF 机器人转换为可加载的 MuJoCo MJCF。"""
from __future__ import annotations

import hashlib
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class URDFConversionError(ValueError):
    """表示输入 URDF 不在当前安全转换器支持范围内。"""


class URDFToMJCFConverter:
    """保留 URDF 的拓扑、惯性、关节限制和基本碰撞几何生成 MJCF。"""

    SUPPORTED_GEOMETRIES = {"box", "cylinder", "sphere"}

    def convert(self, urdf_path: Path, output_path: Path,
                robot_config: Dict[str, Any],
                source_relative: Optional[str] = None) -> Dict[str, Any]:
        """转换 URDF、用真实 MuJoCo 编译校验并原子保存结果和来源元数据。"""
        source = Path(urdf_path).resolve()
        output = Path(output_path).resolve()
        if not source.is_file():
            raise URDFConversionError("URDF 文件不存在：%s" % source.name)
        try:
            import mujoco
        except ImportError as exc:
            raise URDFConversionError("MuJoCo 不可用，不能验证转换结果：%s" % exc)
        source_bytes = source.read_bytes()
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        try:
            root = ET.fromstring(source_bytes)
        except ET.ParseError as exc:
            raise URDFConversionError("URDF XML 无法解析：%s" % exc)
        xml_text = self._build_mjcf(root, robot_config)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", suffix=".xml",
                    prefix=".%s." % output.stem, dir=str(output.parent), delete=False) as temporary:
                temporary.write(xml_text)
                temporary_path = Path(temporary.name)
            model = mujoco.MjModel.from_xml_path(str(temporary_path))
            if model.njnt <= 0 or model.nu <= 0:
                raise URDFConversionError("转换模型缺少可动关节或 actuator")
            expected_actuators = [
                name for name in robot_config.get("default_joint_positions", {})
            ]
            if expected_actuators and model.nu != len(expected_actuators):
                raise URDFConversionError(
                    "转换 actuator 数量不匹配：expected=%d actual=%d" %
                    (len(expected_actuators), model.nu))
            os.replace(str(temporary_path), str(output))
            temporary_path = None
        except URDFConversionError:
            raise
        except Exception as exc:
            raise URDFConversionError("MuJoCo 编译转换模型失败：%s" % str(exc)[:600])
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except OSError:
                    pass
        output_hash = hashlib.sha256(output.read_bytes()).hexdigest()
        metadata = {
            "format": "urdf_to_mjcf_v1",
            "source_urdf": source_relative or source.name,
            "source_sha256": source_hash,
            "generated_mjcf": output.name,
            "generated_sha256": output_hash,
            "converted_model": True,
            "mujoco_version": getattr(mujoco, "__version__", "unknown"),
            "collision_geometry_source": "URDF collision elements; visual DAE meshes omitted",
        }
        metadata_path = output.with_suffix(output.suffix + ".meta.json")
        metadata_path.write_text(
            __import__("json").dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        return metadata

    def _build_mjcf(self, urdf: ET.Element, robot_config: Dict[str, Any]) -> str:
        """从 URDF 根链构造仅使用已支持 URDF 元素的 MJCF 文档。"""
        links = {item.get("name"): item for item in urdf.findall("link") if item.get("name")}
        joints = urdf.findall("joint")
        children: Dict[str, List[ET.Element]] = {}
        child_links = set()
        movable_joints: List[ET.Element] = []
        for joint in joints:
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                raise URDFConversionError("URDF joint 缺少 parent 或 child")
            parent_name, child_name = parent.get("link"), child.get("link")
            if parent_name not in links or child_name not in links:
                raise URDFConversionError("URDF joint 引用了不存在的 link")
            children.setdefault(parent_name, []).append(joint)
            child_links.add(child_name)
            if joint.get("type") != "fixed":
                movable_joints.append(joint)
        roots = sorted(set(links) - child_links)
        if len(roots) != 1:
            raise URDFConversionError("URDF 必须具有唯一根 link，实际为 %d" % len(roots))

        mujoco_root = ET.Element("mujoco", {"model": str(urdf.get("name") or "robot")})
        ET.SubElement(mujoco_root, "compiler", {
            "angle": "radian", "coordinate": "local", "autolimits": "true",
            "inertiafromgeom": "false",
        })
        ET.SubElement(mujoco_root, "option", {
            "timestep": "0.002", "gravity": "0 0 -9.81", "integrator": "implicitfast",
            "iterations": "50",
        })
        default = ET.SubElement(mujoco_root, "default")
        ET.SubElement(default, "joint", {"damping": "0.5", "armature": "0.01"})
        ET.SubElement(default, "geom", {
            "friction": "0.8 0.02 0.001", "condim": "3", "density": "0",
        })
        worldbody = ET.SubElement(mujoco_root, "worldbody")
        ET.SubElement(worldbody, "light", {"pos": "0 0 3", "dir": "0 0 -1"})
        ET.SubElement(worldbody, "geom", {
            "name": "ground", "type": "plane", "pos": "0 0 0",
            "size": "5 5 0.1", "rgba": "0.8 0.8 0.8 1", "friction": "1.0 0.02 0.001",
        })
        root_body = ET.SubElement(worldbody, "body", {
            "name": roots[0], "pos": "0 0 %.9g" % float(robot_config.get("base_initial_height", 0.42)),
        })
        ET.SubElement(root_body, "freejoint", {"name": "root_free"})
        self._add_link_geometry(root_body, links[roots[0]], roots[0])
        visited = {roots[0]}
        for joint in children.get(roots[0], []):
            self._add_child(joint, children, links, root_body, visited)
        if visited != set(links):
            raise URDFConversionError("URDF 包含不可达或循环 link：%s" %
                                      ", ".join(sorted(set(links) - visited)))

        actuators = ET.SubElement(mujoco_root, "actuator")
        defaults = robot_config.get("default_joint_positions", {})
        kp = float(robot_config.get("position_kp", 20.0))
        kd = float(robot_config.get("position_kd", 0.5))
        joint_map = {joint.get("name"): joint for joint in movable_joints}
        for joint_name in defaults:
            joint = joint_map.get(joint_name)
            if joint is None:
                raise URDFConversionError("配置中的 actuator joint 不在 URDF 中：%s" % joint_name)
            limit = joint.find("limit")
            attributes = {"name": "position_" + joint_name, "joint": joint_name,
                          "kp": "%.9g" % kp, "kv": "%.9g" % kd}
            if limit is not None and limit.get("lower") is not None and limit.get("upper") is not None:
                attributes["ctrlrange"] = "%s %s" % (limit.get("lower"), limit.get("upper"))
                attributes["ctrllimited"] = "true"
            if limit is not None and limit.get("effort") is not None:
                effort = abs(float(limit.get("effort")))
                attributes["forcerange"] = "%.9g %.9g" % (-effort, effort)
                attributes["forcelimited"] = "true"
            ET.SubElement(actuators, "position", attributes)
        return ET.tostring(mujoco_root, encoding="unicode", xml_declaration=True)

    def _add_child(self, joint: ET.Element, children: Dict[str, List[ET.Element]],
                   links: Dict[str, ET.Element], parent_body: ET.Element,
                   visited: set) -> None:
        """递归转换 URDF joint 子树并保留 fixed link 的位姿变换。"""
        parent = joint.find("parent")
        child = joint.find("child")
        parent_name, child_name = parent.get("link"), child.get("link")
        if child_name in visited:
            raise URDFConversionError("URDF 中检测到重复或循环 child link：%s" % child_name)
        visited.add(child_name)
        origin = joint.find("origin")
        body_attributes = {"name": child_name}
        if origin is not None:
            body_attributes["pos"] = self._vector(origin.get("xyz"), (0.0, 0.0, 0.0))
            body_attributes["euler"] = self._vector(origin.get("rpy"), (0.0, 0.0, 0.0))
        body = ET.SubElement(parent_body, "body", body_attributes)
        joint_type = joint.get("type", "fixed")
        if joint_type != "fixed":
            if joint_type not in ("revolute", "continuous", "prismatic"):
                raise URDFConversionError("暂不支持 URDF joint 类型：%s" % joint_type)
            axis = joint.find("axis")
            axis_value = self._vector(axis.get("xyz") if axis is not None else None,
                                      (0.0, 0.0, 1.0))
            joint_attributes = {"name": str(joint.get("name")), "axis": axis_value}
            joint_attributes["type"] = "slide" if joint_type == "prismatic" else "hinge"
            limit = joint.find("limit")
            if joint_type != "continuous" and limit is not None:
                if limit.get("lower") is not None and limit.get("upper") is not None:
                    joint_attributes["range"] = "%s %s" % (limit.get("lower"), limit.get("upper"))
                    joint_attributes["limited"] = "true"
            ET.SubElement(body, "joint", joint_attributes)
        self._add_link_geometry(body, links[child_name], child_name)
        for descendant in children.get(child_name, []):
            self._add_child(descendant, children, links, body, visited)

    def _add_link_geometry(self, body: ET.Element, link: ET.Element, link_name: str) -> None:
        """转换每个 link 的惯性和 URDF 基本碰撞体，忽略仅用于显示的网格。"""
        inertial = link.find("inertial")
        if inertial is not None:
            mass = inertial.find("mass")
            inertia = inertial.find("inertia")
            origin = inertial.find("origin")
            if mass is None or inertia is None:
                raise URDFConversionError("link 缺少完整惯性描述：%s" % link_name)
            mass_value = float(mass.get("value", "0"))
            if mass_value <= 0.0:
                if link.findall("collision"):
                    raise URDFConversionError("存在带碰撞几何的零质量 link：%s" % link_name)
                inertial = None
        if inertial is not None:
            mass = inertial.find("mass")
            inertia = inertial.find("inertia")
            origin = inertial.find("origin")
            inertial_attributes = {"mass": str(mass.get("value"))}
            if origin is not None:
                inertial_attributes["pos"] = self._vector(origin.get("xyz"), (0.0, 0.0, 0.0))
            inertial_rpy = self._vector_values(origin.get("rpy") if origin is not None else None)
            values = self._rotate_inertia(inertia, inertial_rpy)
            inertial_attributes["fullinertia"] = self._numbers(values)
            ET.SubElement(body, "inertial", inertial_attributes)
        for index, collision in enumerate(link.findall("collision")):
            geometry = collision.find("geometry")
            if geometry is None or len(geometry) != 1:
                raise URDFConversionError("collision geometry 缺失或含多个元素：%s" % link_name)
            primitive = geometry[0]
            if primitive.tag not in self.SUPPORTED_GEOMETRIES:
                raise URDFConversionError("不支持 %s 的 URDF collision geometry：%s" %
                                          (link_name, primitive.tag))
            attributes = {
                "name": "%s_collision_%d" % (link_name, index),
                "type": primitive.tag,
                "contype": "1", "conaffinity": "1",
                "rgba": "0.35 0.45 0.55 1",
            }
            if primitive.tag == "box":
                size = [float(item) * 0.5 for item in primitive.get("size", "").split()]
                if len(size) != 3:
                    raise URDFConversionError("URDF box size 必须包含三个数值")
                attributes["size"] = self._numbers(size)
            elif primitive.tag == "cylinder":
                attributes["size"] = "%s %s" % (primitive.get("radius"),
                                                    float(primitive.get("length")) * 0.5)
            else:
                attributes["size"] = str(primitive.get("radius"))
            origin = collision.find("origin")
            if origin is not None:
                attributes["pos"] = self._vector(origin.get("xyz"), (0.0, 0.0, 0.0))
                attributes["euler"] = self._vector(origin.get("rpy"), (0.0, 0.0, 0.0))
            ET.SubElement(body, "geom", attributes)

    @staticmethod
    def _vector(value: Optional[str], default: Tuple[float, float, float]) -> str:
        """格式化并校验 URDF 三维向量。"""
        values = URDFToMJCFConverter._vector_values(value, default)
        if len(values) != 3 or not all(math.isfinite(item) for item in values):
            raise URDFConversionError("URDF 三维向量格式无效：%s" % value)
        return URDFToMJCFConverter._numbers(values)

    @staticmethod
    def _vector_values(value: Optional[str],
                       default: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> List[float]:
        """解析 URDF 三维向量为有限浮点数组。"""
        values = list(default) if value is None else [float(item) for item in value.split()]
        if len(values) != 3 or not all(math.isfinite(item) for item in values):
            raise URDFConversionError("URDF 三维向量格式无效：%s" % value)
        return values

    @staticmethod
    def _rotate_inertia(inertia: ET.Element,
                        rpy: List[float]) -> List[float]:
        """将 URDF 惯性张量从 inertial frame 旋转到 MJCF body frame。"""
        values = {key: float(inertia.get(key, "0"))
                  for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
        matrix = [
            [values["ixx"], values["ixy"], values["ixz"]],
            [values["ixy"], values["iyy"], values["iyz"]],
            [values["ixz"], values["iyz"], values["izz"]],
        ]
        roll, pitch, yaw = rpy
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        rotation = [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
        intermediate = [[sum(rotation[i][k] * matrix[k][j] for k in range(3))
                         for j in range(3)] for i in range(3)]
        rotated = [[sum(intermediate[i][k] * rotation[j][k] for k in range(3))
                    for j in range(3)] for i in range(3)]
        result = [rotated[0][0], rotated[1][1], rotated[2][2],
                  rotated[0][1], rotated[0][2], rotated[1][2]]
        if not all(math.isfinite(item) for item in result):
            raise URDFConversionError("URDF inertia 矩阵含有非有限数值")
        return result

    @staticmethod
    def _numbers(values: Union[Tuple[float, ...], List[float]]) -> str:
        """以稳定精度输出 MJCF 数值列表。"""
        return " ".join("%.9g" % float(item) for item in values)
