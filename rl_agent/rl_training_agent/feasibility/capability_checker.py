"""使用本地机器人资产和环境能力清单做确定性前置检查。"""
from __future__ import annotations

import ast
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schema import CapabilityReport, FeasibilityCheck, FeasibilityStatus


class RobotAssetInventory:
    """只从仓库内已配置的 URDF/机器人配置读取关节，不猜测名称。"""

    ROBOTS = {
        "go2": ("resources/robots/go2/urdf/go2.urdf", "legged_gym/envs/go2/go2_config.py"),
        "h1": ("resources/robots/h1/urdf/h1.urdf", "legged_gym/envs/h1/h1_config.py"),
        "h1_2": ("resources/robots/h1_2/h1_2_12dof.urdf", "legged_gym/envs/h1_2/h1_2_config.py"),
        "g1": ("resources/robots/g1_description/g1_12dof.urdf", "legged_gym/envs/g1/g1_config.py"),
    }

    def __init__(self, training_root: Path):
        """保存训练仓库根目录。"""
        self.training_root = training_root

    def inspect(self, robot: str) -> Dict[str, Any]:
        """提取配置 URDF 的活动关节和机器人配置中的默认受控关节。"""
        relative = self.ROBOTS.get(robot)
        if relative is None:
            return {"available": False, "joints": [], "actuators": [], "evidence": []}
        urdf_path = self.training_root / relative[0]
        config_path = self.training_root / relative[1]
        evidence: List[str] = []
        joints: List[str] = []
        actuators: List[str] = []
        if urdf_path.is_file():
            try:
                root = ET.parse(str(urdf_path)).getroot()
                joints = [item.get("name", "") for item in root.findall("joint")
                          if item.get("name") and item.get("type") != "fixed"]
                evidence.append(urdf_path.relative_to(self.training_root).as_posix())
            except (ET.ParseError, OSError):
                pass
        if config_path.is_file():
            try:
                tree = ast.parse(config_path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    value = None
                    if isinstance(node, ast.Assign) and any(
                            isinstance(target, ast.Name) and target.id == "default_joint_angles"
                            for target in node.targets):
                        value = node.value
                    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                            and node.target.id == "default_joint_angles":
                        value = node.value
                    if value is not None:
                        parsed = ast.literal_eval(value)
                        if isinstance(parsed, dict):
                            actuators = sorted(str(name) for name in parsed)
                evidence.append(config_path.relative_to(self.training_root).as_posix())
            except (SyntaxError, OSError, ValueError):
                pass
        missing_actuators = sorted(set(actuators) - set(joints))
        return {
            "available": bool(joints and actuators and not missing_actuators), "joints": joints,
            "actuators": actuators, "missing_actuators": missing_actuators, "evidence": evidence,
        }


class CapabilityChecker:
    """比较任务所需能力与实际机器人、观测、命令及指标注册表。"""

    def __init__(self, training_root: Path):
        """创建基于仓库资产的确定性能力检查器。"""
        self.training_root = training_root
        self.assets = RobotAssetInventory(training_root)

    @staticmethod
    def _task_text(intent: Any) -> str:
        """汇总正向目标与必需行为；禁止行为只用于安全约束，不能反向定义动作。"""
        return " ".join([
            intent.original_instruction, intent.action_name, intent.normalized_goal,
            " ".join(intent.required_behaviors),
        ]).lower()

    @staticmethod
    def _command_range(text: str, name: str) -> Optional[Tuple[float, float]]:
        """从环境配置源码读取命令区间，拒绝执行配置代码。"""
        match = re.search(r"^\s*%s\s*=\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]" % name,
                          text, re.MULTILINE)
        return (float(match.group(1)), float(match.group(2))) if match else None

    def assess(self, intent: Any, manifest: Any) -> CapabilityReport:
        """逐项验证任务需求；未知证据不会被当作能力存在。"""
        data = manifest.dict() if hasattr(manifest, "dict") else dict(manifest)
        robot = intent.robot.lower()
        text = self._task_text(intent)
        checks: List[FeasibilityCheck] = []
        missing: List[str] = []
        risks: List[str] = []
        required: List[str] = []
        unsupported = False
        conditional = False
        known_robots = {str(item).lower() for item in data.get("robots", [])}
        robot_declared = str(data.get("robot", "")).lower()
        robot_ok = robot in known_robots and robot_declared == robot
        required.append("robot:" + robot)
        checks.append(FeasibilityCheck(
            name="robot_model", status=(FeasibilityStatus.SUPPORTED if robot_ok else
                FeasibilityStatus.CONDITIONAL if not known_robots else FeasibilityStatus.UNSUPPORTED),
            summary="机器人型号在当前训练项目中注册" if robot_ok else
                "能力清单未发现机器人型号" if not known_robots else "机器人型号未在当前环境注册",
            evidence=sorted(known_robots)))
        if not robot_ok:
            if known_robots:
                unsupported = True
            else:
                conditional = True
                missing.append("无法读取环境机器人注册表")

        inventory = self.assets.inspect(robot)
        if inventory["available"]:
            checks.append(FeasibilityCheck(
                name="joints_and_actuators", status=FeasibilityStatus.SUPPORTED,
                summary="从机器人 URDF 和环境默认关节配置读取到关节及受控关节",
                evidence=inventory["evidence"]))
        else:
            conditional = True
            missing.append("无法从当前仓库资产确认机器人关节和 actuator 配置")
            checks.append(FeasibilityCheck(
                name="joints_and_actuators", status=FeasibilityStatus.CONDITIONAL,
                summary="机器人关节/执行器资产缺失、关节映射不一致或无法解析",
                evidence=inventory["evidence"] + ["unmatched actuators=" +
                    ",".join(inventory.get("missing_actuators", []))]))

        manipulation = any(token in text for token in ("机械臂", "抓取", "抓住", "grasp", "manipulat", "arm"))
        if manipulation:
            required.append("arm_or_gripper_actuators")
            arm_joints = [name for name in inventory["actuators"]
                          if any(token in name.lower() for token in ("arm", "shoulder", "elbow", "wrist"))]
            gripper_joints = [name for name in inventory["actuators"]
                              if any(token in name.lower() for token in ("gripper", "finger", "hand"))]
            needs_gripper = any(token in text for token in ("抓取", "抓住", "grasp"))
            manipulation_supported = bool(arm_joints and (gripper_joints or not needs_gripper))
            if not manipulation_supported and inventory["available"]:
                unsupported = True
                missing.append("当前机器人受控关节缺少该动作所需的" +
                               ("机械臂与夹爪" if needs_gripper else "机械臂"))
                checks.append(FeasibilityCheck(
                    name="manipulation_actuators", status=FeasibilityStatus.UNSUPPORTED,
                    summary="动作所需机械臂/夹爪 actuator 未在机器人配置中注册",
                    evidence=inventory["evidence"] + arm_joints + gripper_joints))
            elif manipulation_supported:
                checks.append(FeasibilityCheck(
                    name="manipulation_actuators", status=FeasibilityStatus.SUPPORTED,
                    summary="找到动作所需的机械臂/夹爪受控关节",
                    evidence=arm_joints + gripper_joints))
            else:
                conditional = True
                missing.append("机器人资产不可完整解析，需确认机械臂/夹爪 actuator")

        variable_items = data.get("reward_variables", [])
        variables = {str(item.get("name")) for item in variable_items}
        policy_observations = {
            str(item.get("name")) for item in variable_items
            if item.get("available_to_policy") is True
        }
        locomotion_text = any(token in text for token in ("行走", "走路", "倒退", "前进", "跑", "walk", "run", "locomotion"))
        if locomotion_text or any(token in text for token in ("平衡", "站立", "upright", "balance")):
            required.append("observation:base_orientation")
            orientation_evidence = sorted({"rpy", "base_quat", "projected_gravity"} & policy_observations)
            orientation_ok = bool(orientation_evidence)
            checks.append(FeasibilityCheck(
                name="orientation_observation",
                status=FeasibilityStatus.SUPPORTED if orientation_ok else FeasibilityStatus.CONDITIONAL,
                summary="具备躯干姿态观测" if orientation_ok else "未确认躯干姿态观测",
                evidence=orientation_evidence))
            if not orientation_ok:
                conditional = True
                missing.append("base_orientation observation")
        if locomotion_text:
            commands = set(data.get("command_space", []))
            required.append("command:lin_vel_x")
            command_ok = "lin_vel_x" in commands
            command_out_of_range = False
            requested_speed = intent.target_velocity
            speed_match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:m\s*/\s*s|米每秒|米/秒)", text)
            if requested_speed is None and speed_match:
                requested_speed = float(speed_match.group(1))
            if requested_speed is not None and requested_speed > 0.0 and any(
                    token in text for token in ("倒退", "后退", "向后", "backward", "reverse")):
                requested_speed = -abs(float(requested_speed))
            range_evidence: List[str] = []
            if requested_speed is not None:
                config_paths = list((self.training_root / "legged_gym" / "envs" / robot).glob("*_config.py"))
                config_paths.append(self.training_root / "legged_gym" / "envs" / "base" /
                                    "legged_robot_config.py")
                source = "\n".join(path.read_text(encoding="utf-8") for path in config_paths if path.is_file())
                command_range = self._command_range(source, "lin_vel_x")
                if command_range:
                    command_out_of_range = not (command_range[0] <= requested_speed <= command_range[1])
                    command_ok = command_ok and not command_out_of_range
                    range_evidence.append("target lin_vel_x=%.3f; configured range=%s" % (
                        requested_speed, command_range))
                else:
                    conditional = True
                    missing.append("lin_vel_x command range 未能从配置中确认")
            checks.append(FeasibilityCheck(
                name="locomotion_command", status=(FeasibilityStatus.SUPPORTED if command_ok else
                    FeasibilityStatus.UNSUPPORTED if "lin_vel_x" not in commands or command_out_of_range
                    else FeasibilityStatus.CONDITIONAL),
                summary="命令空间支持目标线速度" if command_ok else "目标线速度超出或无法确认命令范围",
                evidence=sorted(commands) + range_evidence))
            if not command_ok:
                if "lin_vel_x" not in commands or command_out_of_range:
                    unsupported = True
                    missing.append("command:lin_vel_x 超出配置范围" if command_out_of_range else
                                   "command:lin_vel_x")
                else:
                    conditional = True
                    missing.append("目标速度超出配置范围或范围证据不足")

        metrics = set(data.get("evaluation_metrics", []))
        locomotion = any(item.endswith(":lin_vel_x") for item in required)
        rear_leg_goal = any(token in text for token in ("后腿", "rear_leg", "hind_leg"))
        front_leg_goal = any(token in text for token in ("前腿站立", "前腿行走", "front_leg_stand", "front_leg_walk"))
        jump_goal = any(token in text for token in ("跳", "jump"))
        if rear_leg_goal and locomotion:
            success_options = {"rear_leg_walk_completion", "rear_leg_walk_velocity_tracking"}
        elif rear_leg_goal:
            success_options = {"rear_leg_stand_duration", "rear_stand_duration"}
        elif front_leg_goal and locomotion:
            success_options = {"front_leg_walk_completion", "front_leg_walk_velocity_tracking",
                               "front_leg_forward_speed"}
        elif front_leg_goal:
            success_options = {"front_leg_stand_duration"}
        elif jump_goal:
            success_options = {"jump_height"}
        elif locomotion:
            success_options = {"tracking_lin_vel", "tracking_error", "walking_speed_tracking"}
        else:
            success_options = {"stable_stand_duration", "rear_stand_duration", "front_leg_stand_duration"}
        success_found = sorted(success_options & metrics)
        failure_options = {"fall_rate", "forbidden_collisions", "abnormal_terminations"}
        failure_found = sorted(failure_options & metrics)
        metric_ok = bool(success_found and failure_found)
        required.extend(["evaluation:task_success_metric", "evaluation:failure_metric"])
        checks.append(FeasibilityCheck(
            name="evaluation_metrics", status=FeasibilityStatus.SUPPORTED if metric_ok else FeasibilityStatus.CONDITIONAL,
            summary="成功指标和失败/安全指标均可用" if metric_ok else "任务成功或失败指标不完整",
            evidence=success_found + failure_found))
        if not metric_ok:
            conditional = True
            if not success_found:
                missing.append("task success metric")
            if not failure_found:
                missing.append("fall/failure metric")

        critical_ambiguities = [item for item in intent.ambiguities if any(
            token in item.lower() for token in ("方向不明", "目标冲突", "动作不明确", "无法区分", "左右不明"))]
        if intent.ambiguities:
            risks.extend("用户意图存在歧义：" + item for item in intent.ambiguities)
        if unsupported:
            status = FeasibilityStatus.UNSUPPORTED
        elif critical_ambiguities:
            status = FeasibilityStatus.NEEDS_CLARIFICATION
        elif conditional:
            status = FeasibilityStatus.CONDITIONAL
        else:
            status = FeasibilityStatus.SUPPORTED
        return CapabilityReport(status=status, required_capabilities=required, checks=checks,
                                missing_requirements=sorted(set(missing)), risks=risks)
