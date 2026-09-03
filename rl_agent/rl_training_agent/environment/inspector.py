from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from ..schemas.task import TaskSpec
from ..utils.io import write_json
from ..utils.paths import relative_display
from .capability_manifest import CapabilityManifest, EnvironmentVariable
from .reward_registry import RewardRegistry


SOURCE = "../unitree_rl_gym/legged_gym/envs/base/legged_robot.py"


def _variable(name: str, symbol: str, shape: List[object], unit: str, frame: str,
              policy: bool, simulation_only: bool = False, derivation: str = "direct") -> EnvironmentVariable:
    """构造一条环境变量能力描述。"""
    return EnvironmentVariable(name=name, source_file=SOURCE, source_symbol=symbol, shape=shape, unit=unit,
                               coordinate_frame=frame, normalized=False, available_to_policy=policy,
                               available_to_reward=True, simulation_only=simulation_only, derivation=derivation)


class EnvironmentInspector:
    def __init__(self, training_root: Path):
        """初始化 EnvironmentInspector 实例及其运行依赖。"""
        self.training_root = training_root

    def inspect(self, robot: str = "go2") -> CapabilityManifest:
        """检查实际项目并返回机器可读的能力清单。"""
        robots = [name for name in ("go2", "h1", "h1_2", "g1")
                  if (self.training_root / "legged_gym" / "envs" / name).is_dir()]
        variables = [
            _variable("base_pos", "LeggedRobot.base_pos", ["num_envs", 3], "m", "world", False),
            _variable("base_quat", "LeggedRobot.base_quat", ["num_envs", 4], "quaternion", "world", False),
            _variable("rpy", "LeggedRobot.rpy", ["num_envs", 3], "rad", "world", False, derivation="base_quat"),
            _variable("base_lin_vel", "LeggedRobot.base_lin_vel", ["num_envs", 3], "m/s", "body", True),
            _variable("base_ang_vel", "LeggedRobot.base_ang_vel", ["num_envs", 3], "rad/s", "body", True),
            _variable("projected_gravity", "LeggedRobot.projected_gravity", ["num_envs", 3], "normalized", "body", True),
            _variable("commands", "LeggedRobot.commands", ["num_envs", 4], "mixed", "body", True),
            _variable("dof_pos", "LeggedRobot.dof_pos", ["num_envs", "num_dof"], "rad", "joint", True),
            _variable("dof_vel", "LeggedRobot.dof_vel", ["num_envs", "num_dof"], "rad/s", "joint", True),
            _variable("actions", "LeggedRobot.actions", ["num_envs", "num_actions"], "normalized", "joint", True),
            _variable("torques", "LeggedRobot.torques", ["num_envs", "num_actions"], "N*m", "joint", False, True),
            _variable("contact_forces", "LeggedRobot.contact_forces", ["num_envs", "num_bodies", 3], "N", "world", False, True),
            _variable("feet_contacts", "contact_forces[:, feet_indices, 2] > threshold", ["num_envs", "num_feet"], "bool", "world", False, True, "contact_forces"),
            _variable("feet_air_time", "LeggedRobot.feet_air_time", ["num_envs", "num_feet"], "s", "world", False, True),
        ]
        rewards = RewardRegistry(self.training_root).inspect(robot)
        return CapabilityManifest(
            project="../unitree_rl_gym", robot=robot, robots=robots,
            observations=[item for item in variables if item.available_to_policy], reward_variables=variables,
            rewards=rewards,
            terminations=["forbidden body contact > 1 N", "abs(pitch) > 1.0 rad", "abs(roll) > 0.8 rad", "episode timeout"],
            command_space=["lin_vel_x", "lin_vel_y", "ang_vel_yaw", "heading"],
            checkpoint_format="torch.save dict: model_state_dict, optimizer_state_dict, iter, infos",
            logger="TensorBoard SummaryWriter; Episode/rew_<name>, Loss/*, Policy/*, Perf/*, Train/*",
            training_entry="../unitree_rl_gym/legged_gym/scripts/train.py:train",
            evaluation_entry="../unitree_rl_gym/legged_gym/scripts/play.py:play",
            evaluation_metrics=[
                "tracking_error", "tracking_lin_vel", "episode_survival", "fall_rate",
                "max_abs_roll", "max_abs_pitch", "roll_limit", "pitch_limit",
                "forbidden_contact_force", "forbidden_collisions", "body_collision_count",
                "abnormal_terminations", "rear_stand_duration", "front_contact_fraction",
                "jump_height", "orientation", "energy", "nan_count",
                "joint_limit_violations", "torque_limit_violations",
                "rear_leg_stand_duration", "rear_leg_walk_completion",
                "rear_leg_walk_velocity_tracking", "front_leg_off_ground_ratio",
                "forbidden_body_contact",
                "stable_stand_duration", "walking_speed_tracking", "body_pitch_within_limit",
                "front_leg_stand_duration", "front_leg_walk_completion",
                "front_leg_walk_velocity_tracking", "rear_leg_off_ground_ratio",
            ])

    def write(self, path: Path, robot: str = "go2") -> CapabilityManifest:
        """将检查结果写入指定的机器可读文件。"""
        manifest = self.inspect(robot)
        write_json(path, manifest)
        return manifest

    def validate_task(self, task: TaskSpec, manifest: CapabilityManifest) -> Tuple[List[str], List[str]]:
        """核对任务所需物理量是否可用、可推导或不支持。"""
        available = {item.name for item in manifest.reward_variables}
        aliases: Dict[str, str] = {"foot_contacts": "feet_contacts", "base_height": "base_pos",
                                   "roll": "rpy", "pitch": "rpy", "yaw": "rpy"}
        unsupported: List[str] = []
        derivable: List[str] = []
        for requirement in task.required_observations:
            if requirement in available:
                continue
            if requirement in aliases and aliases[requirement] in available:
                derivable.append(requirement)
            else:
                unsupported.append(requirement)
        supported_metrics = set(manifest.evaluation_metrics)
        for threshold in list(task.success_metrics) + list(task.safety_constraints):
            if threshold.required and threshold.name not in supported_metrics:
                unsupported.append("evaluation_metric:%s" % threshold.name)
        task.unsupported_requirements = unsupported
        task.derivable_requirements = derivable
        return unsupported, derivable
