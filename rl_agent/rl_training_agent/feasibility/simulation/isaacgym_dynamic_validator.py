"""在 Unitree PPO Isaac Gym 环境中执行 Pinocchio 足端轨迹的短时物理预检。"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..ik.pinocchio_solver import PinocchioIKSolver
from ..motion_prototype.schema import (DynamicMotionPrototype, GaitPattern,
                                       GaitType, MotionType)
from ..motion_prototype.foot_trajectory import FootTrajectoryGenerator
from .isaacgym_validator import IsaacGymFeasibilityValidator
from .schema import SimulationReport


class IsaacGymDynamicValidator(IsaacGymFeasibilityValidator):
    """复用 Unitree Go2 asset/PD/PhysX 参数执行有限动态 rollout。"""

    GO2_LEGS = ("FL", "FR", "RL", "RR")

    def __init__(self, training_root: Path, seed: int = 1, max_seconds: float = 5.0,
                 max_steps: int = 250, velocity_error_limit: float = 0.35,
                 foot_slip_limit: float = 0.25, no_contact_limit_seconds: float = 0.8,
                 environment_factory: Optional[Callable[..., Any]] = None,
                 ik_solver: Optional[Any] = None,
                 process_runner: Optional[Callable[..., Any]] = None,
                 visualize: bool = False):
        """设置 rollout 预算和动态验收阈值。"""
        super().__init__(training_root, seed=seed, max_seconds=max_seconds,
                         max_steps=max_steps, environment_factory=environment_factory,
                         visualize=visualize)
        self.backend = ("mock-isaacgym-dynamic" if environment_factory is not None else
                        "isaacgym")
        self.velocity_error_limit = max(0.01, float(velocity_error_limit))
        self.foot_slip_limit = max(0.01, float(foot_slip_limit))
        self.ik_solver = ik_solver or PinocchioIKSolver(max_iterations=40, tolerance=2.0e-3)
        self.no_contact_limit_seconds = max(0.02, float(no_contact_limit_seconds))
        self.process_runner = process_runner or subprocess.run

    def validate(self, robot_model: Any,
                 trajectory: DynamicMotionPrototype) -> SimulationReport:
        """把真实 Isaac Gym 调用隔离到子进程；Mock 测试替身仍在当前进程执行。"""
        if self.environment_factory is not None:
            return self._validate_in_process(robot_model, trajectory)
        if str(getattr(self.ik_solver, "backend", "")).lower() != "pinocchio":
            return self._unavailable("真实 Isaac Gym 预检要求 Pinocchio IK backend")
        if trajectory.robot.lower() != "go2" or str(
                self._value(robot_model, "robot_name", "")).lower() != "go2":
            return self._unavailable("当前动态探针只实现并校验了 Go2 四足关节映射")
        if trajectory.motion_type != MotionType.LOCOMOTION:
            return self._unavailable("真实动态预检只覆盖 LOCOMOTION，不验证 %s" %
                                      trajectory.motion_type.value)
        if trajectory.gait is None or trajectory.gait.type == GaitType.UNKNOWN:
            return self._unavailable("动态物理验证要求显式、受支持的 gait pattern")
        if trajectory.foot_trajectory is None:
            return self._unavailable("动态物理验证要求显式 FootTrajectory")
        if trajectory.velocity is None or abs(
                trajectory.velocity.duration - trajectory.duration) > 1.0e-6:
            return self._unavailable("动态物理验证要求与原型时长一致的 velocity trajectory")
        if trajectory.duration > self.max_seconds + 1.0e-9 or not trajectory.phases:
            return self._unavailable("动态轨迹缺少阶段或超出完整 rollout 预算")
        return self._validate_isolated(robot_model, trajectory)

    def _validate_isolated(self, robot_model: Any,
                           trajectory: DynamicMotionPrototype) -> SimulationReport:
        """启动独立 Python worker，捕获原生扩展崩溃、超时和无效报告。"""
        if isinstance(robot_model, dict):
            model_payload = {key: value for key, value in robot_model.items()
                             if key != "_descriptor"}
        elif hasattr(robot_model, "runtime_dict"):
            model_payload = robot_model.runtime_dict()
            model_payload.pop("_descriptor", None)
        else:
            return self._unavailable("真实 worker 需要可序列化的 RobotModel descriptor")
        solver_config = {
            "max_iterations": int(getattr(self.ik_solver, "max_iterations", 40)),
            "tolerance": float(getattr(self.ik_solver, "tolerance", 2.0e-3)),
            "damping": float(getattr(self.ik_solver, "damping", 1.0e-2)),
        }
        request = {
            "training_root": str(self.training_root),
            "robot_model": model_payload,
            "trajectory": trajectory.dict(),
            "seed": self.seed,
            "max_seconds": self.max_seconds,
            "max_steps": self.max_steps,
            "velocity_error_limit": self.velocity_error_limit,
            "foot_slip_limit": self.foot_slip_limit,
            "no_contact_limit_seconds": self.no_contact_limit_seconds,
            "visualize": self.visualize,
            "ik_solver": solver_config,
        }
        agent_root = Path(__file__).resolve().parents[3]
        command = [sys.executable, "-m",
                   "rl_training_agent.feasibility.isaacgym_worker"]
        timeout = max(30.0, self.max_seconds * 15.0 + 30.0)
        try:
            completed = self.process_runner(
                command, input=json.dumps(request, ensure_ascii=False, default=str),
                text=True, capture_output=True, timeout=timeout,
                cwd=str(agent_root), check=False,
            )
        except subprocess.TimeoutExpired:
            return self._unavailable(
                "Isaac Gym Feasibility worker 超时（%.1f 秒）；未启动 PPO，需检查环境或缩短预检预算" %
                timeout)
        except Exception as exc:
            return self._unavailable(
                "无法启动隔离的 Isaac Gym Feasibility worker：%s" % str(exc)[:300])
        if int(getattr(completed, "returncode", 1)) != 0:
            code = int(getattr(completed, "returncode", 1))
            detail = str(getattr(completed, "stderr", "") or "").strip()
            return self._unavailable(
                "Isaac Gym Feasibility worker 异常退出（exit code %s）；可能是原生扩展崩溃；"
                "未启动 PPO%s" % (code, "：" + detail[-300:] if detail else ""))
        output = str(getattr(completed, "stdout", "") or "")
        marker = "__DYNAMIC_FEASIBILITY_REPORT__"
        marker_index = output.rfind(marker)
        if marker_index < 0:
            return self._unavailable("Isaac Gym Feasibility worker 未返回结构化报告")
        try:
            payload = json.loads(output[marker_index + len(marker):].strip())
            return SimulationReport.parse_obj(payload)
        except (ValueError, TypeError) as exc:
            return self._unavailable(
                "Isaac Gym Feasibility worker 报告无法解析：%s" % str(exc)[:300])

    def _validate_in_process(self, robot_model: Any,
                             trajectory: DynamicMotionPrototype) -> SimulationReport:
        """在子进程中创建 Unitree 环境并执行实际短时物理 rollout。"""
        if trajectory.robot.lower() != "go2" or str(
                self._value(robot_model, "robot_name", "")).lower() != "go2":
            return self._unavailable("当前动态探针只实现并校验了 Go2 四足关节映射")
        if trajectory.motion_type != MotionType.LOCOMOTION:
            return self._unavailable(
                "动态 Isaac Gym 开环探针当前仅覆盖前进/后退线速度步态，不验证 %s" %
                trajectory.motion_type.value)
        if trajectory.gait is None or trajectory.gait.type == GaitType.UNKNOWN:
            return self._unavailable("动态物理验证要求显式、受支持的 gait pattern")
        if trajectory.foot_trajectory is None:
            return self._unavailable("动态物理验证要求显式四足 FootTrajectory；不会回退到固定关节波形")
        if trajectory.velocity is None or abs(
                trajectory.velocity.duration - trajectory.duration) > 1.0e-6:
            return self._unavailable("动态物理验证要求与原型时长一致的 velocity trajectory")
        if trajectory.duration > self.max_seconds + 1.0e-9:
            return self._unavailable("轨迹 %.2f 秒超过预检预算 %.2f 秒；拒绝截断后冒充完整验证" %
                                      (trajectory.duration, self.max_seconds))
        if not trajectory.phases:
            return self._unavailable("动态轨迹没有动作阶段")

        env = None
        try:
            if self.environment_factory is not None:
                import torch
                from types import SimpleNamespace
                gymtorch = SimpleNamespace(unwrap_tensor=lambda tensor: tensor,
                                           wrap_tensor=lambda tensor: tensor)
            else:
                torch, gymtorch = self._load_runtime_modules()
            expected_urdf = self._value(robot_model, "urdf_path")
            env = (self.environment_factory(self.training_root, trajectory.robot, self.seed)
                   if self.environment_factory else self._create_ppo_environment(
                       trajectory.robot, expected_urdf))
            self._validate_environment_identity(env, robot_model)
            self._validate_go2_dynamic_environment(env)
            target_speed = self._validate_command_range(env, trajectory)
            dof_names = [str(name) for name in env.dof_names[:env.num_actions]]
            reference_plan = None
            if self.backend == "isaacgym":
                reference_plan = self._compile_ik_reference_actions(
                    robot_model, trajectory, env, dof_names, torch)
                if not reference_plan.get("success"):
                    return SimulationReport(
                        status="UNAVAILABLE", success=None, backend=self.backend,
                        validated=False, validation_level="CAPABILITY_ONLY",
                        metrics=dict(reference_plan.get("metrics", {})),
                        reason=reference_plan.get("reason", "Pinocchio joint reference 编译失败"),
                    )
                actions = None
            else:
                # 测试替身只校验张量管线；该开环波形明确属于 Mock 证据。
                actions = self._compile_open_loop_actions(
                    env, dof_names, torch, target_speed, trajectory.gait)
                if actions is None:
                    return self._unavailable("Mock Go2 actuator/scale mapping is invalid")
            foot_states = self._acquire_foot_state(env, gymtorch)
            if foot_states is None:
                return self._unavailable("Isaac Gym 未提供刚体状态张量，无法验证足端滑移")
            self._set_exact_initial_state(env, torch, gymtorch)
            return self._rollout_dynamic(
                env, trajectory, actions, foot_states, torch, reference_plan)
        except Exception as exc:
            return self._unavailable("Isaac Gym 动态预检初始化/执行失败：%s" % str(exc)[:500])
        finally:
            self._destroy_environment(env)

    @staticmethod
    def _validate_go2_dynamic_environment(env: Any) -> None:
        """确认动态探针面对的是原生 12 自由度 Go2 Unitree 环境。"""
        names = [str(name) for name in getattr(env, "dof_names", [])]
        if int(getattr(env, "num_envs", 1)) != 1:
            raise RuntimeError("动态预检必须使用单环境，避免资源和证据混淆")
        if int(getattr(env, "num_actions", -1)) != 12 or int(
                getattr(env, "num_dof", -1)) != 12 or len(names) < 12:
            raise RuntimeError("Go2 动态探针要求 Unitree 配置中的 12 个受控自由度")
        if not all(any(name.startswith(leg + "_") for name in names)
                   for leg in ("FL", "FR", "RL", "RR")):
            raise RuntimeError("当前 actuator 名称未覆盖 Go2 四条腿")
        if len(getattr(env, "feet_indices", [])) != 4:
            raise RuntimeError("当前 Go2 环境没有提供四足接触索引")
        if len(getattr(env, "dof_vel_limits", [])) != 12:
            raise RuntimeError("Unitree 环境未提供 12 个关节速度限制")

    @staticmethod
    def _validate_command_range(env: Any, trajectory: DynamicMotionPrototype) -> float:
        """确保步态阶段与参数化速度轨迹一致且目标位于 Unitree 命令区间。"""
        moving = [phase.target_velocity.get("x", 0.0) for phase in trajectory.phases
                  if phase.name == "locomotion"]
        if len(moving) != 1:
            raise ValueError("当前动态探针需要且只支持一个 locomotion 阶段")
        speed = float(moving[0])
        if trajectory.velocity is None or abs(
                speed - float(trajectory.velocity.target_linear_velocity.get("x", float("inf")))
        ) > 1.0e-6:
            raise ValueError("velocity trajectory 与 locomotion phase 的 x 速度目标不一致")
        ranges = getattr(env, "command_ranges", {})
        limits = ranges.get("lin_vel_x") if isinstance(ranges, dict) else None
        if limits is None or len(limits) != 2:
            raise ValueError("无法从 Unitree Go2 环境读取 lin_vel_x 命令范围")
        lower, upper = float(limits[0]), float(limits[1])
        if not all(math.isfinite(value) for value in (speed, lower, upper)) or lower > upper:
            raise ValueError("Unitree lin_vel_x 范围或目标速度不是有效有限数值")
        if speed < lower - 1.0e-6 or speed > upper + 1.0e-6:
            raise ValueError("目标速度 %.3f m/s 超出 Unitree lin_vel_x 范围 %s" %
                             (speed, list(limits)))
        return speed

    @classmethod
    def _compile_open_loop_actions(cls, env: Any, dof_names: Sequence[str], torch: Any,
                                   target_speed: float, gait: GaitPattern):
        """校验参数化步态波形可映射到 PPO 归一化关节位置动作范围。"""
        if len(dof_names) != 12 or not all(
                all("%s_%s_joint" % (leg, suffix) in dof_names
                    for suffix in ("hip", "thigh", "calf"))
                for leg in cls.GO2_LEGS):
            return None
        if gait.type == GaitType.UNKNOWN or set(gait.phase_offsets) != set(cls.GO2_LEGS):
            return None
        scale = float(env.cfg.control.action_scale)
        clip = float(env.cfg.normalization.clip_actions)
        if not math.isfinite(scale) or scale <= 1.0e-12 or not math.isfinite(clip) or clip <= 0.0:
            return None
        amplitude = 0.25 + 0.30 * min(abs(float(target_speed)), 1.0)
        actions = torch.zeros((1, 12), dtype=torch.float, device=env.device)
        for leg in cls.GO2_LEGS:
            thigh_name = "%s_thigh_joint" % leg
            calf_name = "%s_calf_joint" % leg
            thigh_index = dof_names.index(thigh_name)
            calf_index = dof_names.index(calf_name)
            actions[0, thigh_index] = amplitude / scale
            actions[0, calf_index] = 1.5 * amplitude / scale
        if bool(torch.any(torch.abs(actions) > clip + 1.0e-6).item()):
            return None
        return actions

    @staticmethod
    def _acquire_foot_state(env: Any, gymtorch: Any):
        """从现有 Gym 仿真获取刚体状态张量，以读取真实足端速度。"""
        acquire = getattr(env.gym, "acquire_rigid_body_state_tensor", None)
        refresh = getattr(env.gym, "refresh_rigid_body_state_tensor", None)
        if acquire is None or refresh is None:
            return None
        raw = acquire(env.sim)
        tensor = gymtorch.wrap_tensor(raw)
        if tensor.ndim == 2:
            tensor = tensor.reshape(1, -1, 13)
        elif tensor.ndim != 3 or tensor.shape[-1] != 13:
            return None
        return tensor

    def _compile_ik_reference_actions(self, robot_model: Any,
                                      trajectory: DynamicMotionPrototype,
                                      env: Any, dof_names: Sequence[str],
                                      torch: Any) -> Dict[str, Any]:
        """逐仿真步采样足端轨迹、调用真实 Pinocchio IK 并编译 PPO 位置动作。"""
        metrics: Dict[str, Any] = {
            "foot_trajectory_consumed": False,
            "joint_reference_trajectory_generated": False,
            "ik_samples": 0,
            "ik_max_residual": None,
        }
        if str(getattr(self.ik_solver, "backend", "")).lower() != "pinocchio":
            return {"success": False, "metrics": metrics,
                    "reason": "真实 Isaac Gym 动态验证要求 Pinocchio IK；Mock IK 不能提供关节参考"}
        try:
            if trajectory.foot_trajectory is None or trajectory.gait is None:
                raise ValueError("缺少 FootTrajectory 或 gait")
            descriptor = (robot_model.get("_descriptor") if isinstance(robot_model, dict)
                          else robot_model)
            if descriptor is None:
                raise ValueError("RobotModel descriptor 不含真实 Go2 URDF/foot frames")
            from ..robot_models.go2 import Go2RobotModel
            nominal = Go2RobotModel.nominal_feet_positions(descriptor, {})
            nominal_by_leg = {str(frame).split("_")[0].upper(): position
                              for frame, position in nominal.items()}
            foot_frames = list(self._value(robot_model, "foot_frames", []) or [])
            frame_for_leg = {}
            for leg in self.GO2_LEGS:
                matches = [name for name in foot_frames
                           if str(name).split("_")[0].upper() == leg]
                if len(matches) != 1 or leg not in nominal_by_leg:
                    raise ValueError("无法将 Go2 足端 frame 唯一映射到 %s" % leg)
                frame_for_leg[leg] = matches[0]
            dt = float(env.dt)
            steps = int(math.ceil(trajectory.duration / dt - 1.0e-4))
            if not math.isfinite(dt) or dt <= 0.0 or steps < 1 or steps > self.max_steps:
                raise ValueError("Isaac Gym dt/step budget cannot cover the full trajectory")
            scale = float(env.cfg.control.action_scale)
            clip = float(env.cfg.normalization.clip_actions)
            if not math.isfinite(scale) or scale <= 1.0e-12 or not math.isfinite(clip) or clip <= 0:
                raise ValueError("Unitree PPO position action scale/clip is invalid")
            defaults = {name: float(value) for name, value in zip(
                dof_names, env.default_dof_pos[0].detach().cpu().tolist())}
            lower = env.dof_pos_limits[:, 0].detach().cpu().tolist()
            upper = env.dof_pos_limits[:, 1].detach().cpu().tolist()
            velocity_limits = env.dof_vel_limits.detach().cpu().tolist()
            warm_start = dict(defaults)
            previous = dict(defaults)
            last_locomotion = dict(defaults)
            action_rows = []
            reference_positions = []
            residuals = []
            max_reference_velocity = 0.0
            for index in range(steps):
                elapsed = index * dt
                phase = self._phase_at(trajectory, elapsed)
                if phase.name == "locomotion":
                    foot_sample = FootTrajectoryGenerator.sample(
                        nominal_by_leg, trajectory.gait, trajectory.foot_trajectory,
                        max(0.0, elapsed - phase.start))
                    target = {
                        "time_seconds": elapsed,
                        "feet": {frame_for_leg[leg]: foot_sample[leg]
                                 for leg in self.GO2_LEGS},
                        "base_height": float(self._value(
                            robot_model, "base_initial_height", env.cfg.init_state.pos[2])),
                        "base_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                    result = self.ik_solver.solve_ik(robot_model, target, warm_start)
                    if result.backend != "pinocchio" or result.success is not True:
                        raise ValueError("Pinocchio IK failed at t=%.3f: %s" %
                                         (elapsed, result.reason or result.status))
                    current = {name: float(result.joint_positions[name])
                               for name in dof_names if name in result.joint_positions}
                    if len(current) != len(dof_names):
                        raise ValueError("Pinocchio IK 缺少 Unitree action 对应关节")
                    warm_start = dict(current)
                    last_locomotion = dict(current)
                    if result.residual is not None:
                        residuals.append(float(result.residual))
                    metrics["ik_samples"] += 1
                elif "stop" in phase.name:
                    blend = min(1.0, max(0.0, (elapsed - phase.start + dt) /
                                          max(dt, phase.end - phase.start)))
                    current = {name: last_locomotion[name] * (1.0 - blend) +
                               defaults[name] * blend for name in dof_names}
                else:
                    current = dict(defaults)
                actions = []
                for joint_index, name in enumerate(dof_names):
                    q = current[name]
                    if not math.isfinite(q) or q < float(lower[joint_index]) - 1.0e-5 or \
                            q > float(upper[joint_index]) + 1.0e-5:
                        raise ValueError("joint position reference violates PPO limit: %s" % name)
                    qdot = abs(q - previous[name]) / dt
                    max_reference_velocity = max(max_reference_velocity, qdot)
                    if qdot > float(velocity_limits[joint_index]) + 1.0e-3:
                        raise ValueError("joint reference velocity violates PPO limit: %s" % name)
                    action = (q - defaults[name]) / scale
                    if not math.isfinite(action) or abs(action) > clip + 1.0e-5:
                        raise ValueError("joint reference exceeds PPO action clip: %s" % name)
                    actions.append(action)
                action_rows.append(torch.tensor(actions, dtype=torch.float,
                                                device=env.device).reshape(1, -1))
                reference_positions.append(dict(current))
                previous = current
            metrics.update({
                "foot_trajectory_consumed": True,
                "joint_reference_trajectory_generated": True,
                "joint_reference_samples": len(action_rows),
                "ik_max_residual": max(residuals) if residuals else None,
                "max_reference_joint_velocity": max_reference_velocity,
                "reference_source": "pinocchio_ik_from_foot_trajectory",
            })
            try:
                from ..validation.trajectory_validator import (
                    JointTrajectory, JointTrajectoryPoint, TrajectoryValidator,
                )
                joint_limits = self._value(robot_model, "limits", {}) or {}
                joint_trajectory = JointTrajectory(
                    source="pinocchio_ik_from_foot_trajectory",
                    points=[
                        JointTrajectoryPoint(time=index * dt, positions=positions)
                        for index, positions in enumerate(reference_positions)
                    ],
                )
                trajectory_check = TrajectoryValidator.validate(
                    joint_trajectory, joint_limits)
                metrics["trajectory_validation"] = trajectory_check.dict()
            except Exception as exc:
                metrics["trajectory_validation"] = {
                    "stage": "joint_trajectory", "status": "INCONCLUSIVE",
                    "backend": "deterministic_finite_difference",
                    "reason": "无法执行关节轨迹独立校验：%s" % str(exc)[:250],
                }
            return {"success": True, "actions": torch.stack(action_rows),
                    "metrics": metrics, "reason": "足端轨迹已逐时刻通过 Pinocchio IK 编译"}
        except Exception as exc:
            return {"success": False, "metrics": metrics,
                    "reason": "FootTrajectory/Pinocchio reference 编译失败：%s" % str(exc)[:400]}

    def _rollout_dynamic(self, env: Any, trajectory: DynamicMotionPrototype,
                         base_actions: Any, foot_states: Any, torch: Any,
                         reference_plan: Optional[Dict[str, Any]] = None) -> SimulationReport:
        """执行限时开环步态并收集稳定性、速度、接触、限位和能耗证据。"""
        dt = float(env.dt)
        expected_steps = int(math.ceil(trajectory.duration / dt - 1.0e-4))
        step_limit = min(self.max_steps, expected_steps)
        if not math.isfinite(dt) or dt <= 0.0 or step_limit < 1:
            return self._unavailable("Unitree PPO 环境 dt/预检步数无效")
        if expected_steps > self.max_steps:
            return self._unavailable("预检步数预算不足以覆盖完整动态轨迹")

        foot_indices = [int(item) for item in env.feet_indices.detach().cpu().tolist()]
        foot_names = ["FL", "FR", "RL", "RR"]
        contact_count = [0, 0, 0, 0]
        locomotion_contact_count = [0, 0, 0, 0]
        locomotion_airborne_steps = [0, 0, 0, 0]
        no_contact_run = [0, 0, 0, 0]
        max_no_contact_run = [0, 0, 0, 0]
        slip_speeds: List[float] = []
        velocity_errors: List[float] = []
        actual_velocities: List[float] = []
        min_height = float("inf")
        max_height = float("-inf")
        max_roll = max_pitch = max_yaw = max_orientation_error = 0.0
        max_position_excess = max_velocity_excess = 0.0
        max_torque = torque_square_sum = 0.0
        torque_samples = torque_saturated_samples = 0
        completed_steps = 0
        locomotion_samples = 0
        violations: List[str] = []
        target_direction = next((phase.target_velocity.get("x", 0.0)
                                  for phase in trajectory.phases
                                  if phase.name == "locomotion"), 0.0)
        gait = trajectory.gait
        if gait is None:
            return self._unavailable("动态物理验证缺少 gait pattern")
        action_scale = float(env.cfg.control.action_scale)
        initial_rpy = [float(value) for value in env.rpy[0].detach().cpu().tolist()]

        def angle_delta(value: float, reference: float) -> float:
            """计算两个角度间的最短有符号差。"""
            return math.atan2(math.sin(value - reference), math.cos(value - reference))

        try:
            with torch.no_grad():
                for step in range(step_limit):
                    elapsed = step * dt
                    phase = self._phase_at(trajectory, elapsed)
                    desired_x = float(phase.target_velocity.get("x", 0.0))
                    desired_y = float(phase.target_velocity.get("y", 0.0))
                    desired_yaw = float(phase.target_velocity.get("yaw", 0.0))
                    env.commands.zero_()
                    env.commands[0, 0] = desired_x
                    env.commands[0, 1] = desired_y
                    if env.commands.shape[1] > 2:
                        env.commands[0, 2] = desired_yaw
                    phase_elapsed = max(0.0, elapsed - phase.start)
                    if reference_plan is not None:
                        actions = reference_plan["actions"][step]
                    else:
                        actions = self._step_action(env, base_actions, phase.name,
                                                    phase_elapsed, gait, target_direction,
                                                    action_scale, torch)
                    _obs, _privileged, _reward, dones, _extras = env.step(actions)
                    completed_steps += 1
                    env.gym.refresh_rigid_body_state_tensor(env.sim)

                    height = float(env.root_states[0, 2].detach().cpu().item())
                    rpy = env.rpy[0].detach().cpu().tolist()
                    roll, pitch = abs(float(rpy[0])), abs(float(rpy[1]))
                    yaw_error = abs(angle_delta(float(rpy[2]), initial_rpy[2]))
                    orientation_error = max(
                        abs(angle_delta(float(rpy[0]), initial_rpy[0])),
                        abs(angle_delta(float(rpy[1]), initial_rpy[1])), yaw_error)
                    if hasattr(env, "base_lin_vel"):
                        actual_x = float(env.base_lin_vel[0, 0].detach().cpu().item())
                    else:
                        actual_x = float(env.root_states[0, 7].detach().cpu().item())
                    if not all(math.isfinite(value) for value in
                               (height, roll, pitch, yaw_error, orientation_error, actual_x)):
                        violations.append("non_finite_base_state")
                        break
                    min_height = min(min_height, height)
                    max_height = max(max_height, height)
                    max_roll = max(max_roll, roll)
                    max_pitch = max(max_pitch, pitch)
                    max_yaw = max(max_yaw, yaw_error)
                    max_orientation_error = max(max_orientation_error, orientation_error)

                    if phase.name == "locomotion" and elapsed >= phase.start + 0.2 * (
                            phase.end - phase.start):
                        actual_velocities.append(actual_x)
                        velocity_errors.append(abs(actual_x - desired_x))
                    if phase.name == "locomotion":
                        locomotion_samples += 1
                    else:
                        no_contact_run = [0, 0, 0, 0]

                    contact_forces = env.contact_forces[0, foot_indices]
                    if not bool(torch.all(torch.isfinite(contact_forces)).item()):
                        violations.append("non_finite_contact_state")
                        break
                    contacts = torch.linalg.vector_norm(contact_forces, dim=-1) > 1.0
                    for index, is_contact in enumerate(contacts.detach().cpu().tolist()):
                        if is_contact:
                            contact_count[index] += 1
                            if phase.name == "locomotion":
                                locomotion_contact_count[index] += 1
                            no_contact_run[index] = 0
                            foot_speed = torch.linalg.vector_norm(
                                foot_states[0, foot_indices[index], 7:9])
                            speed_value = float(foot_speed.detach().cpu().item())
                            if not math.isfinite(speed_value):
                                violations.append("non_finite_foot_velocity")
                                break
                            slip_speeds.append(speed_value)
                        else:
                            if phase.name == "locomotion":
                                locomotion_airborne_steps[index] += 1
                                no_contact_run[index] += 1
                                max_no_contact_run[index] = max(
                                    max_no_contact_run[index], no_contact_run[index])

                    actual_q = env.dof_pos[0]
                    actual_dq = env.dof_vel[0]
                    if not bool(torch.all(torch.isfinite(actual_q)).item()) or not bool(
                            torch.all(torch.isfinite(actual_dq)).item()):
                        violations.append("non_finite_joint_state")
                        break
                    pos_limits = env.dof_pos_limits
                    vel_limits = env.dof_vel_limits
                    pos_excess = torch.maximum(
                        pos_limits[:, 0] - actual_q, actual_q - pos_limits[:, 1]).clamp(min=0.0)
                    velocity_excess = (torch.abs(actual_dq) - vel_limits).clamp(min=0.0)
                    max_position_excess = max(max_position_excess,
                                              float(pos_excess.max().detach().cpu().item()))
                    max_velocity_excess = max(max_velocity_excess,
                                              float(velocity_excess.max().detach().cpu().item()))
                    torques = torch.abs(env.torques[0])
                    if not bool(torch.all(torch.isfinite(torques)).item()):
                        violations.append("non_finite_torque")
                        break
                    max_torque = max(max_torque, float(torques.max().detach().cpu().item()))
                    torque_square_sum += float(torch.sum(torques * torques).detach().cpu().item())
                    torque_samples += int(torques.numel())
                    torque_ratio = torques / torch.clamp(env.torque_limits, min=1.0e-6)
                    torque_saturated_samples += int(torch.sum(torque_ratio >= 0.99).item())

                    done = bool(torch.as_tensor(dones).reshape(-1)[0].item())
                    if done:
                        violations.append("unitree_environment_termination")
                    if height < max(0.12, float(env.cfg.init_state.pos[2]) * 0.5):
                        violations.append("base_height_below_fall_threshold")
                    if roll > self.tilt_roll_limit or pitch > self.tilt_pitch_limit:
                        violations.append("unitree_orientation_limit_exceeded")
                    if max_position_excess > 1.0e-4:
                        violations.append("joint_position_limit_exceeded")
                    if max_velocity_excess > 1.0e-4:
                        violations.append("joint_velocity_limit_exceeded")
                    if len(env.termination_contact_indices) > 0:
                        forbidden = env.contact_forces[0, env.termination_contact_indices]
                        if bool(torch.any(torch.linalg.vector_norm(forbidden, dim=-1) > 1.0).item()):
                            violations.append("termination_body_contact")
                    if done or violations:
                        break

            rollout_complete = completed_steps == expected_steps
            if not rollout_complete:
                violations.append("dynamic_rollout_incomplete")
            no_contact_seconds = [steps * dt for steps in max_no_contact_run]
            no_contact_seconds = [max(value, run * dt)
                                  for value, run in zip(no_contact_seconds, no_contact_run)]
            if max(no_contact_seconds, default=0.0) > self.no_contact_limit_seconds:
                violations.append("foot_no_contact_too_long")
            max_slip = max(slip_speeds, default=float("inf"))
            if max_slip > self.foot_slip_limit:
                violations.append("foot_slip_exceeded")
            mean_velocity_error = (sum(velocity_errors) / len(velocity_errors)
                                   if velocity_errors else float("inf"))
            effective_velocity_error_limit = min(
                self.velocity_error_limit,
                max(0.05, 0.25 * abs(float(target_direction))))
            if (not velocity_errors or
                    mean_velocity_error > effective_velocity_error_limit):
                violations.append("velocity_tracking_error_exceeded")
            if any(count == 0 for count in locomotion_contact_count):
                violations.append("one_or_more_feet_never_contacted")
            foot_contact_ratio = {
                name: count / float(max(1, locomotion_samples))
                for name, count in zip(foot_names, locomotion_contact_count)
            }
            torque_saturation_fraction = torque_saturated_samples / float(max(1, torque_samples))
            if torque_saturation_fraction > 0.20:
                violations.append("actuator_torque_saturation_excessive")

            foot_trajectory_consumed = bool(reference_plan and reference_plan.get(
                "metrics", {}).get("foot_trajectory_consumed"))
            if self.backend == "isaacgym" and not foot_trajectory_consumed:
                violations.append("foot_trajectory_not_consumed")
            violations = sorted(set(violations))
            physical_evidence = (self.backend == "isaacgym" and completed_steps > 0 and
                                 foot_trajectory_consumed)
            if not physical_evidence:
                success = None
            elif rollout_complete:
                success = not violations
            elif any(item in violations for item in (
                    "unitree_environment_termination", "base_height_below_fall_threshold",
                    "unitree_orientation_limit_exceeded", "joint_position_limit_exceeded",
                    "joint_velocity_limit_exceeded", "termination_body_contact")):
                success = False
            else:
                success = None
            report_status = ("PASSED" if success is True else "FAILED" if success is False else
                             "INCONCLUSIVE" if self.backend == "isaacgym" else "MOCK")
            contact_values = list(foot_contact_ratio.values())
            contact_balance = (
                min(contact_values) / max(contact_values)
                if contact_values and max(contact_values) > 0.0 else 0.0)
            height_variation = (max_height - min_height
                                if math.isfinite(max_height) and math.isfinite(min_height) else None)
            metrics = {
                "rollout_complete": rollout_complete,
                "steps": completed_steps,
                "requested_duration": trajectory.duration,
                "policy_dt": dt,
                "min_height": min_height if math.isfinite(min_height) else None,
                "max_height": max_height if math.isfinite(max_height) else None,
                "height_variation": height_variation,
                "max_roll": max_roll,
                "max_pitch": max_pitch,
                "max_yaw": max_yaw,
                "orientation_error": max_orientation_error,
                "fall": any(item in violations for item in (
                    "unitree_environment_termination", "base_height_below_fall_threshold",
                    "unitree_orientation_limit_exceeded")),
                "desired_velocity_x": target_direction,
                "locomotion_sample_count": locomotion_samples,
                "actual_velocity_x_mean": (sum(actual_velocities) / len(actual_velocities)
                                            if actual_velocities else None),
                "velocity_tracking_error": (mean_velocity_error
                                             if math.isfinite(mean_velocity_error) else None),
                "velocity_error": (mean_velocity_error
                                   if math.isfinite(mean_velocity_error) else None),
                "velocity_error_limit": effective_velocity_error_limit,
                "velocity_error_limit_configured": self.velocity_error_limit,
                "foot_contact_ratio": foot_contact_ratio,
                "foot_contact_ratio_all_rollout": {
                    name: count / float(max(1, completed_steps))
                    for name, count in zip(foot_names, contact_count)},
                "contact_balance": contact_balance,
                "airborne_time_seconds_per_foot": {
                    name: steps * dt for name, steps in
                    zip(foot_names, locomotion_airborne_steps)},
                "max_no_contact_seconds_per_foot": {
                    name: seconds for name, seconds in zip(foot_names, no_contact_seconds)},
                "max_foot_slip_speed": max_slip if math.isfinite(max_slip) else None,
                "foot_slip_max": max_slip if math.isfinite(max_slip) else None,
                "foot_slip_limit": self.foot_slip_limit,
                "max_joint_position_limit_excess": max_position_excess,
                "max_joint_velocity_limit_excess": max_velocity_excess,
                "max_torque_magnitude": max_torque,
                "rms_torque_magnitude": math.sqrt(torque_square_sum / max(1, torque_samples)),
                "torque_saturation_fraction": torque_saturation_fraction,
                "gait_frequency_hz": gait.frequency,
                "gait_duty_factor": gait.duty_factor,
                "gait_phase_offsets": dict(gait.phase_offsets),
                "policy_runner_created": False,
                "policy_generated": False,
                "policy_trained": False,
                "action_source": ("pinocchio_ik_joint_reference" if reference_plan is not None
                                   else "mock_fixed_open_loop_probe"),
                "foot_trajectory_consumed": foot_trajectory_consumed,
                "joint_reference_trajectory_generated": bool(
                    reference_plan and reference_plan.get("metrics", {}).get(
                        "joint_reference_trajectory_generated")),
                "joint_reference_metrics": (reference_plan.get("metrics", {})
                                            if reference_plan else {}),
                "action_scale": action_scale,
                "decimation": int(env.cfg.control.decimation),
                "randomization_disabled": list(getattr(
                    env, "_feasibility_randomization_disabled", [])),
            }
            return SimulationReport(
                status=report_status, success=success,
                backend=self.backend, validated=bool(
                    physical_evidence and (rollout_complete or success is False)),
                validation_level=("DYNAMIC_PHYSICS_VALIDATED" if success is True else
                                  "PHYSICS_FAILED" if success is False else
                                  "CAPABILITY_ONLY" if self.backend == "isaacgym" else
                                  "MOCK_VALIDATED"),
                model_source=self._display_asset(env), duration=min(trajectory.duration,
                                                                   completed_steps * dt),
                fall=metrics["fall"], violations=violations, metrics=metrics,
                reason=("真实 Unitree Isaac Gym 足端 IK 轨迹预检通过；只覆盖本次短时 rollout" if success else
                        "Mock 动态探针执行完成；不构成物理结论" if self.backend != "isaacgym" else
                        "动态预检未通过：" + ("、".join(violations) if violations else
                                            "没有真实物理 rollout 证据")),
            )
        except Exception as exc:
            return self._unavailable("Isaac Gym 动态 rollout 异常：%s" % str(exc)[:500])

    @staticmethod
    def _phase_at(trajectory: DynamicMotionPrototype, elapsed: float):
        """返回当前仿真时刻对应的低维轨迹阶段。"""
        for phase in trajectory.phases:
            if phase.start <= elapsed < phase.end:
                return phase
        return trajectory.phases[-1]

    @classmethod
    def _step_action(cls, env: Any, base_actions: Any, phase_name: str,
                     phase_elapsed: float, gait: GaitPattern, direction: float,
                     action_scale: float, torch: Any):
        """将原型步态频率、占空比和四足相位映射为受限开环动作探针。"""
        actions = torch.zeros_like(base_actions)
        if phase_name != "locomotion":
            return actions
        names = [str(name) for name in env.dof_names[:env.num_actions]]
        amplitude = 0.25 + 0.30 * min(abs(float(direction)), 1.0)
        direction_sign = -1.0 if direction < 0.0 else 1.0
        for leg in cls.GO2_LEGS:
            cycle = (gait.frequency * phase_elapsed + gait.phase_offsets[leg]) % 1.0
            if cycle < gait.duty_factor:
                stance_progress = cycle / gait.duty_factor
                stride = 2.0 * stance_progress - 1.0
                swing_lift = 0.0
            else:
                swing_progress = (cycle - gait.duty_factor) / (1.0 - gait.duty_factor)
                stride = 1.0 - 2.0 * swing_progress
                swing_lift = math.sin(math.pi * swing_progress)
            thigh_index = names.index("%s_thigh_joint" % leg)
            calf_index = names.index("%s_calf_joint" % leg)
            actions[0, thigh_index] = direction_sign * amplitude * stride / action_scale
            # Go2 标称 calf 角为负；摆动相向更负方向屈膝以抬脚。
            actions[0, calf_index] = -1.5 * amplitude * swing_lift / action_scale
        clip = float(env.cfg.normalization.clip_actions)
        if bool(torch.any(torch.abs(actions) > clip + 1.0e-6).item()):
            raise ValueError("开环运动探针超出 Unitree PPO action clip；拒绝静默裁剪")
        return actions


class MockIsaacGymDynamicValidator(IsaacGymDynamicValidator):
    """提供显式 Mock 动态报告；任何输入都不能升级为真实动态物理验证。"""

    def __init__(self, training_root: Path = Path(".")):
        """标记测试后端，不加载 Isaac Gym 或创建仿真。"""
        super().__init__(training_root, environment_factory=lambda *_args: None)

    def validate(self, robot_model: Any,
                 trajectory: DynamicMotionPrototype) -> SimulationReport:
        """返回 Mock 占位结果并显式禁止真实 physics 级别。"""
        return SimulationReport(
            status="MOCK", success=True, backend="mock-isaacgym-dynamic",
            validated=False, validation_level="MOCK_VALIDATED",
            duration=trajectory.duration,
            metrics={"policy_runner_created": False, "policy_generated": False,
                     "policy_trained": False, "action_source": "mock"},
            reason="Mock 动态验证仅测试接口；没有 Isaac Gym 物理证据",
        )
