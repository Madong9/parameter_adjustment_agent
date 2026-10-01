"""复用 Unitree PPO Isaac Gym 环境执行不含策略训练的短时物理预检。"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..ik.schema import IKResult
from ..motion_prototype.schema import MotionPrototype
from .schema import SimulationReport


class IsaacGymFeasibilityValidator:
    """使用 Unitree 已注册任务类、asset、控制器与仿真参数做有界 rollout。"""

    def __init__(self, training_root: Path, seed: int = 1, max_seconds: float = 1.0,
                 max_steps: int = 64, tilt_roll_limit: float = 0.8,
                 tilt_pitch_limit: float = 1.0, tracking_error_limit: float = 0.75,
                 environment_factory: Optional[Callable[..., Any]] = None,
                 visualize: bool = False):
        """设置训练工程根目录、短时预算、安全阈值和可注入的测试环境工厂。"""
        self.training_root = Path(training_root).resolve()
        self.seed = int(seed)
        self.max_seconds = max(0.02, float(max_seconds))
        self.max_steps = max(1, int(max_steps))
        self.tilt_roll_limit = float(tilt_roll_limit)
        self.tilt_pitch_limit = float(tilt_pitch_limit)
        self.tracking_error_limit = float(tracking_error_limit)
        self.environment_factory = environment_factory
        self.visualize = bool(visualize)
        # A substituted environment is a test double and is never physics evidence.
        self.backend = "mock-isaacgym" if environment_factory is not None else "isaacgym"

    def is_available(self) -> bool:
        """检查 Unitree Isaac Gym 包和 legged_gym 任务环境源码是否存在。"""
        return all((
            (self.training_root / "isaacgym" / "python" / "isaacgym").is_dir(),
            (self.training_root / "legged_gym" / "envs").is_dir(),
        ))

    def validate(self, robot_model: Any, prototype: MotionPrototype,
                 ik_result: IKResult) -> SimulationReport:
        """兼容单目标接口，将 IK 构型作为一个高层动作阶段执行短 rollout。"""
        rows = [{"phase": prototype.phases[0].name, "result": ik_result,
                 "target": {"time_seconds": prototype.phases[0].duration_seconds,
                            "base_height": ik_result.base_height,
                            "base_orientation_xyzw": ik_result.base_orientation_xyzw}}]
        return self.validate_targets(robot_model, prototype, rows)

    def validate_targets(self, robot_model: Any, prototype: MotionPrototype,
                         ik_results: Sequence[Dict[str, Any]]) -> SimulationReport:
        """创建一个真实 PPO 环境，按 IK 目标编译 action 并限时执行各阶段。"""
        if not ik_results:
            return self._unavailable("没有可执行的 IK 目标")
        if any(not row["result"].success or not row["result"].joint_positions
               for row in ik_results):
            return self._unavailable("至少一个阶段没有成功的 IK 关节构型")

        env = None
        try:
            if self.environment_factory is not None:
                import torch
                from types import SimpleNamespace
                gymtorch = SimpleNamespace(unwrap_tensor=lambda tensor: tensor)
            else:
                torch, gymtorch = self._load_runtime_modules()
            expected_urdf = self._value(robot_model, "urdf_path")
            env = (self.environment_factory(self.training_root, prototype.robot, self.seed)
                   if self.environment_factory else self._create_ppo_environment(
                       prototype.robot, expected_urdf))
            self._validate_environment_identity(env, robot_model)
            dof_names = [str(name) for name in env.dof_names[:env.num_actions]]
            if len(dof_names) != int(env.num_actions) or len(dof_names) != int(env.num_dof):
                return self._unavailable("Unitree policy action 数与 DOF 数不一致，无法安全映射 IK 结果")
            action_targets = [self._compile_action_target(env, row["result"], dof_names, torch)
                              for row in ik_results]
            if any(target is None for target in action_targets):
                return self._unavailable("IK 目标缺少 PPO action 对应关节，或超出 action_space")

            self._set_exact_initial_state(
                env, torch, gymtorch, ik_results[0].get("target", {}))
            return self._rollout(env, prototype, ik_results, action_targets, torch)
        except Exception as exc:
            # Construction/configuration/provider errors are not physical evidence.
            return self._unavailable("Isaac Gym 环境或 rollout 初始化失败：%s" % str(exc)[:500])
        finally:
            self._destroy_environment(env)

    def _load_runtime_modules(self):
        """按 Unitree 的 Isaac Gym 导入顺序加载依赖，不在模块导入时污染其他流程。"""
        if not self.is_available():
            raise RuntimeError("Unitree Isaac Gym / legged_gym environment source is missing")
        from ...training.real_train import (
            _ensure_environment_tools_on_path,
            _install_numpy_compatibility_aliases,
        )

        _install_numpy_compatibility_aliases()
        _ensure_environment_tools_on_path()
        isaac_python = str((self.training_root / "isaacgym" / "python").resolve())
        unitree_root = str(self.training_root)
        for path in (isaac_python, unitree_root):
            if path not in sys.path:
                sys.path.insert(0, path)
        # Isaac Gym must be imported before legged_gym's helper imports torch.
        import isaacgym  # noqa: F401
        from isaacgym import gymtorch
        import torch

        return torch, gymtorch

    def _create_ppo_environment(self, robot_name: str, expected_urdf: Optional[str]):
        """用原始 task_registry 构造 headless 单环境，不创建 PPO runner 或策略。"""
        import legged_gym
        from legged_gym.envs import task_registry
        from legged_gym.utils.helpers import get_args

        if str(robot_name).lower() not in task_registry.task_classes:
            raise RuntimeError("Unitree task_registry 中未注册机器人 %s" % robot_name)
        env_cfg, _train_cfg = task_registry.get_cfgs(str(robot_name).lower())
        env_cfg = copy.deepcopy(env_cfg)
        env_cfg.env.num_envs = 1
        env_cfg.env.test = False
        if hasattr(env_cfg.env, "record_video"):
            env_cfg.env.record_video = False
        env_cfg.noise.add_noise = False
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.push_robots = False
        env_cfg.commands.curriculum = False
        env_cfg.terrain.curriculum = False

        asset_value = str(env_cfg.asset.file).format(
            LEGGED_GYM_ROOT_DIR=str(Path(legged_gym.LEGGED_GYM_ROOT_DIR).resolve()))
        if not expected_urdf or Path(asset_value).resolve() != Path(str(expected_urdf)).resolve():
            raise RuntimeError("Unitree task asset 与 Agent 配置的 robot URDF 不一致")

        original_argv = sys.argv
        sys.argv = [original_argv[0], "--task", str(robot_name).lower(), "--seed",
                    str(self.seed), "--num_envs", "1"]
        if not self.visualize:
            sys.argv.append("--headless")
        try:
            args = get_args()
        finally:
            sys.argv = original_argv
        env, _ = task_registry.make_env(
            name=str(robot_name).lower(), args=args, env_cfg=env_cfg)
        # Keep audit facts on the environment instance without touching PPO config.
        env._feasibility_seed = self.seed
        env._feasibility_asset = str(Path(asset_value).resolve())
        env._feasibility_randomization_disabled = [
            "friction", "pushes", "command_curriculum", "terrain_curriculum", "observation_noise"]
        return env

    @staticmethod
    def _value(robot_model: Any, key: str, default: Any = None) -> Any:
        """從統一机器人 descriptor 或 runtime 字典中读取指定字段。"""
        if isinstance(robot_model, dict):
            return robot_model.get(key, default)
        return getattr(robot_model, key, default)

    def _validate_environment_identity(self, env: Any, robot_model: Any) -> None:
        """确保实际 Isaac Gym 资产及默认关节目标和 PPO 使用的注册配置一致。"""
        robot_name = str(self._value(robot_model, "robot_name", "")).lower()
        env_robot_name = str(getattr(env.cfg.asset, "name", "")).lower()
        if robot_name and env_robot_name != robot_name:
            raise RuntimeError("Isaac Gym 环境 asset 与任务机器人不一致：%s != %s" %
                               (env_robot_name, robot_name))
        configured_positions = dict(self._value(robot_model, "default_joint_positions", {}) or {})
        env_positions = {
            str(name): float(value)
            for name, value in zip(env.dof_names, env.default_dof_pos[0].detach().cpu().tolist())
        }
        if configured_positions and any(
                name not in env_positions or abs(float(value) - env_positions[name]) > 1.0e-6
                for name, value in configured_positions.items()):
            raise RuntimeError("Agent Go2 默认关节角与 Unitree PPO 配置不一致")
        expected_urdf = self._value(robot_model, "urdf_path")
        actual_asset = getattr(env, "_feasibility_asset", None)
        if expected_urdf and actual_asset and Path(str(expected_urdf)).resolve() != Path(
                str(actual_asset)).resolve():
            raise RuntimeError("Isaac Gym 加载的 URDF 与 RobotModelLoader 来源不一致")

    @staticmethod
    def _compile_action_target(env: Any, ik_result: IKResult, dof_names: Sequence[str],
                               torch: Any):
        """把 Pinocchio 关节目标按 PPO default+action_scale 映射到规范化动作。"""
        scale = float(env.cfg.control.action_scale)
        if not math.isfinite(scale) or abs(scale) <= 1.0e-12:
            return None
        positions = ik_result.joint_positions
        if any(name not in positions for name in dof_names):
            return None
        values = [float(positions[name]) for name in dof_names]
        if not all(math.isfinite(value) for value in values):
            return None
        target = torch.tensor(values, dtype=torch.float, device=env.device).reshape(1, -1)
        default = env.default_dof_pos.reshape(1, -1)
        action = (target - default) / scale
        action_limit = float(env.cfg.normalization.clip_actions)
        if torch.any(torch.abs(action) > action_limit + 1.0e-5):
            return None
        return action

    @staticmethod
    def _set_exact_initial_state(env: Any, torch: Any, gymtorch: Any,
                                 motion_target: Optional[Dict[str, Any]] = None) -> None:
        """从 PPO 默认状态出发，并应用原型首个目标中的 base pose。"""
        env.dof_pos[0].copy_(env.default_dof_pos[0])
        env.dof_vel[0].zero_()
        env.root_states[0].copy_(env.base_init_state)
        if hasattr(env, "env_origins"):
            env.root_states[0, :3] += env.env_origins[0]
        if motion_target:
            base_height = motion_target.get("base_height")
            if base_height is not None and float(base_height) > 0.0:
                origin_z = float(env.env_origins[0, 2].item()) if hasattr(env, "env_origins") else 0.0
                env.root_states[0, 2] = float(base_height) + origin_z
            orientation = motion_target.get("base_orientation_xyzw")
            if orientation is not None and len(orientation) == 4:
                env.root_states[0, 3:7] = torch.tensor(
                    [float(value) for value in orientation], dtype=torch.float,
                    device=env.device)
        env.root_states[0, 7:13].zero_()
        env.commands.zero_()
        env.actions.zero_()
        env.last_actions.zero_()
        env.last_dof_vel.zero_()
        env.episode_length_buf.zero_()
        env.reset_buf.zero_()
        env.gym.set_dof_state_tensor(
            env.sim, gymtorch.unwrap_tensor(env.dof_state))
        env.gym.set_actor_root_state_tensor(
            env.sim, gymtorch.unwrap_tensor(env.root_states))
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_net_contact_force_tensor(env.sim)
        env.compute_observations()

    def _rollout(self, env: Any, prototype: MotionPrototype,
                 ik_results: Sequence[Dict[str, Any]], action_targets: Sequence[Any],
                 torch: Any) -> SimulationReport:
        """对每个语义阶段短时插值执行关节目标并聚合安全/稳定指标。"""
        violations: List[str] = []
        min_height = float("inf")
        max_roll = max_pitch = max_joint_limit_excess = 0.0
        max_tracking_error = max_settled_tracking_error = max_torque_ratio = 0.0
        contact_samples = foot_contact_samples = forbidden_contact_samples = 0
        completed_steps = 0
        elapsed = 0.0
        dt = float(env.dt)
        if not math.isfinite(dt) or dt <= 0.0:
            return self._unavailable("Unitree PPO env.dt 无效，无法计算短 rollout")
        phase_by_name = {phase.name: phase for phase in prototype.phases}
        initial_target = env.default_dof_pos.reshape(1, -1).clone()
        previous_target = initial_target
        support_expected = any(
            str(value).lower() == "support"
            for phase in prototype.phases for value in phase.body_goal.values())
        completed_phase_count = 0

        try:
            with torch.no_grad():
                for phase_index, (row, target_actions) in enumerate(zip(ik_results, action_targets)):
                    if completed_steps >= self.max_steps or elapsed + dt > self.max_seconds + 1.0e-9:
                        break
                    phase_name = str(row.get("phase", ""))
                    phase = phase_by_name.get(phase_name)
                    requested_duration = (float(phase.duration_seconds) if phase else
                                          float(row.get("target", {}).get("time_seconds", 0.1) or 0.1))
                    remaining_seconds = self.max_seconds - elapsed
                    phases_left = len(ik_results) - phase_index
                    phase_budget = remaining_seconds / float(max(1, phases_left))
                    remaining_steps = min(
                        self.max_steps - completed_steps,
                        int(math.floor((remaining_seconds + 1.0e-9) / dt)),
                    )
                    if remaining_steps <= 0:
                        break
                    phase_steps = min(
                        max(1, int(math.ceil(min(requested_duration, phase_budget) / dt))),
                        remaining_steps,
                    )
                    if phase_steps <= 0:
                        break
                    target_q = env.default_dof_pos.reshape(1, -1) + (
                        target_actions * float(env.cfg.control.action_scale))
                    phase_tracking_errors = []
                    for step_index in range(phase_steps):
                        blend = float(step_index + 1) / float(phase_steps)
                        desired_q = previous_target + (target_q - previous_target) * blend
                        actions = (desired_q - env.default_dof_pos.reshape(1, -1)) / float(
                            env.cfg.control.action_scale)
                        _obs, _privileged, _reward, dones, _extras = env.step(actions)
                        completed_steps += 1
                        elapsed += dt

                        height = float(env.root_states[0, 2].detach().cpu().item())
                        rpy = env.rpy[0].detach().cpu().tolist()
                        roll, pitch = abs(float(rpy[0])), abs(float(rpy[1]))
                        min_height = min(min_height, height)
                        max_roll, max_pitch = max(max_roll, roll), max(max_pitch, pitch)
                        actual_q = env.dof_pos[0]
                        lower, upper = env.dof_pos_limits[:, 0], env.dof_pos_limits[:, 1]
                        excess = torch.maximum(lower - actual_q, actual_q - upper).clamp(min=0.0)
                        max_joint_limit_excess = max(
                            max_joint_limit_excess, float(excess.max().detach().cpu().item()))
                        current_error = torch.max(torch.abs(actual_q - desired_q.reshape(-1)))
                        current_error_value = float(current_error.detach().cpu().item())
                        phase_tracking_errors.append(current_error_value)
                        max_tracking_error = max(max_tracking_error, current_error_value)
                        torque_limits = torch.clamp(env.torque_limits, min=1.0e-6)
                        torque_ratio = torch.max(torch.abs(env.torques[0]) / torque_limits)
                        max_torque_ratio = max(
                            max_torque_ratio, float(torque_ratio.detach().cpu().item()))
                        if env.contact_forces.shape[1] > 0:
                            total_contact = torch.linalg.vector_norm(env.contact_forces[0], dim=-1)
                            contact_samples += int(torch.any(total_contact > 1.0).item())
                            if len(env.feet_indices) > 0:
                                feet = env.contact_forces[0, env.feet_indices]
                                foot_contact_samples += int(
                                    torch.any(torch.linalg.vector_norm(feet, dim=-1) > 1.0).item())
                            if len(env.termination_contact_indices) > 0:
                                forbidden = env.contact_forces[0, env.termination_contact_indices]
                                if bool(torch.any(torch.linalg.vector_norm(forbidden, dim=-1) > 1.0).item()):
                                    forbidden_contact_samples += 1
                                    violations.append("termination_body_contact")
                        done = bool(torch.as_tensor(dones).reshape(-1)[0].item())
                        if done:
                            violations.append("unitree_environment_termination")
                        if height < max(0.12, float(env.cfg.init_state.pos[2]) * 0.5):
                            violations.append("base_height_below_fall_threshold")
                        if roll > self.tilt_roll_limit or pitch > self.tilt_pitch_limit:
                            violations.append("unitree_orientation_limit_exceeded")
                        if max_joint_limit_excess > 1.0e-4:
                            violations.append("joint_limit_exceeded")
                        if max_torque_ratio > 1.001:
                            violations.append("actuator_torque_limit_exceeded")
                        if violations:
                            break
                    if phase_tracking_errors:
                        # Judge servo tracking after the commanded target has had time to settle;
                        # retain the whole-rollout peak separately as diagnostic evidence.
                        settle_window = min(
                            len(phase_tracking_errors),
                            max(3, int(math.ceil(len(phase_tracking_errors) * 0.2))),
                        )
                        max_settled_tracking_error = max(
                            max_settled_tracking_error,
                            max(phase_tracking_errors[-settle_window:]),
                        )
                        if max_settled_tracking_error > self.tracking_error_limit:
                            violations.append("actuator_settled_tracking_error_exceeded")
                    previous_target = target_q.detach().clone()
                    completed_phase_count += 1
                    if violations:
                        break

            if not violations and completed_phase_count < len(ik_results):
                return self._unavailable(
                    "短 rollout 预算不足以覆盖所有 MotionPrototype 阶段；未给出物理通过/失败结论")
            if support_expected and foot_contact_samples == 0:
                violations.append("required_foot_contact_not_observed")
            violations = sorted(set(violations))
            success = not violations and completed_steps > 0
            self_collision_enabled = int(getattr(env.cfg.asset, "self_collisions", 1)) == 0
            return SimulationReport(
                status="PASSED" if success else "FAILED", success=success,
                backend=self.backend,
                validated=completed_steps > 0 and self.backend == "isaacgym",
                validation_level=(
                    ("STATIC_PHYSICS_VALIDATED" if success else "PHYSICS_FAILED")
                    if self.backend == "isaacgym" else "MOCK_VALIDATED"),
                model_source=self._display_asset(env),
                duration=min(elapsed, self.max_seconds),
                fall=any(item in violations for item in (
                    "base_height_below_fall_threshold", "unitree_environment_termination",
                    "unitree_orientation_limit_exceeded")),
                violations=violations,
                self_collision_checked=self_collision_enabled,
                metrics={
                    "steps": completed_steps,
                    "target_count": completed_phase_count,
                    "max_roll": max_roll,
                    "max_pitch": max_pitch,
                    "min_height": min_height if math.isfinite(min_height) else None,
                    "initial_height": float(env.cfg.init_state.pos[2]),
                    "max_joint_limit_excess": max_joint_limit_excess,
                    "max_actuator_tracking_error": max_tracking_error,
                    "max_settled_actuator_tracking_error": max_settled_tracking_error,
                    "max_torque_limit_ratio": max_torque_ratio,
                    "contact_sample_fraction": contact_samples / float(max(1, completed_steps)),
                    "foot_contact_sample_fraction": foot_contact_samples / float(max(1, completed_steps)),
                    "forbidden_body_contact_samples": forbidden_contact_samples,
                    "self_collision_enabled_by_ppo_config": self_collision_enabled,
                    "policy_runner_created": False,
                    "policy_trained": False,
                    "randomization_disabled": list(getattr(
                        env, "_feasibility_randomization_disabled", [])),
                    "controller": str(env.cfg.control.control_type),
                    "action_scale": float(env.cfg.control.action_scale),
                    "decimation": int(env.cfg.control.decimation),
                },
                reason=("Unitree Isaac Gym PPO 环境短时物理预检通过；不代表目标策略已训练成功"
                        if success else "Unitree Isaac Gym 短时物理预检发现安全/稳定违规"),
            )
        except Exception as exc:
            return self._unavailable("Isaac Gym rollout 执行异常：%s" % str(exc)[:500])

    @staticmethod
    def _destroy_environment(env: Any) -> None:
        """销毁预检独占创建的 viewer/simulation，避免占用训练 GPU 资源。"""
        if env is None:
            return
        try:
            if getattr(env, "viewer", None) is not None:
                env.gym.destroy_viewer(env.viewer)
        except Exception:
            pass
        try:
            sim = getattr(env, "sim", None)
            if sim is not None:
                env.gym.destroy_sim(sim)
        except Exception:
            pass

    def _display_asset(self, env: Any) -> str:
        """将实际 URDF 资产来源转换为相对训练工程的可移植路径。"""
        value = getattr(env, "_feasibility_asset", None)
        if not value:
            return "Unitree task_registry Go2 asset"
        try:
            return Path(str(value)).resolve().relative_to(self.training_root).as_posix()
        except (OSError, ValueError):
            return "Unitree task_registry Go2 asset"

    def _unavailable(self, reason: str) -> SimulationReport:
        """返回没有充分物理证据的后端错误，不将初始化失败伪装为物理失败。"""
        return SimulationReport(
            status="UNAVAILABLE", success=None, backend=self.backend, validated=False,
            validation_level="CAPABILITY_ONLY", reason=reason,
        )
