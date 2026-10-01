"""覆盖动态运动原型、Isaac Gym 适配器和可行性状态门。"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from rl_training_agent.environment.inspector import EnvironmentInspector
from rl_training_agent.feasibility.agent import FeasibilityPipeline
from rl_training_agent.feasibility.ik.mock import MockIKSolver
from rl_training_agent.feasibility.motion_prototype.dynamic_generator import (
    DynamicMotionPrototypeGenerator,
)
from rl_training_agent.feasibility.motion_prototype.schema import (
    DynamicMotionPhase, DynamicMotionPrototype, FootTrajectory, GaitPattern, GaitType,
    MotionType, VelocityTrajectory,
)
from rl_training_agent.feasibility.ik.schema import IKResult
from rl_training_agent.feasibility.ik.pinocchio_solver import PinocchioIKSolver
from rl_training_agent.feasibility.robot_models.loader import RobotModelLoader
from rl_training_agent.feasibility.schema import (
    FeasibilityStatus, ValidationLevel, training_ready,
)
from rl_training_agent.feasibility.simulation.isaacgym_dynamic_validator import (
    IsaacGymDynamicValidator,
)
from rl_training_agent.feasibility.simulation.mujoco_validator import MockMuJoCoValidator
from rl_training_agent.feasibility.simulation.schema import SimulationReport
from rl_training_agent.orchestration.state_machine import AgentState, PersistentStateMachine
from rl_training_agent.schemas.agent_workflow import TaskIntentSpec
from rl_training_agent.settings import load_settings


def _intent(text: str, velocity=None, action="backward_locomotion") -> TaskIntentSpec:
    """创建不依赖 Provider 的任务意图样本。"""
    return TaskIntentSpec(
        original_instruction=text, robot="go2", action_name=action,
        normalized_goal=text, target_velocity=velocity,
        required_behaviors=[],
    )


def _manifest():
    """读取本仓库 Go2 的能力清单。"""
    settings = load_settings()
    return EnvironmentInspector(settings.training_root).inspect("go2")


def _dynamic_prototype(duration=0.2, velocity=-0.3):
    """构造短时倒退步态轨迹。"""
    return DynamicMotionPrototype(
        robot="go2", action="backward_locomotion", motion_type=MotionType.LOCOMOTION,
        duration=duration,
        phases=[
            DynamicMotionPhase(name="stand_prepare", start=0.0, end=0.02,
                               target_velocity={"x": 0.0}),
            DynamicMotionPhase(name="locomotion", start=0.02, end=duration - 0.02,
                               target_velocity={"x": velocity}),
            DynamicMotionPhase(name="stop", start=duration - 0.02, end=duration,
                               target_velocity={"x": 0.0}),
        ],
        velocity=VelocityTrajectory(
            duration=duration, target_linear_velocity={"x": velocity}),
        gait=GaitPattern(
            type=GaitType.TROT, frequency=2.0, duty_factor=0.5,
            phase_offsets={"FL": 0.5, "FR": 0.0, "RL": 0.0, "RR": 0.5}),
        foot_trajectory=FootTrajectory(
            step_length=0.15, step_period=0.5, direction_x=-1.0,
            phase_offset={"FL": 0.5, "FR": 0.0, "RL": 0.0, "RR": 0.5}),
        constraints=["avoid_fall", "avoid_joint_limit"],
    )


def _fake_dynamic_env(torch):
    """提供具备 Unitree env.step 张量接口的测试替身，不冒充真实物理后端。"""
    names = ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
             "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
             "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"]
    defaults = torch.zeros((1, 12), dtype=torch.float)
    cfg = SimpleNamespace(
        asset=SimpleNamespace(name="go2", self_collisions=1),
        init_state=SimpleNamespace(pos=[0.0, 0.0, 0.42]),
        control=SimpleNamespace(control_type="P", action_scale=0.25, decimation=4),
        normalization=SimpleNamespace(clip_actions=10.0),
    )

    class FakeGym:
        """实现动态验证初始化、张量刷新与清理接口。"""

        def __init__(self):
            self.destroyed = False

        def set_dof_state_tensor(self, *_args):
            """接受 Agent 设置初始关节状态。"""

        def set_actor_root_state_tensor(self, *_args):
            """接受 Agent 设置初始基座状态。"""

        def refresh_dof_state_tensor(self, *_args):
            """模拟关节状态刷新。"""

        def refresh_actor_root_state_tensor(self, *_args):
            """模拟根状态刷新。"""

        def refresh_net_contact_force_tensor(self, *_args):
            """模拟接触力刷新。"""

        def acquire_rigid_body_state_tensor(self, *_args):
            """返回足端刚体状态测试张量。"""
            return env.rigid_body_states

        def refresh_rigid_body_state_tensor(self, *_args):
            """模拟刚体状态刷新。"""

        def destroy_sim(self, _sim):
            """标记测试仿真被正确销毁。"""
            self.destroyed = True

    env = SimpleNamespace(
        cfg=cfg, dof_names=names, num_actions=12, num_dof=12, num_envs=1,
        device="cpu", dt=0.02, default_dof_pos=defaults,
        dof_state=torch.zeros((1, 12, 2), dtype=torch.float),
        root_states=torch.zeros((1, 13), dtype=torch.float),
        base_init_state=torch.tensor([0.0, 0.0, 0.42, 0.0, 0.0, 0.0, 1.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        env_origins=torch.zeros((1, 3), dtype=torch.float),
        commands=torch.zeros((1, 4), dtype=torch.float),
        actions=torch.zeros((1, 12), dtype=torch.float),
        last_actions=torch.zeros((1, 12), dtype=torch.float),
        last_dof_vel=torch.zeros((1, 12), dtype=torch.float),
        episode_length_buf=torch.zeros((1,), dtype=torch.long),
        reset_buf=torch.zeros((1,), dtype=torch.long),
        dof_pos_limits=torch.tensor([[-2.5, 2.5]] * 12),
        dof_vel_limits=torch.ones((12,), dtype=torch.float) * 20.0,
        torque_limits=torch.ones((12,), dtype=torch.float) * 40.0,
        torques=torch.zeros((1, 12), dtype=torch.float),
        contact_forces=torch.zeros((1, 5, 3), dtype=torch.float),
        feet_indices=torch.tensor([0, 1, 2, 3]),
        termination_contact_indices=torch.tensor([4]),
        rpy=torch.zeros((1, 3), dtype=torch.float),
        base_lin_vel=torch.zeros((1, 3), dtype=torch.float),
        rigid_body_states=torch.zeros((1, 5, 13), dtype=torch.float),
        command_ranges={"lin_vel_x": [-1.0, 1.0]},
        gym=FakeGym(), sim="fake-sim", viewer=None,
        _feasibility_randomization_disabled=["test-only"],
    )
    env.dof_pos = env.dof_state.view(1, 12, 2)[..., 0]
    env.dof_vel = env.dof_state.view(1, 12, 2)[..., 1]

    def step(actions):
        """让替身跟踪 command 并提供稳定接触，供验证接口隔离。"""
        env.actions.copy_(actions)
        env.dof_pos[0].copy_(env.default_dof_pos[0] + actions[0] * cfg.control.action_scale)
        env.base_lin_vel[0, 0] = env.commands[0, 0]
        env.root_states[0, 7] = env.commands[0, 0]
        env.contact_forces[0, :4, 2] = 3.0
        return None, None, None, torch.tensor([False]), {}

    env.step = step
    env.compute_observations = lambda: None
    return env


def test_backward_walk_intent_generates_a_bounded_negative_velocity_trajectory():
    """确认倒退任务生成带 trot 周期参数的低维轨迹，不出现控制策略字段。"""
    prototype = DynamicMotionPrototypeGenerator.generate(_intent("Go2 倒退走 1m/s 5秒", 1.0))

    assert prototype.motion_type == MotionType.LOCOMOTION
    assert prototype.duration == 5.0
    assert [phase.name for phase in prototype.phases] == [
        "stand_prepare", "locomotion", "stop"]
    assert prototype.phases[1].target_velocity == {"x": -1.0}
    assert prototype.velocity.duration == 5.0
    assert prototype.velocity.target_linear_velocity == {"x": -1.0}
    assert prototype.gait.type == GaitType.TROT
    assert prototype.gait.frequency == 2.0
    assert prototype.gait.duty_factor == 0.5
    assert prototype.gait.phase_offsets == {"FL": 0.5, "FR": 0.0, "RL": 0.0, "RR": 0.5}
    assert prototype.foot_trajectory.direction_x == -1.0
    assert not ({"joint_positions", "torques", "policy", "actuator_command"} & set(prototype.dict()))


def test_motion_prototype_accepts_parameterized_gait_and_normalizes_hind_leg_aliases():
    """确认步态频率、占空比、相位与常见 HL/HR 别名可被参数化。"""
    intent = _intent("Go2 倒退走 0.3m/s 5秒", 0.3)
    intent.constraints = {
        "gait_type": "TROT", "gait_frequency": 1.5, "duty_factor": 0.55,
        "gait_phase_offsets": {"FR": 0.0, "HL": 0.0, "FL": 0.5, "HR": 0.5},
    }

    prototype = DynamicMotionPrototypeGenerator.generate(intent)

    assert prototype.gait.frequency == pytest.approx(1.5)
    assert prototype.gait.duty_factor == pytest.approx(0.55)
    assert prototype.gait.phase_offsets == {"FL": 0.5, "FR": 0.0, "RL": 0.0, "RR": 0.5}

    intent.constraints = {"gait_type": "PACE"}
    pace = DynamicMotionPrototypeGenerator.generate(intent)
    assert pace.gait.type == GaitType.PACE
    assert pace.gait.phase_offsets["FL"] == pace.gait.phase_offsets["RL"]
    assert pace.gait.phase_offsets["FR"] == pace.gait.phase_offsets["RR"]


def test_locomotion_prototype_cannot_exist_without_gait_or_matching_velocity():
    """确认 LOCOMOTION schema 强制包含步态和一致的低维速度目标。"""
    values = _dynamic_prototype().dict()
    values["gait"] = None
    with pytest.raises(ValueError, match="known gait pattern"):
        DynamicMotionPrototype(**values)

    values = _dynamic_prototype().dict()
    values["velocity"]["target_linear_velocity"]["x"] = 0.3
    with pytest.raises(ValueError, match="targets must match"):
        DynamicMotionPrototype(**values)


def test_standing_is_static_and_unknown_motion_does_not_get_guessed_as_dynamic():
    """确认站立分流至静态验证，未识别动作分类为 UNKNOWN。"""
    assert DynamicMotionPrototypeGenerator.classify(
        _intent("Go2 保持站立 5秒", action="stable_stand")) == MotionType.STATIC_POSE
    assert DynamicMotionPrototypeGenerator.classify(
        _intent("Go2 做一个说不清的动作", action="custom_motion")) == MotionType.UNKNOWN


def test_ik_reference_compiler_samples_cartesian_feet_each_policy_step():
    """逐时刻把语义足端目标交给 IK，并输出位置参考；该单元测试不是物理验收。"""
    torch = pytest.importorskip("torch")
    pytest.importorskip("pinocchio")
    settings = load_settings()
    model = RobotModelLoader(settings.training_root).load_robot_model(
        "go2", require_mjcf=False)
    env = _fake_dynamic_env(torch)
    env.default_dof_pos[0] = torch.tensor(
        [model.default_joint_positions[name] for name in env.dof_names])

    class RecordingIK:
        """记录目标并将真实 Pinocchio 多脚端 IK 结果返回给轨迹编译器。"""

        backend = "pinocchio"

        def __init__(self):
            self.targets = []
            self.solver = PinocchioIKSolver(max_iterations=60, tolerance=2.0e-3)

        def solve_ik(self, robot_model, target_pose, initial_configuration=None):
            self.targets.append(target_pose)
            return self.solver.solve_ik(
                robot_model, target_pose, initial_configuration)

    solver = RecordingIK()
    validator = IsaacGymDynamicValidator(
        ".", max_seconds=0.2, max_steps=10,
        environment_factory=lambda *_args: env, ik_solver=solver,
    )
    plan = validator._compile_ik_reference_actions(
        model.runtime_dict(), _dynamic_prototype(), env, env.dof_names, torch)

    assert plan["success"], plan.get("reason")
    assert plan["metrics"]["foot_trajectory_consumed"] is True
    assert plan["metrics"]["joint_reference_samples"] == 10
    assert plan["metrics"]["ik_samples"] == len(solver.targets) == 9
    trajectory_check = plan["metrics"]["trajectory_validation"]
    assert trajectory_check["stage"] == "joint_trajectory"
    assert trajectory_check["status"] in ("CONDITIONAL", "PASSED")
    assert tuple(plan["actions"].shape) == (10, 1, 12)
    assert all(set(target["feet"]) == set(model.foot_frames)
               for target in solver.targets)
    assert solver.targets[0]["feet"] != solver.targets[-1]["feet"]


def test_native_isaacgym_crash_is_converted_to_inconclusive_report():
    """模拟 gym_38.so 段错误，确认只返回能力级报告而不杀死调用方进程。"""
    calls = []

    def crashed_worker(command, **kwargs):
        """模拟原生 Gym 扩展导入时的负信号退出。"""
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=-11, stdout="Importing module gym_38.so",
                               stderr="")

    validator = IsaacGymDynamicValidator(
        ".", max_seconds=0.2, max_steps=10, process_runner=crashed_worker,
        visualize=True)
    report = validator.validate({"robot_name": "go2"}, _dynamic_prototype())

    assert calls and calls[0][0][-1] == (
        "rl_training_agent.feasibility.isaacgym_worker")
    request = json.loads(calls[0][1]["input"])
    assert request["visualize"] is True
    assert report.status == "UNAVAILABLE"
    assert report.backend == "isaacgym"
    assert report.success is None and report.validated is False
    assert report.validation_level == ValidationLevel.CAPABILITY_ONLY.value
    assert "exit code -11" in report.reason
    assert "未启动 PPO" in report.reason


def test_injected_dynamic_environment_never_claims_real_physics_or_trained_policy():
    """确认接口替身可以走完整指标采集，但只能产出 MOCK_VALIDATED。"""
    torch = pytest.importorskip("torch")
    env = _fake_dynamic_env(torch)
    joint_defaults = {name: 0.0 for name in env.dof_names}
    validator = IsaacGymDynamicValidator(
        ".", max_seconds=0.2, max_steps=10,
        environment_factory=lambda *_args: env,
    )

    report = validator.validate(
        {"robot_name": "go2", "default_joint_positions": joint_defaults},
        _dynamic_prototype(),
    )

    assert report.backend == "mock-isaacgym-dynamic"
    assert report.validation_level == "MOCK_VALIDATED", report.reason
    assert report.validated is False and report.success is None
    assert report.metrics["steps"] == 10
    assert report.metrics["velocity_tracking_error"] == pytest.approx(0.0, abs=1.0e-6)
    assert report.metrics["foot_contact_ratio"] == {key: 1.0 for key in ("FL", "FR", "RL", "RR")}
    assert report.metrics["gait_frequency_hz"] == pytest.approx(2.0)
    assert report.metrics["gait_duty_factor"] == pytest.approx(0.5)
    assert report.metrics["height_variation"] == pytest.approx(0.0)
    assert report.metrics["contact_balance"] == pytest.approx(1.0)
    assert report.metrics["policy_runner_created"] is False
    assert report.metrics["policy_runner_created"] is False
    assert report.metrics["policy_generated"] is False
    assert report.metrics["policy_trained"] is False
    assert env.gym.destroyed is True


def test_open_loop_action_uses_gait_frequency_duty_and_diagonal_phase_offsets():
    """确认 validator 按步态原型的相位/占空比逐腿产生不同的周期动作。"""
    torch = pytest.importorskip("torch")
    env = _fake_dynamic_env(torch)
    gait = GaitPattern(
        type=GaitType.TROT, frequency=2.0, duty_factor=0.5,
        phase_offsets={"FL": 0.5, "FR": 0.0, "RL": 0.0, "RR": 0.5})
    base = torch.zeros((1, 12), dtype=torch.float)

    at_start = IsaacGymDynamicValidator._step_action(
        env, base, "locomotion", 0.0, gait, -0.3, 0.25, torch)
    during_swing = IsaacGymDynamicValidator._step_action(
        env, base, "locomotion", 0.125, gait, -0.3, 0.25, torch)

    names = env.dof_names
    thigh = {leg: float(at_start[0, names.index("%s_thigh_joint" % leg)].item())
             for leg in ("FL", "FR", "RL", "RR")}
    assert thigh["FL"] == pytest.approx(thigh["RR"])
    assert thigh["FR"] == pytest.approx(thigh["RL"])
    assert thigh["FL"] == pytest.approx(-thigh["FR"])
    calf_indices = [names.index("%s_calf_joint" % leg) for leg in ("FL", "FR", "RL", "RR")]
    assert min(float(during_swing[0, index].item()) for index in calf_indices) < 0.0


def test_low_speed_velocity_threshold_does_not_accept_a_stationary_robot():
    """确认 0.3 m/s 任务不能因固定 0.35 m/s 误差上限而把静止误判为跟踪。"""
    torch = pytest.importorskip("torch")
    env = _fake_dynamic_env(torch)

    def remain_stationary(actions):
        """保持机身静止，同时提供有限关节/接触张量供完整指标流程运行。"""
        env.actions.copy_(actions)
        env.contact_forces[0, :4, 2] = 3.0
        return None, None, None, torch.tensor([False]), {}

    env.step = remain_stationary
    validator = IsaacGymDynamicValidator(
        ".", max_seconds=0.2, max_steps=10,
        environment_factory=lambda *_args: env)
    joint_defaults = {name: 0.0 for name in env.dof_names}

    report = validator.validate(
        {"robot_name": "go2", "default_joint_positions": joint_defaults},
        _dynamic_prototype())

    assert report.metrics["velocity_tracking_error"] == pytest.approx(0.3)
    assert report.metrics["velocity_error_limit"] == pytest.approx(0.075)
    assert "velocity_tracking_error_exceeded" in report.violations


def test_dynamic_validation_refuses_a_gaitless_locomotion_even_if_model_was_mutated():
    """确认绕过 Pydantic 拷贝产生的无步态 locomotion 不会开始仿真或通过。"""
    broken = _dynamic_prototype().copy(update={"gait": None})
    validator = IsaacGymDynamicValidator(".", environment_factory=lambda *_args: pytest.fail(
        "gaitless motion must be blocked before creating the simulator"))

    report = validator.validate({"robot_name": "go2"}, broken)

    assert report.validation_level == ValidationLevel.CAPABILITY_ONLY.value
    assert report.success is None
    assert "gait pattern" in report.reason


def test_mock_dynamic_pipeline_cannot_be_promoted_to_dynamic_physics_validated():
    """确认 Mock rollout 即使接口成功也绝不能放行成动态物理等级。"""
    settings = load_settings()
    pipeline = FeasibilityPipeline.mocked(settings.training_root)

    report = pipeline.assess(_intent("Go2 倒退走 0.3m/s 5秒", 0.3), _manifest())

    assert report.motion_type == MotionType.LOCOMOTION.value
    assert report.backend == "mock-isaacgym-dynamic"
    assert report.status == FeasibilityStatus.CAPABILITY_SUPPORTED
    assert report.validation_level == ValidationLevel.MOCK_VALIDATED.value
    assert report.dynamic_report["validation_level"] == ValidationLevel.MOCK_VALIDATED.value


def test_unknown_motion_is_blocked_without_calling_dynamic_validator():
    """确认 UNKNOWN 不会被交给物理动态控制器。"""
    class NeverRun:
        """捕获意外的动态 validator 调用。"""

        backend = "mock-dynamic"

        def validate(self, *_args):
            """未知类别必须在流水线分流阶段停止。"""
            pytest.fail("unknown motion must not enter dynamic validation")

    settings = load_settings()
    pipeline = FeasibilityPipeline(
        settings.training_root, ik_solver=MockIKSolver(),
        simulation_validator=MockMuJoCoValidator(), dynamic_validator=NeverRun(),
    )

    report = pipeline.assess(
        _intent("Go2 做一个未知动作", action="custom_motion"), _manifest())

    assert report.motion_type == MotionType.UNKNOWN.value
    assert report.validation_level == ValidationLevel.CAPABILITY_ONLY.value
    assert report.simulation_report == {}


def test_real_rollout_failure_contract_maps_to_physics_failed():
    """验证真实动态适配器的失败报告会被汇总为 PHYSICS_FAILED。"""
    class FailedRolloutResult:
        """用结构化后端回包隔离测试失败状态映射，不执行或声称真实仿真。"""

        backend = "isaacgym"

        def validate(self, *_args):
            """模拟已完成 rollout 且安全/跟踪验收失败的 adapter 回包。"""
            return SimulationReport(
                status="FAILED", success=False, backend="isaacgym", validated=True,
                validation_level=ValidationLevel.PHYSICS_FAILED.value,
                violations=["velocity_tracking_error_exceeded"],
                reason="velocity tracking exceeded threshold",
            )

    settings = load_settings()
    pipeline = FeasibilityPipeline(
        settings.training_root, ik_solver=MockIKSolver(),
        simulation_validator=MockMuJoCoValidator(), dynamic_validator=FailedRolloutResult(),
    )
    # Override the general capability checker only for this adapter-contract unit test.
    report = pipeline.assess(_intent("Go2 倒退走 0.3m/s 5秒", 0.3), _manifest())

    assert report.status == FeasibilityStatus.PHYSICS_FAILED
    assert report.validation_level == ValidationLevel.PHYSICS_FAILED.value
    assert report.dynamic_report["violations"] == ["velocity_tracking_error_exceeded"]


def test_training_ready_requires_dynamic_physics_reward_and_evaluation():
    """确认动态物理通过不是充分条件，训练配置和指标定义也必须齐全。"""
    assert not training_ready(ValidationLevel.STATIC_PHYSICS_VALIDATED.value, {"x": 1}, ["m"])
    assert not training_ready(ValidationLevel.DYNAMIC_PHYSICS_VALIDATED.value, {}, ["m"])
    assert not training_ready(ValidationLevel.DYNAMIC_PHYSICS_VALIDATED.value, {"x": 1}, [])
    assert training_ready(ValidationLevel.DYNAMIC_PHYSICS_VALIDATED.value, {"x": 1}, ["m"])


def test_state_machine_records_motion_generation_and_dynamic_validation(tmp_path):
    """确认动态验证阶段可幂等进入并在通过后继续上下文构建。"""
    machine = PersistentStateMachine(tmp_path / "state.json")
    machine.transition(AgentState.ENVIRONMENT_INSPECTED)
    machine.transition(AgentState.TASK_UNDERSTANDING)
    machine.transition(AgentState.TASK_FEASIBILITY_CHECK)
    machine.transition(AgentState.MOTION_PROTOTYPE_GENERATING,
                       operation_id="motion-generation-run-1")
    machine.transition(AgentState.DYNAMIC_MOTION_VALIDATION,
                       operation_id="dynamic-validation-run-1")
    machine.transition(AgentState.RAG_RETRIEVING)

    assert machine.record.state == AgentState.RAG_RETRIEVING
    assert machine.record.history[-3]["state"] == AgentState.MOTION_PROTOTYPE_GENERATING.value
    assert machine.record.history[-2]["state"] == AgentState.DYNAMIC_MOTION_VALIDATION.value
