"""验证 Isaac Gym validator 的 PPO action 映射、短 rollout 和 Mock 隔离。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from rl_training_agent.feasibility.ik.schema import IKResult
from rl_training_agent.feasibility.motion_prototype.schema import MotionPhase, MotionPrototype
from rl_training_agent.feasibility.simulation.isaacgym_validator import IsaacGymFeasibilityValidator


def _torch():
    """在真实测试环境不存在 PyTorch 时跳过依赖 Tensor 的 adapter 测试。"""
    return pytest.importorskip("torch")


def _fake_env(torch):
    """构造与 Unitree LeggedRobot step 接口相同、但不伪装成真实物理的替身。"""
    names = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    ]
    defaults = torch.zeros((1, 12), dtype=torch.float)
    cfg = SimpleNamespace(
        asset=SimpleNamespace(name="go2", self_collisions=1),
        init_state=SimpleNamespace(pos=[0.0, 0.0, 0.42]),
        control=SimpleNamespace(control_type="P", action_scale=0.25, decimation=4),
        normalization=SimpleNamespace(clip_actions=10.0),
    )

    class FakeGym:
        """提供 reset、refresh 和 destroy 所需的最小 Gym API。"""

        def __init__(self):
            self.destroyed = False

        def set_dof_state_tensor(self, *_args):
            """接受 adapter 设置初始关节状态。"""

        def set_actor_root_state_tensor(self, *_args):
            """接受 adapter 设置初始根状态。"""

        def refresh_dof_state_tensor(self, *_args):
            """模拟 Isaac Gym 关节状态刷新。"""

        def refresh_actor_root_state_tensor(self, *_args):
            """模拟 Isaac Gym 根状态刷新。"""

        def refresh_net_contact_force_tensor(self, *_args):
            """模拟 Isaac Gym 接触力刷新。"""

        def destroy_sim(self, _sim):
            """记录仿真句柄已被关闭。"""
            self.destroyed = True

    env = SimpleNamespace(
        cfg=cfg, dof_names=names, num_actions=12, num_dof=12, device="cpu", dt=0.02,
        default_dof_pos=defaults,
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
        dof_pos_limits=torch.tensor([[-3.0, 3.0]] * 12),
        torque_limits=torch.ones((12,), dtype=torch.float) * 50.0,
        torques=torch.zeros((1, 12), dtype=torch.float),
        contact_forces=torch.zeros((1, 5, 3), dtype=torch.float),
        feet_indices=torch.tensor([0, 1, 2, 3]),
        termination_contact_indices=torch.tensor([4]),
        rpy=torch.zeros((1, 3), dtype=torch.float),
        gym=FakeGym(), sim="fake-sim", viewer=None,
        _feasibility_randomization_disabled=["test-only"],
    )
    env.dof_pos = env.dof_state.view(1, 12, 2)[..., 0]
    env.dof_vel = env.dof_state.view(1, 12, 2)[..., 1]

    def step(actions):
        """令替身关节直接跟踪 PPO action，并返回站稳、足端接触状态。"""
        env.actions.copy_(actions)
        env.dof_pos[0].copy_(env.default_dof_pos[0] + actions[0] * cfg.control.action_scale)
        env.contact_forces[0, :4, 2] = 2.0
        return None, None, None, torch.tensor([False]), {}

    env.step = step
    env.compute_observations = lambda: None
    return env


def test_ik_joint_targets_compile_to_the_same_unitree_ppo_action_space():
    """验证关节角按 Unitree target=default+scale*action 的公式映射。"""
    torch = _torch()
    env = _fake_env(torch)
    joint_names = env.dof_names
    ik = IKResult(status="SOLVED", success=True, backend="pinocchio",
                  joint_positions={name: 0.25 for name in joint_names})

    action = IsaacGymFeasibilityValidator._compile_action_target(
        env, ik, joint_names, torch)

    assert action is not None
    assert torch.allclose(action, torch.ones((1, 12)))


def test_target_outside_real_ppo_action_clip_is_not_silently_clipped():
    """关节目标超出 PPO 动作裁剪范围时拒绝仿真，不能静默改写目标。"""
    torch = _torch()
    env = _fake_env(torch)
    ik = IKResult(status="SOLVED", success=True, backend="pinocchio",
                  joint_positions={name: 3.0 for name in env.dof_names})

    action = IsaacGymFeasibilityValidator._compile_action_target(
        env, ik, env.dof_names, torch)

    assert action is None


def test_injected_environment_short_rollout_is_never_labeled_physics_validated():
    """验证短时环境调用流程，同时确保测试替身绝不能产生真实物理等级。"""
    torch = _torch()
    env = _fake_env(torch)
    prototype = MotionPrototype(
        robot="go2", action="stable_stand", source="llm_semantic",
        phases=[MotionPhase(name="hold", duration_seconds=0.08,
                            body_goal={"torso": "upright", "feet": "support"})],
    )
    joint_positions = {name: 0.0 for name in env.dof_names}
    ik = IKResult(status="SOLVED", success=True, backend="pinocchio",
                  joint_positions=joint_positions)
    model = {"robot_name": "go2", "default_joint_positions": joint_positions}
    validator = IsaacGymFeasibilityValidator(
        training_root=".", max_seconds=0.1, max_steps=5,
        environment_factory=lambda *_args: env,
    )

    report = validator.validate_targets(model, prototype, [{
        "phase": "hold", "result": ik,
        "target": {"base_height": 0.42, "base_orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
    }])

    assert report.backend == "mock-isaacgym"
    assert report.status == "PASSED"
    assert report.validation_level == "MOCK_VALIDATED"
    assert report.validated is False
    assert report.metrics["policy_runner_created"] is False
    assert report.metrics["policy_trained"] is False
    assert env.gym.destroyed is True


def test_missing_ik_does_not_construct_or_run_simulation():
    """没有 IK 关节结果时直接报告不可用，不创建物理环境。"""
    prototype = MotionPrototype(
        robot="go2", action="stable_stand", source="llm_semantic",
        phases=[MotionPhase(name="hold", duration_seconds=0.1)],
    )
    validator = IsaacGymFeasibilityValidator(".", environment_factory=lambda *_args: pytest.fail(
        "must not construct environment without IK"))

    report = validator.validate("go2", prototype,
                                IKResult(status="UNAVAILABLE", success=None, backend="pinocchio"))

    assert report.backend == "mock-isaacgym"
    assert report.success is None
    assert report.validation_level == "CAPABILITY_ONLY"


def test_actual_urdf_source_is_reported_relative_to_training_root(tmp_path):
    """验证仿真报告不泄露开发机绝对资产路径，上传后仍保持可移植。"""
    validator = IsaacGymFeasibilityValidator(tmp_path)
    env = SimpleNamespace(_feasibility_asset=str(
        tmp_path / "resources" / "robots" / "go2" / "urdf" / "go2.urdf"))

    assert validator._display_asset(env) == "resources/robots/go2/urdf/go2.urdf"
