"""验证多层级动作分类、静态几何/力学检查和保守轨迹报告。"""
import pytest
import importlib.util

from rl_training_agent.environment.inspector import EnvironmentInspector
from rl_training_agent.feasibility.agent import FeasibilityPipeline
from rl_training_agent.feasibility.motion_prototype.dynamic_generator import (
    DynamicMotionPrototypeGenerator,
)
from rl_training_agent.feasibility.motion_prototype.foot_trajectory import FootTrajectoryGenerator
from rl_training_agent.feasibility.motion_prototype.schema import (
    FootTrajectory, GaitPattern, ManipulationPrototype, MotionPhase, MotionPrototype,
    MotionType, TargetPose,
)
from rl_training_agent.feasibility.motion_prototype.trajectory_generator import (
    TrajectoryPrototypeGenerator,
)
from rl_training_agent.feasibility.schema import (
    FeasibilityLevel, FeasibilityStatus, StageStatus,
)
from rl_training_agent.feasibility.validation.candidate_search import StaticPoseCandidateGenerator
from rl_training_agent.feasibility.validation.friction_validator import FrictionFeasibilityValidator
from rl_training_agent.feasibility.validation.static_validator import StaticPoseValidator
from rl_training_agent.feasibility.validation.torque_validator import TorqueFeasibilityValidator
from rl_training_agent.feasibility.validation.trajectory_validator import (
    JointTrajectory, JointTrajectoryPoint, TrajectoryValidator,
)
from rl_training_agent.schemas.agent_workflow import TaskIntentSpec
from rl_training_agent.settings import load_settings
from rl_training_agent.feasibility.ik.pinocchio_solver import PinocchioIKSolver
from rl_training_agent.feasibility.robot_models.go2 import Go2RobotModel
from rl_training_agent.feasibility.robot_models.loader import RobotModelLoader
from rl_training_agent.feasibility.simulation.mujoco_validator import MockMuJoCoValidator


def _intent(text, action="task", velocity=None):
    """构造无需外部模型调用的结构化任务意图。"""
    return TaskIntentSpec(original_instruction=text, robot="go2", action_name=action,
                          normalized_goal=text, target_velocity=velocity)


def test_action_classifier_separates_walk_jump_acrobatics_and_balance():
    """检查任务分类优先级，避免通用平衡要求吞掉行走意图。"""
    assert DynamicMotionPrototypeGenerator.classify(
        _intent("Go2 倒退走 0.3m/s", "backward_walk")) == MotionType.LOCOMOTION
    assert DynamicMotionPrototypeGenerator.classify(
        _intent("Go2 原地跳跃", "jump")) == MotionType.JUMP
    assert DynamicMotionPrototypeGenerator.classify(
        _intent("Go2 后空翻", "backflip")) == MotionType.ACROBATIC
    assert DynamicMotionPrototypeGenerator.classify(
        _intent("Go2 单腿站立", "single_leg_stand")) == MotionType.BALANCE
    assert DynamicMotionPrototypeGenerator.classify(
        _intent("Go2 单腿行走", "single_leg_walk")) == MotionType.BALANCE
    assert DynamicMotionPrototypeGenerator.classify(
        _intent("Go2 前腿站立", "front_leg_stand")) == MotionType.BALANCE
    assert DynamicMotionPrototypeGenerator.classify(
        _intent("Go2 后腿站立", "hind_leg_stand")) == MotionType.BALANCE


def test_backward_locomotion_contains_semantic_foot_trajectory():
    """确认倒退走原型有步态和足端参数，但不含关节/力矩控制。"""
    prototype = DynamicMotionPrototypeGenerator.generate(
        _intent("Go2 倒退走 0.3m/s", "backward_walk", 0.3))
    assert prototype.motion_type == MotionType.LOCOMOTION
    assert prototype.gait.type.value == "TROT"
    assert prototype.foot_trajectory.step_length == pytest.approx(0.15)
    assert prototype.foot_trajectory.step_period == pytest.approx(0.5)
    assert "joint_positions" not in prototype.dict()


def test_foot_trajectory_generator_samples_four_semantic_cartesian_targets():
    """验证参数化步态可采样四足笛卡尔目标，结果不是 actuator command。"""
    gait = GaitPattern()
    trajectory = FootTrajectory(step_length=0.15, step_period=0.5,
                                phase_offset=gait.phase_offsets)
    feet = {"FL": [0.2, 0.1, -0.3], "FR": [0.2, -0.1, -0.3],
            "RL": [-0.2, 0.1, -0.3], "RR": [-0.2, -0.1, -0.3]}
    sample = FootTrajectoryGenerator.sample(feet, gait, trajectory, 0.1)
    assert set(sample) == set(feet)
    assert all(len(value) == 3 for value in sample.values())
    assert any(value[2] > -0.3 for value in sample.values())


def test_support_polygon_detects_com_outside_and_degenerate_support():
    """验证支撑凸包内部/外部计算，单点足底时保持未知。"""
    polygon = [[-0.2, -0.1], [0.2, -0.1], [0.2, 0.1], [-0.2, 0.1]]
    inside = StaticPoseValidator.support_polygon_margin([0.0, 0.0], polygon)
    outside = StaticPoseValidator.support_polygon_margin([0.3, 0.0], polygon)
    degenerate = StaticPoseValidator.support_polygon_margin([0.0, 0.0], [[0.0, 0.0]])
    assert inside["status"] == "STATIC_STABLE" and inside["margin_m"] > 0.0
    assert outside["status"] == "STATIC_UNSTABLE" and outside["margin_m"] < 0.0
    assert degenerate["status"] == "UNKNOWN"


def test_joint_limit_failure_is_deterministic():
    """确认超过实际关节限位的 IK 解被单独拒绝。"""
    report = StaticPoseValidator.check_joint_limits(
        {"hip": 1.1}, {"hip": {"lower": -1.0, "upper": 1.0}})
    assert report.status == StageStatus.FAILED
    assert report.metrics["violations"] == ["joint_position_limit:hip"]


def test_torque_validator_rejects_limit_excess_and_missing_data_is_unknown():
    """检查力矩比较采用明确输入，缺少真实模型时不猜测。"""
    failed = TorqueFeasibilityValidator.validate({"hip": 11.0}, {"hip": 10.0})
    unknown = TorqueFeasibilityValidator.validate(None, {"hip": 10.0})
    assert failed.status == StageStatus.FAILED
    assert unknown.status == StageStatus.UNKNOWN


def test_friction_validator_checks_coulomb_cone_and_missing_force():
    """检查摩擦锥必要条件及无接触力证据时的 UNKNOWN。"""
    failed = FrictionFeasibilityValidator.validate(
        {"x": 20.0, "y": 0.0, "z": 10.0}, 0.5)
    unknown = FrictionFeasibilityValidator.validate(None, 0.5)
    assert failed.status == StageStatus.FAILED
    assert unknown.status == StageStatus.UNKNOWN


def test_joint_trajectory_validator_reports_velocity_limit_failure():
    """通过有限差分识别离散关节轨迹中的速度/连续性违规。"""
    trajectory = JointTrajectory(points=[
        JointTrajectoryPoint(time=0.0, positions={"hip": 0.0}),
        JointTrajectoryPoint(time=0.1, positions={"hip": 0.5}),
        JointTrajectoryPoint(time=0.2, positions={"hip": 0.0}),
    ])
    result = TrajectoryValidator.validate(
        trajectory, {"hip": {"lower": -1.0, "upper": 1.0, "velocity": 2.0}})
    assert result.status == StageStatus.FAILED
    assert "joint_velocity_limit:hip" in result.metrics["violations"]


def test_jump_and_acrobatic_generate_inconclusive_phase_scaffolds():
    """确认跳跃/空翻只有阶段骨架，不伪装成可执行物理轨迹。"""
    jump = TrajectoryPrototypeGenerator.generate(_intent("Go2 原地跳跃"), MotionType.JUMP)
    flip = TrajectoryPrototypeGenerator.generate(_intent("Go2 后空翻"), MotionType.ACROBATIC)
    assert [phase.name.value for phase in jump.phases] == [
        "PRELOAD", "TAKEOFF", "FLIGHT", "LANDING", "RECOVERY"]
    assert "ROTATION" in [phase.name.value for phase in flip.phases]
    assert any("不是 Go2 实测" in item for item in flip.assumptions)


def test_single_leg_and_handstand_never_claim_physics_validation():
    """离线检查复杂静态平衡任务转人工复核且不会调用 Mock 放行。"""
    settings = load_settings()
    manifest = EnvironmentInspector(settings.training_root).inspect("go2")
    pipeline = FeasibilityPipeline.mocked(settings.training_root)
    single = pipeline.assess(_intent("Go2 单腿站立 3 秒"), manifest)
    handstand = pipeline.assess(_intent("Go2 倒立站立"), manifest)
    assert single.status == FeasibilityStatus.CONDITIONAL
    assert single.static_dynamics_report["candidate_search"]["total_candidates"] == 4
    assert single.static_dynamics_report["candidate_search"]["evaluated_candidates"] == 0
    assert handstand.status == FeasibilityStatus.CONDITIONAL
    assert single.validation_level == "CAPABILITY_ONLY"
    assert single.feasibility_level != FeasibilityLevel.LEVEL_5_PHYSICS


def test_jump_does_not_route_through_locomotion_probe():
    """确保跳跃任务走阶段报告，不调用只支持 LOCOMOTION 的 Isaac Gym 探针。"""
    settings = load_settings()
    manifest = EnvironmentInspector(settings.training_root).inspect("go2")
    pipeline = FeasibilityPipeline.mocked(settings.training_root)
    report = pipeline.assess(_intent("Go2 原地跳跃"), manifest)
    assert report.status == FeasibilityStatus.CONDITIONAL
    assert report.motion_type == MotionType.JUMP.value
    assert report.simulation_result is None
    assert report.trajectory_report["status"] == "INCONCLUSIVE"


def test_report_exposes_each_validation_stage_and_confidence_basis():
    """验证统一报告输出分级字段，confidence 明确不是学习成功率。"""
    settings = load_settings()
    manifest = EnvironmentInspector(settings.training_root).inspect("go2")
    pipeline = FeasibilityPipeline.mocked(settings.training_root)
    report = pipeline.assess(_intent("Go2 倒退走 0.3m/s", "backward_walk", 0.3), manifest)
    assert report.action_type == MotionType.LOCOMOTION.value
    assert report.feasibility_level in FeasibilityLevel
    assert "不是机器人动作成功率" in report.confidence_basis
    assert {item.stage for item in report.stage_reports} == {
        "language_understanding", "capability", "kinematic", "static_dynamics",
        "trajectory", "physics", "motion_planning", "whole_body_solve"}
    assert report.evidence


def test_local_deterministic_motion_fallback_is_not_mislabeled_as_mock():
    """本地规则回退不是 Mock backend，但仍由 provider_failed 阻止完整任务放行。"""
    prototype = MotionPrototype(
        robot="go2", action="stable_stand", source="deterministic_fallback",
        phases=[MotionPhase(name="stand", duration_seconds=1.0)],
    )
    evidence_mode = FeasibilityPipeline._evidence_mode(
        [type("IK", (), {"backend": "pinocchio"})()],
        [type("Physics", (), {"backend": "isaacgym"})()], prototype)
    assert evidence_mode == "REAL"


def test_real_pinocchio_standing_candidate_reports_com_and_gravity_torque():
    """用仓库真实 Go2 URDF 离线计算站姿 CoM 和固定根重力力矩估算。"""
def test_front_and_hind_support_candidate_search_has_three_lift_heights():
    """验证前/后腿支撑候选枚举不同抬脚高度且只输出足端笛卡尔目标。"""
    pytest.importorskip("pinocchio")
    settings = load_settings()
    model = RobotModelLoader(settings.training_root).load_robot_model("go2")
    front = StaticPoseCandidateGenerator.leg_pair_candidates(model, "front")
    hind = StaticPoseCandidateGenerator.leg_pair_candidates(model, "hind")

    assert len(front) == len(hind) == 3
    assert all(item.support_legs == ["FL", "FR"] for item in front)
    assert all(item.support_legs == ["RL", "RR"] for item in hind)
    assert front[0].target["feet"]["RL_foot"][2] < front[-1].target["feet"]["RL_foot"][2]
    assert all("joint_positions" not in item.target for item in front + hind)


def test_single_leg_pose_candidate_search_uses_real_ik_but_remains_conditional():
    """验证四种支撑腿都运行真实 IK，但不把运动学候选当成平衡通过。"""
    pytest.importorskip("pinocchio")
    settings = load_settings()
    manifest = EnvironmentInspector(settings.training_root).inspect("go2")
    pipeline = FeasibilityPipeline(
        settings.training_root, ik_solver=PinocchioIKSolver(),
        simulation_validator=MockMuJoCoValidator(),
    )

    report = pipeline.assess(_intent("Go2 单腿站立 3 秒", "single_leg_stand"), manifest)
    search = report.static_dynamics_report["candidate_search"]

    assert report.status == FeasibilityStatus.CONDITIONAL
    assert search["total_candidates"] == 4
    assert search["evaluated_candidates"] == 4
    assert search["search_status"] in ("INCONCLUSIVE", "SEARCH_FAILED")
    assert report.validation_level == "CAPABILITY_ONLY"
    assert search["interpretation"].startswith("有限候选搜索结果")


def test_manipulation_prototype_is_cartesian_only_and_closed_schema():
    """确保机械臂目标 Schema 接收末端笛卡尔目标并拒绝关节控制字段。"""
    prototype = ManipulationPrototype(
        robot="g1", action="reach", end_effector_frame="left_hand",
        target=TargetPose(frame="left_hand", position=[0.3, 0.1, 0.4]),
    )
    assert prototype.target.position == [0.3, 0.1, 0.4]
    with pytest.raises(ValueError, match="extra fields"):
        ManipulationPrototype(
            robot="g1", action="reach", end_effector_frame="left_hand",
            target=TargetPose(frame="left_hand", position=[0.3, 0.1, 0.4]),
            joint_positions={"elbow": 1.2},
        )


    if importlib.util.find_spec("pinocchio") is None:
        pytest.skip("Pinocchio is not installed")
    settings = load_settings()
    model = RobotModelLoader(settings.training_root).load_robot_model("go2", require_mjcf=False)
    feet = Go2RobotModel.nominal_feet_positions(model, {})
    target = {"feet": feet, "base_height": model.base_initial_height,
              "base_orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
    ik = PinocchioIKSolver().solve_ik(model.runtime_dict(), target,
                                      model.default_joint_positions)
    assert ik.success is True, ik.reason
    report = StaticPoseValidator().assess(model.runtime_dict(), target, ik.joint_positions)
    assert report["center_of_mass_support_polygon"]["status"] == StageStatus.PASSED
    torque = report["torque"]
    assert torque["status"] == StageStatus.CONDITIONAL
    assert torque["backend"] == "pinocchio_rnea_fixed_base_gravity_only"
    assert torque["metrics"]["gravity_torque_estimate_by_joint"]
