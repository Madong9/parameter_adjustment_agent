"""验证 Go2 的真实 URDF/Pinocchio 路径与 PPO Isaac Gym 后端选择。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from rl_training_agent.feasibility.agent import FeasibilityPipeline
from rl_training_agent.feasibility.ik.pinocchio_solver import PinocchioIKSolver
from rl_training_agent.feasibility.robot_models.go2 import Go2RobotModel
from rl_training_agent.feasibility.robot_models.loader import RobotModelLoader
from rl_training_agent.feasibility.simulation.isaacgym_validator import IsaacGymFeasibilityValidator
from rl_training_agent.settings import load_settings


def _settings_and_model():
    """加载当前仓库配置的训练根目录及不依赖 MJCF 的 Go2 URDF descriptor。"""
    settings = load_settings()
    loader = RobotModelLoader(settings.training_root)
    return settings, loader, loader.load_robot_model("go2", require_mjcf=False)


def test_go2_model_load_does_not_convert_urdf_to_mjcf():
    """确认 Go2 Feasibility 默认只读取真实 PPO URDF，不生成派生 MJCF。"""
    settings, loader, robot = _settings_and_model()
    environment = loader.inspect_environment()

    assert robot.model_status == "AVAILABLE", robot.model_error
    assert robot.urdf_source == "resources/robots/go2/urdf/go2.urdf"
    assert robot.mjcf_path is None
    assert robot.converted_model is False
    assert len(robot.joint_names) == 12
    assert len(robot.actuators) == 12
    assert environment.go2_urdf == robot.urdf_source
    assert environment.isaacgym_available
    assert environment.unitree_env_available
    assert (settings.training_root / robot.urdf_source).is_file()


def test_pinocchio_ik_converges_for_go2_nominal_standing_pose():
    """使用真实 Go2 URDF 正运动学生成站姿目标并验证 IK 收敛及关节限位。"""
    if importlib.util.find_spec("pinocchio") is None:
        pytest.skip("Pinocchio is not installed")
    _, _, robot = _settings_and_model()
    assert robot.model_status == "AVAILABLE", robot.model_error
    feet = Go2RobotModel.nominal_feet_positions(robot, {})
    target = {"feet": feet, "base_height": robot.base_initial_height,
              "time_seconds": 0.2, "base_orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}

    result = PinocchioIKSolver().solve_ik(
        robot.runtime_dict(), target, robot.default_joint_positions)

    assert result.backend == "pinocchio"
    assert result.success is True, result.reason
    assert result.residual <= 1.0e-3
    assert set(robot.default_joint_positions).issubset(result.joint_positions)
    for joint, value in result.joint_positions.items():
        limit = robot.limits[joint]
        assert limit["lower"] <= value <= limit["upper"]


def test_production_feasibility_backend_is_unitree_isaacgym_not_mujoco():
    """确认默认生产 pipeline 选择 Isaac Gym，且未探测/生成 Go2 MJCF。"""
    settings = load_settings()
    pipeline = FeasibilityPipeline(settings.training_root)

    if importlib.util.find_spec("pinocchio") is not None:
        assert isinstance(pipeline.simulation_validator, IsaacGymFeasibilityValidator)
        assert pipeline.simulation_validator.backend == "isaacgym"
        model = pipeline.robot_model_loader.load_robot_model("go2", require_mjcf=False)
        assert model.model_status == "AVAILABLE"
        assert model.mjcf_path is None
    else:
        # Missing physical dependencies use an explicitly tagged non-production Mock.
        assert "mock" in pipeline.simulation_validator.backend
