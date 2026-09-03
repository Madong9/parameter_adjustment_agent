import json
import math

import pytest

from rl_training_agent.environment.inspector import EnvironmentInspector
from rl_training_agent.rewards.compiler import RewardCompiler
from rl_training_agent.rewards.validator import RewardCodeValidator, RewardPlanValidator, RewardValidationError
from rl_training_agent.schemas.rewards import CurriculumStage, RewardPlan, RewardTerm
from rl_training_agent.settings import load_settings
from rl_training_agent.utils.io import write_json
from rl_training_agent.utils.paths import ensure_within


def term(name="tracking_lin_vel", weight=1.0):
    """执行 term 对应的业务逻辑并返回结果。"""
    return RewardTerm(name=name, implementation="registry:" + name, purpose="test", weight=weight,
                      expected_training_trend="increase")


def plan(terms):
    """生成并验证任务与候选配置但不启动训练。"""
    return RewardPlan(task_id="task-x", version=1, design_rationale=["test"], terms=terms)


def test_registry_parses_actual_project():
    """验证“registry parses actual project”场景的预期行为。"""
    settings = load_settings()
    manifest = EnvironmentInspector(settings.training_root).inspect("go2")
    names = {item.name for item in manifest.rewards}
    assert {"tracking_lin_vel", "torques", "action_rate", "rear_leg_stand", "rear_leg_walk",
            "front_leg_stand", "front_leg_walk"} <= names
    tracking = next(item for item in manifest.rewards if item.name == "tracking_lin_vel")
    assert tracking.default_weight == 1.0


def test_missing_reward_and_large_weight():
    """验证“missing reward and large weight”场景的预期行为。"""
    manifest = EnvironmentInspector(load_settings().training_root).inspect("go2")
    validator = RewardPlanValidator(manifest.rewards, 10)
    with pytest.raises(RewardValidationError, match="not present"):
        validator.validate(plan([term("not_real")]))
    with pytest.raises(RewardValidationError, match="safety limit"):
        validator.validate(plan([term(weight=11)]))


def test_nan_weight_rejected_by_schema():
    """验证“nan weight rejected by schema”场景的预期行为。"""
    with pytest.raises(ValueError):
        term(weight=float("nan"))


def test_penalty_sign_and_parameter_type():
    """验证“penalty sign and parameter type”场景的预期行为。"""
    manifest = EnvironmentInspector(load_settings().training_root).inspect("go2")
    with pytest.raises(RewardValidationError, match="positive sign"):
        RewardPlanValidator(manifest.rewards).validate(plan([term("torques", 1.0)]))
    for name in ("orientation", "base_height"):
        with pytest.raises(RewardValidationError, match="positive sign"):
            RewardPlanValidator(manifest.rewards).validate(plan([term(name, 1.0)]))
    with pytest.raises(ValueError):
        RewardTerm(name="x", implementation="x", purpose="x", weight=1, parameters=[])


def test_positive_reward_cannot_be_compiled_as_penalty():
    """验证任务正奖励不能被模型误设为负权重。"""
    manifest = EnvironmentInspector(load_settings().training_root).inspect("go2")
    with pytest.raises(RewardValidationError, match="unsafe negative sign"):
        RewardPlanValidator(manifest.rewards).validate(plan([term("tracking_lin_vel", -1.0)]))


def test_curriculum_boundaries_must_be_nonnegative_and_ordered():
    """验证模型不能生成负数或起止颠倒的课程迭代区间。"""
    with pytest.raises(ValueError, match="nonnegative"):
        CurriculumStage(name="错误阶段", start_iteration=-1, end_iteration=10)
    with pytest.raises(ValueError, match="must not exceed"):
        CurriculumStage(name="错误阶段", start_iteration=20, end_iteration=10)


def test_json_artifacts_replace_nonfinite_values_with_null(tmp_path):
    """验证缺失指标不会以非标准 NaN/Infinity 文本污染产物或 GPT 证据。"""
    path = tmp_path / "metrics.json"
    write_json(path, {"missing": float("nan"), "overflow": float("inf"), "valid": 1.0})
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"missing": None, "overflow": None, "valid": 1.0}


def test_compiler_copies_config_and_diff(tmp_path):
    """验证“compiler copies config and diff”场景的预期行为。"""
    manifest = EnvironmentInspector(load_settings().training_root).inspect("go2")
    metadata = RewardCompiler(manifest).compile(plan([term()]), tmp_path)
    assert (tmp_path / "config.yaml").is_file() and (tmp_path / "config.diff").is_file()
    assert json.loads((tmp_path / "config.yaml").read_text())["rewards"]["scales"]["tracking_lin_vel"] == 1.0
    assert json.loads((tmp_path / "config.yaml").read_text())["rewards"]["terms"][0]["active_phases"] == ["all"]
    assert len(metadata["config_hash"]) == 64


def test_path_escape(tmp_path):
    """验证“path escape”场景的预期行为。"""
    with pytest.raises(ValueError, match="escapes"):
        ensure_within(tmp_path.parent / "outside", tmp_path / "inside")


def test_ast_dangerous_calls_and_tensor_whitelist():
    """验证“ast dangerous calls and tensor whitelist”场景的预期行为。"""
    validator = RewardCodeValidator(["base_lin_vel"])
    with pytest.raises(RewardValidationError):
        validator.validate_ast("import os\ndef reward(tensors):\n return tensors['base_lin_vel'][:, 0]")
    with pytest.raises(RewardValidationError, match="whitelist"):
        validator.validate_ast("def reward(tensors):\n return tensors['secret'][:, 0]")


def test_tensor_shape_and_finite_smoke():
    """验证“tensor shape and finite smoke”场景的预期行为。"""
    validator = RewardCodeValidator(["base_lin_vel"])
    stats = validator.tensor_smoke_test("def reward(tensors):\n return torch.square(tensors['base_lin_vel'][:, 0])")
    assert math.isfinite(stats["mean"])
    with pytest.raises(RewardValidationError, match="shape"):
        validator.tensor_smoke_test("def reward(tensors):\n return tensors['base_lin_vel']")
    with pytest.raises(RewardValidationError, match="NaN"):
        validator.tensor_smoke_test("def reward(tensors):\n return tensors['base_lin_vel'][:, 0] / 0")
