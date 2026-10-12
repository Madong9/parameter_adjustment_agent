"""覆盖验收可信度、恢复、模型上下文、运行隔离和记忆兼容性缺口。"""
import ast
import json
import math
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from rl_training_agent.evaluation.deterministic import DeterministicEvaluator
from rl_training_agent.memory.compatibility import compatibility_fingerprint
from rl_training_agent.memory.consolidator import SemanticMemoryConsolidatorAgent
from rl_training_agent.memory.store import LongTermMemoryStore
from rl_training_agent.memory.evidence_snapshot import create_evidence_snapshot
from rl_training_agent.metrics.trajectory_metrics import TrajectoryMetrics
from rl_training_agent.orchestration.orchestrator import TrainingOrchestrator
from rl_training_agent.providers.browser_lease import BrowserLease
from rl_training_agent.providers.errors import ProviderTimeout
from rl_training_agent.providers.mock_provider import MockLLMReasoningProvider
from rl_training_agent.providers.opencli_chatgpt import OpenCLIChatGPTWebProvider
from rl_training_agent.schemas.agent_workflow import LongTermMemoryRecord
from rl_training_agent.schemas.experiments import ConversationHandle, ProviderHealth
from rl_training_agent.schemas.task import TaskSpec
from rl_training_agent.settings import OpenCLISettings, Settings, load_settings
from rl_training_agent.storage.locking import FileLock
from rl_training_agent.training import real_rollout
from rl_training_agent.utils.io import write_json


def settings_for(tmp_path, **overrides):
    """构造使用临时产物目录和极小离线预算的配置。"""
    return Settings(**dict(load_settings().dict(), **{
        "experiment_root": str(tmp_path / "experiments"),
        "artifact_root": str(tmp_path / "artifacts"),
        "memory_root": str(tmp_path / "memory"),
        "num_reward_candidates": 1, "smoke_iterations": 1,
        "screening_iterations": 2, "full_iterations": 3,
        "evaluation_seeds": [1, 2], "rollouts_per_seed": 1,
        **overrides}))


def trajectory():
    """构造明确具有两步离地时间差和真实落地角度的四足轨迹。"""
    data = pd.DataFrame({"sim_time": np.arange(7) * 0.02,
                         "roll": 0.0, "pitch": [0, 0, 0, 0, 0, 0.22, 0.1]})
    for foot in ("fl", "fr", "rl", "rr"):
        data["contact_" + foot] = [True, True, True, False, False, True, True]
    data["contact_fl"] = [True, False, False, False, False, True, True]
    data["joint_positions"] = [[0.0, 0.0] for _ in range(7)]
    data["joint_position_limits"] = [[[-1, 1], [-2, 2]] for _ in range(7)]
    data["joint_torques"] = [[0.1, 0.2] for _ in range(7)]
    data["joint_torque_limits"] = [[10, 20] for _ in range(7)]
    return data


def test_jump_metrics_are_measured_and_not_fixed():
    """离地不同步与落地倾斜必须反映在真实验收指标中。"""
    metrics = TrajectoryMetrics().compute(trajectory())
    assert metrics["takeoff_time_spread"] == pytest.approx(0.04)
    assert metrics["landing_pitch_abs"] == pytest.approx(0.22)
    assert metrics["joint_limit_violations"] == metrics["torque_limit_violations"] == 0


def test_jump_acceptance_contract_uses_registered_measured_quantities(tmp_path):
    """真实已实现的跳跃指标须注册，并保留起跳时序和落地姿态而非误映射其他量。"""
    settings = settings_for(tmp_path)
    provider = MockLLMReasoningProvider(1)
    orchestrator = TrainingOrchestrator(settings, provider)
    manifest = orchestrator.inspector.inspect("go2")
    task = TaskSpec.parse_obj(provider.design_task_and_rewards(
        "Go2 原地跳跃", "go2", manifest.dict())["task_spec"])
    from rl_training_agent.environment.metric_registry import normalize_task_metrics
    assert normalize_task_metrics(task, manifest.evaluation_metrics) == []
    assert orchestrator.inspector.validate_task(task, manifest)[0] == []
    assert {item.name for item in task.success_metrics} == {
        "jump_height", "takeoff_time_spread", "landing_pitch_abs"}
    numeric = TrajectoryMetrics().compute(trajectory())
    assert numeric["landing_pitch_abs"] == pytest.approx(0.22)
    # 验收指标可计算，不等于通用跳跃物理规划后端已经实现；这里不绕过准入训练。
    assert all(DeterministicEvaluator()._check(item, numeric).passed
               for item in task.success_metrics if item.name != "jump_height")


@pytest.mark.parametrize("kind", ["no_flight", "no_landing", "reset", "missing_contact"])
def test_missing_jump_evidence_cannot_pass(kind):
    """没有完整离地和落地证据时必须报告未知而非默认达标值。"""
    data = trajectory()
    if kind == "no_flight":
        data[["contact_fl", "contact_fr", "contact_rl", "contact_rr"]] = True
    elif kind == "no_landing":
        data.loc[5:, ["contact_fl", "contact_fr", "contact_rl", "contact_rr"]] = False
    elif kind == "reset":
        data["termination_reason"] = ""
        data.loc[4, "termination_reason"] = "reset"
    else:
        data = data.drop(columns="contact_rr")
    metrics = TrajectoryMetrics().compute(data)
    assert math.isnan(metrics["takeoff_time_spread"])
    assert math.isnan(metrics["landing_pitch_abs"])


@pytest.mark.parametrize("column", ["joint_positions", "joint_velocities", "actions", "joint_torques"])
def test_nested_nonfinite_values_are_detected(column):
    """关节、速度、动作和力矩中的嵌套 NaN/Inf 不能逃过数值安全检查。"""
    data = trajectory()
    data[column] = [[0.0, float("nan"), float("inf")] for _ in range(7)]
    assert TrajectoryMetrics().compute(data)["nan_count"] == 14


def test_real_model_limits_and_parquet_arrays(tmp_path):
    """实际模型限位可从 Parquet 嵌套数组读取，越限与缺失均不能按零处理。"""
    data = trajectory()
    data.at[2, "joint_positions"] = [1.2, 0]
    data.at[3, "joint_torques"] = [0, -21]
    path = tmp_path / "trajectory.parquet"
    data.to_parquet(path)
    metrics = TrajectoryMetrics().compute(pd.read_parquet(path))
    assert metrics["joint_limit_violations"] == metrics["torque_limit_violations"] == 1
    data = data.drop(columns=["joint_position_limits", "joint_torque_limits"])
    metrics = TrajectoryMetrics().compute(data)
    assert math.isnan(metrics["joint_limit_violations"])
    assert math.isnan(metrics["torque_limit_violations"])


def test_reward_soft_limits_are_not_misreported_as_physical_hard_limits():
    """奖励软限位越界单独记录，不得冒充 URDF/Isaac Gym 硬关节限位越界。"""
    data = trajectory()
    data.at[2, "joint_positions"] = [0.9, 0]
    data["joint_soft_position_limits"] = [[[-0.8, 0.8], [-1.8, 1.8]] for _ in range(7)]
    metrics = TrajectoryMetrics().compute(data)
    assert metrics["joint_limit_violations"] == 0
    assert metrics["joint_soft_limit_violations"] == 1


def test_missing_safety_evidence_blocks_joint_acceptance():
    """任务与视觉都通过时，缺少力矩/关节安全证据仍不得完成验收。"""
    provider = MockLLMReasoningProvider(1)
    task = TaskSpec.parse_obj(provider.design_task_and_rewards("稳定行走", "go2", {})["task_spec"])
    visual = provider.critique_visual_behavior(task, [])
    metrics = {"tracking_error": 0, "fall_rate": 0, "nan_count": 0,
               "joint_limit_violations": 0, "torque_limit_violations": 0,
               "forbidden_collisions": 0, "abnormal_terminations": 0}
    evaluator = DeterministicEvaluator()
    assert evaluator.evaluate(task, metrics, visual).completed
    for name in ("joint_limit_violations", "torque_limit_violations"):
        missing = dict(metrics)
        missing.pop(name)
        result = evaluator.evaluate(task, missing, visual)
        assert not result.completed and "safety_evidence:" + name in result.violations


def test_rollout_video_sampling_cannot_skip_physics_recording():
    """结构性保护控制步记录：视频分支之外记录轨迹，不允许 continue 跳过物理数据。"""
    tree = ast.parse(Path(real_rollout.__file__).read_text(encoding="utf-8"))
    loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For) and
                isinstance(node.target, ast.Name) and node.target.id == "step")
    assert not any(isinstance(node, ast.Continue) for node in ast.walk(loop))
    assert any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and
               isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "append" and
               isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == "rows"
               for node in loop.body)


def test_json_recovery_replays_original_context_and_files(tmp_path, monkeypatch):
    """JSON 修复超时换会话时，原始动作、能力和视觉附件必须全部保留。"""
    class Reply(BaseModel):
        """定义测试用结构化结果。"""
        ok: bool

    provider = OpenCLIChatGPTWebProvider(OpenCLISettings(max_retries=1))
    calls = []
    monkeypatch.setattr(provider, "new_conversation", lambda title: ConversationHandle(
        conversation_id=title, title_hint=title))
    monkeypatch.setattr(provider, "_latest_assistant", lambda: "")

    def send(conversation, prompt, title, files=None):
        """模拟原回复错误、原会话修复超时、新会话成功。"""
        calls.append((conversation.conversation_id, prompt, files))
        if len(calls) == 1:
            return "not JSON"
        if len(calls) == 2:
            raise ProviderTimeout("发送确认失败")
        return '{"ok":true}'

    monkeypatch.setattr(provider, "_send_model_prompt", send)
    original = "Go2 后腿站立五秒；ENVIRONMENT_MANIFEST:真实关节与接触能力"
    files = [tmp_path / "contact_sheet.png"]
    assert provider._request_model(original, Reply, "design", files).ok
    assert len(calls) == 3 and "recovery" in calls[-1][0]
    assert all(original in prompt and attachments == files for _, prompt, attachments in calls)


def _acquire_browser_in_child(root, acquired):
    """在独立进程尝试相同 profile 的浏览器租约。"""
    from rl_training_agent.providers import browser_lease
    browser_lease.AGENT_ROOT = Path(root)
    with BrowserLease().acquire("shared-profile"):
        acquired.set()


def test_browser_profile_lock_is_cross_process_and_reentrant(tmp_path, monkeypatch):
    """不同进程共用 profile 必须等待；同 Provider 嵌套调用不能自锁。"""
    monkeypatch.setattr("rl_training_agent.providers.browser_lease.AGENT_ROOT", tmp_path)
    context = multiprocessing.get_context("spawn")
    acquired = context.Event()
    child = context.Process(target=_acquire_browser_in_child, args=(str(tmp_path), acquired))
    lease = BrowserLease()
    try:
        with lease.acquire("shared-profile"):
            with lease.acquire("shared-profile"):
                child.start()
                assert not acquired.wait(0.3)
        assert acquired.wait(10)
        child.join(10)
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.terminate()
            child.join(5)


def test_task_lock_blocks_training_and_resume_before_writes(tmp_path):
    """同任务运行或恢复不能覆盖正在进行的状态与验收合同。"""
    settings = settings_for(tmp_path)
    orchestrator = TrainingOrchestrator(settings, MockLLMReasoningProvider(1))
    task_id = orchestrator._task_id("测试向前走", "go2")
    directory = orchestrator.store.task_dir(task_id)
    with FileLock(directory / ".run.lock"):
        for call in (lambda: orchestrator.train("测试向前走", "go2", True),
                     lambda: orchestrator.resume(task_id, True)):
            with pytest.raises(RuntimeError, match="已有运行进程"):
                call()
        assert not (directory / "state.json").exists()


def test_doctor_checks_configured_roles_without_bailian(tmp_path, monkeypatch):
    """全 ChatGPT 角色不依赖百炼，也不会误检查未配置的豆包主备。"""
    import torch
    from rl_training_agent import cli
    settings = settings_for(tmp_path, provider="multi-agent", task_planner_provider="opencli",
                            reward_designer_provider="opencli", visual_critic_provider="opencli",
                            diagnosis_provider="opencli")
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    health = ProviderHealth(available=True, image_upload_supported=True, opencli_available=True,
                            extension_connected=True, chatgpt_logged_in=True)
    monkeypatch.setattr(TrainingOrchestrator, "provider_for", staticmethod(
        lambda settings, name: SimpleNamespace(doctor=lambda: health)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())

    def forbidden_bailian():
        """当前未配置百炼，禁止健康检查实例化该服务。"""
        raise AssertionError("不应检查百炼")

    monkeypatch.setattr(cli, "BailianGLMProvider", forbidden_bailian)
    report = cli.doctor()
    assert report["production_ready"] and report["bailian"]["required"] is False


class RepairingProvider(MockLLMReasoningProvider):
    """首次生成非法奖励，再根据审查意见恢复真实注册奖励。"""

    def __init__(self, always_bad=False):
        """记录设计次数和重试上下文。"""
        super().__init__(1)
        self.calls = []
        self.always_bad = always_bad

    def design_task_and_rewards(self, instruction, robot, capabilities):
        """模拟错误奖励与模型改写原始指令，但不改变真实测试目标。"""
        self.calls.append(capabilities)
        design = super().design_task_and_rewards(instruction, robot, capabilities)
        design["task_spec"]["original_instruction"] = "模型擅自改成跳跃"
        if len(self.calls) == 1 or self.always_bad:
            design["reward_plans"][0]["terms"][0]["name"] = "invented_reward"
        return design


def test_business_review_retry_preserves_user_target_and_contract(tmp_path):
    """全部拒绝后回传业务意见重新设计，不消耗训练修订额度、不替换用户目标。"""
    provider = RepairingProvider()
    settings = settings_for(tmp_path)
    result = TrainingOrchestrator(settings, provider).train("稳定向前行走", "go2", True)
    assert result["state"] == "COMPLETED" and len(provider.calls) == 2
    feedback = provider.calls[1]["reward_review_feedback"]
    assert feedback["review"]["rejected_candidate_indexes"] == [1]
    assert feedback["fixed_task_spec"]["original_instruction"] == "稳定向前行走"
    task_dir = settings.experiments_path / result["task_id"]
    task = json.loads((task_dir / "task_spec.json").read_text())
    assert task["original_instruction"] == "稳定向前行走"
    assert result["used_revisions"] == 0
    assert (task_dir / "reward_review_attempt_01.json").is_file()
    original = json.loads((task_dir / "design_response_attempt_00.json").read_text())
    assert original["reward_plans"][0]["terms"][0]["name"] == "invented_reward"
    assert (task_dir / "plan_normalization_attempt_01.json").is_file()


def test_business_review_retry_is_bounded_and_never_trains_rejected_rewards(tmp_path):
    """持续错误的候选在有限重试后仍停在人工复核，不能启动训练。"""
    provider = RepairingProvider(always_bad=True)
    settings = settings_for(tmp_path, reward_design_review_retries=1)
    result = TrainingOrchestrator(settings, provider).train("稳定向前行走", "go2", True)
    assert result["state"] == "HUMAN_REVIEW" and len(provider.calls) == 2
    assert not list(settings.experiments_path.rglob("model_*.pt"))


def test_restore_rehashes_actual_config_content(tmp_path):
    """元数据哈希一致也不能掩盖配置文件被改动。"""
    settings = settings_for(tmp_path)
    orchestrator = TrainingOrchestrator(settings, MockLLMReasoningProvider(1))
    result = orchestrator.train("稳定向前行走", "go2", True)
    task_dir = settings.experiments_path / result["task_id"]
    candidate = result["selected_experiment"]
    assert orchestrator._restore_selected_candidate(task_dir, candidate)["checkpoint"].is_file()
    config = task_dir / "candidates" / candidate / "config.yaml"
    config.write_text(config.read_text() + "\n ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="配置哈希"):
        orchestrator._restore_selected_candidate(task_dir, candidate)


def test_full_training_without_new_checkpoint_cannot_reuse_smoke_checkpoint(tmp_path, monkeypatch):
    """真实训练进程即使退出码为零，也不能用冒烟旧 checkpoint 充当新种子结果。"""
    settings = settings_for(tmp_path)
    orchestrator = TrainingOrchestrator(settings, MockLLMReasoningProvider(1))
    monkeypatch.setattr(orchestrator, "_dry_train", orchestrator._real_train)
    monkeypatch.setattr(orchestrator, '_pilot_reward_quality', lambda *args: True)
    from rl_training_agent.metrics.tensorboard_reader import TensorBoardReader
    monkeypatch.setattr(TensorBoardReader, "latest", lambda self, directory: {
        "Episode/rew_raw_mean_tracking_lin_vel": 0.8})

    def smoke(experiment_id, robot, config, iterations, seed, directory):
        """只为冒烟阶段生成旧 checkpoint，后续阶段不产生新文件。"""
        checkpoint = directory / "checkpoints/smoke/model_1.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text("old smoke model")
        return SimpleNamespace(exit_code=0, timed_out=False)

    monkeypatch.setattr(orchestrator.controller, "run_smoke_training", smoke)
    monkeypatch.setattr(orchestrator.controller, "continue_training", lambda *args: SimpleNamespace(exit_code=0))
    monkeypatch.setattr(orchestrator.controller, "run_full_training", lambda *args: SimpleNamespace(exit_code=0))
    with pytest.raises(RuntimeError, match="produced no checkpoint for seed"):
        orchestrator.train("稳定向前行走", "go2", True)
    task_id = orchestrator._task_id("稳定向前行走", "go2")
    assert json.loads((settings.experiments_path / task_id / "state.json").read_text())["state"] == "FAILED"


def episodic(index, robot="go2", env="env-v1", registry="reward-v1"):
    """构造版本可追溯的已验证情景记忆。"""
    return LongTermMemoryRecord(
        memory_id="memory-%d" % index, task_id="task-%d" % index, robot=robot,
        action_name="后腿站立行走", normalized_goal="后腿站立行走", outcome="completed",
        reward_version=1, reward_terms=[{"name": "rear_leg_stand"}, {"name": "rear_leg_walk"}],
        environment_version=env, reward_registry_version=registry,
        created_at="2026-10-03T00:00:00+00:00")


def attach_episodic_snapshot(store, record, tmp_path):
    """给纯检索合成样本保存最小真值快照，不放宽生产证据门控。"""
    task_dir = tmp_path / "snapshot-sources" / record.task_id
    values = {
        "task_spec.json": {"task_id": record.task_id, "robot": record.robot},
        "final/reward_plan.json": {"version": record.reward_version, "terms": record.reward_terms},
        "rollouts/evaluation.json": {"completed": True, "hard_constraints_passed": True,
                                     "task_metrics_passed": True, "visual_alignment_passed": True},
    }
    for source, value in values.items():
        write_json(task_dir / source, value)
    root, hashes = create_evidence_snapshot(store.root, record.memory_id, task_dir,
                                           [task_dir / source for source in values])
    record.evidence_snapshot_root = root
    record.evidence_snapshot_hashes = hashes


def test_memory_hard_filters_robot_environment_and_reward_registry(tmp_path):
    """跨机器人、环境或奖励版本的经验与未标版本的旧记忆不得送入生产提示词。"""
    store = LongTermMemoryStore(tmp_path / "memory", tmp_path, max_context_chars=20000)
    for record in (episodic(1), episodic(2, robot="g1"), episodic(3, env="env-v2"),
                   episodic(4, registry="reward-v2"), episodic(5, env="unknown")):
        attach_episodic_snapshot(store, record, tmp_path)
        store.promote(record)
    result = store.retrieve("后腿站立", "go2", environment_version="env-v1",
                            reward_registry_version="reward-v1")
    assert [hit["record"]["memory_id"] for hit in result["hits"]] == ["memory-1"]


def test_semantic_consolidation_cannot_mix_incompatible_evidence(tmp_path):
    """不同环境的单条证据不能合成一条达到支持数阈值的活跃规律。"""
    store = LongTermMemoryStore(tmp_path / "memory", tmp_path)
    consolidator = SemanticMemoryConsolidatorAgent()
    first = consolidator.consolidate(store, episodic(1), 2)
    other = consolidator.consolidate(store, episodic(2, env="env-v2"), 2)
    assert first.semantic_id != other.semantic_id
    assert first.status == other.status == "candidate"
    active = consolidator.consolidate(store, episodic(3), 2)
    assert active.status == "active"
    assert store.retrieve("后腿站立", "go2", environment_version="env-v2",
                          reward_registry_version="reward-v1")["hits"] == []


def test_environment_fingerprint_changes_when_sources_or_registry_change(tmp_path):
    """真实环境源码、模型和奖励注册表变更必须形成不同兼容版本。"""
    source = tmp_path / "legged_gym/envs/go2/config.py"
    source.parent.mkdir(parents=True)
    source.write_text("version = 1")
    manifest = {"robot": "go2", "rewards": [{"name": "orientation"}]}
    first = compatibility_fingerprint(tmp_path, manifest)
    assert compatibility_fingerprint(tmp_path, manifest) == first
    source.write_text("version = 2")
    assert compatibility_fingerprint(tmp_path, manifest)["environment_version"] != first["environment_version"]
    manifest["rewards"].append({"name": "rear_leg_stand"})
    assert compatibility_fingerprint(tmp_path, manifest)["reward_registry_version"] != first["reward_registry_version"]


def test_candidate_proxy_is_specific_to_posture_and_flip_tasks():
    """前/后腿站立和侧翻不能仅凭普通速度奖励胜出。"""
    provider = MockLLMReasoningProvider(1)
    task = TaskSpec.parse_obj(provider.design_task_and_rewards("后腿站立行走", "go2", {})["task_spec"])
    score, terms = TrainingOrchestrator._task_reward_proxy(task, {
        "Episode/rew_raw_mean_tracking_lin_vel": 999,
        "Episode/rew_raw_mean_rear_leg_stand": 0.4,
        "Episode/rew_raw_mean_rear_leg_walk": 0.2})
    assert terms == ["rear_leg_stand", "rear_leg_walk"] and score == pytest.approx(0.3)
    task.original_instruction = "侧翻"
    assert "roll_rotation" in TrainingOrchestrator._task_reward_proxy(task, {})[1]


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_missing_safety_from_one_seed_is_not_hidden_by_other_seed(missing):
    """某种子缺失或非法安全证据不能被另一种子的零违规覆盖。"""
    task = TaskSpec.parse_obj(MockLLMReasoningProvider(1).design_task_and_rewards(
        "向前走", "go2", {})["task_spec"])
    aggregate = TrainingOrchestrator._aggregate_rollout_metrics(task, [
        {"metrics": {"joint_limit_violations": 0}},
        {"metrics": {"joint_limit_violations": missing}}])
    assert math.isnan(aggregate["joint_limit_violations"])
    assert math.isfinite(TrainingOrchestrator._rollout_score(task, {"fall_rate": missing}))


def test_browser_lock_releases_after_provider_exception(tmp_path, monkeypatch):
    """Provider 异常后租约必须释放，不能阻塞其他会话请求。"""
    monkeypatch.setattr("rl_training_agent.providers.browser_lease.AGENT_ROOT", tmp_path)
    first = BrowserLease()
    with pytest.raises(ValueError):
        with first.acquire("test"):
            raise ValueError("page failure")
    assert first.depth == 0
    with BrowserLease().acquire("test"):
        pass
