"""覆盖长期证据、语义反证与维护命令的路径安全回归，不调用真实模型。"""
from pathlib import Path

import pytest

from rl_training_agent.maintenance.experiment_cleanup import ExperimentCleaner
from rl_training_agent.memory.consolidator import SemanticMemoryConsolidatorAgent
from rl_training_agent.memory.curator import MemoryCuratorAgent
from rl_training_agent.memory.procedural import ProceduralMemoryAgent
from rl_training_agent.memory.store import LongTermMemoryStore
from rl_training_agent.memory.reward_experience.agent import RewardExperienceAgent
from rl_training_agent.schemas.agent_workflow import LongTermMemoryRecord, TaskIntentSpec
from rl_training_agent.schemas.rewards import RewardPlan
from rl_training_agent.schemas.task import TaskSpec
from rl_training_agent.storage.migrations import ArtifactMigrator
from rl_training_agent.utils.io import read_json, write_json
from test_reward_experience import _fixture, _build, _narrative


def _episodic(index, failed=False):
    """生成版本一致的纯离线情景，用于反证治理而非真实经验晋升。"""
    return LongTermMemoryRecord(
        memory_id="memory-%d" % index, task_id="task-%d" % index, robot="go2",
        action_name="后腿站立行走", normalized_goal="后腿站立行走",
        outcome="verified_failure" if failed else "completed", reward_version=1,
        reward_terms=[{"name": "rear_leg_stand"}, {"name": "rear_leg_walk"}],
        failure_pattern="持续摔倒" if failed else None,
        environment_version="env-v1", reward_registry_version="rewards-v1",
        created_at="2026-10-01T00:00:00+00:00")


def _verified_case(tmp_path):
    """创建内容完整、可经 Validator 验证的合成实验及经验文件。"""
    case = _fixture(tmp_path)
    root = case["task_dir"]
    task = TaskSpec(**{
        **read_json(root / "task_spec.json"), "initial_state": "站立",
        "required_behaviors": [], "forbidden_behaviors": [], "phases": [],
        "required_observations": [], "required_sensors": [],
        "success_metrics": [], "safety_constraints": [], "training_budget": {},
        "visual_evaluation_requirements": [],
    })
    write_json(root / "task_spec.json", task)
    for path in root.glob("candidates/*/reward_plan.json"):
        value = read_json(path)
        value["design_rationale"] = ["该合成任务的离线奖励"]
        for term in value["terms"]:
            term["purpose"] = "离线验证"
        write_json(path, value)
    plan = RewardPlan.parse_obj(read_json(case["selected"]["dir"] / "reward_plan.json"))
    write_json(root / "final" / "reward_plan.json", plan)
    case["selected"]["plan"] = plan
    evidence, eligibility = _build(case)
    experience = RewardExperienceAgent(_narrative).create(evidence, eligibility)
    experience_dir = root / "memory" / "reward_experience" / case["selected"]["id"]
    write_json(experience_dir / "evidence.json", evidence)
    write_json(experience_dir / "eligibility.json", eligibility)
    write_json(experience_dir / "experience.json", experience)
    write_json(experience_dir / "validation.json", {"valid": True})
    write_json(root / "environment_compatibility.json", {
        "environment_version": "env-v1", "reward_registry_version": "rewards-v1"})
    intent = TaskIntentSpec(
        original_instruction=task.original_instruction, robot=task.robot,
        action_name=task.task_name, normalized_goal=task.normalized_description)
    return case, task, intent, experience


def _curate(case, task, intent, experience, memory_root):
    """走真实证据复核和快照流程，模型输出仅使用合成叙述。"""
    curator = MemoryCuratorAgent()
    record = curator.curate(
        intent, task, case["selected"]["plan"], case["summary"], case["outcome"],
        case["selected"], case["task_dir"], experience=experience,
        require_validated_experience=True, memory_root=memory_root)
    return record, curator.last_reason


@pytest.mark.parametrize("new_success", [False, True])
def test_superseded_semantic_rule_cannot_be_reactivated(tmp_path, new_success):
    """旧证据重放或新成功证据均不能自动推翻已经成立的反证替代。"""
    store = LongTermMemoryStore(tmp_path / "memory", tmp_path)
    consolidator = SemanticMemoryConsolidatorAgent()
    for index in (1, 2):
        consolidator.consolidate(store, _episodic(index), 2)
    for index in (3, 4):
        consolidator.consolidate(store, _episodic(index, failed=True), 2)
    before = next(item for item in store.semantic_records() if item.rule_key == "rear-leg-posture-gating")
    assert before.status == "superseded"
    after = consolidator.consolidate(store, _episodic(5 if new_success else 1), 2)
    assert after.status == "superseded" and after.superseded_by == before.superseded_by
    assert after.counterevidence_memory_ids == before.counterevidence_memory_ids
    if not new_success:
        assert after.dict() == before.dict()
    assert all(item.rule_key != "rear-leg-posture-gating" for item in store.semantic_records(active_only=True))


@pytest.mark.parametrize("kind", ["experiment-file", "experiment-directory", "memory-file", "memory-directory"])
def test_migrations_never_follow_artifact_symlinks(tmp_path, kind):
    """文件或父目录链接均不能把格式迁移带到允许范围之外。"""
    experiments = tmp_path / "experiments"
    memory = tmp_path / "memory"
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / ("state.json" if kind.startswith("experiment") else "item.json")
    write_json(victim, {"untouched": True})
    if kind.startswith("experiment"):
        link = experiments / "task-linked"
    else:
        link = memory / "records"
    link.parent.mkdir(parents=True, exist_ok=True)
    if kind.endswith("directory"):
        link.symlink_to(outside, target_is_directory=True)
    else:
        link.mkdir()
        (link / victim.name).symlink_to(victim)
    normal = experiments / "task-safe" / "summary.json"
    write_json(normal, {"state": "FAILED"})
    result = ArtifactMigrator(experiments, memory).migrate(dry_run=False)
    assert read_json(victim) == {"untouched": True}
    assert read_json(normal)["format_version"] == 2
    assert result["changed"] == [str(normal)]


@pytest.mark.parametrize("artifact", ["video", "checkpoint"])
def test_single_task_cleanup_cannot_delete_other_task_symlink_targets(tmp_path, artifact):
    """清理指定任务时保留其他任务最终产物，也不删除指向它们的链接。"""
    task = tmp_path / "task-clean"
    obsolete = task / "candidates" / "obsolete"
    obsolete.mkdir(parents=True)
    write_json(task / "summary.json", {"state": "FAILED", "selected_experiment": "selected"})
    victim = tmp_path / "task-other" / "final" / ("protected.mp4" if artifact == "video" else "checkpoint.pt")
    victim.parent.mkdir(parents=True)
    victim.write_bytes(b"other-task-evidence")
    link = obsolete / ("front.mp4" if artifact == "video" else "model_1.pt")
    link.symlink_to(victim)
    if artifact == "checkpoint":
        (obsolete / "model_2.pt").write_bytes(b"latest")
    result = ExperimentCleaner(tmp_path).run(task.name, apply=True)
    assert victim.read_bytes() == b"other-task-evidence"
    assert link.is_symlink() and result["total_files"] == 0


def test_cleanup_rejects_a_symlink_task_directory(tmp_path):
    """不能借同根目录任务链接扩大单任务清理范围。"""
    target = tmp_path / "task-real"
    target.mkdir()
    (tmp_path / "task-alias").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="符号链接"):
        ExperimentCleaner(tmp_path).run("task-alias", apply=True)


@pytest.mark.parametrize("fake_latest", ["symlink", "directory"])
def test_checkpoint_retention_ignores_unsafe_or_nonfile_artifacts(tmp_path, fake_latest):
    """高迭代名的链接或目录不能挤掉本任务真正最新的策略保留位。"""
    task = tmp_path / "task-safe"
    run = task / "candidates" / "candidate" / "seed-1"
    run.mkdir(parents=True)
    old = run / "model_1.pt"
    latest = run / "model_2.pt"
    old.write_bytes(b"old")
    latest.write_bytes(b"latest")
    fake = run / "model_999.pt"
    if fake_latest == "symlink":
        other = tmp_path / "task-other" / "checkpoint.pt"
        other.parent.mkdir()
        other.write_bytes(b"other")
        fake.symlink_to(other)
    else:
        fake.mkdir()
    result = ExperimentCleaner(tmp_path).run(task.name, apply=True, cleanup_videos=False)
    assert not old.exists() and latest.read_bytes() == b"latest"
    assert result["checkpoint_files"] == 1 and fake.exists()


def test_legacy_records_without_snapshots_are_quarantined_by_default(tmp_path):
    """缺少不可变来源的旧记录和依赖它们的语义规则不能被当作真实经验。"""
    store = LongTermMemoryStore(tmp_path / "memory", tmp_path)
    consolidator = SemanticMemoryConsolidatorAgent()
    for index in (1, 2):
        record = _episodic(index)
        store.promote(record)
        consolidator.consolidate(store, record, 2)
    result = store.retrieve("后腿站立行走", "go2")
    assert result["hits"] == [] and result["unverified_records"] == 2


def test_experience_snapshot_survives_new_task_run_and_tracks_checkpoint_hash(tmp_path):
    """新运行覆盖任务级文件后，旧经验仍能从不可变快照复验且不复制策略。"""
    case, task, intent, experience = _verified_case(tmp_path / "experiments")
    memory_root = tmp_path / "memory"
    record, reason = _curate(case, task, intent, experience, memory_root)
    assert record is not None, reason
    store = LongTermMemoryStore(memory_root, tmp_path, max_context_chars=30000)
    store.promote(record)
    snapshot = memory_root / record.evidence_snapshot_root
    assert not list(snapshot.rglob("*.pt")) and not list(snapshot.rglob("*.mp4"))
    manifest = read_json(snapshot / "snapshot_manifest.json")
    assert len(manifest["checkpoint_provenance"]["sha256"]) == 64
    assert manifest["checkpoint_provenance"]["task_id"] == task.task_id
    assert manifest["checkpoint_provenance"]["run_id"].startswith("run-")
    assert all(not Path(source).is_absolute() for source in record.evidence_sources)
    (case["task_dir"] / "loop_history.json").unlink()
    write_json(case["task_dir"] / "task_spec.json", {"robot": "different-new-task"})
    write_json(case["task_dir"] / "final" / "reward_plan.json", {"version": 99})
    (case["task_dir"] / "final" / "checkpoint.pt").write_bytes(b"new-run-policy")
    assert store.evidence_valid(record)
    assert store.retrieve("Go2 稳定向前行走", "go2")["hits"]


def test_tampered_evidence_snapshot_is_excluded_from_retrieval(tmp_path):
    """不可变来源内容发生变化时立即隔离，不把既有晋升状态当作充分条件。"""
    case, task, intent, experience = _verified_case(tmp_path / "experiments")
    memory_root = tmp_path / "memory"
    record, reason = _curate(case, task, intent, experience, memory_root)
    assert record is not None, reason
    store = LongTermMemoryStore(memory_root, tmp_path)
    store.promote(record)
    write_json(memory_root / record.evidence_snapshot_root / "task_spec.json", {"robot": "changed"})
    assert not store.evidence_valid(record)
    assert store.retrieve("Go2 稳定向前行走", "go2")["hits"] == []


def test_curator_rechecks_actual_evaluation_before_snapshot(tmp_path):
    """传入成功对象但实际评估失败时拒绝晋升，不依赖进程内声称的完成结论。"""
    case, task, intent, experience = _verified_case(tmp_path / "experiments")
    evaluation_path = case["task_dir"] / case["summary"]["rollout"] / "evaluation.json"
    write_json(evaluation_path, {**read_json(evaluation_path), "completed": False})
    record, reason = _curate(case, task, intent, experience, tmp_path / "memory")
    assert record is None and "evaluation.json" in reason


def test_curator_names_the_malformed_evaluation_file(tmp_path):
    """损坏证据被安全拒绝时，错误必须明确指出实际来源文件。"""
    case, task, intent, experience = _verified_case(tmp_path / "experiments")
    write_json(case["task_dir"] / case["summary"]["rollout"] / "evaluation.json", {"completed": False})
    record, reason = _curate(case, task, intent, experience, tmp_path / "memory")
    assert record is None and "evaluation.json 无效" in reason


def test_procedure_snapshot_identity_covers_prompt_and_policy(tmp_path, monkeypatch):
    """版本、工作流、诊断策略或模板内容变化均必须产生新的不可变快照 ID。"""
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    template = prompts / "reward_design_agent.md"
    template.write_text("initial-template", encoding="utf-8")
    monkeypatch.setattr(ProceduralMemoryAgent, "POLICY_FILES", ())
    agent = ProceduralMemoryAgent(prompts)
    first = agent.snapshot()
    assert first.procedure_id == agent.snapshot().procedure_id
    template.write_text("changed-template", encoding="utf-8")
    changed_prompt = agent.snapshot()
    assert changed_prompt.procedure_id != first.procedure_id
    class ChangedPolicy(ProceduralMemoryAgent):
        VERSION = "new-policy-version"
        WORKFLOW_STATES = ["RECEIVED", "DIFFERENT_POLICY", "COMPLETED"]
        DIAGNOSIS_POLICY = {"budget_exhaustion": "FAILED"}
    changed_policy = ChangedPolicy(prompts).snapshot()
    assert changed_policy.procedure_id != changed_prompt.procedure_id
    store = LongTermMemoryStore(tmp_path / "memory", tmp_path)
    first_path = store.save_procedural(first)
    second_path = store.save_procedural(changed_policy)
    assert first_path != second_path and read_json(first_path)["version"] == first.version


def test_procedure_store_rejects_same_id_with_different_policy(tmp_path):
    """伪造相同 ID 的不同程序快照不能使 current 与不可变版本文件分歧。"""
    store = LongTermMemoryStore(tmp_path / "memory", tmp_path)
    record = ProceduralMemoryAgent().snapshot()
    path = store.save_procedural(record)
    forged = record.copy(deep=True)
    forged.workflow_states = ["DIFFERENT"]
    with pytest.raises(ValueError, match="不能覆盖"):
        store.save_procedural(forged)
    assert read_json(store.procedural_dir / "current.json")["workflow_states"] == read_json(path)["workflow_states"]


def test_untrusted_memory_ids_cannot_escape_record_directory(tmp_path):
    """文件中的记忆标识也受白名单约束，检索访问计数不能触发路径跳转。"""
    store = LongTermMemoryStore(tmp_path / "memory", tmp_path)
    record = _episodic(1)
    record.memory_id = "../../victim"
    with pytest.raises(ValueError, match="不安全"):
        store.promote(record)
    write_json(store.records_dir / "poison.json", record)
    assert store.records() == [] and not (tmp_path / "victim.json").exists()
