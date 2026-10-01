"""覆盖奖励经验资格门控、证据归纳和反臆造校验。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from rl_training_agent.memory.reward_experience.agent import RewardExperienceAgent
from rl_training_agent.memory.reward_experience.eligibility import ExperienceEligibilityChecker
from rl_training_agent.memory.reward_experience.evidence_builder import RewardExperienceEvidenceBuilder
from rl_training_agent.memory.reward_experience.schema import RewardExperienceNarrative
from rl_training_agent.memory.reward_experience.validator import (
    RewardExperienceValidationError,
    RewardExperienceValidator,
)
from rl_training_agent.memory.store import LongTermMemoryStore
from rl_training_agent.utils.io import write_json


def _write_plan(path: Path, version: int, velocity: float, orientation: float) -> None:
    """写入测试用、来源明确的奖励配置。"""
    write_json(path, {
        "task_id": "task-test", "version": version,
        "terms": [
            {"name": "velocity", "weight": velocity, "implementation": "registry:velocity",
             "parameters": {}, "active_phases": ["all"], "normalization": "none"},
            {"name": "orientation", "weight": orientation, "implementation": "registry:orientation",
             "parameters": {}, "active_phases": ["all"], "normalization": "none"},
        ],
    })


def _fixture(tmp_path: Path, failed: bool = False, interrupted: bool = False) -> Dict[str, Any]:
    """创建一套与生产产物布局一致的可复核实验目录。"""
    task_dir = tmp_path / "task-test"
    initial_id = "run-test-candidate-01-v01"
    selected_id = "run-test-revision-01-v02"
    initial_dir = task_dir / "candidates" / initial_id
    selected_dir = task_dir / "candidates" / selected_id
    rollout_dir = selected_dir / "rollouts" / "round_01" / "rollout_001"
    initial_dir.mkdir(parents=True)
    selected_dir.mkdir(parents=True)
    rollout_dir.mkdir(parents=True)
    (task_dir / "final").mkdir()
    write_json(task_dir / "task_spec.json", {
        "task_id": "task-test", "robot": "go2", "task_name": "forward_walk",
        "normalized_description": "Go2 稳定向前行走", "original_instruction": "稳定向前走",
    })
    _write_plan(initial_dir / "reward_plan.json", 1, 3.0, 0.5)
    _write_plan(selected_dir / "reward_plan.json", 2,
                5.0 if failed else 1.0, 0.5 if failed else 3.0)
    write_json(initial_dir / "manifest.json", {
        "experiment_id": initial_id, "parent_experiment_id": None,
        "task_id": "task-test", "robot": "go2", "git_commit": "abc123",
        "config_hash": "initial-hash", "reward_version": 1, "training_result": "completed",
    })
    training_result = "failed" if interrupted else "completed"
    write_json(selected_dir / "manifest.json", {
        "experiment_id": selected_id, "parent_experiment_id": initial_id,
        "task_id": "task-test", "robot": "go2", "git_commit": "abc123",
        "config_hash": "config-456", "reward_version": 2,
        "iteration": 0 if interrupted else 1500, "training_result": training_result,
    })
    write_json(selected_dir / "revision_audit.json", {
        "parent_experiment": initial_id, "diagnosis": {"reward_changes": []},
    })
    checkpoint = task_dir / "final" / "checkpoint.pt"
    checkpoint.write_bytes(b"not-a-real-model-but-a-test-artifact")
    (selected_dir / "config.yaml").write_text("test: true\n", encoding="utf-8")
    (task_dir / "final" / "reward_plan.json").write_text(
        (selected_dir / "reward_plan.json").read_text(encoding="utf-8"), encoding="utf-8")

    eval_success = not failed and not interrupted
    visual = {
        "visual_success": eval_success, "alignment_score": 0.9 if eval_success else 0.2,
        "confidence": 0.8, "summary": "动作稳定" if eval_success else "机体跌倒",
        "requires_human_review": False, "failure_modes": [] if eval_success else [
            {"failure_mode": "fall", "evidence_frames": [30], "description": "机体向前跌倒"}],
        "unintended_behaviors": [] if eval_success else ["机体跌倒"],
    }
    metric = {"name": "tracking_error", "value": 0.1 if eval_success else 0.8,
              "unit": "m/s", "passed": eval_success}
    evaluation = {
        "hard_constraints_passed": eval_success, "task_metrics_passed": eval_success,
        "visual_alignment_passed": eval_success, "completed": eval_success,
        "metrics": [metric], "violations": [] if eval_success else ["fall_rate"], "conflicts": [],
    }
    write_json(rollout_dir / "evaluation.json", evaluation)
    write_json(rollout_dir / "numeric_summary.json", {
        "tracking_error": 0.1 if eval_success else 0.8,
        "fall_rate": 0.0 if eval_success else 0.75,
        "joint_limit_violations": 0.0,
    })
    write_json(rollout_dir / "visual_report.json", visual)
    selected_summary = {
        "task_id": "task-test", "state": "COMPLETED" if eval_success else "FAILED",
        "result": "completed" if eval_success else "failed",
        "selected_experiment": selected_id,
        "rollout": "candidates/%s/rollouts/round_01/rollout_001" % selected_id,
        "checkpoint": "final/checkpoint.pt", "dry_run": False,
        "evaluation_seeds": [1, 2, 3],
    }
    snapshot = task_dir / "memory" / "reward_experience" / selected_id / "training_result_snapshot.json"
    write_json(snapshot, selected_summary)
    write_json(task_dir / "loop_history.json", [{"round": 1, "decision": "complete"}])
    selected = {
        "id": selected_id, "dir": selected_dir,
        "checkpoint": checkpoint, "checkpoint_seeds": [1, 2, 3],
        "plan": {"version": 2}, "manifest": {
            "experiment_id": selected_id, "parent_experiment_id": initial_id,
            "git_commit": "abc123", "config_hash": "config-456",
            "training_result": training_result, "iteration": 0 if interrupted else 1500,
        },
    }
    outcome = {"evaluation": evaluation, "visual": visual,
               "physical": {"fall_rate": 0.0 if eval_success else 0.75}}
    return {"task_dir": task_dir, "selected": selected, "summary": selected_summary, "outcome": outcome}


def _narrative(payload: Dict[str, Any]) -> RewardExperienceNarrative:
    """按证据源 ID 构造测试模型的受限经验叙述。"""
    ids = list(payload["evidence_index"])
    change_ids = sorted({item for change in payload["reward_changes"] for item in change["evidence_ids"]})
    refs = change_ids or ids[:2]
    behavior_ids = []
    for evidence_id, source in payload["evidence_index"].items():
        if source["source"].endswith(("evaluation.json", "numeric_summary.json", "visual_report.json")):
            behavior_ids.append(evidence_id)
    diff_facts = []
    observations = []
    for change in payload["reward_changes"]:
        before = (change.get("before") or {}).get("weight", "absent")
        after = (change.get("after") or {}).get("weight", "absent")
        diff_facts.append("奖励项 %s 的权重从 %s 变为 %s。" % (
            change["reward_name"], before, after))
        observations.append({
            "reward_name": change["reward_name"],
            "reward_version": change["reward_version"],
            "observed_behavior_change": "同期 tracking_error 为 %s m/s，视觉结论为%s。" % (
                payload["numeric_evaluation"]["metrics"].get("tracking_error", "unknown"),
                "通过" if payload["visual_evaluation"].get("visual_success") else "失败"),
            "evidence_ids": behavior_ids,
        })
    return RewardExperienceNarrative(
        observed_facts=[{
            "statement": "；".join(diff_facts),
            "evidence_ids": refs,
        }],
        hypotheses=[{
            "statement": "奖励差异与本次行为评价之间可能存在关联，单次实验不足以判定因果。",
            "evidence_ids": refs,
        }],
        reward_change_observations=observations,
        successful_patterns=["在该任务和环境配置下，姿态项与目标速度项共同出现在通过验收的方案中。"],
        failure_patterns=["该失败样本中出现了跌倒及较大的跟踪误差。"],
        limitations=[{"statement": "结论仅来自本实验的有限随机种子和当前仿真环境。", "evidence_ids": ids[:2]}],
    )


def _build(case: Dict[str, Any]):
    """运行证据包构建和资格门控，返回后续阶段所需对象。"""
    evidence = RewardExperienceEvidenceBuilder().build(
        case["task_dir"], case["selected"], case["summary"], case["outcome"])
    eligibility = ExperienceEligibilityChecker().check(evidence)
    return evidence, eligibility


def test_success_experience_records_real_reward_evolution(tmp_path: Path) -> None:
    """验证真实联合验收成功时奖励差异由文件计算并可形成情景经验。"""
    case = _fixture(tmp_path)
    evidence, eligibility = _build(case)
    assert eligibility.eligible is True
    assert eligibility.outcome == "SUCCESS"
    assert {item.reward_name: (item.before, item.after) for item in evidence.reward_changes} == {
        "orientation": ({"weight": 0.5, "implementation": "registry:orientation", "parameters": {},
                         "active_phases": ["all"], "normalization": "none"},
                        {"weight": 3.0, "implementation": "registry:orientation", "parameters": {},
                         "active_phases": ["all"], "normalization": "none"}),
        "velocity": ({"weight": 3.0, "implementation": "registry:velocity", "parameters": {},
                      "active_phases": ["all"], "normalization": "none"},
                     {"weight": 1.0, "implementation": "registry:velocity", "parameters": {},
                      "active_phases": ["all"], "normalization": "none"}),
    }
    experience = RewardExperienceAgent(_narrative).create(evidence, eligibility)
    RewardExperienceValidator().validate(experience, evidence, case["task_dir"])
    assert experience.outcome == "SUCCESS"
    assert len(experience.reward_evolution.changes) == 2
    assert experience.confidence == 0.65


def test_verified_failure_requires_metric_and_behavior_evidence(tmp_path: Path) -> None:
    """验证训练完成但跌倒指标和行为证据失败时生成限定范围的失败经验。"""
    case = _fixture(tmp_path, failed=True)
    evidence, eligibility = _build(case)
    assert eligibility.eligible is True
    assert eligibility.outcome == "VERIFIED_FAILURE"
    assert next(item for item in evidence.reward_changes
                if item.reward_name == "velocity").after["weight"] == 5.0
    experience = RewardExperienceAgent(_narrative).create(evidence, eligibility)
    RewardExperienceValidator().validate(experience, evidence, case["task_dir"])
    assert experience.failure_patterns
    assert experience.failure_patterns[0].applicability["robot"] == "go2"


def test_reward_history_keeps_each_revision_iteration(tmp_path: Path) -> None:
    """验证多轮奖励修订逐次保留版本差异和迭代编号。"""
    case = _fixture(tmp_path)
    task_dir = case["task_dir"]
    previous = case["selected"]
    final_id = "run-test-revision-02-v03"
    final_dir = task_dir / "candidates" / final_id
    final_rollout = final_dir / "rollouts" / "round_02" / "rollout_001"
    final_rollout.mkdir(parents=True)
    _write_plan(final_dir / "reward_plan.json", 3, 0.75, 4.0)
    write_json(final_dir / "manifest.json", {
        "experiment_id": final_id, "parent_experiment_id": previous["id"],
        "task_id": "task-test", "robot": "go2", "git_commit": "abc123",
        "config_hash": "config-789", "reward_version": 3,
        "iteration": 2000, "training_result": "completed",
    })
    write_json(final_dir / "revision_audit.json", {
        "parent_experiment": previous["id"], "diagnosis": {"reward_changes": []},
    })
    old_rollout = task_dir / case["summary"]["rollout"]
    for name in ("evaluation.json", "numeric_summary.json", "visual_report.json"):
        write_json(final_rollout / name, json.loads((old_rollout / name).read_text(encoding="utf-8")))
    case["selected"]["id"] = final_id
    case["selected"]["dir"] = final_dir
    case["selected"]["plan"] = {"version": 3}
    case["selected"]["manifest"] = {
        "experiment_id": final_id, "parent_experiment_id": previous["id"],
        "git_commit": "abc123", "config_hash": "config-789",
        "training_result": "completed", "iteration": 2000,
    }
    case["summary"]["selected_experiment"] = final_id
    case["summary"]["rollout"] = "candidates/%s/rollouts/round_02/rollout_001" % final_id
    snapshot = task_dir / "memory" / "reward_experience" / final_id / "training_result_snapshot.json"
    write_json(snapshot, case["summary"])

    evidence, eligibility = _build(case)
    assert eligibility.outcome == "SUCCESS"
    assert [(item.iteration, item.reward_version) for item in evidence.reward_changes] == [
        (1, 2), (1, 2), (2, 3), (2, 3)]
    experience = RewardExperienceAgent(_narrative).create(evidence, eligibility)
    RewardExperienceValidator().validate(experience, evidence, task_dir)


def test_interrupted_training_is_inconclusive(tmp_path: Path) -> None:
    """验证 PPO 未真实完成时不生成可晋升经验。"""
    case = _fixture(tmp_path, failed=True, interrupted=True)
    evidence, eligibility = _build(case)
    assert eligibility.eligible is False
    assert eligibility.outcome == "INCONCLUSIVE"
    with pytest.raises(ValueError, match="INCONCLUSIVE"):
        RewardExperienceAgent(_narrative).create(evidence, eligibility)


def test_validator_rejects_fabricated_reward_change(tmp_path: Path) -> None:
    """验证模型或后处理篡改奖励权重时被本地 Validator 拒绝。"""
    case = _fixture(tmp_path)
    evidence, eligibility = _build(case)
    experience = RewardExperienceAgent(_narrative).create(evidence, eligibility)
    experience.reward_evolution.changes[0].after = {"weight": 999.0}
    with pytest.raises(RewardExperienceValidationError, match="real configuration diffs"):
        RewardExperienceValidator().validate(experience, evidence, case["task_dir"])


def test_compacted_memory_keeps_cited_evidence_sources() -> None:
    """验证 RAG 压缩记忆时仅保留且不丢失摘要内容引用的证据来源。"""
    record = SimpleNamespace(
        reward_experience={
            "outcome": "SUCCESS",
            "evidence_index": {
                "ev-change": {"source": "candidates/run/reward_plan.json", "sha256": "a"},
                "ev-metric": {"source": "rollout/numeric_summary.json", "sha256": "b"},
                "ev-unused": {"source": "unused.json", "sha256": "c"},
            },
            "reward_evolution": {"changes": [{
                "reward_name": "orientation", "evidence_ids": ["ev-change"],
            }]},
            "observed_facts": [{"statement": "tracking error passed", "evidence_ids": ["ev-metric"]}],
            "hypotheses": [], "successful_patterns": [], "failure_patterns": [],
            "limitations": [],
        },
        memory_id="memory-1", task_id="task-1", robot="go2", action_name="walk",
        normalized_goal="walk forward", outcome="completed", lessons=[],
        failure_pattern=None, confidence=0.6,
    )
    compact = LongTermMemoryStore._compact_payload("episodic", record)
    assert set(compact["reward_experience"]["evidence_index"]) == {"ev-change", "ev-metric"}
