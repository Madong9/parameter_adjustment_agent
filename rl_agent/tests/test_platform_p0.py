"""验证 P0 平台升级及后续工程能力的关键行为。"""
from __future__ import annotations

import json

import pytest

from rl_training_agent.benchmark import BenchmarkRunner
from rl_training_agent.providers.registry import ProviderRegistry
from rl_training_agent.rag.knowledge_base import TrainingKnowledgeBase, tokenize
from rl_training_agent.schemas.visual import VisualBehaviorReport
from rl_training_agent.settings import Settings, load_settings
from rl_training_agent.storage.migrations import ArtifactMigrator
from rl_training_agent.visual.ensemble import ConservativeVisualAggregator


def test_visual_ensemble_selects_extremes_and_uses_worst_case():
    """验证最差、中央、最好样本都被选择，任一失败即阻止视觉通过。"""
    records = [{"score": value, "seed": index} for index, value in enumerate((0.8, 0.1, 0.5, 0.9))]
    assert [item["score"] for item in ConservativeVisualAggregator.select(records)] == [0.1, 0.8, 0.9]
    reports = [
        VisualBehaviorReport(visual_success=True, alignment_score=0.9, confidence=0.9,
                             summary="通过", phase_results=[]),
        VisualBehaviorReport(visual_success=False, alignment_score=0.3, confidence=0.8,
                             summary="摔倒", phase_results=[]),
        VisualBehaviorReport(visual_success=True, alignment_score=0.8, confidence=0.9,
                             summary="通过", phase_results=[]),
    ]
    result = ConservativeVisualAggregator.aggregate(reports, ["worst", "median", "best"])
    assert not result.visual_success and result.alignment_score == 0.3
    assert result.requires_human_review and result.agreement_score == pytest.approx(2 / 3)
    assert result.confidence == pytest.approx(0.8 * 2 / 3)


def test_provider_registry_rejects_role_capability_mismatch():
    """验证百炼不能被误配为需要图片能力的视觉 Agent。"""
    registry = ProviderRegistry()
    registry.validate_role("reward_designer", "bailian")
    with pytest.raises(ValueError, match="不支持角色"):
        registry.validate_role("visual_critic", "bailian")
    assert "visual_critique" in registry.inventory()["opencli"]
    assert "visual_critique" in registry.inventory()["doubao"]
    assert "visual_critique" in registry.inventory()["opencli-doubao"]


def test_hybrid_rag_vector_score_recovers_related_tokens():
    """验证本地向量相似度对相关文本给出高于无关文本的分数。"""
    query = tokenize("后腿站立姿态奖励")
    documents = [tokenize("后腿站立需要姿态门控奖励"), tokenize("前进速度跟踪")]
    scores = TrainingKnowledgeBase._vector_scores(query, documents)
    assert scores[0] > scores[1]


def test_artifact_migration_is_idempotent_and_dry_run_safe(tmp_path):
    """验证格式迁移演练不写盘，正式迁移可重复执行。"""
    experiments = tmp_path / "experiments"
    state = experiments / "task-demo" / "state.json"
    state.parent.mkdir(parents=True)
    state.write_text('{"state":"COMPLETED"}', encoding="utf-8")
    migrator = ArtifactMigrator(experiments, tmp_path / "memory")
    preview = migrator.migrate(dry_run=True)
    assert preview["changed"] and "format_version" not in state.read_text(encoding="utf-8")
    applied = migrator.migrate(dry_run=False)
    assert applied["changed"] and json.loads(state.read_text())["format_version"] == 2
    assert not migrator.migrate(dry_run=False)["changed"]


def test_benchmark_runner_produces_reproducibility_report(tmp_path):
    """验证固定种子基准演练产生版本、环境提交和重复性报告。"""
    suite = tmp_path / "suite.yaml"
    suite.write_text("""
version: 1
suite: smoke_v1
robot: go2
environment_commit: current
seeds: [1]
repetitions: 1
cases:
  - id: walk
    instruction: 测试机器狗稳定向前行走
    required_metrics: [tracking_error, fall_rate]
    thresholds:
      tracking_error: {operator: "<=", value: 0.25}
      fall_rate: {operator: "<=", value: 0.05}
""", encoding="utf-8")
    base = load_settings()
    settings = Settings(**{
        **base.dict(), "experiment_root": str(tmp_path / "experiments"),
        "artifact_root": str(tmp_path / "artifacts"), "num_reward_candidates": 1,
        "smoke_iterations": 2, "screening_iterations": 3, "full_iterations": 5,
        "max_total_iterations": 20, "rollouts_per_seed": 1,
    })
    report = BenchmarkRunner(settings, suite).run("mock", dry_run=True)
    assert report["dry_run"] and not report["real_evidence"]
    assert report["reproducibility"]["walk"]["consistent_terminal_state"]
    assert report["reproducibility"]["walk"]["thresholds_match"]
    assert (settings.artifacts_path / "benchmarks" / report["run_id"] / "report.json").is_file()
