import json

from rl_training_agent.orchestration.orchestrator import TrainingOrchestrator
from rl_training_agent.providers.mock_provider import MockLLMReasoningProvider
from rl_training_agent.rag.knowledge_base import TrainingKnowledgeBase, chunk_text, tokenize
from rl_training_agent.settings import Settings, load_settings


def _knowledge(tmp_path, max_context=6000):
    """创建完全隔离到临时目录的本地训练知识库。"""
    docs = tmp_path / "docs"
    docs.mkdir()
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    return TrainingKnowledgeBase(
        tmp_path, experiments, tmp_path / "artifacts" / "rag" / "index.json",
        [docs], chunk_chars=160, chunk_overlap=20, max_context_chars=max_context), docs, experiments


def _write_experiment(experiments, task_id, robot, instruction, state="COMPLETED"):
    """写入可供 RAG 索引的最小历史实验。"""
    task_dir = experiments / task_id
    candidate = task_dir / "candidates" / "candidate-v01"
    candidate.mkdir(parents=True)
    (task_dir / "task_request.txt").write_text(instruction, encoding="utf-8")
    (task_dir / "task_spec.json").write_text(json.dumps({
        "robot": robot, "task_name": "rear_stand", "original_instruction": instruction,
    }, ensure_ascii=False), encoding="utf-8")
    (task_dir / "summary.json").write_text(json.dumps({
        "state": state, "result": state.lower(), "reason": "通过多种子验收",
    }, ensure_ascii=False), encoding="utf-8")
    (candidate / "reward_plan.json").write_text(json.dumps({
        "design_rationale": ["后腿站立使用姿态门控奖励，避免四足局部最优"],
        "terms": ["rear_leg_stand", "rear_leg_walk"],
    }, ensure_ascii=False), encoding="utf-8")


def test_chinese_tokenization_and_overlapping_chunking():
    """验证中文二元词、英文标记和超长段落窗口均可稳定生成。"""
    tokens = tokenize("Go2 后腿站立 rear_leg_stand")
    assert "go2" in tokens and "后腿" in tokens and "rear_leg_stand" in tokens
    chunks = chunk_text("甲" * 250, maximum=100, overlap=20)
    assert len(chunks) == 3 and all(len(item) <= 100 for item in chunks)


def test_rag_indexes_docs_and_experiments_with_filters(tmp_path):
    """验证文档和历史实验可检索、当前任务可排除且上下文字符受限。"""
    knowledge, docs, experiments = _knowledge(tmp_path, max_context=180)
    (docs / "经验.md").write_text(
        "# 后腿站立经验\n\n后腿站立应使用姿态门控奖励，命令速度必须越过死区。", encoding="utf-8")
    _write_experiment(experiments, "task-past", "go2", "机器狗后腿站立行走")
    _write_experiment(experiments, "task-other", "g1", "人形机器人挥手")
    stats = knowledge.refresh(force=True)
    result = knowledge.retrieve("Go2 后腿站立奖励和速度死区", "task_design", robot="go2")
    assert stats["documents"] >= 7 and stats["chunks"] > 0
    assert result.hits and sum(len(item.text) for item in result.hits) <= 180
    assert any("后腿站立" in item.text for item in result.hits)
    filtered = knowledge.retrieve(
        "后腿站立 rear_leg_stand", "task_design", robot="go2", exclude_task_id="task-past")
    assert all(item.metadata.get("task_id") != "task-past" for item in filtered.hits)


def test_rag_reuses_persisted_index_when_sources_unchanged(tmp_path):
    """验证来源未变化时新实例直接加载持久化索引。"""
    knowledge, docs, experiments = _knowledge(tmp_path)
    (docs / "安全.md").write_text("禁止身体碰撞，必须检查接触力。", encoding="utf-8")
    _write_experiment(experiments, "task-safe", "go2", "稳定向前行走")
    first = knowledge.refresh(force=True)
    restored = TrainingKnowledgeBase(
        tmp_path, experiments, knowledge.index_path, [docs],
        chunk_chars=160, chunk_overlap=20, max_context_chars=6000)
    second = restored.refresh()
    assert second["chunks"] == first["chunks"]
    assert restored.retrieve("身体碰撞接触力", "manual_query").hits


def test_orchestrator_injects_rag_into_design_and_diagnosis(tmp_path):
    """验证任务设计和训练诊断都获得 RAG 上下文，并保存可审计检索产物。"""
    class CapturingProvider(MockLLMReasoningProvider):
        """记录两个模型决策阶段收到的检索上下文。"""

        def __init__(self):
            """初始化捕获字段和单候选 Mock Provider。"""
            super().__init__(1)
            self.design_context = None
            self.diagnosis_context = None

        def design_task_and_rewards(self, instruction, robot, capabilities):
            """记录奖励设计输入后复用确定性设计。"""
            self.design_context = capabilities.get("retrieved_experience")
            return super().design_task_and_rewards(instruction, robot, capabilities)

        def diagnose_training(self, payload):
            """记录训练诊断输入后返回确定性完成判断。"""
            self.diagnosis_context = payload.get("retrieved_experience")
            return super().diagnose_training(payload)

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "行走经验.md").write_text(
        "Go2 稳定向前行走应检查速度跟踪、身体碰撞和多随机种子结果。", encoding="utf-8")
    base = load_settings()
    settings = Settings(**{
        **base.dict(), "experiment_root": str(tmp_path / "experiments"),
        "artifact_root": str(tmp_path / "artifacts"),
        "rag_index_path": str(tmp_path / "artifacts" / "rag" / "index.json"),
        "rag_document_roots": [str(docs)], "num_reward_candidates": 1,
        "smoke_iterations": 5, "screening_iterations": 10, "full_iterations": 30,
        "max_total_iterations": 100, "evaluation_seeds": [1], "rollouts_per_seed": 1,
    })
    provider = CapturingProvider()
    result = TrainingOrchestrator(settings, provider).train(
        "测试 Go2 稳定向前行走", "go2", dry_run=True)
    task_dir = settings.experiments_path / result["task_id"]
    rollout_dir = task_dir / result["rollout"]
    assert provider.design_context["hits"]
    assert provider.diagnosis_context["hits"]
    assert (task_dir / "rag" / "design_retrieval.json").is_file()
    assert (rollout_dir / "rag_diagnosis_context.json").is_file()
    assert (task_dir / "task_intent.json").is_file()
    assert (task_dir / "context_snapshot.json").is_file()
    assert (task_dir / "prompts" / "reward_design_prompt.md").is_file()
    assert (task_dir / "reward_review.json").is_file()
    assert (task_dir / "memory" / "promotion.json").is_file()
    assert (task_dir / "memory" / "working_memory.json").is_file()
    assert (settings.memory_path / "procedural" / "current.json").is_file()
    assert (rollout_dir / "memory_diagnosis_context.json").is_file()
    working = json.loads((task_dir / "memory" / "working_memory.json").read_text(encoding="utf-8"))
    assert working["state"] == "COMPLETED" and working["latest_evaluation"]["completed"]
    promotion = json.loads((task_dir / "memory" / "promotion.json").read_text(encoding="utf-8"))
    assert not promotion["promoted"] and "dry-run" in promotion["reason"]
    assert result["state"] == "COMPLETED"
