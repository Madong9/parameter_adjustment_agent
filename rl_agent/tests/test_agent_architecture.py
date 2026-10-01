import json
from types import SimpleNamespace

from pydantic import BaseModel

from rl_training_agent.agents.context_builder import ContextBuilder
from rl_training_agent.agents.prompt_compiler import RewardPromptCompiler
from rl_training_agent.agents.reward_reviewer import RewardReviewAgent
from rl_training_agent.memory.curator import MemoryCuratorAgent
from rl_training_agent.memory.consolidator import SemanticMemoryConsolidatorAgent
from rl_training_agent.memory.procedural import ProceduralMemoryAgent
from rl_training_agent.memory.store import LongTermMemoryStore
from rl_training_agent.providers.bailian_glm import BailianGLMProvider
from rl_training_agent.providers.mock_provider import MockLLMReasoningProvider
from rl_training_agent.schemas.metrics import EvaluationResult
from rl_training_agent.schemas.rewards import RewardPlan
from rl_training_agent.schemas.task import TaskSpec
from rl_training_agent.settings import BailianSettings, load_settings


class SmallReply(BaseModel):
    """表示百炼协议测试使用的最小结构化回复。"""

    ok: bool


def _mock_bundle(instruction="训练 Go2 稳定向前行走"):
    """构造包含真实任务和奖励 Schema 的离线设计结果。"""
    capabilities = {
        "project": "unitree_rl_gym",
        "robot": "go2",
        "rewards": [{"name": name} for name in (
            "tracking_lin_vel", "tracking_ang_vel", "orientation",
            "torques", "action_rate", "collision",
        )],
        "reward_variables": [],
        "terminations": ["fall"],
        "command_space": ["lin_vel_x"],
        "evaluation_metrics": ["tracking_error", "fall_rate"],
    }
    provider = MockLLMReasoningProvider(1)
    return provider, capabilities, provider.design_task_and_rewards(instruction, "go2", capabilities)


def test_task_intent_context_and_prompt_compiler_are_structured():
    """验证动作输入依次形成任务意图、限界上下文和固定版本提示词。"""
    provider, capabilities, _ = _mock_bundle("训练 Go2 以 0.4 m/s 稳定向前行走")
    capabilities["rewards"][0]["source_file"] = "/host/private/path.py"
    intent = provider.understand_task("训练 Go2 以 0.4 m/s 稳定向前行走", "go2")
    context = ContextBuilder().build(
        intent, capabilities,
        {"enabled": True, "hits": [{"source": "docs/agent/REWARD_DESIGN.md"}]},
        {"enabled": True, "hits": [{"source": "artifacts/memory/records/memory-a.json"}]},
    )
    compiler = RewardPromptCompiler(
        load_settings().agent_root / "rl_training_agent" / "prompts" / "reward_design_agent.md")
    first = compiler.compile(intent, context, 3)
    second = compiler.compile(intent, context, 3)
    assert intent.target_velocity == 0.4
    assert "source_file" not in context["environment_manifest"]["rewards"][0]
    assert first["sha256"] == second["sha256"]
    assert "TASK_INTENT_SPEC" in first["prompt"] and "OUTPUT_JSON_SCHEMA" in first["prompt"]


def test_reward_review_agent_accepts_complete_plan_and_rejects_unknown_reward():
    """验证奖励审查 Agent 能在训练前阻止未注册奖励。"""
    provider, capabilities, bundle = _mock_bundle()
    intent = provider.understand_task("训练 Go2 稳定向前行走", "go2")
    task = TaskSpec.parse_obj(bundle["task_spec"])
    plan = RewardPlan.parse_obj(bundle["reward_plans"][0])
    reviewer = RewardReviewAgent()
    approved = reviewer.review(
        intent, task, [plan], capabilities, bundle["reward_hacking_risks"])
    assert approved.approved
    plan.terms[0].name = "not_registered"
    rejected = reviewer.review(
        intent, task, [plan], capabilities, bundle["reward_hacking_risks"])
    assert not rejected.approved and "未注册奖励" in rejected.conflicts[0]


def test_reward_review_agent_filters_candidates_independently():
    """验证单个坏候选不会连带拒绝同批次的合格候选。"""
    provider, capabilities, _ = _mock_bundle()
    provider = MockLLMReasoningProvider(3)
    bundle = provider.design_task_and_rewards(
        "训练 Go2 稳定向前行走", "go2", capabilities)
    intent = provider.understand_task("训练 Go2 稳定向前行走", "go2")
    task = TaskSpec.parse_obj(bundle["task_spec"])
    plans = [RewardPlan.parse_obj(item) for item in bundle["reward_plans"]]
    plans[1].terms[0].name = "not_registered"

    report = RewardReviewAgent().review(
        intent, task, plans, capabilities, bundle["reward_hacking_risks"])

    assert report.approved
    assert report.passed_candidate_indexes == [1, 3]
    assert report.rejected_candidate_indexes == [2]
    assert report.candidate_results[1].conflicts


def test_memory_curator_only_promotes_verified_completion(tmp_path):
    """验证长期记忆只接收视觉、任务和安全联合通过的实验。"""
    provider, _, bundle = _mock_bundle()
    intent = provider.understand_task("训练 Go2 稳定向前行走", "go2")
    task = TaskSpec.parse_obj(bundle["task_spec"])
    plan = RewardPlan.parse_obj(bundle["reward_plans"][0])
    evaluation = EvaluationResult(
        hard_constraints_passed=True, task_metrics_passed=True,
        visual_alignment_passed=True, completed=True)
    selected = {
        "manifest": SimpleNamespace(git_commit="commit-demo", config_hash="config-demo"),
        "checkpoint_seeds": [1, 2, 3],
        "dir": tmp_path / "candidate",
    }
    rollout = "candidates/candidate-v01/rollouts/round_01/rollout_001"
    summary = {
        "state": "COMPLETED", "selected_experiment": "candidate-v01", "loop_rounds": 1,
        "rollout": rollout, "dry_run": False, "checkpoint": "final/checkpoint.pt",
    }
    evidence_paths = [
        tmp_path / "task_spec.json", tmp_path / "final" / "reward_plan.json",
        tmp_path / "loop_history.json", tmp_path / rollout / "evaluation.json",
    ]
    for path in evidence_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    record = MemoryCuratorAgent().curate(
        intent, task, plan, summary, {"evaluation": evaluation, "physical": {"fall_rate": 0.0}},
        selected, tmp_path)
    assert record is not None and record.confidence == 0.98
    assert record.seed_count == 3 and record.git_commit == "commit-demo"
    assert record.checkpoint == "final/checkpoint.pt" and record.visual_conclusion == {}
    store = LongTermMemoryStore(tmp_path / "memory", tmp_path)
    store.promote(record)
    result = store.retrieve("Go2 稳定向前行走", "go2")
    assert result["hits"][0]["record"]["memory_id"] == record.memory_id
    assert not store.retrieve(
        "Go2 稳定向前行走", "go2", exclude_task_id=task.task_id)["hits"]
    blocked = MemoryCuratorAgent().curate(
        intent, task, plan, {**summary, "state": "HUMAN_REVIEW"},
        {"evaluation": evaluation}, selected, tmp_path)
    assert blocked is None
    dry_run = MemoryCuratorAgent().curate(
        intent, task, plan, {**summary, "dry_run": True},
        {"evaluation": evaluation}, selected, tmp_path)
    assert dry_run is None


def test_memory_requires_multiple_seeds_and_promotes_verified_failure(tmp_path):
    """验证单种子被拒绝，而多种子一致的确定性失败可以形成失败模式记忆。"""
    provider, _, bundle = _mock_bundle()
    intent = provider.understand_task("训练 Go2 稳定向前行走", "go2")
    task = TaskSpec.parse_obj(bundle["task_spec"])
    plan = RewardPlan.parse_obj(bundle["reward_plans"][0])
    evaluation = EvaluationResult(
        hard_constraints_passed=False, task_metrics_passed=False,
        visual_alignment_passed=False, completed=False,
        violations=["fall_rate 超过阈值"],
    )
    rollout = "candidate/rollouts/round_01/rollout_001"
    summary = {
        "state": "FAILED", "selected_experiment": "candidate-v01", "loop_rounds": 2,
        "rollout": rollout, "dry_run": False, "checkpoint": "final/checkpoint.pt",
    }
    for path in (
        tmp_path / "task_spec.json", tmp_path / "final" / "reward_plan.json",
        tmp_path / "loop_history.json", tmp_path / rollout / "evaluation.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    curator = MemoryCuratorAgent()
    selected = {
        "manifest": SimpleNamespace(git_commit="commit", config_hash="hash"),
        "checkpoint_seeds": [1], "dir": tmp_path / "candidate",
    }
    assert curator.curate(
        intent, task, plan, summary, {"evaluation": evaluation}, selected, tmp_path) is None
    assert "两个不同随机种子" in curator.last_reason
    selected["checkpoint_seeds"] = [1, 2, 3]
    record = curator.curate(
        intent, task, plan, summary, {"evaluation": evaluation}, selected, tmp_path)
    assert record is not None and record.outcome == "verified_failure"
    assert "fall_rate" in record.failure_pattern


def test_four_layer_memory_consolidation_procedure_and_forgetting(tmp_path):
    """验证语义规律需独立证据升级，程序快照可追溯，遗忘只归档不删除。"""
    store = LongTermMemoryStore(tmp_path / "memory", tmp_path)
    now = "2026-01-01T00:00:00+00:00"
    from rl_training_agent.schemas.agent_workflow import LongTermMemoryRecord

    def episodic(index, confidence=0.9):
        """构造后腿动作的已验证情景记忆。"""
        return LongTermMemoryRecord(
            memory_id="memory-%d" % index, task_id="task-%d" % index,
            robot="go2", action_name="后腿站立行走", normalized_goal="后腿站立行走",
            outcome="completed", reward_version=index,
            reward_terms=[{"name": "rear_leg_stand"}, {"name": "rear_leg_walk"}],
            confidence=confidence, created_at=now,
        )

    consolidator = SemanticMemoryConsolidatorAgent()
    first = episodic(1)
    store.promote(first)
    candidate = consolidator.consolidate(store, first, min_support=2)
    assert candidate is not None and candidate.status == "candidate"
    second = episodic(2)
    store.promote(second)
    active = consolidator.consolidate(store, second, min_support=2)
    assert active is not None and active.status == "active" and active.evidence_count == 2
    for index in (4, 5):
        failure = episodic(index)
        failure.outcome = "verified_failure"
        failure.failure_pattern = "姿态门控仍导致持续摔倒"
        store.promote(failure)
        failure_rule = consolidator.consolidate(store, failure, min_support=2)
    assert failure_rule is not None and failure_rule.status == "active"
    posture_rule = next(item for item in store.semantic_records()
                        if item.rule_key == "rear-leg-posture-gating")
    assert posture_rule.status == "superseded" and posture_rule.superseded_by == failure_rule.semantic_id
    procedure_path = store.save_procedural(ProceduralMemoryAgent().snapshot())
    assert procedure_path.is_file() and (store.procedural_dir / "current.json").is_file()
    low = episodic(3, confidence=0.1)
    store.promote(low)
    maintenance = store.maintain(max_records=10, max_age_days=10000, min_confidence=0.5)
    assert maintenance["deleted"] == 0 and low.memory_id in maintenance["reasons"]
    assert (store.records_dir / (low.memory_id + ".json")).is_file()
    assert next(item for item in store.records() if item.memory_id == low.memory_id).status == "archived"


def test_bailian_provider_uses_openai_compatible_json_protocol(tmp_path, monkeypatch):
    """验证百炼请求不落盘密钥，并使用非思考 JSON 输出和严格 Schema。"""
    monkeypatch.setenv("TEST_DASHSCOPE_KEY", "secret-for-test")
    settings = BailianSettings(
        model="glm-4.7", base_url="https://example.invalid/compatible-mode/v1",
        api_key_env="TEST_DASHSCOPE_KEY", max_retries=0,
        enable_thinking=False, response_format_json=True)
    provider = BailianGLMProvider(settings, tmp_path / "records")
    captured = {}

    def fake_http(body):
        """捕获百炼兼容请求并返回模拟 Chat Completions 信封。"""
        captured.update(body)
        return json.dumps({"choices": [{"message": {"content": '{"ok": true}'}}]})

    monkeypatch.setattr(provider, "_http_request", fake_http)
    reply = provider.request_model("只返回 JSON", SmallReply, "protocol-test")
    assert reply.ok
    assert captured["model"] == "glm-4.7"
    assert captured["enable_thinking"] is False
    assert captured["response_format"] == {"type": "json_object"}
    request_text = next((tmp_path / "records").glob("*_request.json")).read_text(encoding="utf-8")
    assert "secret-for-test" not in request_text
