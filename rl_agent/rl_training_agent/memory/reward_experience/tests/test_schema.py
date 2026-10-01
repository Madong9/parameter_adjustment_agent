"""验证 Reward Experience 输出 Schema 的基本边界。"""
import pytest
from pydantic import ValidationError

from ..schema import EvidenceStatement, RewardEvolution, RewardExperience


def test_experience_confidence_is_bounded() -> None:
    """拒绝超出 [0, 1] 的最终经验可信度。"""
    with pytest.raises(ValidationError):
        RewardExperience(
            experience_id="experience-test", outcome="SUCCESS", task_context={},
            reward_evolution=RewardEvolution(initial_design={}, final_design={}), confidence=1.1)


def test_evidence_statement_requires_text() -> None:
    """拒绝空白的观察或假设陈述。"""
    with pytest.raises(ValidationError):
        EvidenceStatement(statement="  ", evidence_ids=["ev-test"])
