from __future__ import annotations

from pydantic import BaseModel


class BudgetTracker(BaseModel):
    max_iterations: int
    used_iterations: int = 0
    max_revisions: int
    used_revisions: int = 0
    reserve_for_revisions: bool = False

    def per_seed_allocation(self, requested: int, seed_count: int) -> int:
        """计算本轮每个种子的额度；探索模式为尚未执行的修订轮次保留等额预算。"""
        if requested <= 0 or seed_count <= 0:
            raise ValueError("requested iterations and seed count must be positive")
        remaining = self.max_iterations - self.used_iterations
        future_rounds = max(0, self.max_revisions - self.used_revisions) \
            if self.reserve_for_revisions else 0
        return min(requested, remaining // (seed_count * (1 + future_rounds)))

    def consume_iterations(self, amount: int) -> None:
        """在不超预算的前提下登记训练迭代消耗。"""
        if amount < 0 or self.used_iterations + amount > self.max_iterations:
            raise RuntimeError("training iteration budget exhausted")
        self.used_iterations += amount

    def consume_revision(self) -> None:
        """在不超预算的前提下登记一次奖励修订。"""
        if self.used_revisions + 1 > self.max_revisions:
            raise RuntimeError("reward revision budget exhausted")
        self.used_revisions += 1
