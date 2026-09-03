from __future__ import annotations

from pydantic import BaseModel


class BudgetTracker(BaseModel):
    max_iterations: int
    used_iterations: int = 0
    max_revisions: int
    used_revisions: int = 0

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

