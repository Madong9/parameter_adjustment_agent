from __future__ import annotations

from typing import Dict, Union

from pydantic import BaseModel


class BudgetTracker(BaseModel):
    max_iterations: int
    used_iterations: int = 0
    max_revisions: int
    used_revisions: int = 0
    reserve_for_revisions: bool = False

    @property
    def remaining_iterations(self) -> int:
        """返回未使用的 PPO 训练迭代数，不代表模型 token 或运行时长。"""
        return max(0, self.max_iterations - self.used_iterations)

    @property
    def remaining_revisions(self) -> int:
        """返回尚可启动的奖励修订轮数，不修改已登记的消耗。"""
        return max(0, self.max_revisions - self.used_revisions)

    @property
    def exhausted(self) -> bool:
        """任一训练资源达到上限时禁止启动下一轮自动修订。"""
        return self.used_iterations >= self.max_iterations or self.used_revisions >= self.max_revisions

    def summary_fields(self) -> Dict[str, Union[int, bool]]:
        """提供摘要所需的冻结额度及剩余额度，不退款或自动扩容。"""
        return {
            "max_iterations": self.max_iterations,
            "used_iterations": self.used_iterations,
            "remaining_iterations": self.remaining_iterations,
            "max_revisions": self.max_revisions,
            "used_revisions": self.used_revisions,
            "remaining_revisions": self.remaining_revisions,
            "budget_exhausted": self.exhausted,
        }

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
