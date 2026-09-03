from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..utils.io import atomic_write_text, write_json


class ReportBuilder:
    def build(self, task_id: str, summary: Dict[str, Any], output_dir: Path) -> Path:
        """构建并保存人类可读及机器可读的训练报告。"""
        write_json(output_dir / "summary.json", summary)
        lines = ["# 训练报告：%s" % task_id, "", "- 状态：`%s`" % summary.get("state", "未知"),
                 "- 结果：`%s`" % summary.get("result", "未知"),
                 "- 原因：%s" % summary.get("reason", "未记录"),
                 "- 闭环轮次：%s" % summary.get("loop_rounds", 0),
                 "- 最终奖励版本：v%s" % summary.get("reward_version", "未知"),
                 "- 已用训练迭代：%s" % summary.get("used_iterations", 0),
                 "- 已用奖励修订：%s/%s" % (
                     summary.get("used_revisions", 0), summary.get("max_revisions", 0)),
                 "- 最终实验：`%s`" % summary.get("selected_experiment", "未知"), "",
                 "机器可读证据请查看 `summary.json`、`loop_history.json`、候选 manifest、",
                 "逐项指标、rollout、诊断审计和 `lineage.json`。"]
        path = output_dir / "report.md"
        atomic_write_text(path, "\n".join(lines) + "\n")
        return path
