from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ..utils.io import write_json


class LineageGraph:
    def __init__(self):
        """初始化 LineageGraph 实例及其运行依赖。"""
        self.nodes: List[dict] = []
        self.edges: List[dict] = []

    def add(self, experiment_id: str, parent_id: Optional[str], reward_version: int,
            config_hash: str, result: str = "pending") -> None:
        """向实验谱系中加入节点及可选父子关系。"""
        self.nodes.append({"experiment_id": experiment_id, "reward_version": reward_version,
                           "config_hash": config_hash, "result": result})
        if parent_id:
            self.edges.append({"parent": parent_id, "child": experiment_id})

    def save(self, path: Path) -> None:
        """将实验谱系保存为 JSON。"""
        write_json(path, {"nodes": self.nodes, "edges": self.edges})

