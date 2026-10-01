"""实现无需外部向量服务的中文训练经验 BM25 检索库。"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..settings import Settings
from ..utils.io import read_json, utc_now, write_json
from ..utils.paths import relative_display
from .models import RAGChunk, RAGHit, RAGResult


INDEX_VERSION = 1
EXPERIMENT_FILENAMES = {
    "task_request.txt": "task_request",
    "task_spec.json": "task_spec",
    "summary.json": "experiment_summary",
    "loop_history.json": "loop_history",
    "reward_plan.json": "reward_plan",
    "revision_audit.json": "revision_audit",
    "diagnosis.json": "training_diagnosis",
}


def tokenize(text: str) -> List[str]:
    """将中英文混合文本转换为英文词、中文单字和中文二元词。"""
    lowered = str(text).lower()
    tokens = re.findall(r"[a-z0-9_]+", lowered)
    for sequence in re.findall(r"[\u4e00-\u9fff]+", lowered):
        tokens.extend(sequence)
        tokens.extend(sequence[index:index + 2] for index in range(max(0, len(sequence) - 1)))
    return [token for token in tokens if token.strip()]


def _flatten_json(value: Any, prefix: str = "") -> List[str]:
    """把嵌套 JSON 展开为适合语义检索的字段路径文本。"""
    lines: List[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = "%s.%s" % (prefix, key) if prefix else str(key)
            lines.extend(_flatten_json(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            lines.extend(_flatten_json(item, "%s[%d]" % (prefix, index)))
    elif value is not None:
        lines.append("%s: %s" % (prefix or "value", value))
    return lines


def chunk_text(text: str, maximum: int, overlap: int) -> List[str]:
    """按段落切分文本，并对超长段落使用有限重叠窗口。"""
    if maximum <= 0:
        raise ValueError("RAG chunk size must be positive")
    overlap = min(max(0, overlap), max(0, maximum - 1))
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    pieces: List[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= maximum:
            pieces.append(paragraph)
            continue
        step = maximum - overlap
        for start in range(0, len(paragraph), step):
            window = paragraph[start:start + maximum].strip()
            if window:
                pieces.append(window)
            if start + maximum >= len(paragraph):
                break
    chunks: List[str] = []
    current = ""
    for piece in pieces:
        candidate = piece if not current else current + "\n\n" + piece
        if current and len(candidate) > maximum:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


class TrainingKnowledgeBase:
    """索引项目文档和历史实验，并返回带来源的相关训练经验。"""

    def __init__(self, agent_root: Path, experiments_root: Path, index_path: Path,
                 document_roots: Sequence[Path], chunk_chars: int = 1200,
                 chunk_overlap: int = 120, max_context_chars: int = 6000,
                 bm25_weight: float = 0.65, vector_weight: float = 0.35):
        """保存索引范围和切分参数，但不在构造阶段执行磁盘扫描。"""
        self.agent_root = agent_root.resolve()
        self.experiments_root = experiments_root.resolve()
        self.index_path = index_path.resolve()
        self.document_roots = [path.resolve() for path in document_roots]
        self.chunk_chars = chunk_chars
        self.chunk_overlap = chunk_overlap
        self.max_context_chars = max_context_chars
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.chunks: List[RAGChunk] = []
        self.document_count = 0
        self.generated_at = utc_now()
        self._fingerprint = ""

    @classmethod
    def from_settings(cls, settings: Settings) -> "TrainingKnowledgeBase":
        """根据项目相对路径配置创建训练知识库。"""
        roots = [settings.resolve_agent_path(path) for path in settings.rag_document_roots]
        return cls(
            settings.agent_root, settings.experiments_path, settings.rag_index_file,
            roots, settings.rag_chunk_chars, settings.rag_chunk_overlap,
            settings.rag_max_context_chars, settings.rag_bm25_weight,
            settings.rag_vector_weight)

    def _source_name(self, path: Path) -> str:
        """生成不包含主机专属目录的稳定来源名称。"""
        resolved = path.resolve()
        try:
            return "experiments/" + resolved.relative_to(self.experiments_root).as_posix()
        except ValueError:
            return relative_display(resolved, self.agent_root)

    def _document_paths(self) -> List[Tuple[Path, str]]:
        """枚举允许进入 RAG 的文档和结构化实验文件。"""
        paths: List[Tuple[Path, str]] = []
        for root in self.document_roots:
            if root.is_file() and root.suffix.lower() in (".md", ".txt"):
                paths.append((root, "project_document"))
            elif root.is_dir():
                paths.extend((path, "project_document") for path in root.rglob("*.md") if path.is_file())
        if self.experiments_root.is_dir():
            for task_dir in self.experiments_root.glob("task-*"):
                if not task_dir.is_dir():
                    continue
                for name in ("task_request.txt", "task_spec.json", "summary.json", "loop_history.json"):
                    path = task_dir / name
                    if path.is_file():
                        paths.append((path, EXPERIMENT_FILENAMES[name]))
                for pattern in (
                    "candidates/*/reward_plan.json",
                    "candidates/*/revision_audit.json",
                    "candidates/*/rollouts/round_*/rollout_*/diagnosis.json",
                ):
                    for path in task_dir.glob(pattern):
                        if path.is_file():
                            paths.append((path, EXPERIMENT_FILENAMES[path.name]))
        return sorted(set(paths), key=lambda item: self._source_name(item[0]))

    def _sources_fingerprint(self, paths: Sequence[Tuple[Path, str]]) -> str:
        """根据路径、大小和修改时间计算低成本索引新鲜度指纹。"""
        records = []
        for path, source_type in paths:
            stat = path.stat()
            records.append("%s|%s|%d|%d" % (
                self._source_name(path), source_type, stat.st_size, stat.st_mtime_ns))
        return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()

    def _experiment_metadata(self, path: Path) -> Dict[str, Any]:
        """从实验任务目录提取机器人、状态和动作描述等过滤元数据。"""
        try:
            relative = path.resolve().relative_to(self.experiments_root)
        except ValueError:
            return {}
        if not relative.parts:
            return {}
        task_id = relative.parts[0]
        task_dir = self.experiments_root / task_id
        metadata: Dict[str, Any] = {"task_id": task_id}
        for filename, fields in (
            ("task_spec.json", ("robot", "task_name", "original_instruction")),
            ("summary.json", ("state", "result", "reason", "reward_version")),
        ):
            candidate = task_dir / filename
            if not candidate.is_file():
                continue
            try:
                value = read_json(candidate)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            for field in fields:
                if field in value:
                    metadata[field] = value[field]
        return metadata

    def _read_document(self, path: Path) -> str:
        """读取白名单文本或将 JSON 展开为字段路径文本。"""
        if path.suffix.lower() == ".json":
            value = read_json(path)
            return "\n".join(_flatten_json(value))
        return path.read_text(encoding="utf-8", errors="replace")

    def _build_chunks(self, paths: Sequence[Tuple[Path, str]]) -> List[RAGChunk]:
        """读取全部来源并构造确定性、可追溯的文本片段。"""
        chunks: List[RAGChunk] = []
        for path, source_type in paths:
            try:
                text = self._read_document(path).strip()
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if not text:
                continue
            source = self._source_name(path)
            metadata = self._experiment_metadata(path)
            for index, body in enumerate(chunk_text(text, self.chunk_chars, self.chunk_overlap)):
                digest = hashlib.sha256((source + "\n" + body).encode("utf-8")).hexdigest()
                chunks.append(RAGChunk(
                    chunk_id="%s-%04d" % (digest[:16], index), source=source,
                    source_type=source_type, text=body, content_hash=digest,
                    metadata=metadata))
        return chunks

    def _load_index(self) -> bool:
        """从磁盘加载版本兼容的既有索引。"""
        if not self.index_path.is_file():
            return False
        try:
            payload = read_json(self.index_path)
            if int(payload.get("version", 0)) != INDEX_VERSION:
                return False
            self.chunks = [RAGChunk.parse_obj(item) for item in payload.get("chunks", [])]
            self.document_count = int(payload.get("document_count", 0))
            self.generated_at = str(payload.get("generated_at", utc_now()))
            self._fingerprint = str(payload.get("source_fingerprint", ""))
            return True
        except (OSError, ValueError, json.JSONDecodeError):
            return False

    def refresh(self, force: bool = False) -> Dict[str, Any]:
        """在来源变化时重建索引，否则复用磁盘中的持久化片段。"""
        paths = self._document_paths()
        fingerprint = self._sources_fingerprint(paths)
        loaded = self._load_index()
        if loaded and not force and self._fingerprint == fingerprint:
            return self.stats()
        self.chunks = self._build_chunks(paths)
        self.document_count = len(paths)
        self.generated_at = utc_now()
        self._fingerprint = fingerprint
        write_json(self.index_path, {
            "version": INDEX_VERSION, "generated_at": self.generated_at,
            "source_fingerprint": fingerprint, "document_count": self.document_count,
            "chunk_count": len(self.chunks), "chunks": [item.dict() for item in self.chunks],
        })
        return self.stats()

    def stats(self) -> Dict[str, Any]:
        """返回可供 CLI 和上位机展示的索引统计。"""
        return {
            "index": relative_display(self.index_path, self.agent_root),
            "documents": self.document_count, "chunks": len(self.chunks),
            "generated_at": self.generated_at,
        }

    @staticmethod
    def _bm25_scores(query_tokens: Sequence[str], documents: Sequence[Sequence[str]]) -> List[float]:
        """为已分词文档计算标准 BM25 相关度。"""
        if not query_tokens or not documents:
            return [0.0 for _ in documents]
        total = len(documents)
        average_length = sum(len(item) for item in documents) / max(1, total)
        frequencies = Counter(token for document in documents for token in set(document))
        scores: List[float] = []
        for document in documents:
            counts = Counter(document)
            score = 0.0
            for token in set(query_tokens):
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                document_frequency = frequencies[token]
                inverse = math.log(1.0 + (total - document_frequency + 0.5) /
                                   (document_frequency + 0.5))
                denominator = frequency + 1.5 * (
                    1.0 - 0.75 + 0.75 * len(document) / max(1.0, average_length))
                score += inverse * frequency * 2.5 / denominator
            scores.append(score)
        return scores

    @staticmethod
    def _vector_scores(query_tokens: Sequence[str], documents: Sequence[Sequence[str]]) -> List[float]:
        """用本地 TF-IDF 稀疏向量计算余弦相似度，无需外部嵌入服务。"""
        if not query_tokens or not documents:
            return [0.0 for _ in documents]
        total = len(documents)
        frequencies = Counter(token for document in documents for token in set(document))
        idf = {token: math.log((1.0 + total) / (1.0 + count)) + 1.0
               for token, count in frequencies.items()}
        query_counts = Counter(query_tokens)
        query_vector = {token: count * idf.get(token, 1.0) for token, count in query_counts.items()}
        query_norm = math.sqrt(sum(value * value for value in query_vector.values())) or 1.0
        scores = []
        for document in documents:
            counts = Counter(document)
            vector = {token: count * idf[token] for token, count in counts.items()}
            norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
            dot = sum(query_vector.get(token, 0.0) * value for token, value in vector.items())
            scores.append(dot / (query_norm * norm))
        return scores

    def retrieve(self, query: str, purpose: str, top_k: int = 6,
                 robot: Optional[str] = None, exclude_task_id: Optional[str] = None) -> RAGResult:
        """检索相关片段，应用机器人/完成状态加权并限制注入字符数。"""
        if not self.chunks:
            self.refresh()
        eligible = [chunk for chunk in self.chunks
                    if not exclude_task_id or chunk.metadata.get("task_id") != exclude_task_id]
        document_tokens = [tokenize(chunk.text + " " + json.dumps(
            chunk.metadata, ensure_ascii=False, default=str)) for chunk in eligible]
        query_tokens = tokenize(query)
        lexical = self._bm25_scores(query_tokens, document_tokens)
        vector = self._vector_scores(query_tokens, document_tokens)
        lexical_max = max(lexical or [0.0]) or 1.0
        scores = [self.bm25_weight * (left / lexical_max) + self.vector_weight * right
                  for left, right in zip(lexical, vector)]
        ranked: List[Tuple[float, RAGChunk]] = []
        for score, chunk in zip(scores, eligible):
            if score <= 0.0:
                continue
            if chunk.source_type in ("reward_plan", "revision_audit", "training_diagnosis"):
                score *= 1.15
            if robot and chunk.metadata.get("robot") == robot:
                score *= 1.15
            if chunk.metadata.get("state") == "COMPLETED" or chunk.metadata.get("result") == "completed":
                score *= 1.10
            elif chunk.metadata.get("state") in ("FAILED", "HUMAN_REVIEW"):
                score *= 0.90
            ranked.append((score, chunk))
        ranked.sort(key=lambda item: (-item[0], item[1].source, item[1].chunk_id))
        hits: List[RAGHit] = []
        used = 0
        for score, chunk in ranked:
            if len(hits) >= max(0, top_k) or used >= self.max_context_chars:
                break
            remaining = self.max_context_chars - used
            text = chunk.text[:remaining].strip()
            if not text:
                continue
            hits.append(RAGHit(
                chunk_id=chunk.chunk_id, source=chunk.source,
                source_type=chunk.source_type, score=score, text=text,
                metadata=chunk.metadata))
            used += len(text)
        return RAGResult(
            query=query, purpose=purpose, hits=hits,
            indexed_documents=self.document_count, indexed_chunks=len(self.chunks),
            generated_at=utc_now())
