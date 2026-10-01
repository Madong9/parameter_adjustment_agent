"""提供面向训练文档和历史实验的本地检索增强生成能力。"""

from .knowledge_base import TrainingKnowledgeBase
from .models import RAGChunk, RAGHit, RAGResult

__all__ = ["TrainingKnowledgeBase", "RAGChunk", "RAGHit", "RAGResult"]
