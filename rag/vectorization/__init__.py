"""Module de vectorisation et recherche semantique."""

from .config import VectorConfig
from .embedder import BGEEmbedder
from .indexer import FAISSIndexer
from .retriever import ContextRetriever
from .reranker import CrossEncoderReranker
from .schemas import (
    Embedding,
    RetrievedContext,
    SearchResult,
    SectionContext,
)
from .store import MetadataStore

__all__ = [
    "BGEEmbedder",
    "ContextRetriever",
    "CrossEncoderReranker",
    "Embedding",
    "FAISSIndexer",
    "MetadataStore",
    "RetrievedContext",
    "SearchResult",
    "SectionContext",
    "VectorConfig",
]