"""Pipeline de préparation des données PDF pour vectorisation RAG."""

from .processor import DocumentProcessor
from .schemas import (
    Chunk, Document, Heading, Page, Paragraph, Section, TextBlock,
    TocEntry, HierarchyPath, ContentBlock,
)
from .config import PipelineConfig

__all__ = [
    "DocumentProcessor",
    "PipelineConfig",
    "Chunk",
    "Document",
    "Heading",
    "Page",
    "Paragraph",
    "Section",
    "TextBlock",
    "TocEntry",
    "HierarchyPath",
    "ContentBlock",
]