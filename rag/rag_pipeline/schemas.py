"""Modèles de données immuables pour le pipeline RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Document:
    """Métadonnées du document source (étape 0)."""

    source_id: str
    nom_fichier: str
    chemin_origine: str
    date_ingestion: datetime


@dataclass
class TextBlock:
    """Bloc textuel brut extrait par PyMuPDF (étape 1)."""

    text: str
    bbox: tuple[float, float, float, float]
    font_size: float
    is_bold: bool
    font_name: str
    color: int
    origin: str


@dataclass
class Page:
    """Page extraite avec son contenu brut et ses blocs structurés."""

    doc: Document
    num: int
    raw_text: str
    blocks: list[TextBlock] = field(default_factory=list)
    texte_nettoye: str = ""
    filtered: bool = False
    filter_reason: str | None = None


@dataclass
class Heading:
    """Titre ou sous-titre détecté dans le document."""

    text: str
    level: int
    page: int
    bbox: tuple[float, float, float, float]
    font_size: float


@dataclass
class Paragraph:
    """Paragraphe unitaire au sein d'une section."""

    text: str
    page_start: int
    page_end: int
    blocks: list[TextBlock] = field(default_factory=list)


@dataclass
class Section:
    """Objet sémantique parent : section délimitée par des titres."""

    section_id: str
    source_id: str
    heading: Heading
    paragraphs: list[Paragraph] = field(default_factory=list)
    subsections: list[Section] = field(default_factory=list)

    @property
    def texte_complet(self) -> str:
        """Texte intégral de la section (pour reconstruction contextuelle)."""
        parts = [self.heading.text]
        for para in self.paragraphs:
            parts.append(para.text)
        return "\n\n".join(parts)

    @property
    def pages(self) -> list[int]:
        """Numéros de pages couverts par la section."""
        pages = {self.heading.page}
        for para in self.paragraphs:
            pages.add(para.page_start)
            pages.add(para.page_end)
        return sorted(pages)


@dataclass
class Chunk:
    """Unité textuelle fine destinée à la vectorisation."""

    chunk_id: str
    source_id: str
    page: int
    section_id: str
    titre_section: str
    texte_chunk: str
    texte_section_complete: str
    font_size_avg: float | None = None
    is_bold: bool = False
    position_in_section: int = 0
    total_chunks_in_section: int = 0
    quality_score: float = 0.0
    rejected: bool = False
    rejection_reason: str | None = None
    _blocks_refs: list[TextBlock] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Sérialisation pour stockage JSON."""
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "page": self.page,
            "section_id": self.section_id,
            "titre_section": self.titre_section,
            "texte_chunk": self.texte_chunk,
            "texte_section_complete": self.texte_section_complete,
            "font_size_avg": self.font_size_avg,
            "is_bold": self.is_bold,
            "position_in_section": self.position_in_section,
            "total_chunks_in_section": self.total_chunks_in_section,
            "quality_score": self.quality_score,
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
        }
