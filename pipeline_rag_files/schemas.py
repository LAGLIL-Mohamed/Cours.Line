"""Modeles de donnees immuables pour le pipeline RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List


@dataclass(frozen=True)
class Document:
    """Metadonnees du document source (etape 0)."""

    source_id: str
    nom_fichier: str
    chemin_origine: str
    date_ingestion: datetime


@dataclass
class TextBlock:
    """Bloc textuel brut extrait par PyMuPDF (etape 1)."""

    text: str
    bbox: tuple[float, float, float, float]
    font_size: float
    is_bold: bool
    font_name: str
    color: int
    origin: str


@dataclass
class Page:
    """Page extraite avec son contenu brut et ses blocs structures."""

    doc: Document
    num: int
    raw_text: str
    blocks: list[TextBlock] = field(default_factory=list)
    texte_nettoye: str = ""
    filtered: bool = False
    filter_reason: str | None = None


@dataclass
class Heading:
    """Titre ou sous-titre detecte dans le document - enrichi hierarchie."""

    text: str
    level: int
    page: int
    bbox: tuple[float, float, float, float]
    font_size: float
    source: str = "style"          # "native_toc" | "visual_toc" | "style"
    confidence: float = 0.75
    is_numbered: bool = False
    is_bold: bool = False
    is_uppercase: bool = False


@dataclass
class Paragraph:
    """Paragraphe unitaire au sein d'une section."""

    text: str
    page_start: int
    page_end: int
    blocks: list[TextBlock] = field(default_factory=list)


@dataclass
class Section:
    """Objet semantique parent : section delimitee par des titres."""

    section_id: str
    source_id: str
    heading: Heading
    paragraphs: list[Paragraph] = field(default_factory=list)
    subsections: list[Section] = field(default_factory=list)
    hierarchy_path: list[str] = field(default_factory=list)  # [Chapitre, Titre, Sous-titre]
    parent_id: str | None = None

    @property
    def texte_complet(self) -> str:
        """Texte integral de la section (pour reconstruction contextuelle)."""
        parts = [self.heading.text]
        for para in self.paragraphs:
            parts.append(para.text)
        return "\n\n".join(parts)

    @property
    def pages(self) -> list[int]:
        """Numeros de pages couverts par la section."""
        pages = {self.heading.page}
        for para in self.paragraphs:
            pages.add(para.page_start)
            pages.add(para.page_end)
        return sorted(pages)

    @property
    def chapter_title(self) -> str:
        """Retourne le titre du chapitre racine (premier element du chemin)."""
        return self.hierarchy_path[0] if self.hierarchy_path else self.heading.text

    @property
    def full_path(self) -> str:
        """Chemin hierarchique complet sous forme de chaine."""
        return " > ".join(self.hierarchy_path) if self.hierarchy_path else self.heading.text


@dataclass
class Chunk:
    """Unite textuelle fine destinee a la vectorisation - enrichie hierarchie."""

    chunk_id: str
    source_id: str
    page: int
    section_id: str
    titre_section: str
    texte_chunk: str
    texte_section_complete: str

    # Champs hierarchiques enrichis
    hierarchy_level: int = 1
    hierarchy_path: list[str] = field(default_factory=list)
    chapter_title: str = ""
    parent_sections: list[str] = field(default_factory=list)

    font_size_avg: float | None = None
    is_bold: bool = False
    position_in_section: int = 0
    total_chunks_in_section: int = 0
    quality_score: float = 0.0
    rejected: bool = False
    rejection_reason: str | None = None
    _blocks_refs: list[TextBlock] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialisation pour stockage JSON/JSONL."""
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "page": self.page,
            "section_id": self.section_id,
            "titre_section": self.titre_section,
            "texte_chunk": self.texte_chunk,
            "texte_section_complete": self.texte_section_complete,
            "hierarchy_level": self.hierarchy_level,
            "hierarchy_path": self.hierarchy_path,
            "chapter_title": self.chapter_title,
            "parent_sections": self.parent_sections,
            "font_size_avg": self.font_size_avg,
            "is_bold": self.is_bold,
            "position_in_section": self.position_in_section,
            "total_chunks_in_section": self.total_chunks_in_section,
            "quality_score": self.quality_score,
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
        }
