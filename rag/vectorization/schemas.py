"""Modeles de donnees pour la vectorisation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Embedding:
    """Un chunk avec son vecteur d'embedding."""

    chunk_id: str
    vector: np.ndarray
    texte_chunk: str
    texte_section_complete: str
    titre_section: str
    source_id: str
    page: int
    section_id: str
    position_in_section: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "vector": self.vector.tolist(),
            "texte_chunk": self.texte_chunk,
            "texte_section_complete": self.texte_section_complete,
            "titre_section": self.titre_section,
            "source_id": self.source_id,
            "page": self.page,
            "section_id": self.section_id,
            "position_in_section": self.position_in_section,
        }


@dataclass
class SearchResult:
    """Resultat d'une recherche FAISS."""

    chunk_id: str
    score: float
    rank: int
    texte_chunk: str
    texte_section_complete: str
    titre_section: str
    source_id: str
    page: int
    section_id: str


@dataclass
class RetrievedContext:
    """Contexte assemble pour le LLM."""

    query: str
    sections: list[SectionContext]
    total_tokens_estimate: int


@dataclass
class SectionContext:
    """Une section unique dans le contexte reconstruit."""

    section_id: str
    titre_section: str
    source_id: str
    pages: list[int]
    texte_complet: str
    chunks_sources: list[str] = field(default_factory=list)
