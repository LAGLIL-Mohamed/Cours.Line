"""Modeles de donnees pour la vectorisation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Embedding:
    """Un chunk avec son vecteur d'embedding.

    Correspond au format produit par extract_hierarchy.py :
    chunk_id, text, chapitre, titre, sous_titre, hierarchy_path,
    page_debut, page_fin + source_id (nom du PDF/dossier parent).
    """

    chunk_id: str
    vector: np.ndarray
    text: str
    chapitre: Optional[str]
    titre: Optional[str]
    sous_titre: Optional[str]
    hierarchy_path: list
    page_debut: Optional[int]
    page_fin: Optional[int]
    source_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "vector": self.vector.tolist(),
            "text": self.text,
            "chapitre": self.chapitre,
            "titre": self.titre,
            "sous_titre": self.sous_titre,
            "hierarchy_path": self.hierarchy_path,
            "page_debut": self.page_debut,
            "page_fin": self.page_fin,
            "source_id": self.source_id,
        }


@dataclass
class SearchResult:
    """Resultat d'une recherche FAISS."""

    chunk_id: str
    score: float
    rank: int
    text: str
    chapitre: Optional[str]
    titre: Optional[str]
    sous_titre: Optional[str]
    hierarchy_path: list
    page_debut: Optional[int]
    page_fin: Optional[int]
    source_id: str


@dataclass
class SectionContext:
    """Une section unique dans le contexte reconstruit.

    Identifiee par sa cle de hierarchie (source + chemin hierarchique).
    """

    hierarchy_key: str
    hierarchy_path: list
    source_id: str
    pages: list[int] = field(default_factory=list)
    texte_complet: str = ""
    chunks_sources: list[str] = field(default_factory=list)


@dataclass
class RetrievedContext:
    """Contexte assemble pour le LLM."""

    query: str
    sections: list[SectionContext]
    total_tokens_estimate: int