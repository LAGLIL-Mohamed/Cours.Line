"""Configuration centralisee du pipeline."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PipelineConfig:
    """Tous les parametres ajustables du pipeline."""

    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 50
    chunk_min_chars: int = 30
    chunk_max_chars: int = 2000
    heading_font_ratio: float = 1.15
    heading_font_ratio_strong: float = 1.30
    heading_min_chars: int = 3
    heading_max_chars: int = 200
    skip_patterns: List[str] = field(default_factory=list)
    unique_word_ratio_min: float = 0.25
    alpha_ratio_min: float = 0.50
    min_meaningful_words: int = 5
    dedup_exact: bool = True
    encoding: str = "utf-8"

    # Parametres hierarchie robuste
    toc_max_pages: int = 15
    toc_keyword_ratio: float = 0.25
    kmeans_n_clusters: int = 3
    heading_numbering_patterns: List[str] = field(default_factory=lambda: [
        r"^\d+[.:]\s+",
        r"^\d+\.\d+[.:]\s+",
        r"^\d+\.\d+\.\d+[.:]\s+",
        r"^[IVX]+[.:]\s+",
        r"^[A-Z][.:]\s+",
        r"^\([a-z\d]+\)\s+",
    ])
    chapter_keywords: List[str] = field(default_factory=lambda: [
        r"^(?:chapter|chapitre|part|partie|section|unit|module|leçon|lesson)\s*[IVX\d]+[.:—\-\s]",
        r"^(?:chapter|chapitre|part|partie)\s+[A-ZÀ-ÿ]",
    ])

    def __post_init__(self) -> None:
        if not self.skip_patterns:
            self.skip_patterns = [
                r"^\s*table des matières",
                r"^\s*copyright",
                r"^\s*©",
                r"^\s*all rights reserved",
                r"^\s*avertissement",
                r"^\s*mention légale",
                r"^\s*liste des figures",
                r"^\s*liste des tableaux",
                r"^\s*list of (figures|tables)",
                r"^\s*acknowledgements",
                r"^\s*remerciements",
            ]
