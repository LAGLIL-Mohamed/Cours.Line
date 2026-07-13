"""Configuration centralisée du pipeline."""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Tous les paramètres ajustables du pipeline."""

    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 50
    chunk_min_chars: int = 30
    chunk_max_chars: int = 2000
    heading_font_ratio: float = 1.15
    heading_font_ratio_strong: float = 1.30
    heading_min_chars: int = 3
    heading_max_chars: int = 200
    skip_patterns: list[str] = None
    unique_word_ratio_min: float = 0.25
    alpha_ratio_min: float = 0.50
    min_meaningful_words: int = 5
    dedup_exact: bool = True
    encoding: str = "utf-8"

    def __post_init__(self) -> None:
        if self.skip_patterns is None:
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
