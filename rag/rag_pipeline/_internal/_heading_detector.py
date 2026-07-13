"""Heuristique de détection des titres dans un document PDF."""

from __future__ import annotations

import statistics
from collections import Counter

from ..schemas import Heading, TextBlock


class HeadingDetector:
    """Détecte les titres et sous-titres par analyse typographique."""

    def __init__(
        self,
        font_ratio: float = 1.15,
        font_ratio_strong: float = 1.30,
        min_chars: int = 3,
        max_chars: int = 200,
    ) -> None:
        self.font_ratio = font_ratio
        self.font_ratio_strong = font_ratio_strong
        self.min_chars = min_chars
        self.max_chars = max_chars

    def detect(self, blocks: list[TextBlock], page_height: float) -> list[Heading]:
        """Retourne la liste des headings détectés, triés par position."""
        if not blocks:
            return []

        font_sizes = [b.font_size for b in blocks if b.text.strip()]
        if not font_sizes:
            return []

        mode_size = self._compute_mode(font_sizes)

        headings = []
        for block in blocks:
            text = block.text.strip()
            if not text:
                continue
            if not (self.min_chars <= len(text) <= self.max_chars):
                continue

            is_heading = self._is_heading_block(block, mode_size)
            if is_heading:
                level = self._infer_level(block.font_size, mode_size)
                headings.append(
                    Heading(
                        text=text,
                        level=level,
                        page=0,
                        bbox=block.bbox,
                        font_size=block.font_size,
                    )
                )

        headings.sort(key=lambda h: (h.page, h.bbox[1]))
        return headings

    def _compute_mode(self, values: list[float]) -> float:
        """Calcule le mode (valeur la plus fréquente) des tailles de police."""
        rounded = [round(v, 1) for v in values]
        counter = Counter(rounded)
        most_common = counter.most_common(1)
        return most_common[0][0] if most_common else statistics.median(values)

    def _is_heading_block(self, block: TextBlock, mode_size: float) -> bool:
        """Détermine si un bloc textuel est un titre."""
        text = block.text.strip()
        if not text:
            return False

        ratio = block.font_size / mode_size if mode_size > 0 else 1.0

        if ratio >= self.font_ratio and block.is_bold:
            return True

        if ratio >= self.font_ratio_strong:
            return True

        if block.is_bold and ratio >= 1.0 and len(text) < 100:
            return True

        return False

    def _infer_level(self, font_size: float, mode_size: float) -> int:
        """Infère le niveau hiérarchique (1, 2, 3...) depuis la taille relative."""
        ratio = font_size / mode_size if mode_size > 0 else 1.0
        if ratio >= 1.5:
            return 1
        elif ratio >= 1.25:
            return 2
        elif ratio >= 1.15:
            return 3
        else:
            return 4