"""Etape 3 : Filtrage des contenus non pertinents."""

from __future__ import annotations

import re

from ..schemas import Page
from ..config import PipelineConfig


class Step3Filter:
    """Filtre les pages non pertinentes (couverture, TOC, copyright, etc.)."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def run(self, pages: list[Page]) -> list[Page]:
        for page in pages:
            self._evaluate_page(page)

        kept = [p for p in pages if not p.filtered]
        if not kept and pages:
            kept = [p for p in pages if len(p.texte_nettoye.strip()) > 50]

        return kept

    def _evaluate_page(self, page: Page) -> None:
        text = page.texte_nettoye.lower().strip()

        if len(text) < 50:
            page.filtered = True
            page.filter_reason = "Page quasi vide"
            return

        for pattern in self.config.skip_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                page.filtered = True
                page.filter_reason = f"Pattern match: {pattern}"
                return

        alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
        if alpha_ratio < 0.10 and len(text) > 200:
            page.filtered = True
            page.filter_reason = "Trop peu de texte alphabetique"
            return
