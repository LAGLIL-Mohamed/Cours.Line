"""Étape 2 : Nettoyage du texte extrait."""

from __future__ import annotations

from ..schemas import Page
from .._internal._utils import (
    collapse_whitespace,
    merge_broken_lines,
    normalize_unicode,
)


class Step2Clean:
    """Nettoie le texte brut : espaces, lignes coupées, Unicode, fusion paragraphes."""

    def run(self, pages: list[Page]) -> list[Page]:
        for page in pages:
            page.texte_nettoye = self._clean_text(page.raw_text)
            page.blocks = self._clean_blocks(page.blocks)
        return pages

    def _clean_text(self, text: str) -> str:
        text = normalize_unicode(text)
        text = merge_broken_lines(text)
        text = collapse_whitespace(text)
        return text

    def _clean_blocks(self, blocks: list) -> list:
        from copy import copy
        cleaned = []
        for block in blocks:
            new_block = copy(block)
            new_block.text = self._clean_text(block.text)
            if new_block.text.strip():
                cleaned.append(new_block)
        return cleaned
