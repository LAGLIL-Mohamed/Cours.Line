"""Utilitaires pour la vectorisation."""

from __future__ import annotations


def approximate_token_count(text: str) -> int:
    """Estimation rapide du nombre de tokens (~0.75 token/mot en francais)."""
    words = text.split()
    return int(len(words) * 1.33)


def truncate_context(text: str, max_tokens: int) -> str:
    """Tronque un texte pour respecter une limite de tokens approximative."""
    words = text.split()
    max_words = int(max_tokens / 1.33)
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    return truncated + " [...]"
