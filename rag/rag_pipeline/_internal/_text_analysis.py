"""Analyse linguistique et métriques de qualité du texte."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class QualityMetrics:
    """Métriques calculées sur un chunk pour évaluation qualité."""

    char_count: int
    word_count: int
    unique_word_ratio: float
    alpha_ratio: float
    digit_ratio: float
    punctuation_ratio: float
    avg_word_length: float
    is_meaningful: bool
    rejection_reason: str | None = None


def compute_quality_metrics(text: str) -> QualityMetrics:
    """Calcule l'ensemble des métriques qualité sur un texte."""
    if not text:
        return QualityMetrics(
            char_count=0, word_count=0, unique_word_ratio=0.0,
            alpha_ratio=0.0, digit_ratio=0.0, punctuation_ratio=0.0,
            avg_word_length=0.0, is_meaningful=False,
            rejection_reason="Texte vide",
        )

    clean = text.strip()
    char_count = len(clean)

    words = re.findall(r"[\w'-]+", clean.lower())
    word_count = len(words)

    unique_words = set(words)
    unique_word_ratio = len(unique_words) / word_count if word_count > 0 else 0.0

    alpha_chars = sum(1 for c in clean if c.isalpha())
    digit_chars = sum(1 for c in clean if c.isdigit())
    punct_chars = sum(1 for c in clean if c in ".,;:!?-–—'\"()[]{}«»")

    alpha_ratio = alpha_chars / char_count if char_count > 0 else 0.0
    digit_ratio = digit_chars / char_count if char_count > 0 else 0.0
    punctuation_ratio = punct_chars / char_count if char_count > 0 else 0.0

    avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0.0

    is_meaningful = (
        word_count >= 5
        and unique_word_ratio >= 0.15
        and alpha_ratio >= 0.30
        and digit_ratio < 0.70
        and avg_word_length >= 2.0
    )

    rejection_reason = None
    if not is_meaningful:
        reasons = []
        if word_count < 5:
            reasons.append(f"trop peu de mots ({word_count})")
        if unique_word_ratio < 0.15:
            reasons.append(f"répétition excessive (ratio uniques={unique_word_ratio:.2f})")
        if alpha_ratio < 0.30:
            reasons.append(f"trop peu de caractères alphabétiques ({alpha_ratio:.2f})")
        if digit_ratio >= 0.70:
            reasons.append(f"dominé par des chiffres ({digit_ratio:.2f})")
        rejection_reason = "; ".join(reasons)

    return QualityMetrics(
        char_count=char_count,
        word_count=word_count,
        unique_word_ratio=unique_word_ratio,
        alpha_ratio=alpha_ratio,
        digit_ratio=digit_ratio,
        punctuation_ratio=punctuation_ratio,
        avg_word_length=avg_word_length,
        is_meaningful=is_meaningful,
        rejection_reason=rejection_reason,
    )


def is_pure_number_punctuation(text: str) -> bool:
    """Détecte si le texte est composé uniquement de nombres et ponctuation."""
    stripped = text.strip()
    if not stripped:
        return True
    alpha_count = sum(1 for c in stripped if c.isalpha())
    return alpha_count == 0
