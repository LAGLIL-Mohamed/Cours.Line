"""Fonctions utilitaires partagees par le pipeline."""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..schemas import TextBlock


def normalize_unicode(text: str) -> str:
    """Normalise le texte en NFC et nettoie les caracteres de controle."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) != "Cc" or ch in "\t\n\r"
    )
    return text


def collapse_whitespace(text: str) -> str:
    """Reduit les espaces multiples et les lignes vides successives."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def merge_broken_lines(text: str) -> str:
    """Fusionne les lignes coupees en milieu de phrase."""
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return text


def approximate_token_count(text: str) -> int:
    """Estimation rapide du nombre de tokens (~0.75 token/mot)."""
    words = text.split()
    return int(len(words) * 1.33)


def chars_from_tokens(token_count: int) -> int:
    """Estimation inverse : tokens -> caracteres (~4 caracteres/token)."""
    return token_count * 4


def compute_text_hash(text: str) -> str:
    """Hash SHA-256 normalise pour deduplication exacte."""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def generate_id(prefix: str = "doc") -> str:
    """Genere un identifiant unique court."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def is_in_footer_or_header(
    bbox: tuple[float, float, float, float],
    page_height: float,
    margin_ratio: float = 0.08,
) -> bool:
    """Determine si une boite de texte est dans le header ou le footer."""
    _, y0, _, y1 = bbox
    margin = page_height * margin_ratio
    return y1 < margin or y0 > page_height - margin


def block_to_text(block: TextBlock) -> str:
    """Extrait le texte brut d'un bloc structure."""
    return block.text
