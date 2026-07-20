from __future__ import annotations

import re
import unicodedata

from extract_hierarchy.config import (
    LEVEL_LABELS,
    TOC_HEADING_PATTERNS,
    RE_PHONE_MA,
    RE_EMAIL,
    RE_BIBLIO,
    RE_FORM_DOTS,
    RE_FORM_GENERIC,
    GARBAGE_HEADING_KEYWORDS,
    MEDICAL_KEYWORDS,
    NOISE_CHAR_RATIO_THRESHOLD,
    GARBAGE_HEADING_MIN_WORDS,
)


def normalize(s: str) -> str:
    """Normalise le texte : minuscule, sans accents, espaces unifiés."""
    s = (s or "").strip().lower()
    s = "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
    s = re.sub(r"\s+", " ", s)
    return s


def level_label(level: int) -> str:
    return LEVEL_LABELS.get(level, f"niveau_{level}")


def is_toc_heading(text: str) -> bool:
    t = normalize(text)
    return any(re.match(p, t) for p in TOC_HEADING_PATTERNS)


# ═══════════════════════════════════════════════════════════
#  NOUVEAU : Filtres de bruit structurel
# ═══════════════════════════════════════════════════════════

def is_structural_noise(text: str) -> bool:
    """
    Détecte si un bloc de texte est du bruit structurel
    (téléphone, email, biblio, formulaire, etc.).
    """
    if not text or not text.strip():
        return True

    t = text.strip()

    # 1. Lignes de points pures
    if RE_FORM_DOTS.match(t):
        return True

    # 2. Formulaires génériques courts
    if RE_FORM_GENERIC.search(t) and len(t.split()) < 8:
        return True

    # 3. Téléphones / fax marocains
    if RE_PHONE_MA.search(t):
        return True

    # 4. Emails isolés ou dominants
    emails = RE_EMAIL.findall(t)
    if emails and len(emails) >= len(t.split()) * 0.3:
        return True

    # 5. Bibliographie
    biblio_hits = len(RE_BIBLIO.findall(t))
    if biblio_hits >= 2:
        return True

    # 6. Ratio de caractères non-alphabétiques
    alpha_count = sum(1 for c in t if c.isalpha())
    total_count = len(t.replace(" ", "").replace("\n", ""))
    if total_count > 0 and alpha_count / total_count < (1 - NOISE_CHAR_RATIO_THRESHOLD):
        return True

    return False


def is_garbage_heading(text: str) -> bool:
    """
    Détecte si un heading est une page de garde / titre poubelle
    (Cahier des charges, Guide, Rapport, Ministère, etc.).
    """
    if not text:
        return True

    t_norm = normalize(text)
    words = t_norm.split()
    word_count = len(words)

    # Trop court = suspect
    if word_count < GARBAGE_HEADING_MIN_WORDS:
        # Vérifie s'il contient un mot-clé poubelle
        for kw in GARBAGE_HEADING_KEYWORDS:
            if kw in t_norm:
                return True
        # Vérifie l'absence de terme médical
        has_medical = any(med in t_norm for med in MEDICAL_KEYWORDS)
        if not has_medical:
            return True

    return False


def is_repeated_footer(text: str, recent_buffer: list[str]) -> bool:
    """
    Détecte si un bloc est un footer répété (présent dans le buffer
    des N derniers blocs normalisés).
    """
    if not text or not recent_buffer:
        return False

    t_norm = normalize(text)
    if len(t_norm) < 5:
        return False  # trop court pour être sûr

    for prev in recent_buffer:
        if t_norm == prev:
            return True
    return False