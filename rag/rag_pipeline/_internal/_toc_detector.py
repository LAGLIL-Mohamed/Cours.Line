"""Détection et parsing du sommaire (TOC) dans un PDF."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from ..schemas import TocEntry


# =====================================================================
# PATTERNS DE DÉTECTION
# =====================================================================

TOC_HEADING_PATTERNS = [
    r"^\s*sommaire\s*$",
    r"^\s*table des mati[èe]res\s*$",
    r"^\s*plan du cours\s*$",
    r"^\s*plan du document\s*$",
    r"^\s*plan de cours\s*$",
    r"^\s*table of contents\s*$",
    r"^\s*contents\s*$",
]

KEYWORD_LEVELS = {
    "chapitre": 1,
    "chapter": 1,
    "partie": 1,
    "part": 1,
    "unite": 1,
    "unit": 1,
    "module": 1,
    "section": 2,
    "sous-section": 3,
    "sous section": 3,
}

RE_NUMBERED_WITH_PAGE = re.compile(
    r"^(?P<num>\d+(?:\.\d+)*)[\.\)]?\s+(?P<title>.+?)\s*(?:[\.\-_ ]{2,}|\t)\s*(?P<page>\d{1,4})\s*$"
)
RE_NUMBERED_NO_PAGE = re.compile(
    r"^(?P<num>\d+(?:\.\d+)*)[\.\)]?\s+(?P<title>.+)$"
)
RE_KEYWORD = re.compile(
    r"^(?P<kw>chapitre|chapter|partie|part|unite|unit|module|section)\s+"
    r"(?P<num>[\divxlc]+)\s*[:\-\.]?\s*(?P<title>.+?)"
    r"(?:\s*[\.\-_ ]{2,}\s*(?P<page>\d{1,4}))?\s*$",
    re.IGNORECASE,
)


# =====================================================================
# UTILITAIRES
# =====================================================================

def _normalize(s: str) -> str:
    """Minuscule + sans accents + espaces normalisés."""
    s = (s or "").strip().lower()
    s = "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
    s = re.sub(r"\s+", " ", s)
    return s


def is_toc_heading(text: str) -> bool:
    """Détecte si un texte est un titre de sommaire."""
    t = _normalize(text)
    return any(re.match(p, t) for p in TOC_HEADING_PATTERNS)


# =====================================================================
# PARSING DE LIGNE
# =====================================================================

def parse_toc_line(line: str) -> Optional[dict]:
    """Parse une ligne de sommaire. Retourne dict ou None."""
    line = line.strip()
    if not line or len(line) < 2:
        return None

    # Pattern "1.2 Titre ........ 14"
    m = RE_NUMBERED_WITH_PAGE.match(line)
    if m:
        num = m.group("num")
        return {
            "numbering": num,
            "title": m.group("title").strip(" .-_"),
            "level": num.count(".") + 1,
            "page": int(m.group("page")),
        }

    # Pattern "Chapitre 1 : Introduction ... 5"
    m = RE_KEYWORD.match(line)
    if m:
        kw = m.group("kw").lower()
        page = m.group("page")
        return {
            "numbering": f"{kw} {m.group('num')}",
            "title": m.group("title").strip(" .-_"),
            "level": KEYWORD_LEVELS.get(kw, 1),
            "page": int(page) if page else None,
        }

    # Pattern "1.2 Titre" (sans page)
    m = RE_NUMBERED_NO_PAGE.match(line)
    if m:
        num = m.group("num")
        title = m.group("title").strip(" .-_")
        # Filtre anti faux-positif
        if len(title) > 120 or title.endswith((".", ",", ";")):
            return None
        return {
            "numbering": num,
            "title": title,
            "level": num.count(".") + 1,
            "page": None,
        }

    return None


# =====================================================================
# DÉTECTION DU TOC DANS LES PAGES
# =====================================================================

def find_toc_pages(pages_text: list[str], max_scan_pages: int, toc_max_span: int) -> Optional[set]:
    """
    Cherche une page contenant un titre de sommaire dans les N premières pages.
    Retourne l'ensemble des numéros de page du TOC, ou None.
    """
    toc_page = None
    for i, text in enumerate(pages_text[:max_scan_pages], start=1):
        lines = text.split("\n")
        for line in lines:
            if is_toc_heading(line.strip()):
                toc_page = i
                break
        if toc_page is not None:
            break

    if toc_page is None:
        return None

    return set(range(toc_page, min(toc_page + toc_max_span + 1, len(pages_text) + 1)))


def extract_toc_from_text(pages_text: list[str], max_scan_pages: int, toc_max_span: int) -> list[TocEntry]:
    """
    Extrait le TOC depuis le texte brut des pages.
    Adapté pour PyMuPDF (texte brut par page).
    """
    toc_pages = find_toc_pages(pages_text, max_scan_pages, toc_max_span)
    if not toc_pages:
        return []

    entries = []
    for page_num in toc_pages:
        if page_num > len(pages_text):
            continue
        text = pages_text[page_num - 1]
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if not line or is_toc_heading(line):
                continue
            parsed = parse_toc_line(line)
            if parsed:
                entries.append(
                    TocEntry(
                        raw_line=line,
                        numbering=parsed["numbering"],
                        title=parsed["title"],
                        level=parsed["level"],
                        page_hint=parsed["page"],
                    )
                )

    return entries