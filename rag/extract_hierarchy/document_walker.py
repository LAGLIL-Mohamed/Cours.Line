from __future__ import annotations

from typing import Optional

from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc import DocItemLabel
from rapidfuzz import fuzz

from extract_hierarchy.config import LABELS_BODY, LABELS_HEADING, LABELS_NOISE
from extract_hierarchy.models import TocEntry, HeadingNode, ContentBlock
from extract_hierarchy.utils import (
    level_label,
    normalize,
    is_structural_noise,
    is_garbage_heading,
    is_repeated_footer,
)


def find_best_toc_match(
    text: str, toc_entries: list[TocEntry], min_score: int
) -> Optional[TocEntry]:
    best_entry = None
    best_score = 0
    norm_text = normalize(text)
    for entry in toc_entries:
        if entry.matched:
            continue
        score = fuzz.token_sort_ratio(norm_text, normalize(entry.title))
        if score > best_score:
            best_score = score
            best_entry = entry
    if best_entry is not None and best_score >= min_score:
        return best_entry
    return None


def walk_document(
    doc: DoclingDocument, toc_entries: list[TocEntry], fuzzy_threshold: int
):
    stack: list[HeadingNode] = []
    outline: list[HeadingNode] = []
    blocks: list[ContentBlock] = []
    elements: list[dict] = []
    heading_counter = 0

    # Buffer pour la détection des footers répétés (5 derniers blocs)
    footer_buffer: list[str] = []
    FOOTER_BUFFER_SIZE = 5

    for item, _tree_level in doc.iterate_items():
        label = getattr(item, "label", None)
        text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue
        page = item.prov[0].page_no if getattr(item, "prov", None) else None

        # ═══════════════════════════════════════════════════
        # CORRECTIF 1 : Rejet explicite des labels bruyants
        # ═══════════════════════════════════════════════════
        if label in LABELS_NOISE:
            continue

        # ═══════════════════════════════════════════════════
        # CORRECTIF 3 : Détection des footers répétés
        # ═══════════════════════════════════════════════════
        if is_repeated_footer(text, footer_buffer):
            continue

        # Mise à jour du buffer circulaire
        footer_buffer.append(normalize(text))
        if len(footer_buffer) > FOOTER_BUFFER_SIZE:
            footer_buffer.pop(0)

        # ── Traitement des headings ──
        if label in LABELS_HEADING:
            raw_level = getattr(item, "level", None)
            if label == DocItemLabel.TITLE or raw_level is None:
                raw_level = 1

            source = "layout"
            final_level = raw_level

            # ═══════════════════════════════════════════════
            # CORRECTIF 4 : Filtrage des headings poubelle
            # ═══════════════════════════════════════════════
            if is_garbage_heading(text):
                # On ignore purement le heading poubelle :
                # il ne crée pas de niveau hiérarchique.
                continue

            if toc_entries:
                match = find_best_toc_match(text, toc_entries, fuzzy_threshold)
                if match is not None:
                    match.matched = True
                    final_level = match.level
                    source = "toc_corrige"

            while stack and stack[-1].level >= final_level:
                stack.pop()

            heading_counter += 1
            node = HeadingNode(
                id=f"h{heading_counter}",
                level=final_level,
                label_type=level_label(final_level),
                title=text,
                page_start=page,
                source=source,
            )
            stack.append(node)
            outline.append(node)
            elements.append(
                {"kind": "heading", "level": final_level, "title": text, "page": page}
            )
            continue

        # ── Traitement du body ──
        if label in LABELS_BODY:
            # ═══════════════════════════════════════════════
            # CORRECTIF 2 : Filtre sémantique sur le body
            # ═══════════════════════════════════════════════
            if is_structural_noise(text):
                continue

            path_titles = [n.title for n in stack]
            chapitre = next((n.title for n in stack if n.level == 1), None)
            titre = next((n.title for n in stack if n.level == 2), None)
            sous_titre = next(
                (n.title for n in reversed(stack) if n.level >= 3), None
            )

            block = ContentBlock(
                text=text,
                page=page,
                hierarchy_path=path_titles,
                chapitre=chapitre,
                titre=titre,
                sous_titre=sous_titre,
            )
            blocks.append(block)
            elements.append({"kind": "body", "text": text, "page": page})

    return outline, blocks, elements