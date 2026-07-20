from __future__ import annotations

from typing import Optional
from dataclasses import asdict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from extract_hierarchy.models import ContentBlock, Chunk
from extract_hierarchy.utils import is_structural_noise
from extract_hierarchy.config import MIN_CHUNK_WORDS, CHUNK_MERGE_WORDS


def build_markdown(elements: list[dict], doc_title: str) -> str:
    lines = [f"# {doc_title}", ""]
    for el in elements:
        if el["kind"] == "heading":
            lines.append(f"{'#' * min(el['level'] + 1, 6)} {el['title']}")
            lines.append("")
        else:
            lines.append(el["text"])
            lines.append("")
    return "\n".join(lines)


def build_hierarchy_tree(outline: list, doc_title: str) -> dict:
    root = {"type": "document", "titre": doc_title, "enfants": []}
    stack = [(0, root)]
    for node in outline:
        entry = {
            "type": node.label_type,
            "titre": node.title,
            "page_debut": node.page_start,
            "source": node.source,
            "enfants": [],
        }
        while stack and stack[-1][0] >= node.level:
            stack.pop()
        stack[-1][1]["enfants"].append(entry)
        stack.append((node.level, entry))
    return root


def build_flat_plan(outline: list) -> list[str]:
    plan = []
    for node in outline:
        indent = "  " * (node.level - 1)
        plan.append(
            f"{indent}- [{node.label_type}] {node.title} (p.{node.page_start})"
        )
    return plan


def build_chunks(
    blocks: list[ContentBlock], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    buffer_texts: list[str] = []
    buffer_pages: list[Optional[int]] = []
    current_meta: Optional[ContentBlock] = None
    chunk_counter = 0

    def flush():
        nonlocal buffer_texts, buffer_pages, chunk_counter
        if not buffer_texts or current_meta is None:
            buffer_texts, buffer_pages = [], []
            return
        combined = "\n\n".join(buffer_texts)
        pieces = (
            splitter.split_text(combined)
            if len(combined) > chunk_size
            else [combined]
        )
        pages = [p for p in buffer_pages if p is not None]
        page_debut = min(pages) if pages else None
        page_fin = max(pages) if pages else None

        for piece in pieces:
            # ═══════════════════════════════════════════
            # CORRECTIF 5 : Filtre post-split
            # ═══════════════════════════════════════════
            if is_structural_noise(piece):
                continue

            chunk_counter += 1
            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{chunk_counter:05d}",
                    text=piece.strip(),
                    chapitre=current_meta.chapitre,
                    titre=current_meta.titre,
                    sous_titre=current_meta.sous_titre,
                    hierarchy_path=current_meta.hierarchy_path,
                    page_debut=page_debut,
                    page_fin=page_fin,
                )
            )
        buffer_texts, buffer_pages = [], []

    for block in blocks:
        path_key = tuple(block.hierarchy_path)
        current_key = tuple(current_meta.hierarchy_path) if current_meta else None
        if current_meta is not None and path_key != current_key:
            flush()
        current_meta = block
        buffer_texts.append(block.text)
        buffer_pages.append(block.page)
        if sum(len(t) for t in buffer_texts) > chunk_size * 4:
            flush()
            current_meta = block

    flush()

    # ═══════════════════════════════════════════════════
    # CORRECTIF 5 (suite) : Fusion des chunks courts
    # ═══════════════════════════════════════════════════
    merged: list[Chunk] = []
    i = 0
    while i < len(chunks):
        current = chunks[i]
        word_count = len(current.text.split())

        # Fusion avec le suivant si trop court et même hiérarchie
        while (
            word_count < CHUNK_MERGE_WORDS
            and i + 1 < len(chunks)
            and current.hierarchy_path == chunks[i + 1].hierarchy_path
        ):
            next_chunk = chunks[i + 1]
            current.text = (current.text + "\n\n" + next_chunk.text).strip()
            current.page_fin = next_chunk.page_fin
            word_count = len(current.text.split())
            i += 1

        # Rejet final si toujours trop court
        if word_count >= MIN_CHUNK_WORDS:
            merged.append(current)
        i += 1

    # Ré-indexation des IDs après fusion/rejet
    for idx, c in enumerate(merged, 1):
        c.chunk_id = f"chunk_{idx:05d}"

    return merged