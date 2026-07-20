from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional
from dataclasses import asdict

from extract_hierarchy.pdf_converter import convert_pdf
from extract_hierarchy.toc_extractor import extract_toc
from extract_hierarchy.document_walker import walk_document
from extract_hierarchy.output_generators import (
    build_markdown,
    build_hierarchy_tree,
    build_flat_plan,
    build_chunks,
)


def process_single_pdf(
    pdf_path: Path,
    output_dir: Path,
    max_scan_pages: int,
    toc_max_span: int,
    fuzzy_threshold: int,
    chunk_size: int,
    chunk_overlap: int,
    disable_toc: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"TRAITEMENT : {pdf_path.name}")
    print("=" * 60)

    print(f"[1/4] Conversion Docling de {pdf_path.name} ...")
    doc = convert_pdf(pdf_path)
    doc_title = getattr(doc, "name", None) or pdf_path.stem

    toc_entries = []
    if not disable_toc:
        print("[2/4] Recherche du sommaire (TOC) ...")
        toc_entries = extract_toc(doc, max_scan_pages, toc_max_span)
        if toc_entries:
            print(f"      -> {len(toc_entries)} entrees de sommaire detectees.")
        else:
            print("      -> aucun sommaire detecte, passage en mode 'layout seul'.")
    else:
        print("[2/4] Detection du TOC desactivee.")

    print("[3/4] Parcours du document et rattachement hierarchique ...")
    outline, blocks, elements = walk_document(doc, toc_entries, fuzzy_threshold)
    print(f"      -> {len(outline)} titres detectes, {len(blocks)} blocs de texte.")

    print("[4/4] Generation des fichiers de sortie ...")

    md = build_markdown(elements, doc_title)
    (output_dir / "document.md").write_text(md, encoding="utf-8")

    hierarchy_tree = build_hierarchy_tree(outline, doc_title)
    hierarchy_out = {
        "titre_document": doc_title,
        "arbre": hierarchy_tree,
        "plan_a_plat": build_flat_plan(outline),
    }
    (output_dir / "hierarchy.json").write_text(
        json.dumps(hierarchy_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    chunks = build_chunks(blocks, chunk_size, chunk_overlap)
    with (output_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    if toc_entries:
        (output_dir / "toc_debug.json").write_text(
            json.dumps([asdict(e) for e in toc_entries], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        unmatched = [e.title for e in toc_entries if not e.matched]
        if unmatched:
            print(
                f"      -> ATTENTION : {len(unmatched)} entrees du sommaire "
                f"n'ont pas ete retrouvees (voir warnings.json)."
            )
        (output_dir / "warnings.json").write_text(
            json.dumps(
                {"toc_entries_non_retrouvees": unmatched}, ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )

    print(f"\nTermine. Fichiers ecrits dans : {output_dir}")
    print(f"  - {output_dir / 'document.md'}")
    print(f"  - {output_dir / 'hierarchy.json'}")
    print(f"  - {output_dir / 'chunks.jsonl'} ({len(chunks)} chunks)")


def run(
    pdf_dir: Path,
    output_dir: Optional[Path],
    max_scan_pages: int,
    toc_max_span: int,
    fuzzy_threshold: int,
    chunk_size: int,
    chunk_overlap: int,
    disable_toc: bool,
) -> None:
    if not pdf_dir.exists():
        print(f"Erreur : dossier introuvable : {pdf_dir}", file=sys.stderr)
        sys.exit(1)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"Aucun fichier PDF trouve dans : {pdf_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Dossier source  : {pdf_dir}")
    print(f"Dossier sortie  : {output_dir or pdf_dir} (sous-dossiers par PDF)")
    print(f"PDFs trouves    : {len(pdf_files)}")
    print("-" * 60)

    base_output_dir = output_dir if output_dir is not None else pdf_dir

    for pdf_path in pdf_files:
        pdf_output_dir = base_output_dir / pdf_path.stem
        process_single_pdf(
            pdf_path=pdf_path,
            output_dir=pdf_output_dir,
            max_scan_pages=max_scan_pages,
            toc_max_span=toc_max_span,
            fuzzy_threshold=fuzzy_threshold,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            disable_toc=disable_toc,
        )

    print("\n" + "=" * 60)
    print(f"TOUS LES PDF TRAITES : {len(pdf_files)}")
    print("=" * 60)