#!/usr/bin/env python3
"""CLI : lance le pipeline sur les PDFs du dossier INPUT_DIR."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# PATHS CONFIGURABLES
PROJECT_ROOT = Path(r"C:\Users\dell\Desktop\Projet python - RAG\rag")
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_FILE = PROJECT_ROOT / "data" / "output" / "chunks.jsonl"
VALID_ONLY = True
CHUNK_MAX_TOKENS = 512

sys.path.insert(0, str(PROJECT_ROOT))

from rag_pipeline import DocumentProcessor, PipelineConfig


def main() -> int:
    config = PipelineConfig(chunk_max_tokens=CHUNK_MAX_TOKENS)
    processor = DocumentProcessor(config)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"Aucun PDF trouve dans : {INPUT_DIR}")
        return 1

    all_chunks = []
    for pdf in pdf_files:
        print(f"\n📄 Traitement : {pdf.name}")
        chunks = processor.process(pdf)
        if VALID_ONLY:
            chunks = [c for c in chunks if not c.rejected]
        all_chunks.extend(chunks)
        print(f"   ✅ {len(chunks)} chunks generes")

        if chunks:
            first = chunks[0]
            print(f"   📌 Exemple : [{first.chapter_title}] > {' > '.join(first.hierarchy_path)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"✅ Exporte : {OUTPUT_FILE}")
    print(f"   Total chunks : {len(all_chunks)}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
