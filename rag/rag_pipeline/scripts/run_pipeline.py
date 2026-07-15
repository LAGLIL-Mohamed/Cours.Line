#!/usr/bin/env python3
"""CLI : lance le pipeline sur les PDFs du dossier INPUT_DIR."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ── PATHS CONFIGURABLES ──
PROJECT_ROOT = Path(r"C:\Users\dell\Desktop\Projet python - RAG\rag")
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
VALID_ONLY = True
CHUNK_MAX_TOKENS = 512
# ──────────────────────────

sys.path.insert(0, str(PROJECT_ROOT))

from rag_pipeline import DocumentProcessor, PipelineConfig


def main() -> int:
    config = PipelineConfig(
        chunk_max_tokens=CHUNK_MAX_TOKENS,
        hierarchy_aware_chunking=True,
        toc_enabled=True,
    )
    processor = DocumentProcessor(config)

    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"Aucun PDF trouvé dans : {INPUT_DIR}")
        return 1

    for pdf in pdf_files:
        print(f"\n{'='*60}")
        print(f"Traitement : {pdf.name}")
        print(f"{'='*60}")

        # Crée un sous-dossier par PDF
        pdf_output = OUTPUT_DIR / pdf.stem
        pdf_output.mkdir(parents=True, exist_ok=True)

        result = processor.process_with_outputs(pdf, pdf_output)

        chunks = result["chunks"]
        if VALID_ONLY:
            chunks = [c for c in chunks if not c.rejected]

        # Écrit aussi un chunks_valid.jsonl
        with open(pdf_output / "chunks_valid.jsonl", "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        print(f"  -> {result['chunks_count']} chunks générés")
        print(f"  -> {result['valid_chunks']} chunks valides")
        print(f"  -> Sorties dans : {pdf_output}")

    print(f"\n{'='*60}")
    print("Pipeline terminé !")
    return 0


if __name__ == "__main__":
    sys.exit(main())