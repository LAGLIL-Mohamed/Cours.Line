#!/usr/bin/env python3
"""Outil d'inspection : analyse les chunks rejetés vs acceptés."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# ── PATH CONFIGURABLE ──
PROJECT_ROOT = Path(r"C:\Users\dell\Desktop\Projet python - RAG\rag")
CHUNKS_FILE = PROJECT_ROOT / "data" / "output" / "chunks.jsonl"
# ──────────────────────

sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    if not CHUNKS_FILE.exists():
        print(f"Fichier non trouvé : {CHUNKS_FILE}")
        print("Vérifie que le pipeline a bien généré le fichier de sortie.")
        return 1

    chunks = []
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    total = len(chunks)
    accepted = [c for c in chunks if not c["rejected"]]
    rejected = [c for c in chunks if c["rejected"]]

    print(f"=== Rapport d'inspection : {CHUNKS_FILE.name} ===\n")
    print(f"Total chunks  : {total}")
    if total > 0:
        print(f"Acceptés    : {len(accepted)} ({len(accepted)/total*100:.1f}%)")
        print(f"Rejetés     : {len(rejected)} ({len(rejected)/total*100:.1f}%)\n")
    else:
        print("Aucun chunk trouvé.")
        return 0

    if rejected:
        print("--- Raisons de rejet ---")
        reasons = Counter(c["rejection_reason"] for c in rejected)
        for reason, count in reasons.most_common():
            print(f"  {count:4d} × {reason}")

        print(f"\n--- Exemples de rejetés (3 premiers) ---")
        for c in rejected[:3]:
            preview = c["texte_chunk"][:120].replace("\n", " ")
            print(f"  [{c['chunk_id']}] {c['rejection_reason']}")
            print(f"      → {preview}...\n")

    if accepted:
        scores = [c["quality_score"] for c in accepted]
        print(f"--- Scores qualité (acceptés) ---")
        print(f"  Min : {min(scores):.3f}")
        print(f"  Max : {max(scores):.3f}")
        print(f"  Moy : {sum(scores)/len(scores):.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
