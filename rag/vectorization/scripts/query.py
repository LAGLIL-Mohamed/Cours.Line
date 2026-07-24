#!/usr/bin/env python3
"""CLI : question complete avec reconstruction contexte."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vectorization import ContextRetriever, VectorConfig


def main() -> int:
    config = VectorConfig()

    print("=" * 60)
    print("RECONSTRUCTION DU CONTEXTE")
    print("=" * 60)
    print()

    retriever = ContextRetriever(config=config)
    print("Chargement de l'index...")
    retriever.load_index()
    print(f"  -> {retriever.indexer.n_vectors} vecteurs charges\n")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Question : ")

    if not query.strip():
        print("Aucune question fournie.")
        return 1

    print(f"\nQuestion : \"{query}\"\n")

    context = retriever.retrieve_context(query)

    print(f"Contexte reconstruit : {len(context.sections)} section(s)")
    print(f"Estimation tokens    : ~{context.total_tokens_estimate}\n")

    print("=" * 60)
    print("CONTEXTE FORMATE POUR LLM")
    print("=" * 60)
    print()

    formatted = retriever.format_context_for_llm(context)
    print(formatted)

    print()
    print("=" * 60)
    print("DETAIL DES SECTIONS")
    print("=" * 60)
    for i, sec in enumerate(context.sections, 1):
        print(f"\n--- Section {i} ---")
        print(f"  Source : {sec.source_id}")
        path = " > ".join(sec.hierarchy_path) if sec.hierarchy_path else "N/A"
        print(f"  Chemin : {path}")
        print(f"  Pages  : {sorted(sec.pages)}")
        print(f"  Chunks : {sec.chunks_sources}")
        print(f"  Longueur : {len(sec.texte_complet)} caracteres")

    return 0


if __name__ == "__main__":
    sys.exit(main())