#!/usr/bin/env python3
"""CLI : question complete avec reconstruction contexte."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(r"C:\\Users\\dell\\Desktop\\Projet python - RAG\\rag")
sys.path.insert(0, str(PROJECT_ROOT))

from vectorization import ContextRetriever, VectorConfig


def query_once(retriever: ContextRetriever, query: str, use_reranker: bool) -> None:
    """Effectue une requete et affiche le contexte reconstruit."""
    print(f'\nQuestion : "{query}"')
    if use_reranker:
        print("  [Re-ranking active avec cross-encoder]")
    print()

    context = retriever.retrieve_context(query, use_reranker=use_reranker)

    print(f"Contexte reconstruit : {len(context.sections)} section(s)")
    print(f"Estimation tokens    : ~{context.total_tokens_estimate}\n")

    print("=" * 50)
    print("CONTEXTE FORMATE POUR LLM")
    print("=" * 50)
    print()

    formatted = retriever.format_context_for_llm(context)
    print(formatted)

    print()
    print("=" * 50)
    print("DETAIL DES SECTIONS")
    print("=" * 50)
    for i, sec in enumerate(context.sections, 1):
        print(f"\n--- Section {i} ---")
        print(f"  ID    : {sec.section_id}")
        print(f"  Titre : {sec.titre_section}")
        print(f"  Pages : {sorted(sec.pages)}")
        print(f"  Source: {sec.source_id}")
        print(f"  Chunks: {sec.chunks_sources}")
        print(f"  Longueur: {len(sec.texte_complet)} caracteres")


def main() -> int:
    config = VectorConfig()

    print("=" * 50)
    print("RECONSTRUCTION DU CONTEXTE")
    print("=" * 50)
    print()

    # Detecte --rerank dans les arguments
    use_reranker = "--rerank" in sys.argv
    if use_reranker:
        sys.argv.remove("--rerank")

    retriever = ContextRetriever(config=config, use_reranker=use_reranker)
    print("Chargement de l'index...")
    retriever.load_index()
    print(f"  -> {retriever.indexer.n_vectors} vecteurs charges\n")

    # Mode interactif en boucle
    while True:
        try:
            query = input("Question (ou 'q' pour quitter) : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break

        if query.lower() in ("q", "quit", "exit", ""):
            print("Au revoir !")
            break

        query_once(retriever, query, use_reranker)

    return 0


if __name__ == "__main__":
    sys.exit(main())