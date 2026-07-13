#!/usr/bin/env python3
"""CLI : recherche semantique dans l'index FAISS."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(r"C:\\Users\\dell\\Desktop\\Projet python - RAG\\rag")
sys.path.insert(0, str(PROJECT_ROOT))

from vectorization import ContextRetriever, VectorConfig


def search_once(retriever: ContextRetriever, query: str, use_reranker: bool) -> None:
    """Effectue une recherche et affiche les resultats."""
    print(f'\nRecherche : "{query}"')
    if use_reranker:
        print("  [Re-ranking active avec cross-encoder]")
    print()

    results = retriever.search(query, use_reranker=use_reranker)

    print(f"Resultats ({len(results)}) :\n")
    for r in results:
        print("=" * 60)
        print(f"  [{r.rank}] Score: {r.score:.4f} | Chunk: {r.chunk_id}")
        print(f"      Section : {r.titre_section}")
        print(f"      Source  : {r.source_id}, Page: {r.page}")
        print("-" * 60)
        print(f"      TEXTE COMPLET :\n")
        print(f"      {r.texte_chunk}")
        print("=" * 60)
        print()


def main() -> int:
    config = VectorConfig()

    print("=" * 50)
    print("RECHERCHE SEMANTIQUE")
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

        search_once(retriever, query, use_reranker)

    return 0


if __name__ == "__main__":
    sys.exit(main())