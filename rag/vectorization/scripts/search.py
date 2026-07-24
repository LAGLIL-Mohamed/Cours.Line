#!/usr/bin/env python3
"""CLI : recherche semantique dans l'index FAISS."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vectorization import ContextRetriever, VectorConfig


def main() -> int:
    config = VectorConfig()

    print("=" * 60)
    print("RECHERCHE SEMANTIQUE")
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

    print(f"\nRecherche : \"{query}\"\n")

    results = retriever.search(query)

    print(f"Resultats ({len(results)}) :\n")
    for r in results:
        print(f"  [{r.rank}] Score: {r.score:.4f} | Chunk: {r.chunk_id}")
        print(f"      Source : {r.source_id}")
        if r.chapitre:
            print(f"      Chapitre : {r.chapitre}")
        if r.titre:
            print(f"      Titre : {r.titre}")
        if r.sous_titre:
            print(f"      Sous-titre : {r.sous_titre}")
        if r.hierarchy_path:
            print(f"      Chemin : {' > '.join(r.hierarchy_path)}")
        print(f"      Pages : {r.page_debut or '?'} - {r.page_fin or '?'}")
        print(f"      Texte : {r.text[:200]}...")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())