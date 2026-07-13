#!/usr/bin/env python3
"""CLI : construit l'index FAISS a partir de chunks.jsonl."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(r"C:\\Users\\dell\\Desktop\\Projet python - RAG\\rag")
sys.path.insert(0, str(PROJECT_ROOT))

from vectorization import BGEEmbedder, FAISSIndexer, MetadataStore, VectorConfig


def main() -> int:
    config = VectorConfig()

    print("=" * 50)
    print("CONSTRUCTION DE L'INDEX FAISS")
    print("=" * 50)
    print()

    embedder = BGEEmbedder(config)
    embeddings = embedder.embed_chunks_from_file()

    if not embeddings:
        print("Aucun chunk a indexer.")
        return 1

    indexer = FAISSIndexer(config)
    indexer.build(embeddings)
    indexer.save()

    store = MetadataStore(config.metadata_path)
    store.save(embeddings)

    print()
    print("=" * 50)
    print("INDEX CONSTRUIT AVEC SUCCES")
    print(f"  -> Index FAISS : {config.faiss_index_path}")
    print(f"  -> Metadonnees : {config.metadata_path}")
    print(f"  -> Vecteurs    : {indexer.n_vectors}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
