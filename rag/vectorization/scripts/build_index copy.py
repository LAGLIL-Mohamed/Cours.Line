#!/usr/bin/env python3
"""CLI : construit l'index FAISS a partir de chunks.jsonl (multi-PDF)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vectorization import BGEEmbedder, FAISSIndexer, MetadataStore, VectorConfig


def main() -> int:
    config = VectorConfig()

    print("=" * 60)
    print("CONSTRUCTION DE L'INDEX FAISS (MULTI-PDF)")
    print("=" * 60)
    print()

    # Decouverte automatique des fichiers chunks.jsonl
    chunks_files = config.discover_chunks_files()

    if not chunks_files:
        print("ERREUR : Aucun chunks.jsonl trouve.")
        print()
        print("Verifiez que :")
        print(f"  1. Le dossier existe : {config.chunks_input_dir}")
        print(f"  2. Des sous-dossiers contiennent un fichier 'chunks.jsonl'")
        print()
        return 1

    print(f"Fichiers detectes : {len(chunks_files)}")
    for cf in chunks_files:
        print(f"  -> {cf}")
    print()

    embedder = BGEEmbedder(config)
    embeddings = embedder.embed_chunks_from_files(chunks_files)

    if not embeddings:
        print("Aucun chunk a indexer.")
        return 1

    indexer = FAISSIndexer(config)
    indexer.build(embeddings)
    indexer.save()

    store = MetadataStore(config.metadata_path)
    store.save(embeddings)

    print()
    print("=" * 60)
    print("INDEX CONSTRUIT AVEC SUCCES")
    print(f"  -> Index FAISS : {config.faiss_index_path}")
    print(f"  -> Metadonnees : {config.metadata_path}")
    print(f"  -> Vecteurs    : {indexer.n_vectors}")
    print(f"  -> Sources     : {len(chunks_files)} PDF(s)")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
