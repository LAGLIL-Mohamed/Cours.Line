"""Generation des embeddings avec BGE."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import VectorConfig
from .schemas import Embedding


class BGEEmbedder:
    """Wrappe le modele BGE pour encoder chunks et requetes."""

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, config: VectorConfig | None = None) -> None:
        self.config = config or VectorConfig()
        print(f"Chargement du modele BGE : {self.config.model_name}")
        self.model = SentenceTransformer(self.config.model_name, device=self.config.device)
        self._dim = self.model.get_sentence_embedding_dimension()
        print(f"  -> Dimensions : {self._dim}")

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_chunks(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        prefixed = self.QUERY_PREFIX + query
        embedding = self.model.encode(
            prefixed,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embedding

    def embed_chunks_from_files(self, chunks_files: list[Path]) -> list[Embedding]:
        """Lit plusieurs fichiers chunks.jsonl (un par PDF) et genere les embeddings.

        Les chunk_id sont prefixes par le nom du dossier parent (nom du PDF)
        pour garantir l'unicite globale.
        """
        all_chunks = []

        for chunks_file in chunks_files:
            source_id = chunks_file.parent.name  # ex: "mon_pdf"
            print(f"  Lecture : {chunks_file} (source_id={source_id})")
            with open(chunks_file, "r", encoding="utf-8") as f:
                for line in f:
                    chunk = json.loads(line)
                    # Prefixe l'ID pour eviter les collisions entre PDFs
                    chunk["chunk_id"] = f"{source_id}::{chunk['chunk_id']}"
                    chunk["source_id"] = source_id
                    all_chunks.append(chunk)

        valid_chunks = [c for c in all_chunks if not c.get("rejected", False)]
        print(f"  -> {len(valid_chunks)} chunks valides / {len(all_chunks)} total")

        if not valid_chunks:
            return []

        texts = [c["text"] for c in valid_chunks]
        vectors = self.embed_chunks(texts)

        return [
            Embedding(
                chunk_id=c["chunk_id"],
                vector=vectors[i],
                text=c["text"],
                chapitre=c.get("chapitre"),
                titre=c.get("titre"),
                sous_titre=c.get("sous_titre"),
                hierarchy_path=c.get("hierarchy_path", []),
                page_debut=c.get("page_debut"),
                page_fin=c.get("page_fin"),
                source_id=c["source_id"],
            )
            for i, c in enumerate(valid_chunks)
        ]