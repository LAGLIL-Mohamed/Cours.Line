"""Generation des embeddings avec MiniLM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import VectorConfig
from .schemas import Embedding


class BGEEmbedder:
    """Wrappe le modele MiniLM pour encoder chunks et requetes."""

    def __init__(self, config: VectorConfig | None = None) -> None:
        self.config = config or VectorConfig()
        print(f"Chargement du modele MiniLM : {self.config.model_name}")
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
        # MiniLM n'a pas besoin de préfixe spécifique
        embedding = self.model.encode(
            query,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embedding

    def embed_chunks_from_file(self, chunks_file: Path | None = None) -> list[Embedding]:
        if chunks_file is None:
            chunks_file = self.config.chunks_input

        print(f"Lecture des chunks : {chunks_file}")
        chunks = []
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))

        valid_chunks = [c for c in chunks if not c.get("rejected", False)]
        print(f"  -> {len(valid_chunks)} chunks valides / {len(chunks)} total")

        texts = [c["texte_chunk"] for c in valid_chunks]
        vectors = self.embed_chunks(texts)

        embeddings = []
        for chunk, vec in zip(valid_chunks, vectors):
            embeddings.append(Embedding(
                chunk_id=chunk["chunk_id"],
                vector=vec,
                texte_chunk=chunk["texte_chunk"],
                texte_section_complete=chunk["texte_section_complete"],
                titre_section=chunk["titre_section"],
                source_id=chunk["source_id"],
                page=chunk["page"],
                section_id=chunk["section_id"],
                position_in_section=chunk["position_in_section"],
            ))

        return embeddings