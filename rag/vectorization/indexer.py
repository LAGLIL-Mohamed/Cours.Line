"""Indexation FAISS des embeddings."""

from __future__ import annotations

import numpy as np

from .config import VectorConfig
from .schemas import Embedding


class FAISSIndexer:
    """Gere la creation, la sauvegarde et le chargement d'un index FAISS."""

    def __init__(self, config: VectorConfig | None = None) -> None:
        self.config = config or VectorConfig()
        self.index = None
        self._dimension: int | None = None

    @property
    def is_trained(self) -> bool:
        return self.index is not None

    @property
    def n_vectors(self) -> int:
        if self.index is None:
            return 0
        return self.index.ntotal

    def build(self, embeddings: list[Embedding]) -> None:
        import faiss

        if not embeddings:
            raise ValueError("Aucun embedding fourni")

        vectors = np.vstack([emb.vector for emb in embeddings]).astype("float32")
        self._dimension = vectors.shape[1]

        print(f"Construction index FAISS : {len(embeddings)} vecteurs, dim={self._dimension}")

        if self.config.index_type == "flat_ip":
            self.index = faiss.IndexFlatIP(self._dimension)
            print("  -> Type : IndexFlatIP (exact, cosine similarity)")

        elif self.config.index_type == "ivf_flat":
            nlist = min(self.config.nlist, len(embeddings) // 10)
            nlist = max(nlist, 1)
            quantizer = faiss.IndexFlatIP(self._dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self._dimension, nlist)
            print(f"  -> Type : IndexIVFFlat (approximatif, nlist={nlist})")
            print("  -> Entrainement de l'index...")
            self.index.train(vectors)

        else:
            raise ValueError(f"Type d'index inconnu : {self.config.index_type}")

        self.index.add(vectors)
        print(f"  -> Index construit : {self.n_vectors} vecteurs")

    def search(self, query_vector: np.ndarray, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        import faiss

        if self.index is None:
            raise RuntimeError("L'index n'est pas construit.")

        if k is None:
            k = self.config.top_k

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        query_vector = query_vector.astype("float32")

        if hasattr(self.index, "nprobe"):
            self.index.nprobe = max(self.config.nlist // 10, 1)

        distances, indices = self.index.search(query_vector, k)
        return distances, indices

    def save(self, path: str | None = None) -> None:
        import faiss

        if self.index is None:
            raise RuntimeError("Aucun index a sauvegarder")

        target = path or str(self.config.faiss_index_path)
        faiss.write_index(self.index, target)
        print(f"Index sauvegarde : {target}")

    def load(self, path: str | None = None) -> None:
        import faiss

        target = path or str(self.config.faiss_index_path)
        self.index = faiss.read_index(target)
        self._dimension = self.index.d
        print(f"Index charge : {target} ({self.n_vectors} vecteurs, dim={self._dimension})")
