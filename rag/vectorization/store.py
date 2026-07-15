"""Stockage et recuperation des metadonnees chunks."""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import Embedding, SearchResult


class MetadataStore:
    """Gere le mapping ID FAISS <-> metadonnees chunks."""

    def __init__(self, metadata_path: Path | None = None) -> None:
        self.metadata_path = metadata_path
        self._data: list[dict] = []
        self._id_to_index: dict[str, int] = {}

    def load(self, path: Path | None = None) -> None:
        target = path or self.metadata_path
        if target is None or not target.exists():
            raise FileNotFoundError(f"Fichier metadonnees non trouve : {target}")

        self._data = []
        self._id_to_index = {}
        with open(target, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                record = json.loads(line)
                self._data.append(record)
                self._id_to_index[record["chunk_id"]] = i

        print(f"Metadonnees chargees : {len(self._data)} entrees")

    def save(self, embeddings: list[Embedding], path: Path | None = None) -> None:
        target = path or self.metadata_path
        if target is None:
            raise ValueError("Aucun chemin de sauvegarde specifie")

        target.parent.mkdir(parents=True, exist_ok=True)

        with open(target, "w", encoding="utf-8") as f:
            for emb in embeddings:
                record = {
                    "chunk_id": emb.chunk_id,
                    "text": emb.text,
                    "chapitre": emb.chapitre,
                    "titre": emb.titre,
                    "sous_titre": emb.sous_titre,
                    "hierarchy_path": emb.hierarchy_path,
                    "page_debut": emb.page_debut,
                    "page_fin": emb.page_fin,
                    "source_id": emb.source_id,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._data = [json.loads(line) for line in open(target, "r", encoding="utf-8")]
        self._id_to_index = {r["chunk_id"]: i for i, r in enumerate(self._data)}
        print(f"Metadonnees sauvegardees : {target}")

    def get_by_index(self, index: int) -> dict:
        return self._data[index]

    def get_by_chunk_id(self, chunk_id: str) -> dict:
        idx = self._id_to_index.get(chunk_id)
        if idx is None:
            raise KeyError(f"chunk_id inconnu : {chunk_id}")
        return self._data[idx]

    def build_search_results(
        self,
        indices: list[int],
        distances: list[float],
    ) -> list[SearchResult]:
        results = []
        for rank, (idx, dist) in enumerate(zip(indices, distances), 1):
            meta = self._data[idx]
            results.append(SearchResult(
                chunk_id=meta["chunk_id"],
                score=float(dist),
                rank=rank,
                text=meta["text"],
                chapitre=meta.get("chapitre"),
                titre=meta.get("titre"),
                sous_titre=meta.get("sous_titre"),
                hierarchy_path=meta.get("hierarchy_path", []),
                page_debut=meta.get("page_debut"),
                page_fin=meta.get("page_fin"),
                source_id=meta["source_id"],
            ))
        return results