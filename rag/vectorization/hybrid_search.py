"""Recherche hybride : combinaison sémantique + BM25."""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from .schemas import SearchResult


class BM25Searcher:
    """Index BM25 pour la recherche par mots-cles."""

    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.corpus = [c["texte_chunk"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        tokenized = query.lower().split()
        scores = self.bm25.get_scores(tokenized)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]


def fuse_results(
    semantic_results: list[SearchResult],
    bm25_results: list[tuple[int, float]],
    alpha: float = 0.5,
    top_k: int = 5,
) -> list[SearchResult]:
    """Fusionne les scores sémantiques et BM25 avec Reciprocal Rank Fusion."""

    # Normaliser les scores sémantiques
    sem_scores = {r.chunk_id: r.score for r in semantic_results}

    # Normaliser les scores BM25
    bm25_scores = {semantic_results[idx].chunk_id: score for idx, score in bm25_results if idx < len(semantic_results)}

    # RRF : Reciprocal Rank Fusion
    all_ids = set(sem_scores.keys()) | set(bm25_scores.keys())
    fused = {}

    for cid in all_ids:
        rank_sem = 1 / (1 + list(sem_scores.keys()).index(cid)) if cid in sem_scores else 0
        rank_bm25 = 1 / (1 + list(bm25_scores.keys()).index(cid)) if cid in bm25_scores else 0
        fused[cid] = alpha * rank_sem + (1 - alpha) * rank_bm25

    # Re-tri
    sorted_ids = sorted(fused.keys(), key=lambda x: fused[x], reverse=True)
    chunk_map = {r.chunk_id: r for r in semantic_results}

    final = []
    for cid in sorted_ids[:top_k]:
        if cid in chunk_map:
            r = chunk_map[cid]
            r.score = fused[cid]
            final.append(r)

    return final