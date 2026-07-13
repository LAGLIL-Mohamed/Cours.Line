"""Re-ranking des resultats avec cross-encoder."""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from .schemas import SearchResult


class CrossEncoderReranker:
    """Recalcule la pertinence requete-chunk avec un cross-encoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        print(f"Chargement du cross-encoder : {model_name}")
        self.model = CrossEncoder(model_name)
        print("  -> OK")

    def rerank(self, query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        """Recalcule les scores et re-tri les resultats."""
        if not results:
            return []

        pairs = [(query, r.texte_chunk) for r in results]
        scores = self.model.predict(pairs)

        # Attacher les nouveaux scores
        for r, score in zip(results, scores):
            r.score = float(score)

        # Re-tri par score decroissant
        reranked = sorted(results, key=lambda x: x.score, reverse=True)
        return reranked[:top_k]