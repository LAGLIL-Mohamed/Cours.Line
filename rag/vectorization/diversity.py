"""Diversification des resultats avec MMR (Maximal Marginal Relevance)."""

from __future__ import annotations

import numpy as np


def mmr_select(
    query_vector: np.ndarray,
    candidate_vectors: list[np.ndarray],
    candidate_results: list,
    lambda_param: float = 0.5,
    top_k: int = 5,
) -> list:
    """Selectionne des resultats diversifies."""
    selected = []
    remaining = list(range(len(candidate_vectors)))

    # Premier : le plus similaire a la requete
    similarities = [cosine_sim(query_vector, v) for v in candidate_vectors]
    first = int(np.argmax(similarities))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < top_k and remaining:
        mmr_scores = []
        for i in remaining:
            rel = cosine_sim(query_vector, candidate_vectors[i])
            div = max(cosine_sim(candidate_vectors[i], candidate_vectors[j]) for j in selected)
            mmr = lambda_param * rel - (1 - lambda_param) * div
            mmr_scores.append((i, mmr))

        best = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(best)
        remaining.remove(best)

    return [candidate_results[i] for i in selected]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))