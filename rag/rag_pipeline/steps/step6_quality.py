"""Étape 6 : Contrôle qualité des chunks."""

from __future__ import annotations

from ..schemas import Chunk
from .._internal._text_analysis import compute_quality_metrics
from .._internal._utils import compute_text_hash
from ..config import PipelineConfig


class Step6Quality:
    """Filtre et note les chunks selon les critères de qualité."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def run(self, chunks: list[Chunk]) -> list[Chunk]:
        seen_hashes: set[str] = set()
        for chunk in chunks:
            self._evaluate_chunk(chunk, seen_hashes)
            if not chunk.rejected and self.config.dedup_exact:
                seen_hashes.add(compute_text_hash(chunk.texte_chunk))
        return chunks

    def _evaluate_chunk(self, chunk: Chunk, seen_hashes: set[str]) -> None:
        text = chunk.texte_chunk

        if not text or not text.strip():
            chunk.rejected = True
            chunk.rejection_reason = "Chunk vide"
            return

        if len(text) < self.config.chunk_min_chars:
            chunk.rejected = True
            chunk.rejection_reason = f"Trop court ({len(text)} < {self.config.chunk_min_chars})"
            return

        if len(text) > self.config.chunk_max_chars:
            chunk.rejected = True
            chunk.rejection_reason = f"Trop long ({len(text)} > {self.config.chunk_max_chars})"
            return

        metrics = compute_quality_metrics(text)
        if not metrics.is_meaningful:
            chunk.rejected = True
            chunk.rejection_reason = metrics.rejection_reason
            return

        if self.config.dedup_exact:
            text_hash = compute_text_hash(text)
            if text_hash in seen_hashes:
                chunk.rejected = True
                chunk.rejection_reason = "Doublon exact"
                return

        if metrics.word_count < self.config.min_meaningful_words:
            chunk.rejected = True
            chunk.rejection_reason = f"Trop peu de mots ({metrics.word_count})"
            return

        chunk.quality_score = self._compute_score(metrics)
        chunk.rejection_reason = None

    def _compute_score(self, metrics) -> float:
        scores = [
            min(metrics.unique_word_ratio / 0.5, 1.0),
            min(metrics.alpha_ratio / 0.8, 1.0),
            min(metrics.word_count / 50, 1.0),
            1.0 - metrics.digit_ratio,
        ]
        return sum(scores) / len(scores)