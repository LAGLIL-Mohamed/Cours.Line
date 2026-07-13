"""Recherche semantique et reconstruction du contexte."""

from __future__ import annotations

from .config import VectorConfig
from .embedder import BGEEmbedder
from .indexer import FAISSIndexer
from .reranker import CrossEncoderReranker
from .schemas import RetrievedContext, SearchResult, SectionContext
from .store import MetadataStore
from ._internal._utils import approximate_token_count


class ContextRetriever:
    """Orchestre la recherche et la reconstruction du contexte."""

    def __init__(
        self,
        embedder: BGEEmbedder | None = None,
        indexer: FAISSIndexer | None = None,
        store: MetadataStore | None = None,
        config: VectorConfig | None = None,
        use_reranker: bool = False,
    ) -> None:
        self.config = config or VectorConfig()
        self.embedder = embedder or BGEEmbedder(self.config)
        self.indexer = indexer or FAISSIndexer(self.config)
        self.store = store or MetadataStore(self.config.metadata_path)
        self.reranker = CrossEncoderReranker() if use_reranker else None

    def load_index(self) -> None:
        self.indexer.load()
        self.store.load()

    def search(
        self,
        query: str,
        k: int | None = None,
        use_reranker: bool = False,
    ) -> list[SearchResult]:
        """Recherche avec re-ranking optionnel."""
        query_vector = self.embedder.embed_query(query)
        distances, indices = self.indexer.search(query_vector, k)

        results = self.store.build_search_results(
            indices=indices[0].tolist(),
            distances=distances[0].tolist(),
        )

        # Re-ranking si demande et disponible
        if use_reranker and self.reranker:
            results = self.reranker.rerank(query, results, top_k=k or self.config.top_k)

        return results

    def retrieve_context(
        self,
        query: str,
        k: int | None = None,
        use_reranker: bool = False,
    ) -> RetrievedContext:
        """Reconstruction du contexte avec re-ranking optionnel."""
        results = self.search(query, k=k, use_reranker=use_reranker)

        # ... reste identique (sections, tokens, etc.)
        sections_map: dict[str, SectionContext] = {}
        for r in results:
            if r.section_id not in sections_map:
                sections_map[r.section_id] = SectionContext(
                    section_id=r.section_id,
                    titre_section=r.titre_section,
                    source_id=r.source_id,
                    pages=[r.page],
                    texte_complet=r.texte_section_complete,
                    chunks_sources=[r.chunk_id],
                )
            else:
                sec = sections_map[r.section_id]
                if r.page not in sec.pages:
                    sec.pages.append(r.page)
                sec.chunks_sources.append(r.chunk_id)

        seen = set()
        ordered_sections = []
        for r in results:
            if r.section_id not in seen:
                ordered_sections.append(sections_map[r.section_id])
                seen.add(r.section_id)

        total_tokens = 0
        final_sections = []
        for sec in ordered_sections:
            sec_tokens = approximate_token_count(sec.texte_complet)
            if total_tokens + sec_tokens > self.config.max_context_tokens and final_sections:
                break
            total_tokens += sec_tokens
            final_sections.append(sec)

        return RetrievedContext(
            query=query,
            sections=final_sections,
            total_tokens_estimate=total_tokens,
        )

    def format_context_for_llm(self, context: RetrievedContext) -> str:
        parts = []
        for sec in context.sections:
            header = f"[Source: {sec.source_id}, Pages: {sorted(sec.pages)}]"
            if sec.titre_section and sec.titre_section != "(sans titre)":
                header += f"\nSection: {sec.titre_section}"
            parts.append(f"{header}\n\n{sec.texte_complet}")

        return "\n\n---\n\n".join(parts)