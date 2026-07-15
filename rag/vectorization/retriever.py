"""Recherche semantique et reconstruction du contexte."""

from __future__ import annotations

from .config import VectorConfig
from .embedder import BGEEmbedder
from .indexer import FAISSIndexer
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
    ) -> None:
        self.config = config or VectorConfig()
        self.embedder = embedder or BGEEmbedder(self.config)
        self.indexer = indexer or FAISSIndexer(self.config)
        self.store = store or MetadataStore(self.config.metadata_path)

    def load_index(self) -> None:
        self.indexer.load()
        self.store.load()

    def search(self, query: str, k: int | None = None) -> list[SearchResult]:
        query_vector = self.embedder.embed_query(query)
        distances, indices = self.indexer.search(query_vector, k)

        results = self.store.build_search_results(
            indices=indices[0].tolist(),
            distances=distances[0].tolist(),
        )
        return results

    def retrieve_context(self, query: str, k: int | None = None) -> RetrievedContext:
        results = self.search(query, k)

        # Regroupement par chemin hierarchique complet (source + hierarchy_path)
        sections_map: dict[str, SectionContext] = {}
        for r in results:
            path_str = "|".join(r.hierarchy_path) if r.hierarchy_path else ""
            key = f"{r.source_id}::{path_str}"

            if key not in sections_map:
                sections_map[key] = SectionContext(
                    hierarchy_key=key,
                    hierarchy_path=r.hierarchy_path,
                    source_id=r.source_id,
                    pages=[r.page_debut] if r.page_debut is not None else [],
                    texte_complet=r.text,
                    chunks_sources=[r.chunk_id],
                )
            else:
                sec = sections_map[key]
                if r.page_debut is not None and r.page_debut not in sec.pages:
                    sec.pages.append(r.page_debut)
                sec.texte_complet += "\n\n" + r.text
                sec.chunks_sources.append(r.chunk_id)

        # Conservation de l'ordre de pertinence FAISS (deduplique)
        seen = set()
        ordered_sections = []
        for r in results:
            path_str = "|".join(r.hierarchy_path) if r.hierarchy_path else ""
            key = f"{r.source_id}::{path_str}"
            if key not in seen:
                ordered_sections.append(sections_map[key])
                seen.add(key)

        # Respect de la limite de tokens
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
            header = f"[Source: {sec.source_id}]"
            if sec.hierarchy_path:
                path_str = " > ".join(sec.hierarchy_path)
                header += f"\nChemin: {path_str}"
            if sec.pages:
                header += f"\nPages: {sorted(sec.pages)}"
            parts.append(f"{header}\n\n{sec.texte_complet}")

        return "\n\n---\n\n".join(parts)