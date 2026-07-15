"""Moteur RAG : recherche multi-questions via FAISS."""

import sys
from pathlib import Path

# Cherche 'vectorization' en remontant l'arborescence (max 5 niveaux)
_current = Path(__file__).resolve().parent
project_root = None

for _ in range(5):
    # Cas 1 : vectorization directement ici
    if (_current / "vectorization").exists() and (_current / "vectorization").is_dir():
        project_root = _current
        break
    # Cas 2 : vectorization dans un sous-dossier 'rag/'
    if (_current / "rag" / "vectorization").exists() and (_current / "rag" / "vectorization").is_dir():
        project_root = _current / "rag"
        break
    _current = _current.parent

if project_root is None:
    raise RuntimeError(
        "Dossier 'vectorization' introuvable. "
        "Assurez-vous que le pipeline RAG est installé."
    )

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from vectorization import ContextRetriever, VectorConfig


class RAGEngine:
    """Wrapper RAG autour du module vectorization existant."""

    def __init__(self, top_k: int = 5, max_context_tokens: int = 4000) -> None:
        self.config = VectorConfig()
        self.retriever = ContextRetriever(config=self.config)
        print("[RAG] Chargement de l'index FAISS...")
        self.retriever.load_index()
        print(f"[RAG] Index chargé : {self.retriever.indexer.n_vectors} vecteurs")
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens

    def gather_context(self, questions: list[str]) -> tuple[str, list[str]]:
        """Recherche multi-questions, déduplication et troncature."""
        all_results = []
        for q in questions:
            results = self.retriever.search(q, k=self.top_k)
            all_results.extend(results)

        # Déduplication par chunk_id
        seen = set()
        unique_results = []
        for r in all_results:
            if r.chunk_id not in seen:
                seen.add(r.chunk_id)
                unique_results.append(r)

        # Formatage et troncature
        context_parts = []
        current_tokens = 0
        sources = []

        for r in unique_results:
            header = f"[Source: {r.source_id} | Pages: {r.page_debut or '?'} - {r.page_fin or '?'}]"
            body = r.text.strip()
            part = f"{header}\n{body}\n"
            tokens = int(len(part.split()) * 1.33)

            if current_tokens + tokens > self.max_context_tokens and context_parts:
                break

            context_parts.append(part)
            current_tokens += tokens
            sources.append(r.chunk_id)

        full_context = "\n---\n".join(context_parts)
        return full_context, sources