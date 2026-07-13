"""Configuration du module de vectorisation."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class VectorConfig:
    """Parametres de vectorisation et recherche."""

    project_root: Path = Path(r"C:\\Users\\dell\\Desktop\\Projet python - RAG\\rag")
    chunks_input: Path = None
    vector_store_dir: Path = None

    model_name: str = "BAAI/bge-m3"
    device: str = "cpu"
    normalize_embeddings: bool = True

    index_type: str = "flat_ip"
    nlist: int = 100

    top_k: int = 5
    max_context_tokens: int = 4000
    dedup_sections: bool = True

    def __post_init__(self) -> None:
        if self.chunks_input is None:
            self.chunks_input = self.project_root / "data" / "output" / "chunks.jsonl"
        if self.vector_store_dir is None:
            self.vector_store_dir = self.project_root / "data" / "vector_store"
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)

    @property
    def faiss_index_path(self) -> Path:
        return self.vector_store_dir / "faiss.index"

    @property
    def metadata_path(self) -> Path:
        return self.vector_store_dir / "metadata.jsonl"
