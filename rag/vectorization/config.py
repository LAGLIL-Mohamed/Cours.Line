"""Configuration du module de vectorisation."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class VectorConfig:
    """Parametres de vectorisation et recherche."""

    project_root: Path = Path(__file__).resolve().parent.parent

    # Mode legacy : un seul fichier chunks.jsonl
    chunks_input: Path = None

    # Mode multi-PDF : dossier contenant les sous-dossiers (un par PDF)
    # Chaque sous-dossier doit contenir un fichier chunks.jsonl
    chunks_input_dir: Path = None

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
        # Par defaut, on scanne le dossier data/output (sortie d'extract_hierarchy.py)
        if self.chunks_input is None and self.chunks_input_dir is None:
            self.chunks_input_dir = self.project_root / "data" / "output"

        if self.vector_store_dir is None:
            self.vector_store_dir = self.project_root / "data" / "vector_store"
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)

    @property
    def faiss_index_path(self) -> Path:
        return self.vector_store_dir / "faiss.index"

    @property
    def metadata_path(self) -> Path:
        return self.vector_store_dir / "metadata.jsonl"

    def discover_chunks_files(self) -> list[Path]:
        """Retourne la liste de tous les chunks.jsonl a indexer.

        Priorite :
        1. Si `chunks_input` est defini et existe -> fichier unique
        2. Sinon scanne `chunks_input_dir` pour trouver */chunks.jsonl
        """
        if self.chunks_input and self.chunks_input.exists():
            return [self.chunks_input]

        files = []
        if self.chunks_input_dir and self.chunks_input_dir.exists():
            files = sorted(self.chunks_input_dir.glob("*/chunks.jsonl"))
        return files