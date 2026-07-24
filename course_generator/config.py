"""Configuration du générateur de cours."""

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class CourseConfig:
    """Paramètres de génération de cours."""

    project_root: Path = Path(__file__).resolve().parent

    hierarchy_summary_path: Path = None
    vector_store_dir: Path = None
    output_dir: Path = None

    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")
    mistral_model: str = "mistral-large-latest"
    min_delay_seconds: float = 1.5
    max_retries: int = 3

    top_k_per_question: int = 5
    max_context_tokens: int = 4000

    def __post_init__(self) -> None:
        # Détecte si le pipeline est dans un sous-dossier 'rag/'
        base = self.project_root
        while base.parent != base and not (base / "rag").exists():
            base = base.parent
        if not (base / "rag").exists():
            base = self.project_root
        else:
            base = base / "rag"

        if self.hierarchy_summary_path is None:
            self.hierarchy_summary_path = base / "data" / "output" / "hierarchy_summary.json"
        if self.vector_store_dir is None:
            self.vector_store_dir = base / "data" / "vector_store"
        if self.output_dir is None:
            self.output_dir = base / "data" / "output" / "course_generated"
        if self.mistral_api_key is None:
            self.mistral_api_key = os.getenv("MISTRAL_API_KEY")
        self.output_dir.mkdir(parents=True, exist_ok=True)