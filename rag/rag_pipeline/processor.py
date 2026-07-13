"""Orchestrateur principal du pipeline RAG."""

from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .schemas import Chunk, Document
from .steps import (
    Step0Metadata,
    Step1Extract,
    Step2Clean,
    Step3Filter,
    Step4Structure,
    Step5Chunks,
    Step6Quality,
)


class DocumentProcessor:
    """Enchaîne les 7 étapes de préparation des données PDF."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._init_steps()

    def _init_steps(self) -> None:
        """Instancie chaque étape avec la config partagée."""
        self.step0 = Step0Metadata()
        self.step1 = Step1Extract()
        self.step2 = Step2Clean()
        self.step3 = Step3Filter(self.config)
        self.step4 = Step4Structure(self.config)
        self.step5 = Step5Chunks(self.config)
        self.step6 = Step6Quality(self.config)

    def process(self, pdf_path: Path | str) -> list[Chunk]:
        """Exécute le pipeline complet sur un PDF."""
        path = Path(pdf_path)

        doc: Document = self.step0.run(path)
        pages = self.step1.run(doc)
        pages = self.step2.run(pages)
        pages = self.step3.run(pages)
        sections = self.step4.run(pages)
        chunks = self.step5.run(sections)
        chunks = self.step6.run(chunks)

        return chunks

    def process_valid_only(self, pdf_path: Path | str) -> list[Chunk]:
        """Retourne uniquement les chunks acceptés (non rejetés)."""
        all_chunks = self.process(pdf_path)
        return [c for c in all_chunks if not c.rejected]
