"""Étape 0 : Métadonnées du document."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ..schemas import Document
from .._internal._utils import generate_id


class Step0Metadata:
    """Génère les métadonnées initiales du document."""

    def run(self, pdf_path: Path) -> Document:
        path = Path(pdf_path).resolve()
        return Document(
            source_id=generate_id("doc"),
            nom_fichier=path.name,
            chemin_origine=str(path),
            date_ingestion=datetime.now(timezone.utc),
        )
