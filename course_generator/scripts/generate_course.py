#!/usr/bin/env python3
"""CLI : génère un cours complet via Mistral + RAG."""

from __future__ import annotations

import sys
from pathlib import Path

# Détection automatique de la racine du projet
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from course_generator import CourseOrchestrator, CourseConfig


def main() -> int:
    print("=" * 60)
    print("COURSE GENERATOR - Mistral + RAG")
    print("=" * 60)
    print()

    config = CourseConfig()
    if not config.mistral_api_key:
        print("ERREUR : Définissez la variable d'environnement MISTRAL_API_KEY")
        print("  Windows : $env:MISTRAL_API_KEY='votre_cle'")
        return 1

    print(f"Modèle Mistral : {config.mistral_model}")
    print(f"Structure source : {config.hierarchy_summary_path}")
    print(f"Sortie : {config.output_dir}")
    print()

    orchestrator = CourseOrchestrator(config)
    orchestrator.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())