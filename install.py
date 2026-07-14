"""Script d'installation : pipeline RAG avec extraction hierarchique robuste."""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\dell\Desktop\Projet python - RAG\rag")
SOURCE_DIR = Path(__file__).parent / "pipeline_rag_files"


def copy_file(src_name: str, dest_rel_path: str):
    """Copie un fichier source vers la destination relative dans PROJECT_ROOT."""
    src = SOURCE_DIR / src_name
    dest = PROJECT_ROOT / dest_rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"  ✅ {dest_rel_path}")


def install():
    print("=" * 60)
    print("Installation du pipeline RAG (version hierarchique robuste)")
    print("=" * 60)
    print()

    if not SOURCE_DIR.exists():
        print(f"ERREUR: Dossier source non trouve: {SOURCE_DIR}")
        print("Assure-toi que le dossier 'pipeline_rag_files' est au meme endroit que ce script.")
        return

    # Structure des fichiers
    files_to_copy = [
        ("rag_pipeline___init__.py", "rag_pipeline/__init__.py"),
        ("config.py", "rag_pipeline/config.py"),
        ("schemas.py", "rag_pipeline/schemas.py"),
        ("processor.py", "rag_pipeline/processor.py"),
        ("_utils.py", "rag_pipeline/_internal/_utils.py"),
        ("_text_analysis.py", "rag_pipeline/_internal/_text_analysis.py"),
        ("_heading_detector.py", "rag_pipeline/_internal/_heading_detector.py"),
        ("steps___init__.py", "rag_pipeline/steps/__init__.py"),
        ("step0_metadata.py", "rag_pipeline/steps/step0_metadata.py"),
        ("step1_extract.py", "rag_pipeline/steps/step1_extract.py"),
        ("step2_clean.py", "rag_pipeline/steps/step2_clean.py"),
        ("step3_filter.py", "rag_pipeline/steps/step3_filter.py"),
        ("step4_structure.py", "rag_pipeline/steps/step4_structure.py"),
        ("step5_chunks.py", "rag_pipeline/steps/step5_chunks.py"),
        ("step6_quality.py", "rag_pipeline/steps/step6_quality.py"),
        ("run_pipeline.py", "rag_pipeline/scripts/run_pipeline.py"),
        ("inspect_chunks.py", "rag_pipeline/scripts/inspect_chunks.py"),
    ]

    for src_name, dest_path in files_to_copy:
        copy_file(src_name, dest_path)

    # __init__.py vides pour les sous-packages
    for init_path in [
        "rag_pipeline/_internal/__init__.py",
        "rag_pipeline/tests/__init__.py",
        "rag_pipeline/tests/fixtures/__init__.py",
    ]:
        full_path = PROJECT_ROOT / init_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        if not full_path.exists():
            with open(full_path, "w", encoding="utf-8") as f:
                f.write("")
            print(f"  ✅ {init_path}")

    print()
    print("=" * 60)
    print("Installation terminee !")
    print("=" * 60)
    print()
    print("🆕 Ameliorations apportees :")
    print("   • Detection multi-strategie (TOC native + visuelle + style)")
    print("   • Clustering KMeans pour niveaux hierarchiques")
    print("   • Arbre Sections/Sous-sections reconstruit")
    print("   • Chunks enrichis : chapter_title, hierarchy_path, hierarchy_level")
    print("   • JSONL avec contexte hierarchique complet pour le LLM")
    print()
    print("Prochaines etapes :")
    print(f"   1. Creer le dossier : {PROJECT_ROOT / 'data' / 'input'}")
    print(f"   2. Y placer tes fichiers PDF")
    print(f"   3. Lancer : python {PROJECT_ROOT / 'rag_pipeline' / 'scripts' / 'run_pipeline.py'}")
    print()
    print("Pour inspecter les resultats :")
    print(f"   python {PROJECT_ROOT / 'rag_pipeline' / 'scripts' / 'inspect_chunks.py'}")


if __name__ == "__main__":
    install()
