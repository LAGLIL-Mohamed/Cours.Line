# fix_all.py - à placer dans C:\Users\dell\Desktop\Projet python - RAG\rag\
import os
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\dell\Desktop\Projet python - RAG\rag")

files_to_check = {
    "rag_pipeline/_internal/_utils.py": ["generate_id", "normalize_unicode"],
    "rag_pipeline/schemas.py": ["Chunk", "Document", "Section"],
    "rag_pipeline/config.py": ["PipelineConfig"],
    "rag_pipeline/processor.py": ["DocumentProcessor"],
    "rag_pipeline/__init__.py": ["DocumentProcessor", "PipelineConfig"],
    "rag_pipeline/steps/__init__.py": ["Step0Metadata", "Step1Extract"],
    "rag_pipeline/steps/step0_metadata.py": ["Step0Metadata"],
    "rag_pipeline/steps/step1_extract.py": ["Step1Extract"],
    "rag_pipeline/steps/step2_clean.py": ["Step2Clean"],
}

print("Vérification des fichiers...")
for rel_path, required in files_to_check.items():
    full_path = PROJECT_ROOT / rel_path
    if not full_path.exists():
        print(f"❌ MANQUANT: {rel_path}")
    else:
        content = full_path.read_text(encoding="utf-8")
        missing = [r for r in required if r not in content]
        if missing:
            print(f"⚠️  INCOMPLET: {rel_path} (manque: {', '.join(missing)})")
        else:
            print(f"✅ OK: {rel_path}")

print("\nSi des fichiers sont manquants ou incomplets, copie le code depuis la conversation.")