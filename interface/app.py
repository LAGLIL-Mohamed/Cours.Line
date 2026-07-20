#!/usr/bin/env python3
"""Interface web FastAPI pour le pipeline RAG.

Lancement :
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Variables d'environnement :
    RAG_ORIGINAL_PROJECT : chemin vers le projet RAG original
                           (defaut: C:/Users/dell/Desktop/Projet python - RAG)
    MISTRAL_API_KEY      : cle API Mistral (peut aussi etre passee dans config)

Modifications (integration PDF) :
    - Conversion PDF effectuee dans le process principal APRES le pipeline
      (pandoc/xelatex ne tournent PAS dans le thread isole).
    - keep_workspace=True pendant la conversion et la creation du ZIP.
    - Nettoyage du workspace via BackgroundTasks (apres envoi du ZIP).
    - Le ZIP ne contient PLUS les .md/.json : uniquement les PDFs uploades
      + course.pdf + td.pdf.
    - Import dynamique de md_to_academic_pdf.py depuis le projet original.
"""

import os
import re
import json
import sys
import shutil
import tempfile
import zipfile
import io
import importlib.util
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional
from urllib.parse import quote

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from workspace_manager import WorkspaceManager

# ---------------------------------------------------------------------
# CHARGEMENT DU .ENV
# ---------------------------------------------------------------------
def _load_env_file(env_path: Path):
    """Charge un fichier .env dans os.environ."""
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key] = value


# Charger le .env du projet original si present
_env_path = Path(r"C:/Users/dell/Desktop/Projet python - RAG") / ".env"
_load_env_file(_env_path)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
ORIGINAL_PROJECT_ROOT = Path(os.environ.get(
    "RAG_ORIGINAL_PROJECT",
    r"C:/Users/dell/Desktop/Projet python - RAG"
))

MAX_FILE_SIZE_MB = 50
MAX_TOTAL_SIZE_MB = 200
MAX_FILES = 10

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _sanitize_session_id(value: Optional[str]) -> Optional[str]:
    if value and _SESSION_ID_RE.match(value):
        return value
    return None


def _workspaces_dir() -> Path:
    return ORIGINAL_PROJECT_ROOT.parent / "web_sessions"


# ---------------------------------------------------------------------
# CHARGEMENT DU CONVERTISSEUR PDF (depuis le projet original)
# ---------------------------------------------------------------------
def _load_pdf_converter():
    """Charge dynamiquement md_to_academic_pdf.py depuis le projet original."""
    candidates = [
        ORIGINAL_PROJECT_ROOT / "MdToPdf" / "md_to_academic_pdf.py",
        ORIGINAL_PROJECT_ROOT / "md_to_academic_pdf.py",
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("md_to_academic_pdf", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["md_to_academic_pdf"] = mod
            spec.loader.exec_module(mod)
            return mod.convert_md_to_pdf
    return None


convert_md_to_pdf = _load_pdf_converter()


# ---------------------------------------------------------------------
# TEMPLATES
# ---------------------------------------------------------------------
HTML_PATH = Path(__file__).parent / "templates" / "index.html"


def _read_html() -> str:
    if HTML_PATH.exists():
        return HTML_PATH.read_text(encoding="utf-8")
    return "<h1>Erreur : templates/index.html introuvable</h1>"


# ---------------------------------------------------------------------
# LIFESPAN
# ---------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 70)
    print("🚀 Demarrage du serveur RAG Web")
    print("=" * 70)

    if not ORIGINAL_PROJECT_ROOT.exists():
        print(f"⚠️  Projet original introuvable : {ORIGINAL_PROJECT_ROOT}")
    else:
        print(f"✅ Projet original detecte : {ORIGINAL_PROJECT_ROOT}")
        if not (ORIGINAL_PROJECT_ROOT / "main_batch.py").exists():
            print("⚠️  main_batch.py manquant.")
        else:
            print("✅ main_batch.py detecte")

    if convert_md_to_pdf:
        print("✅ Convertisseur PDF charge")
    else:
        print("⚠️  Convertisseur PDF introuvable (cherché dans MdToPdf/ et racine)")

    if not os.environ.get("MISTRAL_API_KEY"):
        print("⚠️  MISTRAL_API_KEY non definie.")
    else:
        print("✅ Cle API Mistral detectee")

    print("=" * 70)
    yield
    print("\n🛑 Arret du serveur")


app = FastAPI(
    title="RAG Pipeline Web",
    description="Generation de cours structures par RAG a partir de PDFs",
    version="1.1.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


# ---------------------------------------------------------------------
# PAGE D'ACCUEIL
# ---------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return _read_html()


# ---------------------------------------------------------------------
# SUIVI DE PROGRESSION
# ---------------------------------------------------------------------
@app.get("/api/progress/{session_id}")
async def get_progress(session_id: str):
    safe_id = _sanitize_session_id(session_id)
    if not safe_id:
        raise HTTPException(status_code=400, detail="session_id invalide.")

    progress_file = _workspaces_dir() / f"session_{safe_id}" / "progress.json"

    if not progress_file.exists():
        return {"status": "starting", "steps": {}}

    try:
        return json.loads(progress_file.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "running", "steps": {}}


# ---------------------------------------------------------------------
# GENERATION (avec conversion PDF integree)
# ---------------------------------------------------------------------
@app.post("/generate")
async def generate(
    pdfs: List[UploadFile] = File(...),
    pages: int = Form(10),
    chapters: int = Form(3),
    generate_td: bool = Form(True),
    skip_extraction: bool = Form(False),
    skip_summary: bool = Form(False),
    skip_indexing: bool = Form(False),
    session_id: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = ...,
):
    """
    Endpoint principal : recoit les PDFs, lance le pipeline,
    convertit les .md en PDF, et retourne un ZIP filtre.
    """
    if not ORIGINAL_PROJECT_ROOT.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Projet original introuvable : {ORIGINAL_PROJECT_ROOT}"
        )

    if not (ORIGINAL_PROJECT_ROOT / "main_batch.py").exists():
        raise HTTPException(
            status_code=500,
            detail="main_batch.py n'est pas dans le projet original."
        )

    if not pdfs or all(not f.filename for f in pdfs):
        raise HTTPException(status_code=400, detail="Aucun fichier PDF fourni.")

    if len(pdfs) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_FILES} fichiers PDF.")

    total_size = 0
    for pdf in pdfs:
        if pdf.filename and not pdf.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Fichier non-PDF : {pdf.filename}")

    safe_session_id = _sanitize_session_id(session_id)

    temp_dir = Path(tempfile.mkdtemp(prefix="rag_upload_"))
    pdf_paths = []

    try:
        # --- Sauvegarde des PDFs uploades ---
        for pdf in pdfs:
            if not pdf.filename:
                continue
            pdf_path = temp_dir / pdf.filename
            content = await pdf.read()

            if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail=f"'{pdf.filename}' depasse {MAX_FILE_SIZE_MB} Mo."
                )

            total_size += len(content)
            if total_size > MAX_TOTAL_SIZE_MB * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail=f"Taille totale depasse {MAX_TOTAL_SIZE_MB} Mo."
                )

            with open(pdf_path, "wb") as f:
                f.write(content)
            pdf_paths.append(pdf_path)

        if not pdf_paths:
            raise HTTPException(status_code=400, detail="Aucun PDF valide.")

        # --- Configuration pipeline ---
        config = {
            "max_pages": pages,
            "num_chapters": chapters,
            "chapters": [],
            "skip_extraction": skip_extraction,
            "skip_summary": skip_summary,
            "skip_indexing": skip_indexing,
            "interactive_plan": False,
            "generate_td": generate_td,
            "mistral_api_key": os.environ.get("MISTRAL_API_KEY"),
        }

        # --- Execution du pipeline RAG (thread isole) ---
        manager = WorkspaceManager(ORIGINAL_PROJECT_ROOT)
        try:
            result = await run_in_threadpool(
                manager.run_session,
                pdf_paths,
                config,
                keep_workspace=True,          # CRITIQUE : garder pour conversion PDF
                session_id=safe_session_id,
            )
        except Exception as e:
            detail = str(e)
            if len(detail) > 1500:
                detail = "..." + detail[-1500:]
            raise HTTPException(status_code=500, detail=f"Pipeline echoue :\n{detail}")

        if not result.get("success"):
            raise HTTPException(status_code=500, detail="Le pipeline a echoue.")

        # --- Conversion PDF (process principal) ---
        workspace_dir = _workspaces_dir() / result["session_id"]
        output_dir = workspace_dir / "rag" / "data" / "output"
        search_dirs = [output_dir, output_dir / "course_generated"]

        course_md_path = None
        td_md_path = None
        for d in search_dirs:
            if not d.exists():
                continue
            if not course_md_path and (d / "course.md").exists():
                course_md_path = d / "course.md"
            if not td_md_path and (d / "td.md").exists():
                td_md_path = d / "td.md"

        warnings = list(result.get("warnings", []))
        generated_pdfs = []

        if convert_md_to_pdf:
            if course_md_path:
                course_pdf = temp_dir / "course.pdf"
                try:
                    convert_md_to_pdf(course_md_path, course_pdf)
                    generated_pdfs.append(("course.pdf", course_pdf))
                except Exception as e:
                    warnings.append(f"Conversion course.md -> PDF echouee : {e}")

            if td_md_path:
                td_pdf = temp_dir / "td.pdf"
                try:
                    convert_md_to_pdf(td_md_path, td_pdf)
                    generated_pdfs.append(("td.pdf", td_pdf))
                except Exception as e:
                    warnings.append(f"Conversion td.md -> PDF echouee : {e}")
        else:
            warnings.append("Convertisseur PDF introuvable.")

        # --- Creation du ZIP (filtre : pas de .md/.json) ---
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # PDFs sources uploades
            for pdf_path in pdf_paths:
                zf.write(pdf_path, arcname=pdf_path.name)

            # PDFs generes
            for arcname, pdf_path in generated_pdfs:
                zf.write(pdf_path, arcname=arcname)

        zip_buffer.seek(0)

        # --- Rapport ---
        td_requested = bool(generate_td)
        td_generated = "td.pdf" in [a for a, _ in generated_pdfs]

        if td_requested and not td_generated and not any("TD" in w for w in warnings):
            warnings.append(
                "Le TD demande n'a pas pu etre genere ou converti en PDF. "
                "Le cours complet reste disponible."
            )

        report = {
            "session_id": result.get("session_id"),
            "files": [p.name for p in pdf_paths] + [a for a, _ in generated_pdfs],
            "td_requested": td_requested,
            "td_generated": td_generated,
            "warnings": warnings,
        }

        # --- Nettoyage differe du workspace (apres envoi du ZIP) ---
        if workspace_dir.exists():
            background_tasks.add_task(shutil.rmtree, workspace_dir, ignore_errors=True)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition":
                    f"attachment; filename=cours_rag_{result['session_id']}.zip",
                "X-RAG-Report": quote(json.dumps(report, ensure_ascii=False)),
            }
        )

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


# ---------------------------------------------------------------------
# SANTE / INFO
# ---------------------------------------------------------------------
@app.get("/health")
async def health():
    project_ok = ORIGINAL_PROJECT_ROOT.exists()
    batch_ok = (ORIGINAL_PROJECT_ROOT / "main_batch.py").exists() if project_ok else False
    api_key_ok = bool(os.environ.get("MISTRAL_API_KEY"))
    pdf_ok = convert_md_to_pdf is not None

    status = "ok" if (project_ok and batch_ok and api_key_ok and pdf_ok) else "degraded"

    return {
        "status": status,
        "project_root": str(ORIGINAL_PROJECT_ROOT),
        "project_exists": project_ok,
        "main_batch_exists": batch_ok,
        "api_key_configured": api_key_ok,
        "pdf_converter_loaded": pdf_ok,
    }


@app.get("/api/config")
async def get_config():
    return {
        "project_root": str(ORIGINAL_PROJECT_ROOT),
        "max_file_size_mb": MAX_FILE_SIZE_MB,
        "max_total_size_mb": MAX_TOTAL_SIZE_MB,
        "max_files": MAX_FILES,
    }


# ---------------------------------------------------------------------
# GESTIONNAIRES D'ERREURS
# ---------------------------------------------------------------------
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__}
    )


# ---------------------------------------------------------------------
# LANCEMENT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)