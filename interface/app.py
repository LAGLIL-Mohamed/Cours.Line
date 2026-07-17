#!/usr/bin/env python3
"""Interface web FastAPI pour le pipeline RAG.

Lancement :
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Variables d'environnement :
    RAG_ORIGINAL_PROJECT : chemin vers le projet RAG original
                           (defaut: C:/Users/dell/Desktop/Projet python - RAG)
    MISTRAL_API_KEY      : cle API Mistral (peut aussi etre passee dans config)

Modifications (correctifs front/back) :
    1. Etapes REELLES : le frontend envoie un session_id, puis interroge
       GET /api/progress/<session_id> toutes les 2 s. Cet endpoint lit le
       fichier progress.json ecrit par main_batch.py dans le workspace.
       Fini la simulation au setInterval qui cochait les etapes au hasard.
    2. Loading qui se termine : le formulaire est soumis via fetch() (plus
       de POST natif). Quand le ZIP arrive, on declenche le telechargement
       en JS (blob -> <a download>) puis on masque le spinner et on affiche
       un recapitulatif. En cas d'erreur, la banniere s'affiche et le
       bouton est reactive.
    3. TD visible : le pipeline est execute dans un thread (la boucle
       asyncio n'est plus gelee pendant 5-10 min), les warnings du pipeline
       (ex: TD non genere apres retry) sont renvoyes dans l'en-tete
       X-RAG-Report et affiches a l'utilisateur.
    4. FRONTEND SEPARE : le HTML est dans templates/index.html, lu
       directement sans Jinja2 (evite les bugs de version).
"""

import os
import re
import json
import tempfile
import zipfile
import io
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional
from urllib.parse import quote

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from workspace_manager import WorkspaceManager
from fastapi.staticfiles import StaticFiles

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

MAX_FILE_SIZE_MB = 50  # Taille max par PDF
MAX_TOTAL_SIZE_MB = 200  # Taille totale max
MAX_FILES = 10  # Nombre max de PDFs

# session_id : uuid genere par le frontend (lettres, chiffres, - et _)
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _sanitize_session_id(value: Optional[str]) -> Optional[str]:
    """Valide le session_id recu du client (protection anti path-traversal)."""
    if value and _SESSION_ID_RE.match(value):
        return value
    return None


def _workspaces_dir() -> Path:
    """Doit correspondre a WorkspaceManager.workspaces_dir."""
    return ORIGINAL_PROJECT_ROOT.parent / "web_sessions"


# ---------------------------------------------------------------------
# TEMPLATES (frontend separe dans templates/index.html)
# ---------------------------------------------------------------------
HTML_PATH = Path(__file__).parent / "templates" / "index.html"


def _read_html() -> str:
    """Lit le fichier HTML du frontend."""
    if HTML_PATH.exists():
        return HTML_PATH.read_text(encoding="utf-8")
    # Fallback si le fichier n'existe pas
    return "<h1>Erreur : templates/index.html introuvable</h1>"


# ---------------------------------------------------------------------
# LIFESPAN (startup / shutdown)
# ---------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("=" * 70)
    print("🚀 Demarrage du serveur RAG Web")
    print("=" * 70)

    if not ORIGINAL_PROJECT_ROOT.exists():
        print(f"⚠️  ATTENTION : Projet original introuvable : {ORIGINAL_PROJECT_ROOT}")
        print("     Definissez RAG_ORIGINAL_PROJECT pour pointer vers le bon dossier.")
        print("     Exemple : set RAG_ORIGINAL_PROJECT=C:/Users/.../Projet python - RAG")
    else:
        print(f"✅ Projet original detecte : {ORIGINAL_PROJECT_ROOT}")

        # Verifier que main_batch.py existe dans le projet original
        if not (ORIGINAL_PROJECT_ROOT / "main_batch.py").exists():
            print("⚠️  ATTENTION : main_batch.py n'est pas dans le projet original.")
            print("     Copiez main_batch.py a la racine du projet avant de continuer.")
        else:
            print("✅ main_batch.py detecte dans le projet original")

    if not os.environ.get("MISTRAL_API_KEY"):
        print("⚠️  ATTENTION : MISTRAL_API_KEY non definie.")
        print("     Le pipeline echouera sans cle API.")
    else:
        print("✅ Cle API Mistral detectee")

    print("=" * 70)
    yield
    # Shutdown
    print("\n🛑 Arret du serveur")


app = FastAPI(
    title="RAG Pipeline Web",
    description="Generation de cours structures par RAG a partir de PDFs",
    version="1.0.0",
    lifespan=lifespan
)
from pathlib import Path
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

# ---------------------------------------------------------------------
# PAGE D'ACCUEIL (HTML lu depuis fichier, pas de Jinja2)
# ---------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    """Page d'accueil avec le formulaire de generation."""
    return _read_html()


# ---------------------------------------------------------------------
# ENDPOINT DE SUIVI DE PROGRESSION
# ---------------------------------------------------------------------
@app.get("/api/progress/{session_id}")
async def get_progress(session_id: str):
    """Retourne l'etat d'avancement reel du pipeline pour une session.

    Lit <web_sessions>/session_<id>/progress.json, ecrit par main_batch.py.
    Pendant les premieres secondes (copie du projet), le fichier n'existe
    pas encore -> on renvoie un etat 'starting'.
    """
    safe_id = _sanitize_session_id(session_id)
    if not safe_id:
        raise HTTPException(status_code=400, detail="session_id invalide.")

    progress_file = _workspaces_dir() / f"session_{safe_id}" / "progress.json"

    if not progress_file.exists():
        return {"status": "starting", "steps": {}}

    try:
        return json.loads(progress_file.read_text(encoding="utf-8"))
    except Exception:
        # Fichier en cours d'ecriture : le client reessaiera au prochain tick
        return {"status": "running", "steps": {}}


# ---------------------------------------------------------------------
# ENDPOINT DE GENERATION
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
):
    """
    Endpoint principal : recoit les PDFs et les parametres, lance le pipeline,
    et retourne un ZIP avec tous les fichiers generes.

    L'en-tete X-RAG-Report (JSON URL-encode) resume la generation :
    fichiers produits, TD demande/produit, avertissements.
    """
    # Verifications preliminaires
    if not ORIGINAL_PROJECT_ROOT.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                f"Projet original introuvable : {ORIGINAL_PROJECT_ROOT}. "
                f"Definissez la variable d'environnement RAG_ORIGINAL_PROJECT."
            )
        )

    if not (ORIGINAL_PROJECT_ROOT / "main_batch.py").exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "main_batch.py n'est pas dans le projet original. "
                "Copiez-le a la racine du projet avant de lancer le serveur."
            )
        )

    if not pdfs or all(not f.filename for f in pdfs):
        raise HTTPException(status_code=400, detail="Aucun fichier PDF fourni.")

    if len(pdfs) > MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_FILES} fichiers PDF autorises."
        )

    # Validation des fichiers
    total_size = 0
    for pdf in pdfs:
        if pdf.filename and not pdf.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Fichier non-PDF rejete : {pdf.filename}"
            )

    # session_id valide (sinon None -> le manager en generera un)
    safe_session_id = _sanitize_session_id(session_id)

    # Sauvegarde temporaire des PDFs uploades
    temp_dir = Path(tempfile.mkdtemp(prefix="rag_upload_"))
    pdf_paths = []

    try:
        for pdf in pdfs:
            if not pdf.filename:
                continue

            pdf_path = temp_dir / pdf.filename
            content = await pdf.read()

            if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail=f"Le fichier '{pdf.filename}' depasse {MAX_FILE_SIZE_MB} Mo."
                )

            total_size += len(content)
            if total_size > MAX_TOTAL_SIZE_MB * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail=f"La taille totale depasse {MAX_TOTAL_SIZE_MB} Mo."
                )

            with open(pdf_path, "wb") as f:
                f.write(content)
            pdf_paths.append(pdf_path)

        if not pdf_paths:
            raise HTTPException(status_code=400, detail="Aucun fichier PDF valide fourni.")

        # Configuration du pipeline
        config = {
            "max_pages": pages,
            "num_chapters": chapters,
            "chapters": [],  # Auto-generation des titres par Mistral
            "skip_extraction": skip_extraction,
            "skip_summary": skip_summary,
            "skip_indexing": skip_indexing,
            "interactive_plan": False,  # Toujours mode batch en web
            "generate_td": generate_td,
            "mistral_api_key": os.environ.get("MISTRAL_API_KEY"),
        }

        # Lancement du workspace manager DANS UN THREAD :
        # run_session est bloquant (plusieurs minutes) ; l'appeler
        # directement gelerait toute la boucle asyncio, y compris
        # l'endpoint /api/progress utilise pour le suivi en temps reel.
        manager = WorkspaceManager(ORIGINAL_PROJECT_ROOT)
        try:
            result = await run_in_threadpool(
                manager.run_session,
                pdf_paths,
                config,
                keep_workspace=False,
                session_id=safe_session_id,
            )
        except Exception as e:
            # Remonter un message exploitable au frontend (stderr tronque)
            detail = str(e)
            if len(detail) > 1500:
                detail = "..." + detail[-1500:]
            raise HTTPException(
                status_code=500,
                detail=f"Le pipeline a echoue. Detail :\n{detail}"
            )

        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail="Le pipeline a echoue. Consultez les logs du serveur."
            )

        # Creation du ZIP de resultats
        files_data = result.get("files", {})
        if not files_data:
            raise HTTPException(
                status_code=500,
                detail="Aucun fichier genere par le pipeline."
            )

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename, file_bytes in files_data.items():
                if file_bytes:
                    zf.writestr(filename, file_bytes)

        zip_buffer.seek(0)

        # Rapport de generation pour le frontend
        warnings = list(result.get("warnings", []))
        files_list = sorted(files_data.keys())
        td_requested = bool(generate_td)
        td_generated = "td.md" in files_data
        if td_requested and not td_generated and not any("TD" in w for w in warnings):
            warnings.append(
                "Le TD demande n'a pas pu etre genere. "
                "Le cours complet reste disponible dans le ZIP."
            )

        report = {
            "session_id": result.get("session_id"),
            "files": files_list,
            "td_requested": td_requested,
            "td_generated": td_generated,
            "warnings": warnings,
        }

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
        # Nettoyage des fichiers temporaires uploades
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)


# ---------------------------------------------------------------------
# ENDPOINTS DE SANTE ET INFO
# ---------------------------------------------------------------------
@app.get("/health")
async def health():
    """Verifie l'etat du serveur et la disponibilite du projet."""
    project_ok = ORIGINAL_PROJECT_ROOT.exists()
    batch_ok = (ORIGINAL_PROJECT_ROOT / "main_batch.py").exists() if project_ok else False
    api_key_ok = bool(os.environ.get("MISTRAL_API_KEY"))

    status = "ok" if (project_ok and batch_ok and api_key_ok) else "degraded"

    return {
        "status": status,
        "project_root": str(ORIGINAL_PROJECT_ROOT),
        "project_exists": project_ok,
        "main_batch_exists": batch_ok,
        "api_key_configured": api_key_ok,
    }


@app.get("/api/config")
async def get_config():
    """Retourne la configuration actuelle (sans la cle API)."""
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
    """Gestionnaire d'erreurs generique."""
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "type": type(exc).__name__,
        }
    )


# ---------------------------------------------------------------------
# LANCEMENT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)