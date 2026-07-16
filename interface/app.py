#!/usr/bin/env python3
"""Interface web FastAPI pour le pipeline RAG.

Lancement :
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Variables d'environnement :
    RAG_ORIGINAL_PROJECT : chemin vers le projet RAG original
                           (defaut: C:/Users/dell/Desktop/Projet python - RAG)
    MISTRAL_API_KEY      : cle API Mistral (peut aussi etre passee dans config)
"""

import os
import json
import tempfile
import zipfile
import io
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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

MAX_FILE_SIZE_MB = 50  # Taille max par PDF
MAX_TOTAL_SIZE_MB = 200  # Taille totale max
MAX_FILES = 10  # Nombre max de PDFs


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


# ---------------------------------------------------------------------
# PAGE D'ACCUEIL (formulaire HTML)
# ---------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    """Page d'accueil avec le formulaire de generation."""
    return """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pipeline RAG — Generateur de Cours</title>
        <style>
            :root {
                --primary: #667eea;
                --primary-dark: #5a67d8;
                --secondary: #764ba2;
                --success: #48bb78;
                --danger: #f56565;
                --warning: #ed8936;
                --bg: #f7fafc;
                --card: #ffffff;
                --text: #2d3748;
                --text-light: #718096;
                --border: #e2e8f0;
            }

            * { box-sizing: border-box; margin: 0; padding: 0; }

            body {
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
                color: var(--text);
            }

            .container {
                background: var(--card);
                border-radius: 24px;
                padding: 48px;
                max-width: 640px;
                width: 100%;
                box-shadow: 0 25px 80px rgba(0,0,0,0.25);
                animation: slideUp 0.6s ease-out;
            }

            @keyframes slideUp {
                from { opacity: 0; transform: translateY(30px); }
                to { opacity: 1; transform: translateY(0); }
            }

            h1 {
                font-size: 2rem;
                margin-bottom: 8px;
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }

            .subtitle {
                color: var(--text-light);
                margin-bottom: 32px;
                font-size: 1rem;
                line-height: 1.5;
            }

            .info-box {
                background: linear-gradient(135deg, #ebf4ff 0%, #e6fffa 100%);
                border-left: 4px solid var(--primary);
                padding: 16px 20px;
                border-radius: 12px;
                margin-bottom: 28px;
                font-size: 0.9rem;
                color: var(--text);
            }

            .info-box strong { color: var(--primary-dark); }

            .form-group {
                margin-bottom: 24px;
            }

            label {
                display: block;
                margin-bottom: 10px;
                color: var(--text);
                font-weight: 600;
                font-size: 0.95rem;
            }

            .label-hint {
                font-weight: 400;
                color: var(--text-light);
                font-size: 0.85rem;
                margin-left: 6px;
            }

            input[type="number"], input[type="file"] {
                width: 100%;
                padding: 14px 18px;
                border: 2px solid var(--border);
                border-radius: 12px;
                font-size: 1rem;
                transition: all 0.3s;
                background: var(--bg);
            }

            input[type="number"]:focus {
                outline: none;
                border-color: var(--primary);
                background: white;
                box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
            }

            input[type="file"] {
                padding: 12px;
                cursor: pointer;
            }

            input[type="file"]:hover {
                border-color: var(--primary);
            }

            .file-hint {
                font-size: 0.8rem;
                color: var(--text-light);
                margin-top: 8px;
            }

            .checkbox-group {
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 16px;
                background: var(--bg);
                border-radius: 12px;
                margin-bottom: 20px;
                cursor: pointer;
                transition: background 0.2s;
            }

            .checkbox-group:hover {
                background: #edf2f7;
            }

            input[type="checkbox"] {
                width: 22px;
                height: 22px;
                accent-color: var(--primary);
                cursor: pointer;
            }

            .checkbox-group label {
                margin: 0;
                cursor: pointer;
                flex: 1;
            }

            button {
                width: 100%;
                padding: 18px;
                background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                color: white;
                border: none;
                border-radius: 14px;
                font-size: 1.15rem;
                font-weight: 700;
                cursor: pointer;
                transition: all 0.3s;
                position: relative;
                overflow: hidden;
            }

            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
            }

            button:active {
                transform: translateY(0);
            }

            button:disabled {
                opacity: 0.7;
                cursor: not-allowed;
                transform: none;
            }

            .loading {
                display: none;
                text-align: center;
                margin-top: 28px;
                padding: 24px;
                background: var(--bg);
                border-radius: 16px;
            }

            .spinner {
                width: 48px;
                height: 48px;
                border: 4px solid var(--border);
                border-top: 4px solid var(--primary);
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 16px;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .loading p {
                color: var(--text-light);
                font-size: 0.95rem;
                line-height: 1.6;
            }

            .loading .steps {
                margin-top: 16px;
                text-align: left;
                font-size: 0.85rem;
                color: var(--text-light);
                padding-left: 20px;
            }

            .loading .steps li {
                margin-bottom: 6px;
                list-style: none;
            }

            .loading .steps li::before {
                content: "⏳ ";
            }

            .loading .steps li.done::before {
                content: "✅ ";
            }

            .error-banner {
                display: none;
                background: #fff5f5;
                border: 1px solid #feb2b2;
                color: #c53030;
                padding: 16px;
                border-radius: 12px;
                margin-bottom: 20px;
                font-size: 0.9rem;
            }

            .footer {
                text-align: center;
                margin-top: 24px;
                font-size: 0.8rem;
                color: var(--text-light);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎓 Generateur de Cours RAG</h1>
            <p class="subtitle">
                Uploadez vos documents PDF et laissez l'IA generer un cours structure 
                avec plan, contenu pedagogique et exercices.
            </p>

            <div class="info-box">
                <strong>🔧 Pipeline :</strong> Extraction → Resume → Vectorisation → 
                Plan IA → Cours → TD
            </div>

            <div class="error-banner" id="error-banner"></div>

            <form id="rag-form" action="/generate" method="post" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="pdfs">
                        📄 Documents PDF
                        <span class="label-hint">(max 10 fichiers, 50 Mo chacun)</span>
                    </label>
                    <input type="file" name="pdfs" id="pdfs" accept=".pdf" multiple required>
                    <p class="file-hint">Selectionnez un ou plusieurs fichiers PDF source</p>
                </div>

                <div class="form-group">
                    <label for="pages">📄 Nombre de pages du cours</label>
                    <input type="number" name="pages" id="pages" min="1" max="100" value="10" required>
                </div>

                <div class="form-group">
                    <label for="chapters">📕 Nombre de chapitres</label>
                    <input type="number" name="chapters" id="chapters" min="1" max="20" value="3" required>
                </div>

                <div class="checkbox-group">
                    <input type="checkbox" name="generate_td" id="generate_td" checked>
                    <label for="generate_td">Generer un TD de revision (2 pages max)</label>
                </div>

                <button type="submit" id="submit-btn">
                    🚀 Lancer la generation
                </button>
            </form>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p><strong>Le pipeline RAG est en cours d'execution...</strong></p>
                <p>Cela peut prendre de 2 a 10 minutes selon la taille des documents.</p>
                <ul class="steps" id="steps">
                    <li id="step-1">Extraction hierarchique des PDFs</li>
                    <li id="step-2">Resume et structuration</li>
                    <li id="step-3">Vectorisation FAISS</li>
                    <li id="step-4">Generation du plan de cours</li>
                    <li id="step-5">Redaction du cours complet</li>
                    <li id="step-6">Generation du TD</li>
                </ul>
            </div>

            <div class="footer">
                Workspace isole • Session jetable • Aucun fichier conserve
            </div>
        </div>

        <script>
            document.getElementById('rag-form').addEventListener('submit', function(e) {
                const files = document.getElementById('pdfs').files;
                const errorBanner = document.getElementById('error-banner');

                // Validation
                if (files.length === 0) {
                    e.preventDefault();
                    showError("Veuillez selectionner au moins un fichier PDF.");
                    return;
                }

                if (files.length > 10) {
                    e.preventDefault();
                    showError("Maximum 10 fichiers PDF autorises.");
                    return;
                }

                let totalSize = 0;
                for (let f of files) {
                    totalSize += f.size;
                    if (f.size > 50 * 1024 * 1024) {
                        e.preventDefault();
                        showError(`Le fichier "${f.name}" depasse 50 Mo.`);
                        return;
                    }
                }

                if (totalSize > 200 * 1024 * 1024) {
                    e.preventDefault();
                    showError("La taille totale des fichiers depasse 200 Mo.");
                    return;
                }

                // Animation de chargement
                document.getElementById('loading').style.display = 'block';
                document.getElementById('submit-btn').disabled = true;
                document.getElementById('submit-btn').textContent = '⏳ Traitement en cours...';
                errorBanner.style.display = 'none';

                // Simulation de progression des etapes
                const steps = document.querySelectorAll('.steps li');
                let currentStep = 0;
                const interval = setInterval(() => {
                    if (currentStep < steps.length) {
                        steps[currentStep].classList.add('done');
                        currentStep++;
                    }
                }, 30000); // Change toutes les 30s (approximation)

                // Stocker l'interval pour pouvoir le nettoyer si besoin
                window.ragInterval = interval;
            });

            function showError(msg) {
                const banner = document.getElementById('error-banner');
                banner.textContent = "❌ " + msg;
                banner.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """


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
):
    """
    Endpoint principal : recoit les PDFs et les parametres, lance le pipeline,
    et retourne un ZIP avec tous les fichiers generes.
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

    # Validation des fichiers
    total_size = 0
    for pdf in pdfs:
        if pdf.filename and not pdf.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400, 
                detail=f"Fichier non-PDF rejete : {pdf.filename}"
            )

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

        # Lancement du workspace manager
        manager = WorkspaceManager(ORIGINAL_PROJECT_ROOT)
        result = manager.run_session(pdf_paths, config, keep_workspace=False)

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

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": 
                    f"attachment; filename=cours_rag_{result['session_id']}.zip"
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