#!/usr/bin/env python3
"""Orchestrateur de workspace : cree une copie temporaire du projet pour chaque session.

Usage:
    python workspace_manager.py --project "C:/Users/dell/Desktop/Projet python - RAG" \
                                --pdfs doc1.pdf doc2.pdf \
                                --pages 10 --chapters 3

Modifications (integration web) :
    - run_session(..., session_id=...) : l'id fourni par le frontend est
      utilise pour nommer le workspace, ce qui permet a app.py de lire
      <workspace>/progress.json pendant l'execution (vrai suivi d'etapes).
    - Parsing du resultat via la ligne 'RAG_RESULT_JSON:' emise par
      main_batch.py (l'ancien parsing cherchait un JSON sur UNE ligne,
      or le JSON etait imprime indente -> jamais reconnu -> fallback).
    - Les 'warnings' du pipeline (ex: TD non genere) sont propages
      jusqu'a la reponse HTTP.
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import uuid


class WorkspaceManager:
    """
    Gere le cycle de vie d'un workspace isole pour le pipeline RAG.

    Cycle de vie :
        1. Creation d'un dossier temporaire unique
        2. Copie du code source (sans les donnees generees)
        3. PATCH AUTOMATIQUE des chemins absolus
        4. Injection des PDFs uploades
        5. Execution du pipeline batch
        6. Recuperation des fichiers generes
        7. Nettoyage (optionnel)
    """

    EXCLUDE_DIRS = {
        "__pycache__", ".git", ".venv", "venv", "env",
        "vector_store",
        ".pytest_cache", ".mypy_cache", ".idea", ".vscode",
        "node_modules",
    }

    EXCLUDE_EXTS = {".pyc", ".pyo", ".index", ".faiss", ".bin"}

    def __init__(self, original_project_root: Path):
        """
        Args:
            original_project_root: chemin absolu vers le projet RAG original
                                  (ex: C:/Users/dell/Desktop/Projet python - RAG)
        """
        self.original_project_root = Path(original_project_root).resolve()
        if not self.original_project_root.exists():
            raise FileNotFoundError(
                f"Projet original introuvable : {self.original_project_root}\n"
                f"Verifiez le chemin et assurez-vous que le dossier existe."
            )

        self.workspaces_dir = self.original_project_root.parent / "web_sessions"
        self.workspaces_dir.mkdir(parents=True, exist_ok=True)

        print(f"[OK] WorkspaceManager initialise")
        print(f"     Projet original : {self.original_project_root}")
        print(f"     Dossier sessions : {self.workspaces_dir}")

    def create_workspace(self, session_id: Optional[str] = None) -> Path:
        """Cree un nouveau workspace temporaire unique."""
        if session_id is None:
            session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        workspace = self.workspaces_dir / f"session_{session_id}"

        if workspace.exists():
            raise FileExistsError(f"Le workspace existe deja : {workspace}")

        workspace.mkdir(parents=True, exist_ok=False)

        (workspace / "rag" / "data" / "input").mkdir(parents=True, exist_ok=True)
        (workspace / "rag" / "data" / "output").mkdir(parents=True, exist_ok=True)
        (workspace / "rag" / "data" / "vector_store").mkdir(parents=True, exist_ok=True)

        print(f"[DIR] Workspace cree : {workspace}")
        return workspace

    def _should_copy(self, path: Path) -> bool:
        """Determine si un fichier/dossier doit etre copie dans le workspace."""
        name = path.name
        if name in self.EXCLUDE_DIRS:
            return False
        if path.is_file() and path.suffix.lower() in self.EXCLUDE_EXTS:
            return False
        path_str = str(path).lower()
        if "vector_store" in path_str:
            return False
        return True

    def _patch_absolute_paths(self, workspace: Path) -> None:
        """
        SOLUTION DURABLE : Remplace les chemins absolus vers le projet original
        par le chemin du workspace dans tous les fichiers .py.
        Le projet original reste intact.
        """
        original_str = str(self.original_project_root)
        workspace_str = str(workspace)

        # Variantes du chemin original (backslash et slash, avec et sans /rag)
        variants = [
            original_str,
            original_str.replace("\\", "/"),
            original_str + "\\rag",
            original_str.replace("\\", "/") + "/rag",
        ]

        workspace_variants = [
            workspace_str,
            workspace_str.replace("\\", "/"),
            workspace_str + "\\rag",
            workspace_str.replace("\\", "/") + "/rag",
        ]

        patched_files = []

        for py_file in workspace.rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8")
                new_content = content
                has_patch = False

                for orig_var, ws_var in zip(variants, workspace_variants):
                    if orig_var in new_content:
                        new_content = new_content.replace(orig_var, ws_var)
                        has_patch = True

                if has_patch and new_content != content:
                    py_file.write_text(new_content, encoding="utf-8")
                    patched_files.append(py_file.relative_to(workspace))
            except Exception as e:
                print(f"[WARN] Impossible de patcher {py_file}: {e}")

        if patched_files:
            print(f"[PATCH] {len(patched_files)} fichier(s) corrige(s) :")
            for f in patched_files:
                print(f"        -> {f}")
        else:
            print("[PATCH] Aucun chemin absolu detecte (deja relatif ou dynamique)")

    def copy_project(self, workspace: Path) -> None:
        """Copie le code source du projet original vers le workspace."""
        print(f"\n[COPY] Copie du projet vers {workspace}...")

        def copy_tree(src: Path, dst: Path):
            dst.mkdir(parents=True, exist_ok=True)
            for item in src.iterdir():
                if not self._should_copy(item):
                    continue
                dest_item = dst / item.name
                if item.name in ("input", "output", "vector_store") and dest_item.exists():
                    continue
                if item.is_dir():
                    copy_tree(item, dest_item)
                else:
                    shutil.copy2(item, dest_item)

        copy_tree(self.original_project_root, workspace)

        # SOLUTION DURABLE : patcher les chemins absolus automatiquement
        self._patch_absolute_paths(workspace)

        if not (workspace / "main_batch.py").exists():
            print("[WARN] main_batch.py n'a pas ete trouve dans le workspace.")
            print("       Assurez-vous d'avoir copie main_batch.py dans le projet original.")

        print("[OK] Copie terminee.")

    def inject_pdfs(self, workspace: Path, pdf_files: List[Path]) -> None:
        """Injecte les PDFs uploades dans le workspace."""
        input_dir = workspace / "rag" / "data" / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        for old_pdf in input_dir.glob("*.pdf"):
            old_pdf.unlink()
            print(f"     [DEL] Ancien PDF supprime : {old_pdf.name}")

        for pdf_path in pdf_files:
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF introuvable : {pdf_path}")
            dest = input_dir / pdf_path.name
            shutil.copy2(pdf_path, dest)
            print(f"     [FILE] {pdf_path.name} -> {dest}")

        print(f"[OK] {len(pdf_files)} PDF(s) injecte(s).")

    @staticmethod
    def _parse_pipeline_output(stdout: str) -> Optional[dict]:
        """Extrait le JSON de resultat emis par main_batch.py.

        1. Cherche la ligne prefixee 'RAG_RESULT_JSON:' (fiable).
        2. Fallback : ancienne heuristique (JSON sur une seule ligne).
        """
        lines = [l for l in (stdout or "").strip().split("\n") if l.strip()]

        for line in reversed(lines):
            line = line.strip()
            if line.startswith("RAG_RESULT_JSON:"):
                try:
                    return json.loads(line[len("RAG_RESULT_JSON:"):])
                except json.JSONDecodeError:
                    continue

        for line in reversed(lines):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

        return None

    def get_progress(self, session_id: str) -> Optional[dict]:
        """Lit <workspace>/progress.json pour une session en cours (ou None)."""
        progress_file = self.workspaces_dir / f"session_{session_id}" / "progress.json"
        if not progress_file.exists():
            return None
        try:
            return json.loads(progress_file.read_text(encoding="utf-8"))
        except Exception:
            return None

    def run_pipeline(self, workspace: Path, config: dict) -> dict:
        """Lance le pipeline batch dans le workspace via un sous-processus isole."""
        main_batch_path = workspace / "main_batch.py"
        if not main_batch_path.exists():
            raise FileNotFoundError(
                f"main_batch.py introuvable dans {workspace}\n"
                f"Copiez main_batch.py a la racine du projet original."
            )

        config_path = workspace / "session_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        python_executable = sys.executable

        print(f"\n[START] Lancement du pipeline dans le workspace...")
        print(f"        Python    : {python_executable}")
        print(f"        Config    : {config_path}")
        print(f"        Workspace : {workspace}")

        env = os.environ.copy()
        env["RAG_PROJECT_ROOT"] = str(workspace)
        env["PYTHONIOENCODING"] = "utf-8"

        mistral_key = config.get("mistral_api_key", os.environ.get("MISTRAL_API_KEY", ""))
        if mistral_key:
            env["MISTRAL_API_KEY"] = mistral_key

        current_pp = env.get("PYTHONPATH", "")
        paths = [str(workspace), str(self.original_project_root)]
        if current_pp:
            paths.append(current_pp)
        env["PYTHONPATH"] = os.pathsep.join(paths)

        cmd = [
            python_executable,
            str(main_batch_path),
            "--config", str(config_path),
            "--project-root", str(workspace)
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(workspace),
                env=env,
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8"
            )

            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)

            output = self._parse_pipeline_output(result.stdout)

            if output is None:
                print("[WARN] Impossible de parser le resultat JSON, fallback")
                output = {
                    "success": True,
                    "output_dir": str(workspace / "rag" / "data" / "output"),
                    "files": {},
                    "warnings": [],
                }

            return output

        except subprocess.CalledProcessError as e:
            error_msg = f"""
[ERR] ERREUR DU PIPELINE (code {e.returncode})
{'='*70}
STDOUT:
{e.stdout or '(vide)'}

STDERR:
{e.stderr or '(vide)'}
{'='*70}
"""
            print(error_msg)
            raise RuntimeError(error_msg)

    def collect_results(self, workspace: Path, pipeline_result: Optional[dict] = None) -> Dict[str, bytes]:
        """Recupere les fichiers generes depuis le workspace."""
        output_dir = workspace / "rag" / "data" / "output"
        results = {}

        expected_files = ["course.md", "course.json", "plan.json", "td.md"]

        print(f"\n[GET] Recuperation des resultats depuis {output_dir}...")

        # --- CORRECTION : Chercher aussi dans output/course_generated/ ---
        search_dirs = [output_dir, output_dir / "course_generated"]
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for filename in expected_files:
                if filename in results:  # Déjà trouvé dans un dossier précédent
                    continue
                file_path = search_dir / filename
                if file_path.exists():
                    with open(file_path, "rb") as f:
                        results[filename] = f.read()
                    print(f"      [OK] {filename} ({len(results[filename])} bytes) depuis {search_dir.name}")
                else:
                    print(f"      [WARN] {filename} non trouve dans {search_dir.name}")

        return results

    def cleanup(self, workspace: Path) -> None:
        """Supprime le workspace et tout son contenu."""
        if workspace.exists():
            shutil.rmtree(workspace)
            print(f"[DEL] Workspace supprime : {workspace}")

    def run_session(self,
                    pdf_files: List[Path],
                    config: dict,
                    keep_workspace: bool = False,
                    session_id: Optional[str] = None) -> dict:
        """
        Pipeline complet : creation -> copie -> patch -> injection -> execution -> recuperation -> nettoyage.
        """
        workspace = None
        files_data = {}
        pipeline_result = {"success": False, "warnings": []}
        error_msg = None

        try:
            workspace = self.create_workspace(session_id=session_id)
            self.copy_project(workspace)
            self.inject_pdfs(workspace, pdf_files)
            pipeline_result = self.run_pipeline(workspace, config)
            files_data = self.collect_results(workspace)

        except Exception as e:
            # --- CORRECTION : Récupérer les fichiers partiels avant nettoyage ---
            error_msg = str(e)
            print(f"\n[WARN] Pipeline interrompu : {error_msg}")
            if workspace and workspace.exists():
                files_data = self.collect_results(workspace)
                print(f"[WARN] Fichiers partiels recuperes : {list(files_data.keys())}")
            pipeline_result["success"] = bool(files_data)
            pipeline_result["warnings"] = [error_msg]

        finally:
            if workspace and not keep_workspace:
                self.cleanup(workspace)

        return {
            "success": pipeline_result.get("success", True) or bool(files_data),
            "session_id": workspace.name if workspace else (session_id or "unknown"),
            "files": files_data,
            "paths": pipeline_result.get("files", {}),
            "warnings": pipeline_result.get("warnings", []),
        }

    run_session_sync = run_session


# ---------------------------------------------------------------------
# UTILITAIRES STANDALONE (test manuel)
# ---------------------------------------------------------------------
def main():
    """Test manuel du workspace manager en ligne de commande."""
    import argparse
    parser = argparse.ArgumentParser(description="Test du Workspace Manager")
    parser.add_argument("--project", "-p", required=True,
                        help="Chemin du projet original (ex: C:/Users/.../Projet python - RAG)")
    parser.add_argument("--pdfs", required=True, nargs="+",
                        help="Chemins des PDFs a traiter")
    parser.add_argument("--pages", type=int, default=10,
                        help="Nombre de pages du cours")
    parser.add_argument("--chapters", type=int, default=3,
                        help="Nombre de chapitres")
    parser.add_argument("--keep", action="store_true",
                        help="Garder le workspace apres execution (debug)")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Sauter l'extraction si des chunks existent")
    parser.add_argument("--skip-summary", action="store_true",
                        help="Sauter le resume hierarchique")
    parser.add_argument("--skip-indexing", action="store_true",
                        help="Sauter la vectorisation")
    parser.add_argument("--no-td", action="store_true",
                        help="Ne pas generer le TD")
    args = parser.parse_args()

    manager = WorkspaceManager(Path(args.project))

    config = {
        "max_pages": args.pages,
        "num_chapters": args.chapters,
        "chapters": [],
        "skip_extraction": args.skip_extraction,
        "skip_summary": args.skip_summary,
        "skip_indexing": args.skip_indexing,
        "interactive_plan": False,
        "generate_td": not args.no_td,
        "mistral_api_key": os.environ.get("MISTRAL_API_KEY"),
    }

    pdf_files = [Path(p) for p in args.pdfs]

    for p in pdf_files:
        if not p.exists():
            print(f"[ERR] PDF introuvable : {p}")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("CONFIGURATION DE LA SESSION")
    print("=" * 70)
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print("=" * 70)

    result = manager.run_session(pdf_files, config, keep_workspace=args.keep)

    print("\n" + "=" * 70)
    print("RESULTAT")
    print("=" * 70)
    print(f"Succes      : {result['success']}")
    print(f"Session ID  : {result['session_id']}")
    print(f"Fichiers    : {list(result['files'].keys())}")
    print(f"Warnings    : {result.get('warnings', [])}")
    print("=" * 70)

    if result["files"]:
        test_output = Path("test_output")
        test_output.mkdir(exist_ok=True)
        for fname, fbytes in result["files"].items():
            if fbytes:
                out_path = test_output / fname
                with open(out_path, "wb") as f:
                    f.write(fbytes)
                print(f"[SAVE] {fname} sauvegarde dans {out_path}")


if __name__ == "__main__":
    main()