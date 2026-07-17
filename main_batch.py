#!/usr/bin/env python3
"""Version batch (non-interactive) du pipeline RAG.

Modifications (integration web) :
    - ProgressReporter : ecrit progress.json a la racine du workspace
      pour que l'interface web suive les etapes EN TEMPS REEL.
    - Phase 5 (TD) : retry avec backoff + avertissements remontes dans
      le resultat JSON au lieu d'un echec silencieux.
    - Le JSON final est aussi imprime sur UNE ligne prefixee
      RAG_RESULT_JSON: pour etre parse de facon fiable par le
      workspace_manager (l'ancien JSON multi-lignes n'etait jamais
      reconnu -> fallback systematique).
"""

import os
import sys
import json
import time
import importlib.util
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------
# SUIVI DE PROGRESSION (lu par app.py -> /api/progress/<session_id>)
# ---------------------------------------------------------------------
class ProgressReporter:
    """Ecrit l'etat d'avancement dans <workspace>/progress.json.

    Statuts d'etape : pending | running | done | skipped | error
    Statut global   : running | done | error
    """

    STEP_NAMES = {
        1: "Extraction hierarchique des PDFs",
        2: "Resume et structuration",
        3: "Vectorisation FAISS",
        4: "Generation du plan de cours",
        5: "Redaction du cours complet",
        6: "Generation du TD",
    }

    def __init__(self, project_root):
        self.path = Path(project_root) / "progress.json"
        self.steps = {
            str(i): {"name": name, "status": "pending"}
            for i, name in self.STEP_NAMES.items()
        }
        self.status = "running"
        self.error = None
        self.current_step = None
        self._write()

    def _write(self):
        data = {
            "status": self.status,
            "error": self.error,
            "current_step": self.current_step,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "steps": self.steps,
        }
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)  # ecriture atomique
        except Exception as e:
            print(f"[WARN] Impossible d'ecrire progress.json : {e}")

    def start(self, step):
        self.current_step = step
        self.steps[str(step)]["status"] = "running"
        self._write()

    def done(self, step):
        self.steps[str(step)]["status"] = "done"
        self._write()

    def skipped(self, step):
        self.steps[str(step)]["status"] = "skipped"
        self._write()

    def step_error(self, step, message):
        """Erreur NON fatale (ex: TD) : l'etape est en echec mais le
        pipeline continue et livre les autres fichiers."""
        self.steps[str(step)]["status"] = "error"
        self._write()

    def fail(self, step, message):
        """Erreur fatale : tout le pipeline est en echec."""
        if step:
            self.steps[str(step)]["status"] = "error"
        self.status = "error"
        self.error = message
        self._write()

    def finish(self):
        self.status = "done"
        self._write()


def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger {module_name} depuis {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_pipeline(config, project_root):
    # Le reporter est cree AVANT tout le reste pour que meme une erreur
    # d'import soit visible cote interface web.
    reporter = ProgressReporter(project_root)
    try:
        return _run_pipeline_inner(config, project_root, reporter)
    except Exception as e:
        reporter.fail(reporter.current_step, f"{type(e).__name__}: {e}")
        raise


def _run_pipeline_inner(config, project_root, reporter):
    RAG_ROOT = project_root / "rag"
    warnings = []

    # Chargement dynamique des modules
    _reterevel = load_module_from_path("reterevel", project_root / "reterevel.py")
    run_extraction = _reterevel.run

    _hiarchy_summary = load_module_from_path(
        "hiarchy_summary", RAG_ROOT / "vectorization" / "hiarchy_summary.py"
    )
    run_summary = _hiarchy_summary.main

    if str(RAG_ROOT) not in sys.path:
        sys.path.insert(0, str(RAG_ROOT))

    _build_index = load_module_from_path(
        "build_index", RAG_ROOT / "vectorization" / "scripts" / "build_index.py"
    )
    run_indexing = _build_index.main

    # Phase 3 & 4 : course_generator
    from course_generator.config import CourseConfig
    from course_generator.mistral_client import MistralClient
    from course_generator.planner import PlanGenerator
    from course_generator.orchestrator import CourseOrchestrator

    # Parametres
    skip_extraction = config.get("skip_extraction", False)
    skip_summary = config.get("skip_summary", False)
    skip_indexing = config.get("skip_indexing", False)
    max_pages = config.get("max_pages", 10)
    num_chapters = config.get("num_chapters", 3)
    chapters = config.get("chapters", [])
    interactive_plan = config.get("interactive_plan", False)
    pre_validated_plan = config.get("plan", None)
    generate_td = config.get("generate_td", True)
    mistral_api_key = config.get("mistral_api_key", os.environ.get("MISTRAL_API_KEY"))
    plan_modifications = config.get("plan_modifications", [])

    def display_plan(plan_data):
        print("\n" + "=" * 70)
        print(f"TITRE    : {plan_data.get('titre_cours', 'N/A')}")
        print(f"DESCRIPTION : {plan_data.get('description', 'N/A')}")
        mp = plan_data.get('max_pages', 'N/A')
        print(f"PAGES    : {mp}")
        print("=" * 70)

        def _recurse(node, depth=0):
            indent = "   " * depth
            lvl = node.get("niveau", 1)
            badge = {1: "CHAPITRE", 2: "SECTION", 3: "SOUS-TITRE"}.get(lvl, "SECTION")
            print(f"{indent}{badge} : {node.get('titre', 'Sans titre')}  (id={node.get('id', 'N/A')})")
            print(f"{indent}    Objectif : {node.get('objectif', 'N/A')}")
            for enfant in node.get("enfants", []):
                _recurse(enfant, depth + 1)

        for node in plan_data.get("plan", []):
            _recurse(node)
        print("=" * 70 + "\n")

    def refine_plan_with_mistral(client, current_plan, user_request):
        plan_json = json.dumps(current_plan, ensure_ascii=False, indent=2)
        prompt = f"""Tu es un professeur universitaire expert en pedagogie et en structuration de cours.

Voici le plan de cours actuel (format JSON strict) :
{plan_json}

L'utilisateur (ton collegue enseignant) te demande la modification suivante :
"{user_request}"

Modifie le plan en consequence. Tu peux :
- Ajouter, supprimer, renommer ou deplacer des sections
- Modifier les objectifs pedagogiques
- Reorganiser la progression pedagogique
- Ajuster la granularite (decouper ou fusionner des sections)

CONTRAINTES ABSOLUES :
1. Conserve le format JSON exact : {{"titre_cours":"...", "description":"...", "plan":[{{"id":"...", "titre":"...", "niveau":1, "objectif":"...", "enfants":[]}}]}}
2. Chaque element doit avoir : id, titre, niveau (1=chapitre, 2=titre, 3=sous-titre), objectif, enfants (liste, meme vide)
3. Les IDs doivent rester uniques et coherents (ex: ch1, ch1_sec1, ch1_sec1_sub1)
4. Retourne UNIQUEMENT le JSON valide, sans markdown, sans explication, sans balises de code."""

        messages = [
            {
                "role": "system",
                "content": "Tu es un assistant pedagogique qui retourne uniquement du JSON valide representant un plan de cours structure."
            },
            {"role": "user", "content": prompt}
        ]

        try:
            raw = client.chat(messages, temperature=0.35, force_json=True)
            new_plan = json.loads(raw)
            if "plan" not in new_plan:
                raise ValueError("Cle 'plan' manquante")

            def _validate(nodes, parent_id=""):
                for i, node in enumerate(nodes):
                    for k in ("id", "titre", "niveau", "objectif", "enfants"):
                        if k not in node:
                            raise ValueError(f"Champ '{k}' manquant dans le noeud {node.get('id', i)}")
                    if "enfants" in node and node["enfants"]:
                        _validate(node["enfants"], node["id"])

            _validate(new_plan["plan"])
            return new_plan
        except Exception as e:
            print(f"\nWARNING: Mistral n'a pas retourne un JSON valide : {e}")
            print("Le plan precedent est conserve.")
            return current_plan

    def interactive_plan_loop(client, initial_plan, plan_path):
        current_plan = initial_plan
        backup_plan = json.loads(json.dumps(initial_plan))

        print("\n" + "=" * 70)
        print("MODE PLAN : MODIFICATION DU PLAN DE COURS")
        print("=" * 70)
        display_plan(current_plan)

        if plan_modifications:
            for mod in plan_modifications:
                print(f"\nApplication de la modification : {mod}")
                current_plan = refine_plan_with_mistral(client, current_plan, mod)
                plan_path.parent.mkdir(parents=True, exist_ok=True)
                with open(plan_path, "w", encoding="utf-8") as f:
                    json.dump(current_plan, f, ensure_ascii=False, indent=2)
                print("Plan intermediaire sauvegarde.")
                time.sleep(1.0)

        print("\nPlan valide.")

        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(current_plan, f, ensure_ascii=False, indent=2)
        print(f"\nPlan final sauvegarde dans :\n   {plan_path}")
        return current_plan

    def phase1_extraction():
        pdf_dir = RAG_ROOT / "data" / "input"
        output_dir = RAG_ROOT / "data" / "output"

        if not pdf_dir.exists() or not any(pdf_dir.glob("*.pdf")):
            raise FileNotFoundError(f"Aucun PDF trouve dans {pdf_dir}")

        print("\n" + "=" * 70)
        print("PHASE 1 : EXTRACTION HIERARCHIQUE DES PDFs")
        print("=" * 70)

        original_cwd = os.getcwd()
        try:
            os.chdir(str(project_root))
            run_extraction(
                pdf_dir=pdf_dir,
                output_dir=output_dir,
                max_scan_pages=20,
                toc_max_span=8,
                fuzzy_threshold=82,
                chunk_size=1000,
                chunk_overlap=150,
                disable_toc=False,
            )
        finally:
            os.chdir(original_cwd)

        print("Phase 1 terminee.")

    def phase2_summary():
        print("\n" + "=" * 70)
        print("PHASE 2 : RESUME HIERARCHIQUE GLOBAL")
        print("=" * 70)

        original_cwd = os.getcwd()
        try:
            os.chdir(str(project_root))
            rc = run_summary()
        finally:
            os.chdir(original_cwd)

        if rc != 0:
            raise RuntimeError("Phase 2 echouee.")

        # Verifier que le resume a ete genere
        summary_path = RAG_ROOT / "data" / "output" / "hierarchy_summary.json"
        if not summary_path.exists():
            print(f"WARNING: {summary_path} non trouve apres phase 2")
        else:
            print(f"Resume genere : {summary_path}")

        print("Phase 2 terminee.")

    def phase2bis_vectorization():
        print("\n" + "=" * 70)
        print("PHASE 2bis : VECTORISATION ET INDEXATION FAISS")
        print("=" * 70)

        # Verifier que des chunks existent avant d'indexer
        chunks_dir = RAG_ROOT / "data" / "output"
        chunk_files = list(chunks_dir.glob("*/chunks.jsonl"))
        if not chunk_files:
            print(f"WARNING: Aucun chunk trouve dans {chunks_dir}")
            print("La vectorisation sera saute.")
            return

        print(f"Chunks detectes : {len(chunk_files)} fichier(s)")
        for cf in chunk_files:
            print(f"  -> {cf}")

        original_cwd = os.getcwd()
        try:
            os.chdir(str(project_root))
            rc = run_indexing()
        finally:
            os.chdir(original_cwd)

        # Verifier si l'index a ete genere
        vector_store = RAG_ROOT / "data" / "vector_store"
        faiss_file = vector_store / "faiss.index"
        meta_file = vector_store / "metadata.jsonl"

        if rc != 0:
            print(f"WARNING: build_index.py a retourne le code {rc}")
            print("Verification des fichiers generees...")

        if faiss_file.exists() and meta_file.exists():
            print(f"[OK] Index FAISS genere : {faiss_file}")
            print(f"[OK] Metadonnees generees : {meta_file}")
        else:
            print(f"WARNING: Index FAISS introuvable dans {vector_store}")
            if rc != 0:
                raise RuntimeError(f"Phase 2bis echouee (code {rc}). Index non genere.")

        print("Phase 2bis terminee.")

    def phase3_plan():
        print("\n" + "=" * 70)
        print("PHASE 3 : CONFIGURATION ET SUGGESTION DU PLAN")
        print("=" * 70)

        if not mistral_api_key:
            raise ValueError("Cle API Mistral non definie. Definissez MISTRAL_API_KEY.")

        cfg = CourseConfig(project_root=project_root, mistral_api_key=mistral_api_key)

        client = MistralClient(
            api_key=cfg.mistral_api_key,
            model=cfg.mistral_model,
            min_delay=cfg.min_delay_seconds,
        )

        plan_path = cfg.output_dir / "plan.json"

        user_chapters = chapters if chapters else []
        if not user_chapters and num_chapters:
            for i in range(1, num_chapters + 1):
                user_chapters.append({
                    "id": f"ch{i}",
                    "titre": "",
                    "niveau": 1,
                    "objectif": "",
                    "enfants": []
                })

        constraints = {
            "max_pages": max_pages,
            "num_chapters": num_chapters,
            "chapters": user_chapters
        }

        print(f"\nNombre de pages : {max_pages}")
        print(f"Nombre de chapitres : {num_chapters}")

        print("\nMistral suggere un plan adapte a vos contraintes...")
        planner = PlanGenerator(client)

        initial_plan = planner.generate_plan_with_constraints(
            structure_path=cfg.hierarchy_summary_path,
            max_pages=constraints["max_pages"],
            num_chapters=constraints["num_chapters"],
            user_chapters=constraints["chapters"]
        )

        initial_plan["max_pages"] = max_pages

        plan_path.parent.mkdir(parents=True, exist_ok=True)
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(initial_plan, f, ensure_ascii=False, indent=2)

        if pre_validated_plan:
            validated_plan = pre_validated_plan
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(validated_plan, f, ensure_ascii=False, indent=2)
            print("\nPlan pre-valide utilise.")
        elif interactive_plan:
            validated_plan = interactive_plan_loop(client, initial_plan, plan_path)
        else:
            validated_plan = initial_plan
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(validated_plan, f, ensure_ascii=False, indent=2)
            print("\nPlan initial valide automatiquement (mode batch).")

        if validated_plan is None:
            raise RuntimeError("Generation annulee.")

        return validated_plan

    def phase4_generation_cours():
        print("\n" + "=" * 70)
        print("PHASE 4 : GENERATION DU COURS COMPLET")
        print("=" * 70)

        cfg = CourseConfig(project_root=project_root, mistral_api_key=mistral_api_key)
        orchestrator = CourseOrchestrator(cfg)
        orchestrator.run()

        print("\nPhase 4 terminee.")
        out_dir = cfg.output_dir
        print(f"\nFichiers generes dans : {out_dir}")
        for fname in ["course.md", "course.json", "plan.json"]:
            fpath = out_dir / fname
            if fpath.exists():
                print(f"   - {fpath}")

    def phase5_generation_td():
        print("\n" + "=" * 70)
        print("PHASE 5 : GENERATION DU TD")
        print("=" * 70)

        cfg = CourseConfig(project_root=project_root, mistral_api_key=mistral_api_key)
        client = MistralClient(
            api_key=cfg.mistral_api_key,
            model=cfg.mistral_model,
            min_delay=cfg.min_delay_seconds,
        )

        course_json_path = cfg.output_dir / "course.json"
        if not course_json_path.exists():
            msg = "TD non genere : course.json introuvable (le cours doit etre genere d'abord)."
            print(f"ERREUR : {msg}")
            warnings.append(msg)
            reporter.step_error(6, msg)
            return

        with open(course_json_path, "r", encoding="utf-8") as f:
            course_data = json.load(f)

        sections_text = []

        def _extract(nodes):
            for node in nodes:
                titre = node.get("titre", "")
                obj = node.get("objectif", "")
                contenu = node.get("contenu", "")
                if contenu:
                    sections_text.append(f"## {titre}\nObjectif : {obj}\n\n{contenu}")
                if node.get("enfants"):
                    _extract(node["enfants"])

        _extract(course_data.get("plan", []))

        contexte_cours = "\n\n---\n\n".join(sections_text)[:12000]
        titre_cours = course_data.get("titre_cours", "Cours")

        prompt = f"""Tu es un professeur universitaire expert en pedagogie. A partir du cours suivant, cree un TD (Travail Dirige) de revision.

=== CONTEXTE DU COURS ===
Titre : {titre_cours}

{contexte_cours}

=== CONTRAINTES ABSOLUES ===
- Le TD doit faire MAXIMUM 2 pages A4 (police 11pt, interligne 1.5), soit ENVIRON 500 MOTS MAXIMUM.
- Melange d'exercices varies :
  - 3 questions de type QCM (4 propositions dont une seule correcte)
  - 2 questions courtes (reponse en 2-3 lignes max)
  - 1 exercice pratique / cas d'application
- Les exercices doivent porter sur les points ESSENTIELS et les concepts cles du cours.
- Niveau adapte a un examen de revision.
- Format Markdown pur. AUCUN LaTeX structurel.
- Presente d'abord tous les enonces, puis un corrige succinct a la fin.

=== FORMAT ATTENDU ===
# TD : {titre_cours}

## Partie 1 - QCM
...

## Partie 2 - Questions courtes
...

## Partie 3 - Exercice pratique
...

## Corrige succinct
...

Compte tes mots. MAXIMUM 500 mots pour l'ensemble. Si tu depasses, supprime une question ou compresse.
Genere le TD maintenant."""

        messages = [
            {
                "role": "system",
                "content": (
                    "Tu es un assistant pedagogique qui cree des TD concis, varies "
                    "et pertinents en Markdown pur. Tu respectes strictement les limites de mots."
                )
            },
            {"role": "user", "content": prompt}
        ]

        # --- APPEL MISTRAL AVEC RETRY (au lieu d'un echec silencieux) ---
        print("Generation du TD par Mistral...")
        td_content = None
        last_error = None
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                td_content = client.chat(messages, temperature=0.4)
                break
            except Exception as e:
                last_error = e
                print(f"ERREUR TD (tentative {attempt}/{max_attempts}) : {e}")
                if attempt < max_attempts:
                    wait_s = 10 * attempt
                    print(f"Nouvelle tentative dans {wait_s}s...")
                    time.sleep(wait_s)

        if td_content is None:
            msg = (
                f"Le TD n'a pas pu etre genere apres {max_attempts} tentatives "
                f"(derniere erreur : {last_error}). Le cours reste disponible."
            )
            print(f"ERREUR : {msg}")
            warnings.append(msg)
            reporter.step_error(6, msg)
            return

        td_content = td_content.strip()
        if td_content.startswith("```markdown"):
            td_content = td_content[len("```markdown"):].strip()
        if td_content.startswith("```"):
            td_content = td_content[3:].strip()
        if td_content.endswith("```"):
            td_content = td_content[:-3].strip()

        td_path = cfg.output_dir / "td.md"
        with open(td_path, "w", encoding="utf-8") as f:
            f.write(td_content)

        nb_mots = len(td_content.split())
        print(f"TD genere avec succes (~{nb_mots} mots, budget: 500 mots)")
        print(f"Fichier : {td_path}")

    # ORCHESTRATION
    print("=" * 70)
    print("PIPELINE RAG COMPLET - MODE BATCH")
    print("=" * 70)
    print(f"Projet : {project_root}")
    print(f"RAG     : {RAG_ROOT}")
    print("Flux    : Extraction -> Hierarchie -> Vectorisation -> Plan -> Cours -> TD")
    print("=" * 70)

    output_dir = RAG_ROOT / "data" / "output"
    vector_store = RAG_ROOT / "data" / "vector_store"
    has_chunks = output_dir.exists() and any(output_dir.glob("*/chunks.jsonl"))
    has_index = (vector_store / "faiss.index").exists() and (vector_store / "metadata.jsonl").exists()
    has_summary = (output_dir / "hierarchy_summary.json").exists()

    if has_chunks and skip_extraction:
        print("\nPhase 1 ignoree (chunks existants + skip_extraction=True).")
        reporter.skipped(1)
    else:
        reporter.start(1)
        phase1_extraction()
        reporter.done(1)

    if has_summary and skip_summary:
        print("\nPhase 2 ignoree (resume existant + skip_summary=True).")
        reporter.skipped(2)
    else:
        reporter.start(2)
        phase2_summary()
        reporter.done(2)

    if has_index and skip_indexing:
        print("\nPhase 2bis ignoree (index existant + skip_indexing=True).")
        reporter.skipped(3)
    else:
        reporter.start(3)
        phase2bis_vectorization()
        reporter.done(3)

    reporter.start(4)
    phase3_plan()
    reporter.done(4)

    reporter.start(5)
    phase4_generation_cours()
    reporter.done(5)

    if generate_td:
        reporter.start(6)
        phase5_generation_td()
        # Le statut final de l'etape 6 est gere dans phase5 :
        # 'done' si le fichier existe, 'error' sinon (deja signale).
        td_path = CourseConfig(project_root=project_root, mistral_api_key=mistral_api_key).output_dir / "td.md"
        if td_path.exists():
            reporter.done(6)
    else:
        print("\nPhase 5 (TD) ignoree (generate_td=False).")
        reporter.skipped(6)
    
    reporter.skipped(6)
    print("\nPhase 5 ingonere pour test .")
    
    reporter.finish()

    print("\n" + "=" * 70)
    print("TOUT EST TERMINE !")
    print("=" * 70)

    cfg = CourseConfig(project_root=project_root, mistral_api_key=mistral_api_key)
    out_dir = cfg.output_dir

    result = {
        "success": True,
        "output_dir": str(out_dir),
        "files": {
            "course_md": str(out_dir / "course.md") if (out_dir / "course.md").exists() else None,
            "course_json": str(out_dir / "course.json") if (out_dir / "course.json").exists() else None,
            "plan_json": str(out_dir / "plan.json") if (out_dir / "plan.json").exists() else None,
            "td_md": str(out_dir / "td.md") if (out_dir / "td.md").exists() else None,
        },
        "warnings": warnings,
    }

    return result


if __name__ == "__main__":
    import argparse
    import traceback

    parser = argparse.ArgumentParser(description="Pipeline RAG en mode batch")
    parser.add_argument("--config", "-c", required=True, help="Chemin vers le fichier JSON de configuration")
    parser.add_argument("--project-root", "-p", required=True, help="Chemin racine du projet (workspace)")
    args = parser.parse_args()

    try:
        with open(args.config, "r", encoding="utf-8-sig") as f:
            config = json.load(f)

        project_root = Path(args.project_root)
        result = run_pipeline(config, project_root)
        # Ligne machine lisible par workspace_manager (parsing fiable)
        print("RAG_RESULT_JSON:" + json.dumps(result, ensure_ascii=False))
        # Version lisible pour l'humain
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"\nERREUR FATALE DANS main_batch.py:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {e}")
        print("\nTRACEBACK COMPLET:")
        traceback.print_exc()
        sys.exit(1)