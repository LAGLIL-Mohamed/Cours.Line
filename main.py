#!/usr/bin/env python3
"""Main unifié : Extraction+Hiérarchie -> Vectorisation -> Plan interactif -> Cours généré."""

import os
import sys
import json
import time
import importlib.util
from pathlib import Path

# ---------------------------------------------------------------------
# CONFIGURATION RACINE (chemins Windows absolus)
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Users\dell\Desktop\Projet python - RAG")
RAG_ROOT = PROJECT_ROOT / "rag"

# ---------------------------------------------------------------------
# CHARGEMENT DYNAMIQUE DES MODULES (évite les erreurs Pylance)
# ---------------------------------------------------------------------
def load_module_from_path(module_name: str, file_path: Path):
    """Charge un module Python depuis un chemin de fichier absolu."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger {module_name} depuis {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Phase 1+2 : hiarchy_summary.py dans rag\vectorization\hiarchy_summary.py
# Ce fichier gère l'extraction hiérarchique ET le résumé
_hiarchy_summary = load_module_from_path(
    "hiarchy_summary", RAG_ROOT / "vectorization" / "hiarchy_summary.py"
)
run_summary = _hiarchy_summary.main

# Phase 2bis : build_index.py dans rag\vectorization\scripts\build_index.py
# Ajout de RAG_ROOT au sys.path pour que les imports de vectorization fonctionnent
if str(RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_ROOT))
_build_index = load_module_from_path(
    "build_index", RAG_ROOT / "vectorization" / "scripts" / "build_index.py"
)
run_indexing = _build_index.main

# Phase 3 & 4 : modules dans rag\course_generator\
from course_generator.config import CourseConfig
from course_generator.mistral_client import MistralClient
from course_generator.planner import PlanGenerator
from course_generator.orchestrator import CourseOrchestrator


# ---------------------------------------------------------------------
# UTILITAIRES INTERACTIFS
# ---------------------------------------------------------------------
def ask_yes_no(question: str, default: bool = True) -> bool:
    choices = " [O/n] " if default else " [o/N] "
    while True:
        try:
            rep = input(f"{question}{choices}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAbandon.")
            sys.exit(0)
        if rep in ("o", "oui", "y", "yes"):
            return True
        if rep in ("n", "non", "no"):
            return False
        if rep == "":
            return default
        print("Réponse non reconnue.")


def display_plan(plan_data: dict) -> None:
    print("\n" + "=" * 70)
    print(f"📘 TITRE    : {plan_data.get('titre_cours', 'N/A')}")
    print(f"📝 DESCRIPTION : {plan_data.get('description', 'N/A')}")
    print("=" * 70)

    def _recurse(node, depth=0):
        indent = "   " * depth
        lvl = node.get("niveau", 1)
        badge = {1: "📕 CHAPITRE", 2: "📗 TITRE", 3: "📘 SOUS-TITRE"}.get(lvl, "📄 SECTION")
        print(f"{indent}{badge} : {node.get('titre', 'Sans titre')}  (id={node.get('id', 'N/A')})")
        print(f"{indent}    └─ Objectif : {node.get('objectif', 'N/A')}")
        for enfant in node.get("enfants", []):
            _recurse(enfant, depth + 1)

    for node in plan_data.get("plan", []):
        _recurse(node)
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------
# PHASE 3 : BOUCLE INTERACTIVE AVEC MISTRAL
# ---------------------------------------------------------------------
def refine_plan_with_mistral(client: MistralClient, current_plan: dict, user_request: str) -> dict:
    plan_json = json.dumps(current_plan, ensure_ascii=False, indent=2)

    prompt = f"""Tu es un professeur universitaire expert en pédagogie et en structuration de cours.

Voici le plan de cours actuel (format JSON strict) :
{plan_json}

L'utilisateur (ton collègue enseignant) te demande la modification suivante :
"{user_request}"

Modifie le plan en conséquence. Tu peux :
- Ajouter, supprimer, renommer ou déplacer des sections
- Modifier les objectifs pédagogiques
- Réorganiser la progression pédagogique
- Ajuster la granularité (découper ou fusionner des sections)

CONTRAINTES ABSOLUES :
1. Conserve le format JSON exact : {{"titre_cours":"...", "description":"...", "plan":[{{"id":"...", "titre":"...", "niveau":1, "objectif":"...", "enfants":[]}}]}}
2. Chaque élément doit avoir : id, titre, niveau (1=chapitre, 2=titre, 3=sous-titre), objectif, enfants (liste, même vide)
3. Les IDs doivent rester uniques et cohérents (ex: ch1, ch1_sec1, ch1_sec1_sub1)
4. Retourne UNIQUEMENT le JSON valide, sans markdown, sans explication, sans balises de code."""

    messages = [
        {
            "role": "system",
            "content": "Tu es un assistant pédagogique qui retourne uniquement du JSON valide représentant un plan de cours structuré."
        },
        {"role": "user", "content": prompt}
    ]

    try:
        raw = client.chat(messages, temperature=0.35, force_json=True)
        new_plan = json.loads(raw)
        if "plan" not in new_plan:
            raise ValueError("Clé 'plan' manquante")

        def _validate(nodes, parent_id=""):
            for i, node in enumerate(nodes):
                for k in ("id", "titre", "niveau", "objectif", "enfants"):
                    if k not in node:
                        raise ValueError(f"Champ '{k}' manquant dans le nœud {node.get('id', i)}")
                if "enfants" in node and node["enfants"]:
                    _validate(node["enfants"], node["id"])

        _validate(new_plan["plan"])
        return new_plan
    except Exception as e:
        print(f"\n⚠️  Mistral n'a pas retourné un JSON valide : {e}")
        print("ℹ️  Le plan précédent est conservé.")
        return current_plan


def interactive_plan_loop(client: MistralClient, initial_plan: dict, plan_path: Path) -> dict:
    current_plan = initial_plan
    backup_plan = json.loads(json.dumps(initial_plan))

    print("\n" + "=" * 70)
    print("🎓  MODE INTERACTIF : MODIFICATION DU PLAN DE COURS")
    print("=" * 70)
    print("Commandes disponibles :")
    print("  • Tapez une instruction pour modifier le plan (ex: 'Ajoute une section sur les réseaux de neurones')")
    print("  • 'valide'  ou 'je valide'  → confirmer et passer à la génération du cours")
    print("  • 'affiche' ou 'voir'       → réafficher le plan actuel")
    print("  • 'reset'  ou 'reinit'      → revenir au plan initial généré par Mistral")
    print("  • 'quit'   ou 'exit'        → abandonner")
    print("=" * 70)

    while True:
        display_plan(current_plan)

        try:
            user_input = input("\n💬 [VOTRE INSTRUCTION] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n[INFO] Interruption clavier. Sauvegarde du plan actuel.")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("valide", "je valide", "validé", "go", "ok"):
            print("\n✅ Plan validé !")
            break

        if cmd in ("affiche", "voir", "show", "display"):
            continue

        if cmd in ("reset", "reinit", "restaurer"):
            current_plan = json.loads(json.dumps(backup_plan))
            print("\n🔄 Plan réinitialisé à la version initiale.")
            continue

        if cmd in ("quit", "exit", "q", "stop"):
            print("\n❌ Abandon par l'utilisateur.")
            return None

        print("\n🤖 Envoi à Mistral pour modification...")
        current_plan = refine_plan_with_mistral(client, current_plan, user_input)

        plan_path.parent.mkdir(parents=True, exist_ok=True)
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(current_plan, f, ensure_ascii=False, indent=2)
        print("💾 Plan intermédiaire sauvegardé.")
        time.sleep(1.0)

    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(current_plan, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Plan final sauvegardé dans :\n   {plan_path}")
    return current_plan


# ---------------------------------------------------------------------
# ORCHESTRATION DES PHASES
# ---------------------------------------------------------------------
def phase1_extraction():
    """Phase 1 : Extraction hiérarchique des PDFs via hiarchy_summary.py."""
    pdf_dir = RAG_ROOT / "data" / "input"
    output_dir = RAG_ROOT / "data" / "output"

    if not pdf_dir.exists() or not any(pdf_dir.glob("*.pdf")):
        print(f"\n❌ ERREUR : Aucun PDF trouvé dans {pdf_dir}")
        print("   Placez vos fichiers PDF dans ce dossier et relancez.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("📄 PHASE 1 : EXTRACTION HIÉRARCHIQUE DES PDFs")
    print("=" * 70)

    # hiarchy_summary.py utilise OUTPUT_DIR = Path(r"rag\data\output") (relatif)
    # Il faut être dans PROJECT_ROOT pour que ce chemin se résolve correctement
    original_cwd = os.getcwd()
    try:
        os.chdir(str(PROJECT_ROOT))
        rc = run_summary()
    finally:
        os.chdir(original_cwd)

    if rc != 0:
        print("❌ Phase 1 échouée.")
        sys.exit(1)
    print("✅ Phase 1 terminée.")


def phase2bis_vectorization():
    """Phase 2bis : Vectorisation et indexation FAISS."""
    print("\n" + "=" * 70)
    print("🔍 PHASE 2bis : VECTORISATION ET INDEXATION FAISS")
    print("=" * 70)

    rc = run_indexing()

    if rc != 0:
        print("❌ Phase 2bis échouée.")
        sys.exit(1)
    print("✅ Phase 2bis terminée.")


def phase3_plan_interactif():
    """Phase 3 : Génération et validation interactive du plan."""
    print("\n" + "=" * 70)
    print("🎓 PHASE 3 : GÉNÉRATION ET VALIDATION DU PLAN")
    print("=" * 70)

    config = CourseConfig()

    if not config.mistral_api_key:
        print("\n❌ ERREUR : Clé API Mistral non définie.")
        print("   Définissez la variable d'environnement MISTRAL_API_KEY :")
        print("   Windows PowerShell : $env:MISTRAL_API_KEY='votre_cle'")
        print("   Windows CMD        : set MISTRAL_API_KEY=votre_cle")
        sys.exit(1)

    client = MistralClient(
        api_key=config.mistral_api_key,
        model=config.mistral_model,
        min_delay=config.min_delay_seconds,
    )

    planner = PlanGenerator(client)
    plan_path = config.output_dir / "plan.json"

    if plan_path.exists():
        print(f"\nℹ️  Un plan existe déjà : {plan_path}")
        if ask_yes_no("Voulez-vous le reprendre et le modifier ?", default=True):
            with open(plan_path, "r", encoding="utf-8") as f:
                initial_plan = json.load(f)
        else:
            print("\n🤖 Génération d'un nouveau plan initial par Mistral...")
            initial_plan = planner.generate_plan(config.hierarchy_summary_path)
    else:
        print("\n🤖 Génération du plan initial par Mistral...")
        initial_plan = planner.generate_plan(config.hierarchy_summary_path)

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(initial_plan, f, ensure_ascii=False, indent=2)

    validated_plan = interactive_plan_loop(client, initial_plan, plan_path)
    if validated_plan is None:
        print("\n❌ Génération annulée par l'utilisateur.")
        sys.exit(0)

    return validated_plan


def phase4_generation_cours():
    """Phase 4 : Génération du cours complet."""
    print("\n" + "=" * 70)
    print("✍️  PHASE 4 : GÉNÉRATION DU COURS COMPLET")
    print("=" * 70)

    config = CourseConfig()
    orchestrator = CourseOrchestrator(config)
    orchestrator.run()

    print("\n✅ Phase 4 terminée.")
    out_dir = config.output_dir
    print(f"\n📁 Fichiers générés dans : {out_dir}")
    print(f"   • {out_dir / 'course.md'}")
    print(f"   • {out_dir / 'course.json'}")
    print(f"   • {out_dir / 'plan.json'}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("🚀 PIPELINE RAG COMPLET - MODE INTERACTIF")
    print("=" * 70)
    print(f"📂 Projet : {PROJECT_ROOT}")
    print(f"📂 RAG     : {RAG_ROOT}")
    print("🔗 Flux    : Extraction+Hiérarchie → Vectorisation → Plan interactif → Cours")
    print("=" * 70)

    output_dir = RAG_ROOT / "data" / "output"
    vector_store = RAG_ROOT / "data" / "vector_store"
    has_chunks = output_dir.exists() and any(output_dir.glob("*/chunks.jsonl"))
    has_index = (vector_store / "faiss.index").exists() and (vector_store / "metadata.jsonl").exists()
    has_summary = (output_dir / "hierarchy_summary.json").exists()

    # Phase 1 : Extraction hiérarchique (hiarchy_summary.py)
    if has_summary and has_chunks:
        print(f"\nℹ️  Des données extraites existent déjà.")
        if not ask_yes_no("Re-lancer l'extraction hiérarchique ?", default=False):
            print("   → Phase 1 ignorée.")
        else:
            phase1_extraction()
    else:
        phase1_extraction()

    # Phase 2bis : Vectorisation
    if has_index:
        print(f"\nℹ️  Un index FAISS existe déjà.")
        if not ask_yes_no("Re-construire l'index vectoriel ?", default=False):
            print("   → Phase 2bis ignorée.")
        else:
            phase2bis_vectorization()
    else:
        phase2bis_vectorization()

    # Phase 3 : Plan interactif (toujours exécutée)
    phase3_plan_interactif()

    # Phase 4 : Génération du cours
    phase4_generation_cours()

    print("\n" + "=" * 70)
    print("🎉 TOUT EST TERMINÉ !")
    print("=" * 70)


if __name__ == "__main__":
    main()