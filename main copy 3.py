#!/usr/bin/env python3
"""Main unifié : Extraction+Hiérarchie -> Vectorisation -> Plan interactif -> Cours -> TD."""

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
# CHARGEMENT DYNAMIQUE DES MODULES
# ---------------------------------------------------------------------
def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger {module_name} depuis {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Phase 1 : extraction hiérarchique multi-PDF
_reterevel = load_module_from_path("reterevel", PROJECT_ROOT / "reterevel.py")
run_extraction = _reterevel.run

# Phase 2 : résumé hiérarchique global
_hiarchy_summary = load_module_from_path(
    "hiarchy_summary", RAG_ROOT / "vectorization" / "hiarchy_summary.py"
)
run_summary = _hiarchy_summary.main

# Phase 2bis : vectorisation FAISS
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
    max_pages = plan_data.get('max_pages', 'N/A')
    print(f"📄 PAGES    : {max_pages}")
    print("=" * 70)

    def _recurse(node, depth=0):
        indent = "   " * depth
        lvl = node.get("niveau", 1)
        badge = {1: "📕 CHAPITRE", 2: "📗 SECTION", 3: "📘 SOUS-TITRE"}.get(lvl, "📄 SECTION")
        print(f"{indent}{badge} : {node.get('titre', 'Sans titre')}  (id={node.get('id', 'N/A')})")
        print(f"{indent}    └─ Objectif : {node.get('objectif', 'N/A')}")
        for enfant in node.get("enfants", []):
            _recurse(enfant, depth + 1)

    for node in plan_data.get("plan", []):
        _recurse(node)
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------
# COLLECTE MANUELLE DES CONTRAINTES
# ---------------------------------------------------------------------
def collect_user_constraints() -> dict:
    print("\n" + "=" * 70)
    print("📋 CONFIGURATION DU PLAN DE COURS")
    print("=" * 70)
    print("Laissez un champ vide pour que Mistral le propose automatiquement.")
    print("Seuls le nombre de pages et le nombre de chapitres sont obligatoires.")
    print("=" * 70)

    while True:
        pages_input = input("\n📄 Nombre de pages souhaité (obligatoire, ex: 10) : ").strip()
        if pages_input.isdigit() and int(pages_input) > 0:
            max_pages = int(pages_input)
            break
        print("❌ Veuillez entrer un nombre entier positif.")

    while True:
        ch_input = input("📕 Nombre de chapitres souhaité (obligatoire, ex: 3) : ").strip()
        if ch_input.isdigit() and int(ch_input) > 0:
            num_chapters = int(ch_input)
            break
        print("❌ Veuillez entrer un nombre entier positif.")

    chapters = []
    for i in range(1, num_chapters + 1):
        print(f"\n--- Chapitre {i}/{num_chapters} ---")
        title = input("Titre (laisser vide = auto) : ").strip()
        sections_input = input("Nombre de sections (laisser vide = auto) : ").strip()
        num_sections = int(sections_input) if sections_input.isdigit() else None

        sections = []
        if num_sections:
            for j in range(1, num_sections + 1):
                sec_title = input(f"  Section {j} - Titre (laisser vide = auto) : ").strip()
                sections.append({
                    "id": f"ch{i}_sec{j}",
                    "titre": sec_title,
                    "niveau": 2,
                    "objectif": "",
                    "enfants": []
                })

        chapters.append({
            "id": f"ch{i}",
            "titre": title,
            "niveau": 1,
            "objectif": "",
            "enfants": sections
        })

    return {
        "max_pages": max_pages,
        "num_chapters": num_chapters,
        "chapters": chapters
    }


# ---------------------------------------------------------------------
# PHASE 3 : BOUCLE INTERACTIVE
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
    print("  • Tapez une instruction pour modifier le plan")
    print("  • 'valide'  ou 'je valide'  → confirmer et passer à la génération")
    print("  • 'affiche' ou 'voir'       → réafficher le plan actuel")
    print("  • 'reset'  ou 'reinit'      → revenir au plan initial")
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
# PHASES 1-4
# ---------------------------------------------------------------------
def phase1_extraction():
    pdf_dir = RAG_ROOT / "data" / "input"
    output_dir = RAG_ROOT / "data" / "output"

    if not pdf_dir.exists() or not any(pdf_dir.glob("*.pdf")):
        print(f"\n❌ ERREUR : Aucun PDF trouvé dans {pdf_dir}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("📄 PHASE 1 : EXTRACTION HIÉRARCHIQUE DES PDFs")
    print("=" * 70)

    original_cwd = os.getcwd()
    try:
        os.chdir(str(PROJECT_ROOT))
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

    print("✅ Phase 1 terminée.")


def phase2_summary():
    print("\n" + "=" * 70)
    print("📚 PHASE 2 : RÉSUMÉ HIÉRARCHIQUE GLOBAL")
    print("=" * 70)

    original_cwd = os.getcwd()
    try:
        os.chdir(str(PROJECT_ROOT))
        rc = run_summary()
    finally:
        os.chdir(original_cwd)

    if rc != 0:
        print("❌ Phase 2 échouée.")
        sys.exit(1)
    print("✅ Phase 2 terminée.")


def phase2bis_vectorization():
    print("\n" + "=" * 70)
    print("🔍 PHASE 2bis : VECTORISATION ET INDEXATION FAISS")
    print("=" * 70)

    rc = run_indexing()
    if rc != 0:
        print("❌ Phase 2bis échouée.")
        sys.exit(1)
    print("✅ Phase 2bis terminée.")


def phase3_plan_interactif():
    print("\n" + "=" * 70)
    print("🎓 PHASE 3 : CONFIGURATION ET SUGGESTION DU PLAN")
    print("=" * 70)

    config = CourseConfig()
    if not config.mistral_api_key:
        print("\n❌ ERREUR : Clé API Mistral non définie.")
        sys.exit(1)

    client = MistralClient(
        api_key=config.mistral_api_key,
        model=config.mistral_model,
        min_delay=config.min_delay_seconds,
    )

    plan_path = config.output_dir / "plan.json"
    constraints = collect_user_constraints()

    print("\n🤖 Mistral suggère un plan adapté à vos contraintes...")
    planner = PlanGenerator(client)

    initial_plan = planner.generate_plan_with_constraints(
        structure_path=config.hierarchy_summary_path,
        max_pages=constraints["max_pages"],
        num_chapters=constraints["num_chapters"],
        user_chapters=constraints["chapters"]
    )

    initial_plan["max_pages"] = constraints["max_pages"]

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(initial_plan, f, ensure_ascii=False, indent=2)

    validated_plan = interactive_plan_loop(client, initial_plan, plan_path)
    if validated_plan is None:
        print("\n❌ Génération annulée.")
        sys.exit(0)

    return validated_plan


def phase4_generation_cours():
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
# PHASE 5 : GÉNÉRATION DU TD (2 pages max)
# ---------------------------------------------------------------------
def phase5_generation_td():
    """Génère un TD de révision à partir du cours final (course.json)."""
    print("\n" + "=" * 70)
    print("📝 PHASE 5 : GÉNÉRATION DU TD")
    print("=" * 70)

    config = CourseConfig()
    client = MistralClient(
        api_key=config.mistral_api_key,
        model=config.mistral_model,
        min_delay=config.min_delay_seconds,
    )

    # --- Chargement du cours généré (notre "vecteur" de connaissances) ---
    course_json_path = config.output_dir / "course.json"
    if not course_json_path.exists():
        print("❌ ERREUR : course.json introuvable. Générez d'abord le cours.")
        return

    with open(course_json_path, "r", encoding="utf-8") as f:
        course_data = json.load(f)

    # Extraction récursive du contenu pour constituer le contexte
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

    # Tronqué pour rester dans la fenêtre de contexte (~12k chars)
    contexte_cours = "\n\n---\n\n".join(sections_text)[:12000]
    titre_cours = course_data.get("titre_cours", "Cours")

    # --- Prompt TD ---
    prompt = f"""Tu es un professeur universitaire expert en pédagogie. À partir du cours suivant, crée un TD (Travail Dirigé) de révision.

=== CONTEXTE DU COURS ===
Titre : {titre_cours}

{contexte_cours}

=== CONTRAINTES ABSOLUES ===
- Le TD doit faire MAXIMUM 2 pages A4 (police 11pt, interligne 1.5), soit ENVIRON 500 MOTS MAXIMUM.
- Mélange d'exercices variés :
  • 3 questions de type QCM (4 propositions dont une seule correcte)
  • 2 questions courtes (réponse en 2-3 lignes max)
  • 1 exercice pratique / cas d'application
- Les exercices doivent porter sur les points ESSENTIELS et les concepts clés du cours.
- Niveau adapté à un examen de révision.
- Format Markdown pur. AUCUN LaTeX structurel.
- Présente d'abord tous les énoncés, puis un corrigé succinct à la fin.

=== FORMAT ATTENDU ===
# TD : {titre_cours}

## Partie 1 - QCM
...

## Partie 2 - Questions courtes
...

## Partie 3 - Exercice pratique
...

## Corrigé succinct
...

Compte tes mots. MAXIMUM 500 mots pour l'ensemble. Si tu dépasses, supprime une question ou compresse.
Génère le TD maintenant."""

    messages = [
        {
            "role": "system",
            "content": (
                "Tu es un assistant pédagogique qui crée des TD concis, variés "
                "et pertinents en Markdown pur. Tu respectes strictement les limites de mots."
            )
        },
        {"role": "user", "content": prompt}
    ]

    print("🤖 Génération du TD par Mistral...")
    try:
        td_content = client.chat(messages, temperature=0.4)
    except Exception as e:
        print(f"❌ ERREUR lors de la génération du TD : {e}")
        return

    # Nettoyage des balises de code accidentelles
    td_content = td_content.strip()
    if td_content.startswith("```markdown"):
        td_content = td_content[len("```markdown"):].strip()
    if td_content.startswith("```"):
        td_content = td_content[3:].strip()
    if td_content.endswith("```"):
        td_content = td_content[:-3].strip()

    # Sauvegarde
    td_path = config.output_dir / "td.md"
    with open(td_path, "w", encoding="utf-8") as f:
        f.write(td_content)

    nb_mots = len(td_content.split())
    print(f"✅ TD généré avec succès (~{nb_mots} mots, budget: 500 mots)")
    print(f"📁 Fichier : {td_path}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("🚀 PIPELINE RAG COMPLET - MODE INTERACTIF")
    print("=" * 70)
    print(f"📂 Projet : {PROJECT_ROOT}")
    print(f"📂 RAG     : {RAG_ROOT}")
    print("🔗 Flux    : Extraction → Hiérarchie → Vectorisation → Plan → Cours → TD")
    print("=" * 70)

    output_dir = RAG_ROOT / "data" / "output"
    vector_store = RAG_ROOT / "data" / "vector_store"
    has_chunks = output_dir.exists() and any(output_dir.glob("*/chunks.jsonl"))
    has_index = (vector_store / "faiss.index").exists() and (vector_store / "metadata.jsonl").exists()
    has_summary = (output_dir / "hierarchy_summary.json").exists()

    # Phase 1
    if has_chunks:
        print(f"\nℹ️  Des chunks existent déjà dans {output_dir}")
        if not ask_yes_no("Re-lancer l'extraction des PDFs ?", default=False):
            print("   → Phase 1 ignorée.")
        else:
            phase1_extraction()
    else:
        phase1_extraction()

    # Phase 2
    if has_summary:
        print(f"\nℹ️  Un résumé hiérarchique existe déjà.")
        if not ask_yes_no("Re-générer le résumé hiérarchique ?", default=False):
            print("   → Phase 2 ignorée.")
        else:
            phase2_summary()
    else:
        phase2_summary()

    # Phase 2bis
    if has_index:
        print(f"\nℹ️  Un index FAISS existe déjà.")
        if not ask_yes_no("Re-construire l'index vectoriel ?", default=False):
            print("   → Phase 2bis ignorée.")
        else:
            phase2bis_vectorization()
    else:
        phase2bis_vectorization()

    # Phase 3 (toujours exécutée)
    phase3_plan_interactif()

    # Phase 4
    phase4_generation_cours()

    # Phase 5 (optionnelle, confirmation utilisateur)
    print("\n" + "=" * 70)
    if ask_yes_no("📝 Voulez-vous générer un TD (2 pages max) basé sur ce cours ?", default=True):
        phase5_generation_td()
    else:
        print("→ Phase 5 (TD) ignorée.")

    print("\n" + "=" * 70)
    print("🎉 TOUT EST TERMINÉ !")
    print("=" * 70)


if __name__ == "__main__":
    main()