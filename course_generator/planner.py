"""Génération du plan de cours par Mistral."""

import json
import re
from pathlib import Path

from .mistral_client import MistralClient


class PlanGenerator:
    """Génère un plan pédagogique structuré à partir du hierarchy_summary."""

    def __init__(self, client: MistralClient) -> None:
        self.client = client

    def load_structure(self, path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def generate_plan(self, structure_path: Path) -> dict:
        structure = self.load_structure(structure_path)
        structure_json = json.dumps(structure, ensure_ascii=False, indent=2)[:8000]

        prompt_lines = [
            "Tu es un professeur universitaire expert en pédagogie. À partir de la structure de documents suivante (extraite de supports de cours), crée un plan de cours pédagogique concis et progressif.",
            "",
            "CONTRAINTES DE VOLUME STRICTES (à respecter absolument) :",
            "- Le cours final doit faire environ 10 pages, MAXIMUM 20 pages A4 (police 11pt, interligne 1.5)",
            "- Maximum 4 chapitres principaux",
            "- Maximum 2 à 3 sections par chapitre (profondeur max : 2 niveaux : chapitre > section)",
            "- Aucun sous-sous-section (niveau 3 interdit)",
            "- Chaque objectif pédagogique doit tenir en UNE phrase courte et actionnable",
            "- Ne couvre que les points ESSENTIELS et PRINCIPAUX du sujet. Pas d'exhaustivité.",
            "- Pas de chapitre de révision, d'annexes, de bibliographie, ni d'introduction générale dans le plan",
            "- Privilégie la synthèse et la clarté à la densité de contenu",
            "",
            "Structure source :",
            structure_json,
            "",
            "Règles :",
            "1. Respecte la hiérarchie logique : Chapitres > Sections uniquement",
            "2. Chaque section doit avoir un objectif pédagogique clair et actionnable",
            "3. Le plan doit être cohérent, progressif et pédagogique",
            "4. Tu peux réorganiser légèrement si la pédagogie l'exige, mais reste fidèle aux points principaux de la source",
            "5. Les titres et objectifs doivent être en TEXTE BRUT lisible. AUCUNE syntaxe LaTeX (interdit : \\section, \\textbf, \\textit, \\begin, etc.). Le contenu du cours sera rédigé en Markdown natif.",
            "6. Retourne UNIQUEMENT un JSON valide, sans markdown, sans explication",
            "",
            'Format exact : {"titre_cours":"Titre","description":"Desc","plan":[{"id":"ch1","titre":"Titre","niveau":1,"objectif":"Obj","enfants":[]}]}',
        ]
        prompt = "\n".join(prompt_lines)

        messages = [
            {
                "role": "system",
                "content": (
                    "Tu es un assistant pédagogique qui retourne uniquement du JSON valide. "
                    "Tu es extrêmement concis : tu privilégies la qualité pédagogique à la quantité de contenu. "
                    "Tu ne génères jamais de plans dépassant 20 pages de cours final. "
                    "Tu ne crées que des plans légers et synthétiques couvrant les points principaux. "
                    "Tous les nœuds du JSON doivent obligatoirement contenir le champ 'enfants' (même si vide). "
                    "Tu n'utilises JAMAIS de syntaxe LaTeX dans les titres ou objectifs. "
                    "Le cours final sera rédigé en Markdown natif."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        print("[PLAN] Génération du plan de cours (cible : ~10 pages)...")
        raw = self.client.chat(messages, temperature=0.3, force_json=True)

        try:
            plan = json.loads(raw)
            if "plan" not in plan:
                raise ValueError("Clé 'plan' manquante")
            self._validate_plan(plan["plan"])
            total_sections = self._count_sections(plan["plan"])
            print(f"[PLAN] OK : {len(plan['plan'])} chapitre(s), {total_sections} section(s) au total")
            return plan
        except json.JSONDecodeError as e:
            print(f"[PLAN] ERREUR JSON : {raw[:500]}")
            raise

    def generate_plan_with_constraints(
        self,
        structure_path: Path,
        max_pages: int,
        num_chapters: int,
        user_chapters: list
    ) -> dict:
        """Génère un plan en respectant strictement le budget de pages et les contraintes utilisateur."""

        structure_json = ""
        if structure_path.exists():
            with open(structure_path, "r", encoding="utf-8") as f:
                structure = json.load(f)
            structure_json = json.dumps(structure, ensure_ascii=False, indent=2)[:6000]

        skeleton = {
            "titre_cours": "",
            "description": "",
            "max_pages": max_pages,
            "plan": user_chapters
        }
        skeleton_json = json.dumps(skeleton, ensure_ascii=False, indent=2)

        mots_par_page = 250
        max_mots_total = max_pages * mots_par_page

        prompt = f"""Tu es un professeur universitaire expert en pédagogie.
Tu dois construire un plan de cours COMPLET et COHERENT à partir du SQUELETTE fourni.

=== CONTRAINTES ABSOLUES (à respecter impérativement) ===
- Le cours final doit faire EXACTEMENT {max_pages} pages A4 (police 11pt, interligne 1.5).
- Cela représente un TOTAL MAXIMUM de {max_mots_total} mots pour l'ensemble du cours.
- Nombre de chapitres : exactement {num_chapters}.
- Profondeur max : 2 niveaux (chapitre > section). AUCUN niveau 3.
- Chaque objectif pédagogique doit tenir en UNE phrase courte et actionnable.
- Ne couvre que les points ESSENTIELS et PRINCIPAUX du sujet.

=== SQUELETTE (champs vides à compléter par toi) ===
{skeleton_json}

=== CONTEXTE SOURCE (extraits des supports de cours) ===
{structure_json}

=== INSTRUCTIONS ===
1. Conserve les titres déjà fournis par l'utilisateur. Ne les modifie JAMAIS.
2. Pour les titres vides, propose des titres pertinents basés sur le contexte source.
3. Si l'utilisateur n'a pas défini de sections pour un chapitre (enfants vides), propose 2 à 3 sections pertinentes.
4. Si l'utilisateur a défini des sections mais laissé les titres vides, complète-les.
5. Rédige un objectif pédagogique clair pour chaque chapitre et chaque section.
6. Calcule la répartition du volume pour respecter STRICTEMENT les {max_pages} pages ({max_mots_total} mots).
   - Un chapitre = ~{max_mots_total // num_chapters} mots max.
   - Chaque section = proportionnelle au nombre de sections dans le chapitre.
7. Retourne UNIQUEMENT un JSON valide, sans markdown, sans explication, sans balises de code.

Format exact : {{"titre_cours":"...","description":"...","max_pages":{max_pages},"plan":[...]}}
Chaque nœud doit contenir : id, titre, niveau, objectif, enfants (liste, même vide)."""

        messages = [
            {
                "role": "system",
                "content": (
                    "Tu es un assistant pédagogique qui retourne uniquement du JSON valide. "
                    "Tu respectes strictement les contraintes de volume et de structure imposées. "
                    "Tu ne modifies jamais les titres déjà fournis par l'utilisateur."
                )
            },
            {"role": "user", "content": prompt}
        ]

        print(f"[PLAN] Suggestion d'un plan pour {max_pages} pages ({max_mots_total} mots)...")
        raw = self.client.chat(messages, temperature=0.3, force_json=True)

        try:
            plan = json.loads(raw)
            if "plan" not in plan:
                raise ValueError("Clé 'plan' manquante")

            if "max_pages" not in plan:
                plan["max_pages"] = max_pages

            self._validate_plan(plan["plan"])
            total_sections = self._count_sections(plan["plan"])
            print(f"[PLAN] OK : {len(plan['plan'])} chapitre(s), {total_sections} section(s) au total")
            return plan
        except json.JSONDecodeError as e:
            print(f"[PLAN] ERREUR JSON : {raw[:500]}")
            raise

    def _count_sections(self, nodes: list) -> int:
        """Compte le nombre total de sections (chapitres + sections)."""
        count = 0
        for node in nodes:
            count += 1
            if "enfants" in node and node["enfants"]:
                count += self._count_sections(node["enfants"])
        return count

    def _validate_plan(self, nodes: list, parent_id: str = "") -> None:
        if len(nodes) > 4:
            raise ValueError(f"Trop de chapitres : {len(nodes)} (max 4)")

        for i, node in enumerate(nodes):
            if "id" not in node:
                node["id"] = f"{parent_id}_sec{i+1}" if parent_id else f"ch{i+1}"
            if "titre" not in node or "objectif" not in node or "niveau" not in node:
                raise ValueError(f"Section invalide : {node}")

            if "enfants" not in node:
                node["enfants"] = []

            if node["niveau"] > 2:
                raise ValueError(
                    f"Niveau trop profond ({node['niveau']}) pour la section {node['id']}. Max : 2."
                )

            if len(node.get("objectif", "")) > 150:
                raise ValueError(
                    f"Objectif trop long pour {node['id']} : max 150 caractères."
                )

            for field in ["titre", "objectif"]:
                value = node.get(field, "")
                if re.search(r"\\[a-zA-Z]+", value):
                    raise ValueError(
                        f"LaTeX détecté dans {field} de {node['id']} : '{value[:50]}...'. "
                        f"Utilise du texte brut uniquement, pas de \\section, \\textbf, etc."
                    )

            if node["enfants"]:
                if len(node["enfants"]) > 3:
                    raise ValueError(
                        f"Trop de sections dans {node['id']} : {len(node['enfants'])} (max 3)"
                    )
                self._validate_plan(node["enfants"], node["id"])