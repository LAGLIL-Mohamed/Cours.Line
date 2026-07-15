"""Orchestration complète : plan → questions → RAG → rédaction."""

import json
import time
from pathlib import Path

from .config import CourseConfig
from .mistral_client import MistralClient
from .planner import PlanGenerator
from .question_generator import QuestionGenerator
from .rag_engine import RAGEngine
from .writer import SectionWriter
from .storage import CourseStorage


class CourseOrchestrator:
    """Orchestre la génération complète du cours."""

    def __init__(self, config: CourseConfig = None) -> None:
        self.config = config or CourseConfig()
        self.client = MistralClient(
            api_key=self.config.mistral_api_key,
            model=self.config.mistral_model,
            min_delay=self.config.min_delay_seconds,
        )
        self.planner = PlanGenerator(self.client)
        self.question_gen = QuestionGenerator(self.client)
        self.writer = SectionWriter(self.client)
        self.rag_engine = RAGEngine(
            top_k=self.config.top_k_per_question,
            max_context_tokens=self.config.max_context_tokens,
        )
        self.storage = CourseStorage(self.config.output_dir)
        self.progress = self.storage.load_progress()

    def run(self) -> None:
        print("=" * 60)
        print("GENERATION DE COURS (Mistral + RAG)")
        print("=" * 60)

        # 1. Chargement ou génération du plan
        plan_path = self.config.output_dir / "plan.json"
        if plan_path.exists():
            print("[INIT] Plan existant trouvé, chargement...")
            with open(plan_path, "r", encoding="utf-8") as f:
                plan_data = json.load(f)
        else:
            print("[INIT] Génération du plan...")
            plan_data = self.planner.generate_plan(
                self.config.hierarchy_summary_path
            )
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(plan_data, f, ensure_ascii=False, indent=2)

        # Passer le plan au stockage
        self.storage.set_metadata(
            plan_data["titre_cours"], 
            plan_data["description"], 
            plan_data.get("plan", [])
        )

        # Configuration dynamique du writer selon max_pages du plan
        max_pages = plan_data.get("max_pages")
        if max_pages:
            sections_par_niveau = self._count_sections_by_level(plan_data["plan"])
            self.writer = SectionWriter(
                self.client,
                max_pages=max_pages,
                sections_par_niveau=sections_par_niveau,
                max_retries=self.writer.max_retries,
            )
            print(f"[INIT] Budget de pages configuré : {max_pages} pages")

        # 2. Aplatissement du plan (DFS)
        flat_sections = self._flatten_plan(plan_data["plan"])
        total = len(flat_sections)
        print(f"[INIT] Plan : {total} section(s) à générer\n")

        completed = set(self.progress.get("completed", []))
        failed = set(self.progress.get("failed", []))

        # Suivi du budget global pour ajustement glissant
        max_pages_total = plan_data.get("max_pages", 10)
        mots_total_budget = max_pages_total * self.writer._MOTS_PAR_PAGE
        mots_utilises = 0

        for i, section in enumerate(flat_sections, 1):
            sid = section["id"]
            if sid in completed:
                print(f"[{i}/{total}] {sid} : déjà complété, skip")
                continue

            print(f"\n[{i}/{total}] Traitement : {section['titre']}")

            # Ajustement glissant du budget si on a trop consommé
            sections_restantes = total - len(completed) - len(failed) - (i - 1)
            if sections_restantes > 0 and mots_utilises > 0:
                mots_restants = max(0, mots_total_budget - mots_utilises)
                max_pages_virtuel = max(3, mots_restants / self.writer._MOTS_PAR_PAGE)
                sections_par_niveau_restant = self._count_sections_by_level(flat_sections[i-1:])
                self.writer = SectionWriter(
                    self.client,
                    max_pages=max_pages_virtuel,
                    sections_par_niveau=sections_par_niveau_restant,
                    max_retries=self.writer.max_retries,
                )
                print(f"  [BUDGET] Ajustement : {mots_utilises}/{mots_total_budget} mots utilisés, "
                      f"{max_pages_virtuel:.1f} pages restantes pour {sections_restantes} section(s)")

            try:
                # 2.1 Questions
                questions = self.question_gen.generate(section)
                time.sleep(self.config.min_delay_seconds)

                # 2.2 RAG
                context, sources = self.rag_engine.gather_context(questions)

                # 2.3 Rédaction (Markdown natif)
                content = self.writer.write(section, context)
                mots_utilises += self.writer._compter_mots(content)
                time.sleep(self.config.min_delay_seconds)

                # 2.4 Stockage
                self.storage.update_section(sid, questions, content, sources)
                completed.add(sid)
                self.storage.save_progress(list(completed), list(failed))

                print(f"[{i}/{total}] ✓ Section terminée ({mots_utilises}/{mots_total_budget} mots)")

            except Exception as e:
                print(f"[{i}/{total}] ✗ ERREUR : {e}")
                failed.add(sid)
                self.storage.save_progress(list(completed), list(failed))

        print("\n" + "=" * 60)
        print("GENERATION TERMINEE")
        print(f"  Complétées : {len(completed)}")
        print(f"  Échouées   : {len(failed)}")
        print(f"  JSON : {self.storage.json_path}")
        print(f"  MD   : {self.storage.md_path}")
        print("=" * 60)

    def _flatten_plan(self, nodes: list, result: list = None) -> list[dict]:
        """Parcours en profondeur préfixe."""
        if result is None:
            result = []
        for node in nodes:
            result.append(node)
            if node.get("enfants"):
                self._flatten_plan(node["enfants"], result)
        return result

    def _count_sections_by_level(self, nodes: list) -> dict:
        """Compte les sections par niveau dans une liste de nœuds plats."""
        counts = {1: 0, 2: 0, 3: 0}
        for node in nodes:
            lvl = node.get("niveau", 1)
            counts[lvl] = counts.get(lvl, 0) + 1
        return counts