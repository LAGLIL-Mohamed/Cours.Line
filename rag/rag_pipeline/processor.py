"""Orchestrateur principal du pipeline RAG."""

from __future__ import annotations

import json
from pathlib import Path

from .config import PipelineConfig
from .schemas import Chunk, Document
from .steps import (
    Step0Metadata,
    Step1Extract,
    Step2Clean,
    Step3Filter,
    Step4Structure,
    Step5Chunks,
    Step6Quality,
)


class DocumentProcessor:
    """Enchaîne les 7 étapes de préparation des données PDF."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._init_steps()

    def _init_steps(self) -> None:
        """Instancie chaque étape avec la config partagée."""
        self.step0 = Step0Metadata()
        self.step1 = Step1Extract()
        self.step2 = Step2Clean()
        self.step3 = Step3Filter(self.config)
        self.step4 = Step4Structure(self.config)
        self.step5 = Step5Chunks(self.config)
        self.step6 = Step6Quality(self.config)

    def process(self, pdf_path: Path | str) -> list[Chunk]:
        """Exécute le pipeline complet sur un PDF."""
        path = Path(pdf_path)

        doc: Document = self.step0.run(path)
        pages = self.step1.run(doc)
        pages = self.step2.run(pages)
        pages = self.step3.run(pages)
        sections = self.step4.run(pages)
        chunks = self.step5.run(sections)
        chunks = self.step6.run(chunks)

        return chunks

    def process_valid_only(self, pdf_path: Path | str) -> list[Chunk]:
        """Retourne uniquement les chunks acceptés (non rejetés)."""
        all_chunks = self.process(pdf_path)
        return [c for c in all_chunks if not c.rejected]

    def process_with_outputs(self, pdf_path: Path | str, output_dir: Path | str) -> dict:
        """
        Exécute le pipeline complet et génère tous les fichiers de sortie
        (adapté de ton extract_hierarchy.py).
        """
        path = Path(pdf_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Pipeline
        doc: Document = self.step0.run(path)
        pages = self.step1.run(doc)
        pages = self.step2.run(pages)
        pages = self.step3.run(pages)
        sections = self.step4.run(pages)
        chunks = self.step5.run(sections)
        chunks = self.step6.run(chunks)

        # --- Génération des sorties ---

        # 1. chunks.jsonl
        chunks_file = out_dir / "chunks.jsonl"
        with open(chunks_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        # 2. hierarchy.json (arbre + plan à plat)
        hierarchy_tree = self._build_hierarchy_tree(sections, doc.nom_fichier)
        hierarchy_file = out_dir / "hierarchy.json"
        with open(hierarchy_file, "w", encoding="utf-8") as f:
            json.dump(hierarchy_tree, f, ensure_ascii=False, indent=2)

        # 3. document.md
        md_file = out_dir / "document.md"
        md_content = self._build_markdown(sections, doc.nom_fichier)
        md_file.write_text(md_content, encoding="utf-8")

        # 4. toc_debug.json (si TOC détecté)
        if hasattr(self.step4, 'get_unmatched_toc_entries'):
            unmatched = self.step4.get_unmatched_toc_entries()
            toc_debug = out_dir / "toc_debug.json"
            with open(toc_debug, "w", encoding="utf-8") as f:
                json.dump(
                    [{"title": e.title, "level": e.level, "matched": e.matched} for e in getattr(self.step4, 'toc_entries', [])],
                    f, ensure_ascii=False, indent=2
                )

            # 5. warnings.json
            if unmatched:
                warnings_file = out_dir / "warnings.json"
                with open(warnings_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {"toc_entries_non_retrouvees": [e.title for e in unmatched]},
                        f, ensure_ascii=False, indent=2
                    )

        return {
            "chunks": chunks,
            "output_dir": out_dir,
            "chunks_count": len(chunks),
            "valid_chunks": len([c for c in chunks if not c.rejected]),
        }

    def _build_hierarchy_tree(self, sections: list, doc_title: str) -> dict:
        """Construit l'arbre hiérarchique et le plan à plat."""
        root = {"type": "document", "titre": doc_title, "enfants": []}
        stack = [(0, root)]

        plan_a_plat = []

        for section in sections:
            node = section.heading
            entry = {
                "type": f"niveau_{node.level}",
                "titre": node.text,
                "page_debut": node.page,
                "source": getattr(node, "source", "layout"),
                "enfants": [],
            }

            while stack and stack[-1][0] >= node.level:
                stack.pop()

            stack[-1][1]["enfants"].append(entry)
            stack.append((node.level, entry))

            # Plan à plat
            indent = "  " * (node.level - 1)
            plan_a_plat.append(f"{indent}- [{entry['type']}] {node.text} (p.{node.page})")

        return {
            "titre_document": doc_title,
            "arbre": root,
            "plan_a_plat": plan_a_plat,
        }

    def _build_markdown(self, sections: list, doc_title: str) -> str:
        """Construit le markdown du document."""
        lines = [f"# {doc_title}", ""]

        for section in sections:
            level = min(section.heading.level + 1, 6)
            lines.append(f"{'#' * level} {section.heading.text}")
            lines.append("")

            for para in section.paragraphs:
                lines.append(para.text)
                lines.append("")

        return "\n".join(lines)