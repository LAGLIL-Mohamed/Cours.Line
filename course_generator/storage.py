"""Stockage incrémental du cours (JSON + MD)."""

import json
from pathlib import Path
from datetime import datetime


class CourseStorage:
    """Gère la persistance incrémentale du cours."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.output_dir / "course.json"
        self.md_path = self.output_dir / "course.md"
        self.progress_path = self.output_dir / "progress.json"
        self.data = self._load_or_init()

    def _load_or_init(self) -> dict:
        if self.json_path.exists():
            with open(self.json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "titre_cours": "",
            "description": "",
            "date_generation": datetime.now().isoformat(),
            "plan": [],
        }

    def set_metadata(self, titre: str, description: str, plan: list = None) -> None:
        self.data["titre_cours"] = titre
        self.data["description"] = description
        if plan is not None:
            if not self.data.get("plan"):
                self.data["plan"] = plan
            else:
                self._merge_plan(self.data["plan"], plan)
        self._save()

    def _merge_plan(self, old_nodes: list, new_nodes: list) -> None:
        """Fusionne le nouveau plan avec l'ancien en préservant le contenu généré."""
        old_by_id = {n.get("id"): n for n in old_nodes}
        for new_node in new_nodes:
            nid = new_node.get("id")
            if nid in old_by_id:
                old = old_by_id[nid]
                old["titre"] = new_node.get("titre", old.get("titre"))
                old["objectif"] = new_node.get("objectif", old.get("objectif"))
                old["niveau"] = new_node.get("niveau", old.get("niveau"))
                if "enfants" in new_node and new_node["enfants"]:
                    if "enfants" not in old:
                        old["enfants"] = []
                    self._merge_plan(old["enfants"], new_node["enfants"])
            else:
                old_nodes.append(new_node)

    def update_section(
        self, section_id: str, questions: list, contenu: str, sources: list
    ) -> None:
        updated = self._update_node_recursive(
            self.data["plan"], section_id, {
                "questions": questions,
                "contenu": contenu,
                "sources_chunks": sources,
            }
        )
        if not updated:
            print(f"  [WARN] Section {section_id} non trouvée dans le plan")
        self._save()

    def _update_node_recursive(
        self, nodes: list, target_id: str, fields: dict
    ) -> bool:
        for node in nodes:
            if node.get("id") == target_id:
                node.update(fields)
                return True
            if "enfants" in node and node["enfants"]:
                if self._update_node_recursive(node["enfants"], target_id, fields):
                    return True
        return False

    def _save(self) -> None:
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        self._rebuild_md()

    def _rebuild_md(self) -> None:
        lines = []
        # YAML frontmatter
        lines.append("---")
        lines.append(f'title: "{self.data.get("titre_cours", "Cours")}"')
        lines.append(f'date: "{datetime.now().isoformat()}"')
        lines.append("---")
        lines.append("")

        if self.data.get("description"):
            lines.append(self.data["description"])
            lines.append("")
            lines.append("---")
            lines.append("")

        for node in self.data.get("plan", []):
            lines.extend(self._render_node_md(node, 0))

        with open(self.md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _render_node_md(self, node: dict, depth: int) -> list[str]:
        lines = []
        titre = node.get("titre", "")
        contenu = node.get("contenu", "")
        objectif = node.get("objectif", "")

        if objectif:
            lines.append(f"*Objectif : {objectif}*")
            lines.append("")

        if contenu:
            lines.append(contenu)
            lines.append("")
        else:
            lines.append(f"**{titre}**")
            lines.append("")
            lines.append("*[Section en attente de génération]*")
            lines.append("")

        for enfant in node.get("enfants", []):
            lines.extend(self._render_node_md(enfant, depth + 1))

        return lines

    def load_progress(self) -> dict:
        if self.progress_path.exists():
            with open(self.progress_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"completed": [], "failed": []}

    def save_progress(self, completed: list, failed: list) -> None:
        with open(self.progress_path, "w", encoding="utf-8") as f:
            json.dump({"completed": completed, "failed": failed}, f, indent=2)