#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_hierarchy_summary.py
===============================================================
Génère un résumé hiérarchique minimaliste de tous les PDF traités.
Lit les hierarchy.json existants dans chaque sous-dossier de OUTPUT_DIR.

CORRECTION : reconstruit la vraie hiérarchie parent-enfant en se
basant sur la numérotation décimale (1 > 1.1 > 1.1.1) et les niveaux
implicites, au lieu de se fier aveuglément au type "chapitre".
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


# =====================================================================
# CONFIG
# =====================================================================

OUTPUT_DIR = Path("rag/data/output")

# Regex pour extraire la numérotation : "1.2.3 Titre" -> ["1","2","3"]
RE_NUMBERING = re.compile(r"^(?:(\d+(?:\.\d+)*)[.\)\s]+|([IVX]+)[.\)\s]+|([A-Z])[.\)\s]+)")


# =====================================================================
# DÉTECTION DU NIVEAU PAR NUMÉROTATION
# =====================================================================

def extract_numbering(title: str) -> tuple[str | None, int]:
    """
    Extrait la numérotation d'un titre et détermine son niveau.
    Retourne (numérotation, niveau) ou (None, 0) si pas de numérotation.
    """
    m = RE_NUMBERING.match(title.strip())
    if not m:
        return None, 0

    num = (m.group(1) or m.group(2) or m.group(3))
    if num is None:
        return None, 0

    # Niveau = nombre de segments dans la numérotation
    if '.' in num:
        level = num.count('.') + 1
    elif num.isdigit():
        level = 1
    elif num.isalpha() and len(num) == 1 and num.isupper():
        # Lettre majuscule isolée (A, B, C...) = niveau 2 (sous-chapitre)
        level = 2
    elif num.isalpha() and len(num) == 1 and num.islower():
        # Lettre minuscule = niveau 3
        level = 3
    elif all(c in 'IVX' for c in num):
        # Chiffres romains = niveau 1
        level = 1
    else:
        level = 1

    return num, level


def infer_level(node: dict, parent_level: int = 0) -> int:
    """
    Déduit le niveau d'un nœud par sa numérotation, ou par rapport au parent.
    """
    title = node.get("titre", "")
    num, level = extract_numbering(title)

    if level > 0:
        return level

    # Pas de numérotation : on regarde les enfants pour déduire
    enfants = node.get("enfants", [])
    if enfants:
        # Si le premier enfant a une numérotation, ce nœud est son parent
        child_num, child_level = extract_numbering(enfants[0].get("titre", ""))
        if child_level > 1:
            return child_level - 1

    # Fallback : niveau du parent + 1
    return parent_level + 1


# =====================================================================
# RECONSTRUCTION DE LA HIÉRARCHIE
# =====================================================================

def build_hierarchy_tree(nodes: list[dict]) -> list[dict]:
    """
    Reconstruit un arbre hiérarchique à partir d'une liste plate de nœuds.
    Utilise la numérotation et les niveaux pour déterminer les parents.
    """
    if not nodes:
        return []

    # D'abord, on détermine le niveau de chaque nœud
    leveled_nodes = []
    for node in nodes:
        level = infer_level(node)
        leveled_nodes.append({**node, "_level": level, "_enfants": []})

    # Construction de l'arbre avec une pile
    root = {"_level": 0, "_enfants": [], "titre": "ROOT"}
    stack = [root]

    for node in leveled_nodes:
        level = node["_level"]

        # Remonter la pile jusqu'à trouver le bon parent
        while stack and stack[-1]["_level"] >= level:
            stack.pop()

        if stack:
            parent = stack[-1]
            parent["_enfants"].append(node)
            stack.append(node)
        else:
            # Orphelin -> racine
            root["_enfants"].append(node)
            stack = [root, node]

    return root["_enfants"]


def parse_hierarchy_file(hierarchy_path: Path) -> dict[str, Any] | None:
    """Lit un hierarchy.json et retourne la structure reconstruite."""
    if not hierarchy_path.exists():
        return None

    with open(hierarchy_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    titre_doc = data.get("titre_document", hierarchy_path.parent.name)
    arbre = data.get("arbre", {})

    # Extraire tous les nœuds de l'arbre source (parcours en profondeur)
    all_nodes = []

    def collect(node: dict):
        type_node = node.get("type", "")
        if type_node != "document":
            all_nodes.append({
                "titre": node.get("titre", ""),
                "type": type_node,
                "page_debut": node.get("page_debut"),
                "source": node.get("source", ""),
            })
        for enfant in node.get("enfants", []):
            collect(enfant)

    collect(arbre)

    # Reconstruire la hiérarchie
    tree = build_hierarchy_tree(all_nodes)

    return {
        "titre_document": titre_doc,
        "structure": tree,
    }


# =====================================================================
# GÉNÉRATION JSON
# =====================================================================

def clean_node_for_json(node: dict) -> dict:
    """Nettoie un nœud pour le JSON (enlève les champs internes)."""
    result = {
        "titre": node.get("titre", ""),
        "type": node.get("type", ""),
        "page_debut": node.get("page_debut"),
        "source": node.get("source", ""),
    }
    enfants = node.get("_enfants", [])
    if enfants:
        result["enfants"] = [clean_node_for_json(e) for e in enfants]
    return result


def build_json_output(documents: list[dict]) -> dict[str, Any]:
    """Construit le JSON final."""
    return {
        "total_documents": len(documents),
        "documents": [
            {
                "titre_document": d["titre_document"],
                "structure": [clean_node_for_json(n) for n in d["structure"]],
            }
            for d in documents
        ],
    }


# =====================================================================
# GÉNÉRATION MD
# =====================================================================

def render_md_node(node: dict, depth: int = 0) -> list[str]:
    """Rend un nœud et ses enfants en Markdown indenté."""
    lines = []
    titre = node.get("titre", "")
    page_debut = node.get("page_debut")
    enfants = node.get("_enfants", [])

    if not titre:
        return lines

    indent = "  " * depth
    page_str = f" (p.{page_debut})" if page_debut else ""

    # Format selon le niveau
    if depth == 0:
        lines.append(f"{indent}- **{titre}{page_str}**")
    elif depth == 1:
        lines.append(f"{indent}- *{titre}{page_str}*")
    else:
        lines.append(f"{indent}- {titre}{page_str}")

    for enfant in enfants:
        lines.extend(render_md_node(enfant, depth + 1))

    return lines


def build_md_output(documents: list[dict]) -> str:
    """Construit le Markdown final avec hiérarchie correcte."""
    lines = []
    lines.append("# Résumé hiérarchique des supports de cours")
    lines.append("")
    lines.append(f"**Nombre de documents** : {len(documents)}")
    lines.append("")

    for i, doc in enumerate(documents, 1):
        titre_doc = doc.get("titre_document", f"Document {i}")
        structure = doc.get("structure", [])

        lines.append(f"## {i}. {titre_doc}")
        lines.append("")

        if not structure:
            lines.append("*Aucune hiérarchie détectée.*")
            lines.append("")
            continue

        for node in structure:
            lines.extend(render_md_node(node, depth=0))
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# =====================================================================
# ORCHESTRATION
# =====================================================================

def main() -> int:
    if not OUTPUT_DIR.exists():
        print(f"Erreur : dossier introuvable : {OUTPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    hierarchy_files = sorted(OUTPUT_DIR.glob("*/hierarchy.json"))

    if not hierarchy_files:
        print(f"Aucun hierarchy.json trouvé dans : {OUTPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Fichiers hierarchy.json trouvés : {len(hierarchy_files)}")
    for hf in hierarchy_files:
        print(f"  -> {hf}")

    documents: list[dict] = []
    for hf in hierarchy_files:
        doc = parse_hierarchy_file(hf)
        if doc:
            documents.append(doc)

    summary_json = build_json_output(documents)
    summary_md = build_md_output(documents)

    json_path = OUTPUT_DIR / "hierarchy_summary.json"
    md_path = OUTPUT_DIR / "hierarchy_summary.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(summary_md)

    print(f"\nFichiers générés avec succès :")
    print(f"  - JSON : {json_path}")
    print(f"  - MD   : {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())