"""
Extraction de la carte hierarchique brute pour le LLM.
Lit chunks.jsonl et produit un squelette epure : chapitres > titres > sous-titres.

Usage:
    python hierarchy_for_llm.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
CHUNKS_FILE: str = r"rag\data\input\chunks.jsonl"
OUTPUT_DIR: str = r"C:\Users\dell\Desktop\Projet python - RAG\rag\data\output"


# ═══════════════════════════════════════════════════════════════════════════
# MODELES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HierarchyNode:
    """Noeud epure de la hierarchie."""
    title: str
    level: int
    page: int
    children: List["HierarchyNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {"title": self.title, "level": self.level, "page": self.page}
        if self.children:
            result["children"] = [c.to_dict() for c in self.children]
        return result


@dataclass  
class DocumentMap:
    """Carte hierarchique epuree d'un document."""
    source_id: str
    title: str
    total_pages: int
    structure: List[HierarchyNode]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_title": self.title,
            "source_id": self.source_id,
            "total_pages": self.total_pages,
            "structure": [n.to_dict() for n in self.structure],
        }


# ═══════════════════════════════════════════════════════════════════════════
# LECTURE
# ═══════════════════════════════════════════════════════════════════════════

def load_chunks(filepath: str) -> List[Dict[str, Any]]:
    chunks = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# CONSTRUCTION DE LA HIERARCHIE EPUREE
# ═══════════════════════════════════════════════════════════════════════════

def extract_unique_headings(chunks: List[Dict]) -> List[Dict]:
    """
    Extrait les headings uniques depuis les chunks.
    Gere aussi les documents sans hierarchie detectee.
    """
    seen: Set[Tuple[str, ...]] = set()
    headings: List[Dict] = []
    has_hierarchy = False

    for chunk in chunks:
        if chunk.get("rejected", False):
            continue

        hierarchy_path = tuple(chunk.get("hierarchy_path", []))

        # Si hierarchy_path contient quelque chose de significatif
        if hierarchy_path and hierarchy_path != ("(document entier)",) and hierarchy_path != ("(sans titre)",):
            has_hierarchy = True

        if hierarchy_path in seen:
            continue
        seen.add(hierarchy_path)

        # Si hierarchy_path est vide ou generique, utiliser titre_section
        if not hierarchy_path or hierarchy_path == ("(document entier)",) or hierarchy_path == ("(sans titre)",):
            title = chunk.get("titre_section", "Sans titre")
            if title and title not in ("(document entier)", "(sans titre)"):
                headings.append({
                    "path": (title,),
                    "level": 1,
                    "page": chunk.get("page", 1),
                })
        else:
            headings.append({
                "path": hierarchy_path,
                "level": chunk.get("hierarchy_level", 1),
                "page": chunk.get("page", 1),
            })

    # Si aucun heading avec vraie hierarchie, retourner ce qu'on a
    headings.sort(key=lambda h: (h["page"], h["level"]))
    return headings


def build_tree(headings: List[Dict]) -> List[HierarchyNode]:
    """Construit l'arbre hierarchique depuis les chemins plats."""
    if not headings:
        return []

    nodes: Dict[Tuple[str, ...], HierarchyNode] = {}
    roots: List[HierarchyNode] = []

    for heading in headings:
        path = heading["path"]
        if not path:
            continue

        for i in range(len(path)):
            sub_path = path[:i + 1]
            title = path[i]
            level = i + 1

            if sub_path not in nodes:
                node = HierarchyNode(title=title, level=level, page=heading["page"])
                nodes[sub_path] = node

                if i == 0:
                    # Verifier si ce root existe deja
                    existing = [r for r in roots if r.title == title]
                    if not existing:
                        roots.append(node)
                else:
                    parent_path = sub_path[:-1]
                    if parent_path in nodes:
                        parent = nodes[parent_path]
                        if not any(c.title == title for c in parent.children):
                            parent.children.append(node)

    return roots


def estimate_total_pages(chunks: List[Dict]) -> int:
    """Estime le nombre total de pages depuis les chunks."""
    pages = set()
    for chunk in chunks:
        pages.add(chunk.get("page", 1))
    return max(pages) if pages else 1


def extract_document_map(source_id: str, chunks: List[Dict]) -> DocumentMap:
    """Construit la carte hierarchique epuree d'un document."""
    headings = extract_unique_headings(chunks)
    tree = build_tree(headings)
    total_pages = estimate_total_pages(chunks)

    # Titre du document
    title = "Document"
    if tree:
        title = tree[0].title
    elif chunks:
        # Essayer de trouver un titre dans les chunks
        for chunk in chunks:
            ts = chunk.get("titre_section", "")
            if ts and ts not in ("(document entier)", "(sans titre)"):
                title = ts
                break

    return DocumentMap(
        source_id=source_id,
        title=title,
        total_pages=total_pages,
        structure=tree,
    )


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT EPURE
# ═══════════════════════════════════════════════════════════════════════════

def export_for_llm(maps: List[DocumentMap], output_dir: str) -> Tuple[Path, Path]:
    """Exporte les cartes hierarchiques dans un format optimise pour le LLM."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- JSON epure ---
    data = {
        "role": "carte_hierarchique",
        "description": "Squelette des documents pour orientation de la recherche semantique",
        "generated_at": datetime.now().isoformat(),
        "document_count": len(maps),
        "documents": [m.to_dict() for m in maps],
    }

    json_path = output_dir / "llm_hierarchy_map.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ✅ JSON: {json_path}")

    # --- Markdown epure ---
    md_path = output_dir / "llm_hierarchy_map.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Carte Hierarchique des Documents\n\n")
        f.write("Utilise cette carte pour orienter ta recherche dans la base vectorielle FAISS.\n\n")
        f.write("---\n\n")

        for i, doc in enumerate(maps, 1):
            f.write(f"## Document {i}: {doc.title}\n\n")
            f.write(f"- **Pages:** {doc.total_pages}\n")
            f.write(f"- **ID:** `{doc.source_id}`\n\n")

            def write_node(node: HierarchyNode, depth: int = 0):
                indent = "  " * depth
                marker = "#" * min(node.level + 1, 6)
                f.write(f"{indent}{marker} {node.title} (p.{node.page})\n\n")
                for child in node.children:
                    write_node(child, depth + 1)

            if doc.structure:
                for root in doc.structure:
                    write_node(root)
            else:
                f.write("*Aucune structure detectee - document traite comme bloc unique*\n\n")

            f.write("---\n\n")

    print(f"  ✅ Markdown: {md_path}")
    return json_path, md_path


# ═══════════════════════════════════════════════════════════════════════════
# POINT D'ENTREE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Carte Hierarchique pour LLM")
    print("=" * 60)
    print(f"\n📁 Source: {CHUNKS_FILE}")
    print(f"📁 Sortie: {OUTPUT_DIR}\n")

    try:
        chunks = load_chunks(CHUNKS_FILE)
    except FileNotFoundError:
        print(f"❌ Fichier non trouve: {CHUNKS_FILE}")
        return

    print(f"✅ {len(chunks)} chunks charges\n")

    # Grouper par document
    docs = defaultdict(list)
    for chunk in chunks:
        docs[chunk.get("source_id", "unknown")].append(chunk)

    print(f"📄 {len(docs)} document(s) trouve(s)\n")

    maps: List[DocumentMap] = []
    for source_id, doc_chunks in docs.items():
        doc_map = extract_document_map(source_id, doc_chunks)
        maps.append(doc_map)
        node_count = sum(1 for _ in walk_tree(doc_map.structure))
        print(f"  ✅ {doc_map.title}: {node_count} noeuds, {doc_map.total_pages} pages")

    print()
    json_path, md_path = export_for_llm(maps, OUTPUT_DIR)

    print(f"\n{'=' * 60}")
    print("Termine!")
    print(f"{'=' * 60}")
    print(f"\n📊 Resume:")
    for m in maps:
        print(f"   • {m.title}: {m.total_pages} pages")


def walk_tree(nodes: List[HierarchyNode]):
    for node in nodes:
        yield node
        yield from walk_tree(node.children)


if __name__ == "__main__":
    main()