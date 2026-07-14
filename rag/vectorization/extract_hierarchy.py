"""
Script d'extraction de hiérarchie PDF - Multi-documents
Stocke toutes les hiérarchies dans un JSON et un MD unifiés.

Usage:
    python extract_hierarchy.py

Configuration:
    Modifie la variable PDF_PATHS ci-dessous avec tes chemins PDF.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, List, Dict, Optional, Set, Tuple
from collections import defaultdict

import fitz  # PyMuPDF
import numpy as np
from sklearn.cluster import KMeans


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION - MODIFIE CES LIGNES
# ═══════════════════════════════════════════════════════════════════════════
PDF_PATHS: List[str] = [
    # r"C:\Users\dell\Desktop\Projet python - RAG\mon_doc1.pdf",
    # r"C:\Users\dell\Desktop\Projet python - RAG\mon_doc2.pdf",
]

OUTPUT_DIR: str = r"C:\Users\dell\Desktop\Projet python - RAG\output_hierarchies"
VERBOSE: bool = True


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════

TOC_KEYWORDS = [
    "table of contents", "contents", "sommaire",
    "table des matieres", "table des matières",
    "índice", "indice", "contenido", "inhalt",
]

NUMBERING_PATTERNS = [
    re.compile(r"^\d+[.:]\s+"),
    re.compile(r"^\d+\.\d+[.:]\s+"),
    re.compile(r"^\d+\.\d+\.\d+[.:]\s+"),
    re.compile(r"^[IVX]+[.:]\s+", re.IGNORECASE),
    re.compile(r"^[A-Z][.:]\s+"),
    re.compile(r"^\([a-z\d]+\)\s+"),
]

CHAPTER_PATTERNS = [
    re.compile(r"^(?:chapter|chapitre|part|partie|section|unit|module|leçon|lesson)\s*[IVX\d]+[.:—\-\s]", re.IGNORECASE),
    re.compile(r"^(?:chapter|chapitre|part|partie)\s+[A-ZÀ-ÿ]", re.IGNORECASE),
]


# ═══════════════════════════════════════════════════════════════════════════
# MODELES DE DONNEES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Heading:
    """Titre detecte dans un document."""
    level: int
    title: str
    page: int
    font_size: float
    is_bold: bool
    is_uppercase: bool
    source: str
    confidence: float = 0.75


@dataclass
class TOCItem:
    """Element de table des matieres hierarchique."""
    level: int
    title: str
    page: int
    children: List["TOCItem"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "title": self.title,
            "page": self.page,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class DocumentHierarchy:
    """Hierarchie complete d'un document."""
    filename: str
    page_count: int
    body_font_size: float
    headings_count: int
    sources: Dict[str, int]
    flat_headings: List[Dict[str, Any]]
    hierarchy: List[Dict[str, Any]]
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "page_count": self.page_count,
            "body_font_size": self.body_font_size,
            "headings_count": self.headings_count,
            "sources": self.sources,
            "flat_headings": self.flat_headings,
            "hierarchy": self.hierarchy,
            "extracted_at": self.extracted_at,
        }


# ═══════════════════════════════════════════════════════════════════════════
# EXTRACTEUR DE HIERARCHIE
# ═══════════════════════════════════════════════════════════════════════════

class HierarchyExtractor:
    """Extrait la hierarchie d'un PDF par multi-strategies."""

    def __init__(self, pdf_path: str | Path, verbose: bool = False):
        self.pdf_path = Path(pdf_path)
        self.doc: fitz.Document = fitz.open(str(pdf_path))
        self.verbose = verbose
        self.body_font_size: float = 11.0
        self.page_count: int = len(self.doc)
        self._span_cache: List[Dict] = []

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [Extractor] {msg}")

    # --- 1. TOC Native ---
    def _extract_native_toc(self) -> List[Heading]:
        toc = self.doc.get_toc()
        results = []
        for level, title, page in toc:
            results.append(Heading(
                level=level, title=title.strip(), page=max(0, page - 1),
                font_size=0.0, is_bold=False, is_uppercase=title.isupper(),
                source="native_toc", confidence=0.95,
            ))
        self._log(f"TOC native: {len(results)} entrees")
        return results

    # --- 2. TOC Visuelle ---
    def _is_toc_page(self, page: fitz.Page) -> bool:
        text = page.get_text().lower()
        has_kw = any(kw in text for kw in TOC_KEYWORDS)
        if not has_kw:
            return False
        lines = [l.strip() for l in page.get_text().split("\n") if l.strip()]
        if len(lines) < 5:
            return False
        page_num_lines = sum(1 for l in lines if re.search(r"\d+\s*$", l))
        return (page_num_lines / len(lines)) > 0.25

    def _parse_toc_page(self, page: fitz.Page) -> List[Heading]:
        headings = []
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                line_text = "".join(s["text"] for s in line["spans"]).strip()
                if len(line_text) < 3:
                    continue
                m = re.search(r"(.*?)(\d+)\s*$", line_text)
                if not m:
                    continue
                title_part = m.group(1).strip(" .-")
                try:
                    target_page = int(m.group(2)) - 1
                except ValueError:
                    continue
                if not title_part or len(title_part) < 2:
                    continue
                x0 = line["bbox"][0]
                level = self._infer_level(title_part, x0)
                headings.append(Heading(
                    level=level, title=title_part, page=max(0, min(target_page, self.page_count - 1)),
                    font_size=line["spans"][0]["size"] if line["spans"] else 0.0,
                    is_bold=bool(line["spans"][0]["flags"] & 2**4) if line["spans"] else False,
                    is_uppercase=title_part.isupper(), source="visual_toc", confidence=0.85,
                ))
        return headings

    def _extract_visual_toc(self) -> List[Heading]:
        results = []
        for page_num in range(min(15, self.page_count)):
            page = self.doc[page_num]
            if self._is_toc_page(page):
                self._log(f"Page {page_num + 1} = TOC visuelle")
                results.extend(self._parse_toc_page(page))
                break
        self._log(f"TOC visuelle: {len(results)} entrees")
        return results

    # --- 3. Style Typographique ---
    def _collect_spans(self) -> List[Dict]:
        if self._span_cache:
            return self._span_cache
        spans = []
        for page_num in range(self.page_count):
            blocks = self.doc[page_num].get_text("dict")["blocks"]
            for block in blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        spans.append({
                            "text": span["text"], "size": span["size"], "flags": span["flags"],
                            "page": page_num, "y": span["bbox"][1], "x": span["bbox"][0],
                        })
        self._span_cache = spans
        return spans

    def _estimate_body_size(self) -> float:
        spans = self._collect_spans()
        if not spans:
            return 11.0
        sizes = [s["size"] for s in spans]
        try:
            body = statistics.mode(sizes)
        except statistics.StatisticsError:
            body = statistics.median(sizes)
        self._log(f"Taille corps: {body:.2f}pt")
        return body

    def _is_heading(self, text: str, size: float, flags: int, body: float) -> bool:
        text = text.strip()
        if not text or len(text) < 3 or len(text) > 200:
            return False
        is_bold = bool(flags & 2**4)
        is_upper = text.isupper()
        has_num = any(p.match(text) for p in NUMBERING_PATTERNS)
        has_chap = any(p.match(text) for p in CHAPTER_PATTERNS)
        ratio = size / body if body > 0 else 1.0
        if has_chap:
            return True
        if ratio >= 1.3:
            return True
        if ratio >= 1.15 and (is_bold or is_upper or has_num):
            return True
        if is_bold and has_num:
            return True
        return False

    def _infer_level(self, text: str, x: float = 0.0) -> int:
        text = text.strip()
        if any(p.match(text) for p in CHAPTER_PATTERNS):
            return 1
        if re.match(r"^\d+\.\d+\.\d+[.:]\s+", text):
            return 3
        if re.match(r"^\d+\.\d+[.:]\s+", text):
            return 2
        if re.match(r"^\d+[.:]\s+", text):
            return 1
        if re.match(r"^[A-Z][.:]\s+", text):
            return 2
        if re.match(r"^[IVX]+[.:]\s+", text, re.IGNORECASE):
            return 1
        if x > 80:
            return 3
        if x > 40:
            return 2
        return 1

    def _cluster_levels(self, candidates: List[Heading]) -> List[Heading]:
        if not candidates:
            return candidates
        sizes = np.array([[h.font_size] for h in candidates if h.font_size > 0])
        if len(sizes) < 2:
            for h in candidates:
                h.level = self._infer_level(h.title, 0)
            return candidates
        n_clusters = min(3, len(set(sizes.flatten())))
        n_clusters = max(2, min(n_clusters, len(candidates)))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(sizes)
        means = {i: np.mean(sizes[labels == i]) for i in range(n_clusters)}
        sorted_c = sorted(means.items(), key=lambda x: x[1], reverse=True)
        c_to_l = {c[0]: i + 1 for i, c in enumerate(sorted_c)}
        idx = 0
        for h in candidates:
            if h.font_size > 0:
                h.level = c_to_l[labels[idx]]
                idx += 1
            else:
                h.level = self._infer_level(h.title, 0)
        return candidates

    def _extract_by_style(self) -> List[Heading]:
        self.body_font_size = self._estimate_body_size()
        candidates = []
        seen: Set[Tuple[str, int]] = set()
        for page_num in range(self.page_count):
            blocks = self.doc[page_num].get_text("dict")["blocks"]
            for block in blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    line_text = "".join(s["text"] for s in line["spans"]).strip()
                    if not line_text:
                        continue
                    dom = max(line["spans"], key=lambda s: (s["size"], bool(s["flags"] & 2**4)))
                    size, flags = dom["size"], dom["flags"]
                    if self._is_heading(line_text, size, flags, self.body_font_size):
                        key = (line_text, page_num)
                        if key in seen:
                            continue
                        seen.add(key)
                        candidates.append(Heading(
                            level=self._infer_level(line_text, line["bbox"][0]),
                            title=line_text, page=page_num, font_size=size,
                            is_bold=bool(flags & 2**4), is_uppercase=line_text.isupper(),
                            source="style", confidence=0.75,
                        ))
        candidates = self._cluster_levels(candidates)
        self._log(f"Titres par style: {len(candidates)}")
        return candidates

    # --- 4. Fusion & Arbre ---
    def _merge(self, native: List[Heading], visual: List[Heading], style: List[Heading]) -> List[Heading]:
        all_h = []
        seen: Set[Tuple[str, int]] = set()
        for source, prio in [(native, 3), (visual, 2), (style, 1)]:
            for h in source:
                key = (h.title, h.page)
                if key not in seen:
                    seen.add(key)
                    h.confidence = h.confidence * (prio / 3)
                    all_h.append(h)
        all_h.sort(key=lambda h: (h.page, 0))
        return all_h

    def _build_tree(self, headings: List[Heading]) -> List[TOCItem]:
        if not headings:
            return []
        levels = sorted(set(h.level for h in headings))
        lmap = {old: new for new, old in enumerate(levels, start=1)}
        for h in headings:
            h.level = lmap.get(h.level, h.level)
        root: List[TOCItem] = []
        stack: List[TOCItem] = []
        for h in headings:
            item = TOCItem(level=h.level, title=h.title, page=h.page)
            if not stack:
                root.append(item)
                stack.append(item)
                continue
            while stack and stack[-1].level >= h.level:
                stack.pop()
            if stack:
                stack[-1].children.append(item)
            else:
                root.append(item)
            stack.append(item)
        return root

    def extract(self) -> DocumentHierarchy:
        self._log(f"Analyse: {self.pdf_path.name} ({self.page_count} pages)")
        native = self._extract_native_toc()
        visual = self._extract_visual_toc()
        style = self._extract_by_style()
        merged = self._merge(native, visual, style)
        tree = self._build_tree(merged)
        return DocumentHierarchy(
            filename=self.pdf_path.name,
            page_count=self.page_count,
            body_font_size=self.body_font_size,
            headings_count=len(merged),
            sources={"native_toc": len(native), "visual_toc": len(visual), "style_detection": len(style)},
            flat_headings=[{
                "level": h.level, "title": h.title, "page": h.page + 1,
                "font_size": round(h.font_size, 2), "is_bold": h.is_bold,
                "is_uppercase": h.is_uppercase, "source": h.source,
                "confidence": round(h.confidence, 2),
            } for h in merged],
            hierarchy=[h.to_dict() for h in tree],
        )

    def close(self):
        self.doc.close()


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT JSON + MD
# ═══════════════════════════════════════════════════════════════════════════

def export_hierarchies(
    hierarchies: List[DocumentHierarchy],
    output_dir: str | Path,
    verbose: bool = True,
) -> Tuple[Path, Path]:
    """Exporte toutes les hierarchies dans un JSON et un MD unifies."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- JSON ---
    data = {
        "extraction_date": datetime.now().isoformat(),
        "total_documents": len(hierarchies),
        "documents": [h.to_dict() for h in hierarchies],
    }
    json_path = output_dir / "hierarchies.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"\n  ✅ JSON: {json_path}")

    # --- Markdown ---
    md_path = output_dir / "hierarchies.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Hierarchies de Documents\n\n")
        f.write(f"**Date d'extraction:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Nombre de documents:** {len(hierarchies)}\n\n")
        f.write("---\n\n")

        for i, doc in enumerate(hierarchies, 1):
            f.write(f"## {i}. {doc.filename}\n\n")
            f.write(f"- **Pages:** {doc.page_count}\n")
            f.write(f"- **Taille corps:** {doc.body_font_size:.2f}pt\n")
            f.write(f"- **Titres detectes:** {doc.headings_count}\n")
            f.write(f"- **Sources:** Native={doc.sources['native_toc']}, "
                    f"Visuelle={doc.sources['visual_toc']}, "
                    f"Style={doc.sources['style_detection']}\n\n")

            def write_tree(items: List[Dict], depth: int = 0):
                for item in items:
                    indent = "  " * depth
                    f.write(f"{indent}- **{item['title']}** (p. {item['page'] + 1})\n")
                    if item.get("children"):
                        write_tree(item["children"], depth + 1)

            f.write("### Structure hierarchique\n\n")
            if doc.hierarchy:
                write_tree(doc.hierarchy)
            else:
                f.write("*Aucune structure detectee*\n")

            f.write("\n---\n\n")

    if verbose:
        print(f"  ✅ Markdown: {md_path}")

    return json_path, md_path


# ═══════════════════════════════════════════════════════════════════════════
# POINT D'ENTREE
# ═══════════════════════════════════════════════════════════════════════════

def process_pdfs(pdf_paths: List[str], output_dir: str, verbose: bool = True) -> List[DocumentHierarchy]:
    """Traite plusieurs PDFs et retourne leurs hierarchies."""
    hierarchies = []
    for path in pdf_paths:
        path = Path(path)
        if not path.exists():
            print(f"  ⚠️ Fichier non trouve: {path}")
            continue
        extractor = HierarchyExtractor(path, verbose=verbose)
        try:
            hierarchy = extractor.extract()
            hierarchies.append(hierarchy)
            print(f"  ✅ {path.name}: {hierarchy.headings_count} titres")
        finally:
            extractor.close()
    return hierarchies


def main():
    print("=" * 60)
    print("Extraction de Hierarchies PDF - Multi-documents")
    print("=" * 60)

    if not PDF_PATHS:
        print("\n⚠️ Aucun PDF configure!")
        print("Modifie la variable PDF_PATHS en haut du script.")
        return

    print(f"\n📁 Dossier de sortie: {OUTPUT_DIR}")
    print(f"📄 Documents a traiter: {len(PDF_PATHS)}\n")

    hierarchies = process_pdfs(PDF_PATHS, OUTPUT_DIR, verbose=VERBOSE)

    if hierarchies:
        json_path, md_path = export_hierarchies(hierarchies, OUTPUT_DIR, verbose=VERBOSE)
        print(f"\n{'=' * 60}")
        print("Extraction terminee!")
        print(f"{'=' * 60}")
        print(f"\n📊 Resume:")
        for h in hierarchies:
            print(f"   • {h.filename}: {h.headings_count} titres, {h.page_count} pages")
    else:
        print("\n❌ Aucun document traite.")


if __name__ == "__main__":
    main()
