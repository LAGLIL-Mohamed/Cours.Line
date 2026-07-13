"""
PDF Structure Extractor — Pipeline robuste pour extraction de chapitres, titres et sous-titres
Sans dépendance à Docling. Basé sur PyMuPDF + heuristiques + clustering.

Usage direct (hardcodé) :
    # Modifie la variable PDF_PATH ci-dessous, puis :
    python pdf_structure_extractor.py

Usage CLI (optionnel) :
    python pdf_structure_extractor.py mon_fichier.pdf --output resultats/
"""

import fitz  # PyMuPDF
import json
import re
import argparse
import statistics
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple, Set
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION — MODIFIE CETTE LIGNE AVEC TON CHEMIN PDF
# ═══════════════════════════════════════════════════════════════════════════
PDF_PATH: str | None = None  # Ex: r"C:\Users\dell\Desktop\mon_doc.pdf"
OUTPUT_DIR: str = "."
VERBOSE: bool = True


# ---------------------------------------------------------------------------
# Modèles de données
# ---------------------------------------------------------------------------

@dataclass
class Heading:
    """Représente un titre détecté dans le document."""
    level: int
    title: str
    page: int
    font_size: float
    font_flags: int
    is_bold: bool
    is_italic: bool
    is_uppercase: bool
    y_position: float
    x_position: float
    source: str  # "native_toc" | "visual_toc" | "style_detection"
    confidence: float = 1.0


@dataclass
class TOCItem:
    """Élément de table des matières hiérarchique."""
    level: int
    title: str
    page: int
    children: List["TOCItem"] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "level": self.level,
            "title": self.title,
            "page": self.page,
            "children": [c.to_dict() for c in self.children],
        }


# ---------------------------------------------------------------------------
# Constantes & Patterns
# ---------------------------------------------------------------------------

TOC_KEYWORDS = [
    "table of contents", "contents", "sommaire",
    "table des matières", "table des matieres",
    "índice", "indice", "contenido", "inhalt",
]

CHAPTER_PATTERNS = [
    re.compile(r"^(?:chapter|chapitre|part|partie|section|unit|module|leçon|lesson)\s*[IVX\d]+[.:—\-\s]", re.IGNORECASE),
    re.compile(r"^(?:chapter|chapitre|part|partie)\s+[A-ZÀ-ÿ][a-zà-ÿ]*", re.IGNORECASE),
]

NUMBERING_PATTERNS = [
    re.compile(r"^\d+[.:]\s+"),                    # 1.  1:
    re.compile(r"^\d+\.\d+[.:]\s+"),               # 1.1.  1.1:
    re.compile(r"^\d+\.\d+\.\d+[.:]\s+"),        # 1.1.1.
    re.compile(r"^[IVX]+[.:]\s+", re.IGNORECASE),     # I.  II:
    re.compile(r"^[A-Z][.:]\s+"),                     # A.  B:
    re.compile(r"^\([a-z\d]+\)\s+"),               # (a)  (1)
]


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------

class PDFStructureExtractor:
    def __init__(self, pdf_path: str | Path, verbose: bool = False):
        self.pdf_path = Path(pdf_path)
        self.doc: fitz.Document = fitz.open(str(pdf_path))
        self.verbose = verbose
        self.headings: List[Heading] = []
        self.body_font_size: float = 11.0
        self.page_count: int = len(self.doc)
        self._span_cache: List[Dict] = []  # cache de tous les spans pour analyse globale

    def _log(self, msg: str):
        if self.verbose:
            print(f"[PDFExtractor] {msg}")

    # -----------------------------------------------------------------------
    # 1. TOC Native (métadonnées PDF)
    # -----------------------------------------------------------------------
    def extract_native_toc(self) -> List[Heading]:
        """Extrait la table des matières native si le PDF en possède une."""
        toc = self.doc.get_toc()
        results: List[Heading] = []
        for level, title, page in toc:
            # PyMuPDF renvoie des pages 1-based ; on garde 0-based en interne
            results.append(Heading(
                level=level,
                title=title.strip(),
                page=max(0, page - 1),
                font_size=0.0,
                font_flags=0,
                is_bold=False,
                is_italic=False,
                is_uppercase=title.isupper(),
                y_position=0.0,
                x_position=0.0,
                source="native_toc",
                confidence=0.95,
            ))
        self._log(f"TOC native : {len(results)} entrées trouvées")
        return results

    # -----------------------------------------------------------------------
    # 2. TOC Visuelle (page dédiée)
    # -----------------------------------------------------------------------
    def _is_toc_page(self, page: fitz.Page, page_num: int) -> bool:
        """Heuristique : une page de TOC a beaucoup de lignes se terminant par un numéro."""
        text = page.get_text()
        lower_text = text.lower()

        # Vérifier présence d'un mot-clé TOC (sauf si c'est un titre de chapitre "Table of Contents")
        has_keyword = any(kw in lower_text for kw in TOC_KEYWORDS)
        if not has_keyword:
            return False

        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            return False

        # Lignes se terminant par un nombre (numéro de page)
        page_number_lines = sum(
            1 for l in lines if re.search(r"\d+\s*$", l)
        )
        ratio = page_number_lines / len(lines)

        # Au moins 25% des lignes se terminent par un nombre ET plus de 5 lignes
        return ratio > 0.25 and len(lines) > 5

    def _parse_toc_lines(self, page: fitz.Page) -> List[Heading]:
        """Parse les lignes d'une page TOC pour en extraire titre + numéro de page."""
        headings: List[Heading] = []
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                line_text = "".join(span["text"] for span in line["spans"]).strip()
                if not line_text or len(line_text) < 3:
                    continue

                # Détection numéro de page en fin de ligne
                m = re.search(r"(.*?)(\d+)\s*$", line_text)
                if not m:
                    continue

                title_part = m.group(1).strip()
                page_num_str = m.group(2)
                try:
                    target_page = int(page_num_str) - 1  # 0-based
                except ValueError:
                    continue

                if not title_part or len(title_part) < 2:
                    continue

                # Détection du niveau par indentation (x0) ou pattern numéroté
                x0 = line["bbox"][0]
                level = self._infer_level_from_text(title_part, x0)

                headings.append(Heading(
                    level=level,
                    title=title_part,
                    page=max(0, min(target_page, self.page_count - 1)),
                    font_size=line["spans"][0]["size"] if line["spans"] else 0.0,
                    font_flags=line["spans"][0]["flags"] if line["spans"] else 0,
                    is_bold=bool(line["spans"][0]["flags"] & 2 ** 4) if line["spans"] else False,
                    is_italic=bool(line["spans"][0]["flags"] & 2 ** 1) if line["spans"] else False,
                    is_uppercase=title_part.isupper(),
                    y_position=line["bbox"][1],
                    x_position=x0,
                    source="visual_toc",
                    confidence=0.85,
                ))
        return headings

    def extract_visual_toc(self, max_pages: int = 15) -> List[Heading]:
        """Cherche une page de table des matières parmi les premières pages."""
        results: List[Heading] = []
        for page_num in range(min(max_pages, self.page_count)):
            page = self.doc[page_num]
            if self._is_toc_page(page, page_num):
                self._log(f"Page {page_num + 1} identifiée comme TOC visuelle")
                results.extend(self._parse_toc_lines(page))
                # On s'arrête à la première TOC trouvée (ou on pourrait chercher plusieurs)
                break
        self._log(f"TOC visuelle : {len(results)} entrées trouvées")
        return results

    # -----------------------------------------------------------------------
    # 3. Détection par style typographique (page par page)
    # -----------------------------------------------------------------------
    def _collect_all_spans(self) -> List[Dict]:
        """Collecte tous les spans de texte avec leurs propriétés pour analyse globale."""
        if self._span_cache:
            return self._span_cache

        spans: List[Dict] = []
        for page_num in range(self.page_count):
            page = self.doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        spans.append({
                            "text": span["text"],
                            "size": span["size"],
                            "flags": span["flags"],
                            "font": span["font"],
                            "page": page_num,
                            "y": span["bbox"][1],
                            "x": span["bbox"][0],
                            "bbox": span["bbox"],
                        })
        self._span_cache = spans
        return spans

    def _estimate_body_font_size(self) -> float:
        """Estime la taille de police du corps du texte (mode statistique)."""
        spans = self._collect_all_spans()
        if not spans:
            return 11.0
        sizes = [s["size"] for s in spans]
        try:
            body_size = statistics.mode(sizes)
        except statistics.StatisticsError:
            body_size = statistics.median(sizes)
        self._log(f"Taille de police du corps estimée : {body_size:.2f}pt")
        return body_size

    def _is_heading_candidate(self, text: str, size: float, flags: int, body_size: float) -> bool:
        """Détermine si un span est un candidat titre selon plusieurs critères."""
        text = text.strip()
        if not text or len(text) < 3 or len(text) > 200:
            return False

        is_bold = bool(flags & 2 ** 4)
        is_italic = bool(flags & 2 ** 1)
        is_upper = text.isupper()
        has_numbering = any(p.match(text) for p in NUMBERING_PATTERNS)
        has_chapter_keyword = any(p.match(text) for p in CHAPTER_PATTERNS)

        # Critère taille
        size_ratio = size / body_size if body_size > 0 else 1.0
        size_heading = size_ratio >= 1.15

        # Un titre doit être soit plus gros, soit gras, soit numéroté, soit chapitre
        if has_chapter_keyword:
            return True
        if size_heading and (is_bold or is_upper or has_numbering):
            return True
        if size_ratio >= 1.3:
            return True
        if is_bold and has_numbering:
            return True
        if size_ratio >= 1.15 and is_bold:
            return True

        return False

    def _infer_level_from_text(self, text: str, x_position: float = 0.0) -> int:
        """Infère le niveau hiérarchique à partir du pattern de numérotation."""
        text = text.strip()

        # Patterns de niveau 1
        if any(p.match(text) for p in CHAPTER_PATTERNS):
            return 1
        if re.match(r"^\d+[.:]\s+", text) and not re.match(r"^\d+\.", text):
            return 1

        # Niveau par profondeur numérique
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

        # Indentation comme proxy de niveau (si x > 50pt, probablement sous-niveau)
        if x_position > 80:
            return 3
        if x_position > 40:
            return 2
        return 1

    def _cluster_heading_levels(self, candidates: List[Heading]) -> List[Heading]:
        """Utilise KMeans sur les tailles de police pour assigner les niveaux 1-3."""
        if not candidates:
            return candidates

        sizes = np.array([[h.font_size] for h in candidates])
        n_clusters = min(3, len(set(sizes.flatten())))
        if n_clusters < 2:
            n_clusters = 2
        if len(candidates) < n_clusters:
            n_clusters = len(candidates)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(sizes)

        # Associer le cluster le plus petit au niveau le plus élevé (1 = grand titre)
        cluster_means = {i: np.mean(sizes[labels == i]) for i in range(n_clusters)}
        sorted_clusters = sorted(cluster_means.items(), key=lambda x: x[1], reverse=True)
        cluster_to_level = {c[0]: i + 1 for i, c in enumerate(sorted_clusters)}

        for h, lbl in zip(candidates, labels):
            h.level = cluster_to_level[lbl]

        return candidates

    def extract_headings_by_style(self) -> List[Heading]:
        """Parcourt toutes les pages et détecte les titres par analyse typographique."""
        self.body_font_size = self._estimate_body_font_size()
        candidates: List[Heading] = []

        for page_num in range(self.page_count):
            page = self.doc[page_num]
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    line_text = "".join(span["text"] for span in line["spans"]).strip()
                    if not line_text:
                        continue

                    # On prend le span dominant (plus gros ou plus gras) de la ligne
                    dominant_span = max(line["spans"], key=lambda s: (s["size"], bool(s["flags"] & 2 ** 4)))
                    size = dominant_span["size"]
                    flags = dominant_span["flags"]

                    if self._is_heading_candidate(line_text, size, flags, self.body_font_size):
                        level = self._infer_level_from_text(line_text, line["bbox"][0])
                        candidates.append(Heading(
                            level=level,
                            title=line_text,
                            page=page_num,
                            font_size=size,
                            font_flags=flags,
                            is_bold=bool(flags & 2 ** 4),
                            is_italic=bool(flags & 2 ** 1),
                            is_uppercase=line_text.isupper(),
                            y_position=line["bbox"][1],
                            x_position=line["bbox"][0],
                            source="style_detection",
                            confidence=0.75,
                        ))

        # Dédoublonnage simple : si deux titres identiques sur la même page, on garde le premier
        seen: Set[Tuple[str, int]] = set()
        unique_candidates: List[Heading] = []
        for h in candidates:
            key = (h.title, h.page)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(h)

        # Clustering pour affiner les niveaux
        unique_candidates = self._cluster_heading_levels(unique_candidates)
        self._log(f"Titres détectés par style : {len(unique_candidates)}")
        return unique_candidates

    # -----------------------------------------------------------------------
    # 4. Fusion & Hiérarchisation
    # -----------------------------------------------------------------------
    def merge_headings(self, native: List[Heading], visual: List[Heading], style: List[Heading]) -> List[Heading]:
        """Fusionne les trois sources en évitant les doublons et privilégie la meilleure confiance."""
        all_headings: List[Heading] = []
        seen: Set[Tuple[str, int]] = set()

        # Ordre de priorité : native > visual > style
        for source_list, priority in [(native, 3), (visual, 2), (style, 1)]:
            for h in source_list:
                key = (h.title, h.page)
                if key not in seen:
                    seen.add(key)
                    h.confidence = h.confidence * (priority / 3)  # bonus priorité
                    all_headings.append(h)

        # Tri par page puis position Y
        all_headings.sort(key=lambda h: (h.page, h.y_position))
        return all_headings

    def build_hierarchy(self, headings: List[Heading]) -> List[TOCItem]:
        """Construit l'arbre hiérarchique (chapitre > titre > sous-titre)."""
        if not headings:
            return []

        # Normaliser les niveaux : s'assurer qu'ils commencent à 1 et sont contigus
        levels = sorted(set(h.level for h in headings))
        level_map = {old: new for new, old in enumerate(levels, start=1)}
        for h in headings:
            h.level = level_map.get(h.level, h.level)

        # Construire l'arbre
        root: List[TOCItem] = []
        stack: List[TOCItem] = []

        for h in headings:
            item = TOCItem(level=h.level, title=h.title, page=h.page)

            if not stack:
                root.append(item)
                stack.append(item)
                continue

            # Remonter la pile jusqu'à trouver le parent approprié
            while stack and stack[-1].level >= h.level:
                stack.pop()

            if stack:
                stack[-1].children.append(item)
            else:
                root.append(item)

            stack.append(item)

        return root

    # -----------------------------------------------------------------------
    # Pipeline principal
    # -----------------------------------------------------------------------
    def extract(self) -> Dict:
        """Exécute le pipeline complet et retourne un dict structuré."""
        self._log(f"Analyse de : {self.pdf_path.name} ({self.page_count} pages)")

        native = self.extract_native_toc()
        visual = self.extract_visual_toc()
        style = self.extract_headings_by_style()

        merged = self.merge_headings(native, visual, style)
        hierarchy = self.build_hierarchy(merged)

        return {
            "file": str(self.pdf_path.name),
            "page_count": self.page_count,
            "body_font_size": self.body_font_size,
            "headings_count": len(merged),
            "sources": {
                "native_toc": len(native),
                "visual_toc": len(visual),
                "style_detection": len(style),
            },
            "flat_headings": [
                {
                    "level": h.level,
                    "title": h.title,
                    "page": h.page + 1,  # 1-based pour l'affichage
                    "font_size": round(h.font_size, 2),
                    "is_bold": h.is_bold,
                    "is_uppercase": h.is_uppercase,
                    "source": h.source,
                    "confidence": round(h.confidence, 2),
                }
                for h in merged
            ],
            "hierarchy": [h.to_dict() for h in hierarchy],
        }

    def save(self, output_dir: Path, data: Dict):
        """Sauvegarde les résultats en JSON et Markdown."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        base = output_dir / self.pdf_path.stem

        # JSON
        json_path = base.with_suffix(".structure.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self._log(f"JSON sauvegardé : {json_path}")

        # Markdown — on travaille avec les dicts (data["hierarchy"] est déjà sérialisé)
        md_path = base.with_suffix(".structure.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Structure de : {data['file']}\n\n")
            f.write(f"- **Pages** : {data['page_count']}\n")
            f.write(f"- **Taille corps** : {data['body_font_size']:.2f}pt\n")
            f.write(f"- **Titres détectés** : {data['headings_count']}\n")
            f.write(f"- **Sources** : Native={data['sources']['native_toc']}, "
                    f"Visuelle={data['sources']['visual_toc']}, "
                    f"Style={data['sources']['style_detection']}\n\n")
            f.write("---\n\n")

            def write_md(items: List[Dict], depth: int = 0):
                for item in items:
                    indent = "  " * depth
                    f.write(f"{indent}- **{item['title']}** (p. {item['page'] + 1})\n")
                    if item.get("children"):
                        write_md(item["children"], depth + 1)

            write_md(data["hierarchy"])
        self._log(f"Markdown sauvegardé : {md_path}")

        return json_path, md_path

    def close(self):
        self.doc.close()


# ---------------------------------------------------------------------------
# Point d'entrée — utilise la variable PDF_PATH si définie, sinon CLI
# ---------------------------------------------------------------------------

def run(pdf_path: str | None = None, output_dir: str = ".", verbose: bool = True) -> Dict:
    """
    Fonction principale utilisable directement depuis un autre script.

    Args:
        pdf_path: Chemin vers le PDF. Si None, utilise la variable globale PDF_PATH.
        output_dir: Dossier de sortie pour les fichiers JSON/Markdown.
        verbose: Affiche les logs d'exécution.

    Returns:
        Le dictionnaire de résultats structurés.
    """
    path = pdf_path or PDF_PATH
    if not path:
        raise ValueError(
            "Aucun chemin PDF fourni. Définissez la variable PDF_PATH en haut du fichier "
            "ou passez pdf_path=... à la fonction run()."
        )

    extractor = PDFStructureExtractor(path, verbose=verbose)
    try:
        data = extractor.extract()
        extractor.save(Path(output_dir), data)
        print(f"\n✅ Extraction terminée : {data['headings_count']} titres trouvés")
        print(f"   JSON  : {output_dir}/{Path(path).stem}.structure.json")
        print(f"   MD    : {output_dir}/{Path(path).stem}.structure.md")
        return data
    finally:
        extractor.close()


def main():
    """Point d'entrée CLI (optionnel, pour compatibilité terminal)."""
    parser = argparse.ArgumentParser(description="Extrait la structure hiérarchique d'un PDF")
    parser.add_argument("pdf", nargs="?", help="Chemin vers le fichier PDF")
    parser.add_argument("--output", "-o", default=".", help="Dossier de sortie")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mode verbeux")
    args = parser.parse_args()

    pdf = args.pdf or PDF_PATH
    if not pdf:
        parser.error("Spécifiez un PDF en argument ou définissez PDF_PATH dans le script.")

    run(pdf_path=pdf, output_dir=args.output, verbose=args.verbose or VERBOSE)


# ═══════════════════════════════════════════════════════════════════════════
# EXEMPLE D'UTILISATION DIRECTE (hardcodé)
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Option 1 : Hardcoder le chemin ici (décommente et modifie)
    PDF_PATH = r"C:\Users\dell\Desktop\UML_Chatbot\Soutenance\Rapport pfe version finale corrigé chatbot 2025 (1).pdf"
    # run()

    # Option 2 : Lancer depuis le terminal (CLI)
    main()