#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génère un PDF académique structuré à partir d'un cours Markdown ou JSON.

Réécriture complète du générateur original. Corrige notamment :
  - le contenu dupliqué (le fichier était rendu deux fois)
  - le sommaire mal placé et aux numéros de page faux
  - le parseur Markdown artisanal qui confondait les commentaires Python
    ("# ...") avec des titres, cassant la pagination et le sommaire
  - les titres justifiés avec des espaces géants
  - le préfixe "#"/"##" affiché littéralement dans les titres
  - les sauts de page mal placés (veuves/orphelines)
  - le rendu des formules LaTeX en texte brut illisible
  - les listes imbriquées aplaties
  - le chemin Windows codé en dur et le téléchargement de polices non mis en cache

Usage:
    python build_pdf_v2.py --input course.md  --output course.pdf
    python build_pdf_v2.py --input course.json --json --output course.pdf
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import unicodedata
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mistune
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from pypdf import PdfReader, PdfWriter

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


# =============================================================================
# CONFIGURATION
# =============================================================================

COLORS = {
    "primary": (31, 78, 120),
    "accent": (192, 80, 77),
    "text": (30, 30, 30),
    "secondary": (100, 100, 100),
    "light": (240, 240, 240),
    "code_bg": (245, 245, 245),
    "inline_code_bg": (238, 238, 238),
    "border": (200, 200, 200),
    "objectif_bg": (255, 248, 240),
}

MARGIN_LEFT = 20
MARGIN_RIGHT = 20
MARGIN_TOP = 25
MARGIN_BOTTOM = 20
PAGE_W = 210.0
PAGE_H = 297.0
CONTENT_W = PAGE_W - MARGIN_LEFT - MARGIN_RIGHT
LINE_HEIGHT = 5.5

# En dessous de cette marge restante (mm), on force un saut de page avant un
# titre / encadré plutôt que de le laisser "orphelin" en bas de page.
MIN_SPACE_BEFORE_BLOCK = 30
MIN_SPACE_BEFORE_HEADING = 35

FONT_CACHE_DIR = Path.home() / ".cache" / "course_pdf_fonts"
FONT_URLS = {
    "DejaVuSans.ttf": "https://github.com/py-pdf/fpdf2/raw/master/test/fonts/DejaVuSans.ttf",
    "DejaVuSans-Bold.ttf": "https://github.com/py-pdf/fpdf2/raw/master/test/fonts/DejaVuSans-Bold.ttf",
    "DejaVuSans-Oblique.ttf": "https://github.com/py-pdf/fpdf2/raw/master/test/fonts/DejaVuSans-Oblique.ttf",
    "DejaVuSans-BoldOblique.ttf": "https://github.com/py-pdf/fpdf2/raw/master/test/fonts/DejaVuSans-BoldOblique.ttf",
    "DejaVuSansMono.ttf": "https://github.com/py-pdf/fpdf2/raw/master/test/fonts/DejaVuSansMono.ttf",
}


def ensure_fonts(cache_dir: Path = FONT_CACHE_DIR) -> Path:
    """Télécharge les polices DejaVu une seule fois et les met en cache
    dans le dossier utilisateur (portable, pas de chemin Windows codé en dur)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for fname, url in FONT_URLS.items():
        fpath = cache_dir / fname
        if fpath.exists() and fpath.stat().st_size > 1000:
            continue
        print(f"  Téléchargement police : {fname} ...")
        try:
            urllib.request.urlretrieve(url, fpath)
        except Exception as e:
            raise RuntimeError(
                f"Impossible de télécharger {fname} ({e}). "
                f"Placez les .ttf DejaVu manuellement dans {cache_dir}"
            )
    return cache_dir


# =============================================================================
# MODELE DE DONNEES
# =============================================================================

@dataclass
class Run:
    text: str
    bold: bool = False
    italic: bool = False
    code: bool = False


@dataclass
class TOCEntry:
    level: int
    title: str
    page: int
    numbering: str = ""


@dataclass
class Block:
    kind: str  # heading, paragraph, list, code, blockquote, objectif, separator, math
    content: Any
    level: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# PARSER MARKDOWN (CommonMark via mistune -> Block[])
# =============================================================================

RE_OBJECTIF = re.compile(r'^objectif\s*:\s*', re.IGNORECASE)


class MarkdownToBlocks:
    """Convertit l'AST mistune (CommonMark) en une liste de Block plats.

    Contrairement au parseur original (regex ligne par ligne), celui-ci
    respecte la spécification CommonMark : un bloc de code ```lang ... ```
    n'est jamais ré-interprété comme du Markdown, quel que soit son contenu
    (donc plus de faux titres à partir de commentaires Python "# ...").
    """

    def __init__(self, dedup_titles: bool = True):
        self.md = mistune.create_markdown(
            renderer=None, plugins=["math", "strikethrough", "table"]
        )
        self.dedup_titles = dedup_titles
        self._seen_headings: List[str] = []

    def parse(self, text: str) -> List[Block]:
        tokens = self.md(text)
        blocks: List[Block] = []
        self._walk(tokens, blocks, base_level=0)
        return blocks

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        return re.sub(r"\s+", " ", s).strip().lower()

    def _runs_from_inline(self, children: List[dict]) -> List[Run]:
        runs: List[Run] = []

        def walk(nodes, bold=False, italic=False, code=False):
            for n in nodes:
                t = n.get("type")
                if t == "text":
                    runs.append(Run(n["raw"], bold, italic, code))
                elif t == "codespan":
                    runs.append(Run(n["raw"], bold, italic, True))
                elif t == "strong":
                    walk(n.get("children", []), True, italic, code)
                elif t == "emphasis":
                    walk(n.get("children", []), bold, True, code)
                elif t in ("linebreak", "softbreak"):
                    runs.append(Run("\n", bold, italic, code))
                elif t == "block_math" or t == "inline_math":
                    runs.append(Run(f"${n.get('raw','')}$", bold, italic, True))
                elif "children" in n:
                    walk(n["children"], bold, italic, code)
                elif "raw" in n:
                    runs.append(Run(n["raw"], bold, italic, code))

        walk(children)
        return runs

    @staticmethod
    def _plain_text(runs: List[Run]) -> str:
        return "".join(r.text for r in runs)

    def _is_objectif_paragraph(self, runs: List[Run]) -> Optional[str]:
        # Convention du cours : un paragraphe entièrement en italique
        # "*Objectif : ...*" doit devenir un encadré "objectif".
        if not runs:
            return None
        if not all(r.italic for r in runs):
            return None
        text = self._plain_text(runs).strip()
        if RE_OBJECTIF.match(text):
            return RE_OBJECTIF.sub("", text).strip()
        return None

    def _looks_like_duplicate_dump(self, code_text: str, lang: str) -> bool:
        """Heuristique de sécurité : certains pipelines de génération (RAG)
        ré-injectent tout le Markdown déjà structuré dans un bloc
        ```markdown ... ``` en fin de section. Rendu tel quel, cela double
        le contenu du document. On ne le fait que si le bloc contient des
        titres déjà vus ET que son langage évoque du Markdown."""
        if not self.dedup_titles:
            return False
        if lang.lower() not in ("markdown", "md", ""):
            return False
        hits = 0
        for line in code_text.splitlines():
            m = re.match(r"^#{1,6}\s+(.*)$", line.strip())
            if m and self._norm(m.group(1)) in self._seen_headings:
                hits += 1
        return hits >= 2  # au moins deux titres déjà connus retrouvés

    # -- walker ------------------------------------------------------------
    def _walk(self, tokens: List[dict], blocks: List[Block], base_level: int, list_depth: int = 0):
        for tok in tokens:
            t = tok.get("type")

            if t == "heading":
                level = min(tok["attrs"]["level"] + base_level, 6)
                runs = self._runs_from_inline(tok.get("children", []))
                title = self._plain_text(runs).strip()
                self._seen_headings.append(self._norm(title))
                blocks.append(Block("heading", title, level=level))

            elif t == "paragraph":
                runs = self._runs_from_inline(tok.get("children", []))
                obj = self._is_objectif_paragraph(runs)
                if obj is not None:
                    blocks.append(Block("objectif", obj))
                else:
                    blocks.append(Block("paragraph", runs))

            elif t == "block_text":
                # utilisé dans les items de liste "tight"
                runs = self._runs_from_inline(tok.get("children", []))
                blocks.append(Block("paragraph", runs))

            elif t == "block_code":
                lang = tok.get("attrs", {}).get("info", "") or ""
                code = tok.get("raw", "")
                if self._looks_like_duplicate_dump(code, lang):
                    blocks.append(Block("notice", "[Section dupliquée dans la source — omise]"))
                    continue
                blocks.append(Block("code", code.rstrip("\n"), meta={"lang": lang.strip()}))

            elif t == "block_quote":
                sub: List[Block] = []
                self._walk(tok.get("children", []), sub, base_level, list_depth)
                blocks.append(Block("blockquote", sub))

            elif t == "list":
                ordered = tok.get("attrs", {}).get("ordered", False)
                items = []
                for item in tok.get("children", []):
                    item_blocks: List[Block] = []
                    self._walk(item.get("children", []), item_blocks, base_level, list_depth + 1)
                    items.append(item_blocks)
                blocks.append(Block("list", items, level=list_depth, meta={"ordered": ordered}))

            elif t == "block_math":
                blocks.append(Block("math", tok.get("raw", "").strip()))

            elif t == "thematic_break":
                blocks.append(Block("separator", None))

            elif t == "table":
                blocks.append(self._table_block(tok))

            elif t in ("blank_line", "linebreak", "softbreak"):
                continue

            else:
                # type non géré explicitement : on tente de descendre
                if "children" in tok:
                    self._walk(tok["children"], blocks, base_level, list_depth)

    def _table_block(self, tok: dict) -> Block:
        head, rows = [], []
        for child in tok.get("children", []):
            if child["type"] == "table_head":
                for cell in child.get("children", []):
                    head.append(self._plain_text(self._runs_from_inline(cell.get("children", []))))
            elif child["type"] == "table_body":
                for row in child.get("children", []):
                    cells = [
                        self._plain_text(self._runs_from_inline(c.get("children", [])))
                        for c in row.get("children", [])
                    ]
                    rows.append(cells)
        return Block("table", {"head": head, "rows": rows})


# =============================================================================
# CHARGEMENT DES SOURCES
# =============================================================================

def load_from_markdown(path: Path, parser: MarkdownToBlocks) -> Tuple[str, str, List[Block]]:
    text = path.read_text(encoding="utf-8")
    title_match = re.search(r'^#\s+(.*)$', text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Cours généré"
    desc_match = re.search(r'^#\s+.*\n\n(.+?)(?:\n\n|\n#|\Z)', text, re.DOTALL)
    subtitle = desc_match.group(1).strip() if desc_match else ""
    blocks = parser.parse(text)
    # Le H1 sert déjà de titre de couverture : on ne le re-rend pas comme
    # un titre de chapitre en double au fil du texte s'il correspond
    # exactement au titre du document.
    if blocks and blocks[0].kind == "heading" and blocks[0].level == 1 \
            and blocks[0].content.strip() == title:
        blocks = blocks[1:]
    return title, subtitle, blocks


def load_from_json(path: Path, parser: MarkdownToBlocks) -> Tuple[str, str, List[Block]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    title = data.get("titre_cours", "Cours généré")
    subtitle = data.get("description", "")

    blocks: List[Block] = []

    def walk(nodes: List[dict], depth: int = 0):
        for node in nodes:
            node_title = node.get("titre", "")
            obj = node.get("objectif", "")
            contenu = node.get("contenu", "")
            enfants = node.get("enfants", [])

            if node_title:
                blocks.append(Block("heading", node_title, level=min(depth + 1, 6)))
            if obj:
                blocks.append(Block("objectif", obj))
            if contenu:
                blocks.extend(parser.parse(contenu))
            if enfants:
                walk(enfants, depth + 1)

    walk(data.get("plan", []))
    return title, subtitle, blocks


# =============================================================================
# RENDU LATEX -> IMAGE (mathtext), avec repli propre si échec
# =============================================================================

def render_latex_png(expr: str, out_path: Path, fontsize: int = 16) -> Optional[Tuple[Path, float, float]]:
    """Essaie de typographier `expr` avec mathtext (matplotlib). Renvoie
    (chemin_png, largeur_mm, hauteur_mm) ou None si le rendu échoue (mathtext
    ne supporte pas tout LaTeX ; dans ce cas l'appelant doit se replier sur
    un rendu texte)."""
    if not HAVE_MPL:
        return None
    expr_clean = expr.strip()
    if not expr_clean.startswith("$"):
        expr_clean = f"${expr_clean}$"
    try:
        fig = plt.figure(figsize=(0.1, 0.1))
        text = fig.text(0, 0, expr_clean, fontsize=fontsize)
        fig.canvas.draw()
        bbox = text.get_window_extent()
        width_in = bbox.width / fig.dpi + 0.15
        height_in = bbox.height / fig.dpi + 0.1
        plt.close(fig)

        fig = plt.figure(figsize=(width_in, height_in))
        fig.text(0.02, 0.15, expr_clean, fontsize=fontsize)
        fig.savefig(out_path, dpi=200, transparent=True)
        plt.close(fig)

        px_w, px_h = width_in * 200, height_in * 200
        mm_w, mm_h = px_w / 200 * 25.4, px_h / 200 * 25.4
        return out_path, mm_w, mm_h
    except Exception:
        return None


# =============================================================================
# RENDERER PDF
# =============================================================================

class BasePDF(FPDF):
    """Fonctionnalités communes : polices, marges, helpers de mise en page."""

    def __init__(self, fonts_dir: Path):
        super().__init__(unit="mm", format="A4")
        self.add_font("DejaVu", "", str(fonts_dir / "DejaVuSans.ttf"))
        self.add_font("DejaVu", "B", str(fonts_dir / "DejaVuSans-Bold.ttf"))
        self.add_font("DejaVu", "I", str(fonts_dir / "DejaVuSans-Oblique.ttf"))
        self.add_font("DejaVu", "BI", str(fonts_dir / "DejaVuSans-BoldOblique.ttf"))
        self.add_font("DejaVuMono", "", str(fonts_dir / "DejaVuSansMono.ttf"))
        self.set_auto_page_break(auto=True, margin=MARGIN_BOTTOM)
        self.set_margins(MARGIN_LEFT, MARGIN_TOP, MARGIN_RIGHT)
        self.set_line_width(0.2)

    def space_left(self) -> float:
        return (PAGE_H - MARGIN_BOTTOM) - self.get_y()

    def keep_together(self, estimated_height: float, minimum: float = MIN_SPACE_BEFORE_BLOCK):
        """Evite les veuves/orphelines : force un saut de page si le bloc
        qui arrive ne tiendrait pas (au moins en partie utile) sur la page
        courante."""
        needed = min(estimated_height, minimum)
        if self.space_left() < needed:
            self.add_page()


class ContentPDF(BasePDF):
    """Rend uniquement les pages de contenu (pas de couverture, pas de
    sommaire) : c'est le SEUL passage de rendu complet du document. Les
    numéros de page collectés ici sont relatifs au début de ce document et
    seront décalés (offset) une fois le préfixe (couverture + sommaire)
    connu — voir `build_pdf()`."""

    def __init__(self, fonts_dir: Path, course_title: str):
        super().__init__(fonts_dir)
        self.course_title = course_title
        self.toc_entries: List[TOCEntry] = []
        self._current_chapter = ""
        self._chapter_counters = [0, 0, 0, 0, 0, 0]
        self._latex_tmp = Path(tempfile.gettempdir()) / "course_pdf_latex"
        self._latex_tmp.mkdir(exist_ok=True)
        self._latex_i = 0

    # -- entête / pied de page ------------------------------------------
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("DejaVu", "", 8)
        self.set_text_color(*COLORS["secondary"])
        self.set_y(12)
        self.set_x(MARGIN_LEFT)
        self.cell(CONTENT_W * 0.55, 6, self._truncate(self.course_title, 60), new_x=XPos.LEFT, new_y=YPos.TOP)
        self.set_x(MARGIN_LEFT + CONTENT_W * 0.55)
        self.cell(CONTENT_W * 0.45, 6, self._truncate(self._current_chapter, 45), align="R",
                  new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.set_y(19)
        self.set_draw_color(*COLORS["border"])
        self.line(MARGIN_LEFT, self.get_y(), PAGE_W - MARGIN_RIGHT, self.get_y())
        self.set_y(MARGIN_TOP)

    def footer(self):
        # Pas de numérotation ici : elle est tamponnée après coup une fois
        # le décalage (offset des pages de couverture/sommaire) connu -
        # voir stamp_page_numbers().
        pass

    @staticmethod
    def _truncate(text: str, n: int) -> str:
        return text if len(text) <= n else text[: n - 1] + "…"

    def register_toc(self, level: int, title: str):
        numbering = ""
        if level <= 6:
            self._chapter_counters[level - 1] += 1
            for i in range(level, 6):
                self._chapter_counters[i] = 0
            numbering = ".".join(str(c) for c in self._chapter_counters[:level] if c)
        self.toc_entries.append(TOCEntry(level, title, self.page_no(), numbering))
        return numbering

    # -- blocs -------------------------------------------------------------
    def render_blocks(self, blocks: List[Block]):
        if self.page_no() == 0:
            self.add_page()
        for b in blocks:
            getattr(self, f"_render_{b.kind}")(b)

    def _render_heading(self, block: Block):
        level, title = block.level, block.content
        if level == 1:
            self.add_page()
            self._current_chapter = title
        else:
            self.keep_together(MIN_SPACE_BEFORE_HEADING)

        numbering = self.register_toc(level, title)

        sizes = {1: 19, 2: 14, 3: 12, 4: 11, 5: 10, 6: 10}
        weights = {1: "B", 2: "B", 3: "B", 4: "B", 5: "B", 6: "B"}
        colors = {1: COLORS["primary"], 2: COLORS["primary"], 3: COLORS["text"]}

        self.set_font("DejaVu", weights.get(level, "B"), sizes.get(level, 10))
        self.set_text_color(*colors.get(level, COLORS["text"]))

        display = f"{numbering}  {title}" if numbering and level <= 4 else title
        self.ln(5 if level > 1 else 6)
        self.set_x(MARGIN_LEFT)
        self.multi_cell(0, 8 if level == 1 else 7, display, align="L",
                         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1.5)

        if level == 1:
            self.set_draw_color(*COLORS["primary"])
            self.set_line_width(0.4)
            self.line(MARGIN_LEFT, self.get_y(), PAGE_W - MARGIN_RIGHT, self.get_y())
            self.ln(4)
        self.set_line_width(0.2)

    def _render_paragraph(self, block: Block):
        self.set_x(MARGIN_LEFT)
        self._write_runs(block.content, align="J")
        self.ln(2.5)

    def _render_list(self, block: Block, indent0: int = 0):
        items: List[List[Block]] = block.content
        ordered: bool = block.meta.get("ordered", False)
        depth: int = block.level
        indent = 6 + depth * 6
        for idx, item_blocks in enumerate(items, 1):
            bullet = f"{idx}." if ordered else ("•" if depth == 0 else "–")
            first = True
            for sub in item_blocks:
                if sub.kind in ("paragraph",):
                    self.set_x(MARGIN_LEFT + indent - 6)
                    self.set_font("DejaVu", "", 10)
                    self.set_text_color(*COLORS["text"])
                    if first:
                        self.cell(6, LINE_HEIGHT, bullet, new_x=XPos.RIGHT, new_y=YPos.TOP)
                    else:
                        self.cell(6, LINE_HEIGHT, "", new_x=XPos.RIGHT, new_y=YPos.TOP)
                    self._write_runs(sub.content, indent=indent, align="L")
                    first = False
                elif sub.kind == "list":
                    self._render_list(sub, indent0=indent)
                else:
                    getattr(self, f"_render_{sub.kind}")(sub)
        self.ln(1.5)

    def _render_code(self, block: Block):
        code = block.content
        lang = block.meta.get("lang", "")
        self.set_font("DejaVuMono", "", 8.3)
        avail_w = CONTENT_W - 6
        wrapped: List[str] = []
        for raw_line in code.split("\n"):
            line = raw_line.replace("\t", "    ").replace("\r", "")
            wrapped.extend(self._wrap_mono(line, avail_w) or [""])

        line_h = 4.3
        header_h = 5.5 if lang else 2
        height = header_h + len(wrapped) * line_h + 3
        self.keep_together(min(height, 60))

        # Si le bloc est plus long qu'une page, on laisse fpdf le couper
        # naturellement (auto page break) plutôt que de le tronquer.
        self.ln(2)
        y0 = self.get_y()
        self.set_fill_color(*COLORS["code_bg"])
        self.set_draw_color(*COLORS["border"])
        if lang:
            self.set_font("DejaVu", "I", 7)
            self.set_text_color(*COLORS["secondary"])
            self.set_xy(MARGIN_LEFT, self.get_y())
            self.cell(0, header_h, f"  {lang.upper()}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_font("DejaVuMono", "", 8.3)
        self.set_text_color(*COLORS["text"])
        block_start_y = self.get_y()
        for line in wrapped:
            self.set_x(MARGIN_LEFT + 3)
            self.cell(CONTENT_W - 6, line_h, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        block_end_y = self.get_y()

        # Cadre dessiné a posteriori autour du texte réellement écrit
        # (donc jamais désynchronisé, contrairement à l'original).
        self.rect(MARGIN_LEFT, y0, CONTENT_W, block_end_y - y0 + 1.5, style="D")
        self.ln(3)

    def _wrap_mono(self, line: str, avail_w: float) -> List[str]:
        self.set_font("DejaVuMono", "", 8.3)
        if self.get_string_width(line) <= avail_w:
            return [line]
        out, cur = [], ""
        for ch in line:
            if self.get_string_width(cur + ch) > avail_w:
                out.append(cur)
                cur = ch
            else:
                cur += ch
        if cur:
            out.append(cur)
        return out

    def _render_blockquote(self, block: Block):
        self.keep_together(20)
        self.ln(2)
        sub_blocks: List[Block] = block.content
        y0 = self.get_y()
        self.set_x(MARGIN_LEFT + 6)
        self.set_font("DejaVu", "I", 10)
        self.set_text_color(*COLORS["secondary"])
        for sub in sub_blocks:
            if sub.kind == "paragraph":
                self.set_x(MARGIN_LEFT + 6)
                self._write_runs(sub.content, indent=6, align="L", italic_default=True)
            else:
                getattr(self, f"_render_{sub.kind}")(sub)
        y1 = self.get_y()
        self.set_draw_color(*COLORS["accent"])
        self.set_line_width(1)
        self.line(MARGIN_LEFT + 2, y0, MARGIN_LEFT + 2, y1)
        self.set_line_width(0.2)
        self.ln(2)

    def _render_objectif(self, block: Block):
        text = block.content
        self.set_font("DejaVu", "I", 10)
        lines = self.multi_cell(CONTENT_W - 22, LINE_HEIGHT, text, dry_run=True, output="LINES")
        h = len(lines) * LINE_HEIGHT + 5
        self.keep_together(h)
        self.ln(2)
        y0 = self.get_y()
        self.set_fill_color(*COLORS["objectif_bg"])
        self.set_draw_color(*COLORS["accent"])
        self.set_line_width(0.3)
        self.rect(MARGIN_LEFT, y0, CONTENT_W, h, style="DF")

        self.set_xy(MARGIN_LEFT + 3, y0 + 2)
        self.set_font("DejaVu", "BI", 10)
        self.set_text_color(*COLORS["accent"])
        self.cell(20, LINE_HEIGHT, "Objectif :", new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.set_font("DejaVu", "I", 10)
        self.multi_cell(CONTENT_W - 26, LINE_HEIGHT, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_line_width(0.2)
        self.set_y(y0 + h + 2)

    def _render_math(self, block: Block):
        expr = block.content
        self.keep_together(20)
        self.ln(2)
        self._latex_i += 1
        png_path = self._latex_tmp / f"eq_{self._latex_i}.png"
        result = render_latex_png(expr, png_path)
        if result:
            _, w_mm, h_mm = result
            w_mm = min(w_mm, CONTENT_W - 10)
            x = MARGIN_LEFT + (CONTENT_W - w_mm) / 2
            self.image(str(png_path), x=x, y=self.get_y(), w=w_mm)
            self.ln(h_mm + 3)
        else:
            self.set_font("DejaVuMono", "I", 9.5)
            self.set_text_color(*COLORS["text"])
            self.set_x(MARGIN_LEFT + 8)
            self.multi_cell(0, LINE_HEIGHT, expr, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.ln(2)

    def _render_separator(self, block: Block):
        self.ln(3)
        self.set_draw_color(*COLORS["border"])
        cx = PAGE_W / 2
        y = self.get_y()
        self.line(cx - 20, y, cx + 20, y)
        self.ln(4)

    def _render_notice(self, block: Block):
        self.set_font("DejaVu", "I", 8.5)
        self.set_text_color(*COLORS["secondary"])
        self.set_x(MARGIN_LEFT)
        self.multi_cell(0, 4.5, block.content, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def _render_table(self, block: Block):
        head = block.content["head"]
        rows = block.content["rows"]
        n = max(len(head), 1)
        col_w = CONTENT_W / n
        self.keep_together(10 + 7 * (len(rows) + 1))
        self.ln(2)
        self.set_font("DejaVu", "B", 9)
        self.set_fill_color(*COLORS["light"])
        self.set_x(MARGIN_LEFT)
        for cell in head:
            self.cell(col_w, 7, cell, border=1, fill=True)
        self.ln(7)
        self.set_font("DejaVu", "", 9)
        for row in rows:
            self.set_x(MARGIN_LEFT)
            for cell in row:
                self.cell(col_w, 7, cell, border=1)
            self.ln(7)
        self.ln(2)

    # -- texte enrichi -------------------------------------------------------
    def _write_runs(self, runs: List[Run], indent: int = 0, align: str = "J",
                     italic_default: bool = False):
        """Ecrit une liste de Run (gras/italique/code) avec un vrai retour
        à la ligne géré par fpdf2 (fragments), sans regex fragile ni
        justification hasardeuse sur des titres courts."""
        self.set_x(MARGIN_LEFT + indent)
        fragments = []
        for r in runs:
            if r.text == "\n":
                fragments.append(("\n", {}))
                continue
            style = ""
            if r.bold:
                style += "B"
            if r.italic or italic_default:
                style += "I"
            fragments.append((r.text, {"style": style or "", "code": r.code}))

        # fpdf2 write_html/ fragments API: on utilise multi_cell avec
        # marqueurs de style successifs via write() pour chaque run, ce qui
        # gère nativement le retour à la ligne mot par mot.
        for text, meta in fragments:
            if text == "\n":
                self.ln(LINE_HEIGHT)
                self.set_x(MARGIN_LEFT + indent)
                continue
            if meta.get("code"):
                self.set_font("DejaVuMono", meta["style"].replace("I", ""), 9)
                self.set_text_color(*COLORS["text"])
                w = self.get_string_width(text) + 2
                x, y = self.get_x(), self.get_y()
                self.set_fill_color(*COLORS["inline_code_bg"])
                self.cell(w, LINE_HEIGHT, "", fill=True)
                self.set_xy(x + 1, y)
                self.write(LINE_HEIGHT, text)
                self.set_xy(x + w, y)
            else:
                self.set_font("DejaVu", meta["style"], 10)
                self.set_text_color(*COLORS["text"])
                self.write(LINE_HEIGHT, text)
        self.ln(LINE_HEIGHT)


class PrefixPDF(BasePDF):
    """Couverture + sommaire. Document séparé, fusionné ensuite devant le
    contenu — c'est ce qui permet un sommaire fiable sans devoir rendre
    tout le cours deux fois."""

    def __init__(self, fonts_dir: Path, title: str, subtitle: str):
        super().__init__(fonts_dir)
        self.course_title = title
        self.course_subtitle = subtitle

    def header(self):
        pass

    def footer(self):
        pass

    def add_cover_page(self):
        self.add_page()
        self.set_y(80)
        self.set_draw_color(*COLORS["primary"])
        self.set_line_width(1.5)
        self.line(MARGIN_LEFT, self.get_y(), PAGE_W - MARGIN_RIGHT, self.get_y())
        self.ln(12)

        self.set_font("DejaVu", "B", 24)
        self.set_text_color(*COLORS["primary"])
        self.multi_cell(0, 12, self.course_title, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(4)

        if self.course_subtitle:
            self.set_font("DejaVu", "I", 12)
            self.set_text_color(*COLORS["secondary"])
            self.multi_cell(0, 8, self.course_subtitle, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.ln(8)

        self.set_draw_color(*COLORS["primary"])
        self.set_line_width(1.5)
        self.line(MARGIN_LEFT, self.get_y(), PAGE_W - MARGIN_RIGHT, self.get_y())
        self.ln(15)

        self.set_font("DejaVu", "", 10)
        self.set_text_color(*COLORS["secondary"])
        meta = f"Document généré automatiquement\n{datetime.now().strftime('%d/%m/%Y')}"
        self.multi_cell(0, 6, meta, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def add_toc_pages(self, entries: List[TOCEntry], max_level: int = 3):
        self.add_page()
        self.set_font("DejaVu", "B", 18)
        self.set_text_color(*COLORS["primary"])
        self.cell(0, 12, "Table des matières", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)
        self.set_draw_color(*COLORS["primary"])
        self.set_line_width(0.5)
        self.line(MARGIN_LEFT, self.get_y(), PAGE_W - MARGIN_RIGHT, self.get_y())
        self.ln(6)

        for entry in entries:
            if entry.level > max_level:
                continue
            indent = (entry.level - 1) * 6
            self.set_x(MARGIN_LEFT + indent)
            if entry.level == 1:
                self.set_font("DejaVu", "B", 11)
                self.set_text_color(*COLORS["primary"])
            elif entry.level == 2:
                self.set_font("DejaVu", "", 10)
                self.set_text_color(*COLORS["text"])
            else:
                self.set_font("DejaVu", "", 9)
                self.set_text_color(*COLORS["secondary"])

            label = f"{entry.numbering}  {entry.title}" if entry.numbering else entry.title
            avail = 145 - indent
            if self.get_string_width(label) > avail:
                while self.get_string_width(label + "…") > avail and len(label) > 5:
                    label = label[:-1]
                label += "…"

            if self.space_left() < 10:
                self.add_page()
            self.cell(avail, 7, label, new_x=XPos.RIGHT, new_y=YPos.TOP)
            self.cell(0, 7, str(entry.page), align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)


# =============================================================================
# ORCHESTRATION
# =============================================================================

def stamp_page_numbers(pdf_path: Path, out_path: Path, fonts_dir: Path,
                        skip_first: bool = True):
    """Tamponne 'Page X / Y' en pied de page sur le PDF final déjà fusionné
    (couverture + sommaire + contenu), une fois le nombre total de pages
    connu avec certitude."""
    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)

    overlay = BasePDF(fonts_dir)
    overlay.set_auto_page_break(False)
    for i in range(total):
        overlay.add_page()
        if skip_first and i == 0:
            continue
        overlay.set_font("DejaVu", "", 8)
        overlay.set_text_color(*COLORS["secondary"])
        overlay.set_xy(0, PAGE_H - 15)
        overlay.cell(PAGE_W, 10, f"Page {i + 1} / {total}", align="C")
    overlay_path = pdf_path.with_suffix(".overlay.pdf")
    overlay.output(str(overlay_path))

    overlay_reader = PdfReader(str(overlay_path))
    writer = PdfWriter()
    for i, page in enumerate(reader.pages):
        page.merge_page(overlay_reader.pages[i])
        writer.add_page(page)
    with open(out_path, "wb") as f:
        writer.write(f)
    overlay_path.unlink(missing_ok=True)


def build_pdf(input_path: Path, output_path: Path, from_json: bool = False,
              dedup: bool = True, max_toc_level: int = 3):
    print("=" * 60)
    print("GÉNÉRATION DU PDF ACADÉMIQUE")
    print("=" * 60)

    fonts_dir = ensure_fonts()
    parser = MarkdownToBlocks(dedup_titles=dedup)

    if from_json or input_path.suffix.lower() == ".json":
        print(f"[1/5] Lecture JSON : {input_path}")
        title, subtitle, blocks = load_from_json(input_path, parser)
    else:
        print(f"[1/5] Lecture Markdown : {input_path}")
        title, subtitle, blocks = load_from_markdown(input_path, parser)
    print(f"      Titre : {title}")
    print(f"      Blocs : {len(blocks)}")

    # --- passage UNIQUE de rendu du contenu -------------------------------
    print("[2/5] Rendu du contenu (passage unique)...")
    content_pdf = ContentPDF(fonts_dir, course_title=title)
    content_pdf.render_blocks(blocks)
    content_pages = content_pdf.page_no()
    tmp_dir = Path(tempfile.gettempdir())
    tmp_content = tmp_dir / "_course_content.pdf"
    content_pdf.output(str(tmp_content))
    print(f"      Pages de contenu : {content_pages}")
    print(f"      Entrées de sommaire : {len(content_pdf.toc_entries)}")

    # --- préfixe (couverture + sommaire), taille encore inconnue ----------
    print("[3/5] Construction du sommaire (estimation de longueur)...")
    prefix_probe = PrefixPDF(fonts_dir, title, subtitle)
    prefix_probe.add_cover_page()
    prefix_probe.add_toc_pages(content_pdf.toc_entries, max_level=max_toc_level)
    prefix_len = prefix_probe.page_no()

    # --- reconstruction du préfixe avec les BONS numéros de page ----------
    final_entries = [
        TOCEntry(e.level, e.title, e.page + prefix_len, e.numbering)
        for e in content_pdf.toc_entries
    ]
    prefix_pdf = PrefixPDF(fonts_dir, title, subtitle)
    prefix_pdf.add_cover_page()
    prefix_pdf.add_toc_pages(final_entries, max_level=max_toc_level)
    tmp_prefix = tmp_dir / "_course_prefix.pdf"
    prefix_pdf.output(str(tmp_prefix))
    print(f"      Pages de préfixe : {prefix_len}")

    # --- fusion -------------------------------------------------------------
    print("[4/5] Fusion couverture + sommaire + contenu...")
    writer = PdfWriter()
    writer.append(str(tmp_prefix))
    writer.append(str(tmp_content))
    tmp_merged = tmp_dir / "_course_merged.pdf"
    with open(tmp_merged, "wb") as f:
        writer.write(f)

    # --- numérotation finale --------------------------------------------
    print("[5/5] Numérotation des pages...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stamp_page_numbers(tmp_merged, output_path, fonts_dir)

    total_pages = prefix_len + content_pages
    print("\n" + "=" * 60)
    print(f"PDF GÉNÉRÉ : {output_path}")
    print(f"  Pages       : {total_pages}")
    print(f"  Entrées TOC : {len(final_entries)}")
    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    # -------------------------------------------------------------------
    # CHEMINS INJECTÉS EN DUR : le flux s'exécute automatiquement sans
    # arguments, pour être branché en fin de pipeline. Modifier ici selon
    # l'environnement. DEFAULT_INPUT est cherché tel quel ; si absent, on
    # essaie DEFAULT_INPUT_FALLBACK (utile si le pipeline produit tantôt
    # du .md, tantôt du .json).
    # -------------------------------------------------------------------
    DEFAULT_INPUT = Path("rag/data/output/course_generated/course.md")
    DEFAULT_INPUT_FALLBACK = Path("rag/data/output/course_generated/course.json")
    DEFAULT_OUTPUT = Path("rag/data/output/course_generated/course.pdf")
    DEFAULT_DEDUP = True
    DEFAULT_TOC_DEPTH = 3

    ap = argparse.ArgumentParser(
        description="Génère un PDF académique à partir du cours. "
                     "Sans arguments, utilise les chemins injectés en haut de main()."
    )
    ap.add_argument("--input", "-i", type=Path, default=None,
                     help=f"Fichier source (def: {DEFAULT_INPUT} ou {DEFAULT_INPUT_FALLBACK})")
    ap.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT,
                     help=f"Chemin du PDF de sortie (def: {DEFAULT_OUTPUT})")
    ap.add_argument("--json", action="store_true", help="Force la lecture comme JSON")
    ap.add_argument("--no-dedup", action="store_true",
                     help="Désactive la détection des blocs Markdown dupliqués")
    ap.add_argument("--toc-depth", type=int, default=DEFAULT_TOC_DEPTH,
                     help=f"Profondeur maximale de titres affichés dans le sommaire (def: {DEFAULT_TOC_DEPTH})")
    args = ap.parse_args()

    # Résolution automatique de la source si rien n'est passé en ligne de
    # commande : .md prioritaire, sinon .json.
    input_path = args.input
    force_json = args.json
    if input_path is None:
        if DEFAULT_INPUT.exists():
            input_path = DEFAULT_INPUT
        elif DEFAULT_INPUT_FALLBACK.exists():
            input_path = DEFAULT_INPUT_FALLBACK
            force_json = True
        else:
            print("ERREUR : Aucun fichier source trouvé.")
            print(f"  Recherché : {DEFAULT_INPUT}")
            print(f"              {DEFAULT_INPUT_FALLBACK}")
            print("  Fournissez --input/-i explicitement si les chemins ont changé.")
            return 1

    if not input_path.exists():
        print(f"ERREUR : Fichier introuvable : {input_path}")
        return 1

    build_pdf(input_path, args.output, from_json=force_json,
              dedup=(not args.no_dedup) and DEFAULT_DEDUP, max_toc_level=args.toc_depth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())