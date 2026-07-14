"""Etape 1 : Extraction du contenu textuel du PDF avec PyMuPDF."""

from __future__ import annotations

import fitz  # PyMuPDF

from ..schemas import Document, Page, TextBlock


class Step1Extract:
    """Extrait le texte brut et structure page par page."""

    def run(self, doc: Document) -> list[Page]:
        pages = []
        with fitz.open(doc.chemin_origine) as pdf:
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                page_obj = self._extract_page(doc, page, page_num + 1)
                pages.append(page_obj)
        return pages

    def _extract_page(self, doc: Document, page: fitz.Page, page_num: int) -> Page:
        raw_text = page.get_text()
        blocks = self._extract_blocks(page)
        return Page(
            doc=doc,
            num=page_num,
            raw_text=raw_text,
            blocks=blocks,
        )

    def _extract_blocks(self, page: fitz.Page) -> list[TextBlock]:
        blocks = []
        dict_data = page.get_text("dict")

        for block in dict_data.get("blocks", []):
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                line_text = ""
                line_font_sizes = []
                line_is_bold = False
                line_font_name = ""
                line_color = 0

                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text.strip():
                        continue

                    line_text += text
                    line_font_sizes.append(span.get("size", 11.0))

                    font_name = span.get("font", "").lower()
                    if "bold" in font_name or "black" in font_name or "heavy" in font_name:
                        line_is_bold = True

                    line_font_name = span.get("font", "")
                    line_color = span.get("color", 0)

                if not line_text.strip():
                    continue

                bbox = tuple(line.get("bbox", (0, 0, 0, 0)))
                avg_font_size = sum(line_font_sizes) / len(line_font_sizes) if line_font_sizes else 11.0

                blocks.append(
                    TextBlock(
                        text=line_text,
                        bbox=bbox,
                        font_size=avg_font_size,
                        is_bold=line_is_bold,
                        font_name=line_font_name,
                        color=line_color,
                        origin="line",
                    )
                )

        return blocks
