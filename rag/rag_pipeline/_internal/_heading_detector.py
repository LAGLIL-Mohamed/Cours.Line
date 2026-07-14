"""Detection robuste des titres - multi-strategie sans Docling."""

from __future__ import annotations

import re
import statistics
from typing import TYPE_CHECKING

import fitz
import numpy as np
from sklearn.cluster import KMeans

if TYPE_CHECKING:
    from ..schemas import TextBlock, Heading, Page
    from ..config import PipelineConfig


TOC_KEYWORDS = [
    "table of contents", "contents", "sommaire",
    "table des matieres", "table des matières",
    "índice", "indice", "contenido", "inhalt",
]


class RobustHeadingDetector:
    """Detecte les titres par 3 strategies : TOC native, TOC visuelle, Style typographique."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._numbering_patterns = [re.compile(p) for p in config.heading_numbering_patterns]
        self._chapter_patterns = [re.compile(p, re.IGNORECASE) for p in config.chapter_keywords]

    def detect(self, doc_path: str, pages: list[Page]) -> list[Heading]:
        """Retourne la liste fusionnee et hierarchisee des headings."""
        native = self._extract_native_toc(doc_path)
        visual = self._extract_visual_toc(pages)
        style = self._extract_by_style(pages)
        merged = self._merge_sources(native, visual, style)
        return self._cluster_and_normalize_levels(merged)

    def _extract_native_toc(self, doc_path: str) -> list[Heading]:
        from ..schemas import Heading
        headings = []
        try:
            with fitz.open(doc_path) as doc:
                toc = doc.get_toc()
                for level, title, page in toc:
                    headings.append(Heading(
                        text=title.strip(),
                        level=level,
                        page=max(0, page - 1),
                        bbox=(0, 0, 0, 0),
                        font_size=0.0,
                        source="native_toc",
                        confidence=0.95,
                    ))
        except Exception:
            pass
        return headings

    def _extract_visual_toc(self, pages: list[Page]) -> list[Heading]:
        from ..schemas import Heading
        results = []
        for page in pages[:self.config.toc_max_pages]:
            if self._is_toc_page(page):
                results.extend(self._parse_toc_page(page))
        return results

    def _is_toc_page(self, page: Page) -> bool:
        text = page.raw_text.lower()
        has_kw = any(kw in text for kw in TOC_KEYWORDS)
        if not has_kw:
            return False
        lines = [l.strip() for l in page.raw_text.split("\n") if l.strip()]
        if len(lines) < 5:
            return False
        page_num_lines = sum(1 for l in lines if re.search(r"\d+\s*$", l))
        return (page_num_lines / len(lines)) > self.config.toc_keyword_ratio

    def _parse_toc_page(self, page: Page) -> list[Heading]:
        from ..schemas import Heading
        headings = []
        for block in page.blocks:
            text = block.text.strip()
            if len(text) < 3:
                continue
            m = re.search(r"(.*?)(\d+)\s*$", text)
            if not m:
                continue
            title_part = m.group(1).strip(" .-")
            try:
                target_page = int(m.group(2)) - 1
            except ValueError:
                continue
            if not title_part or len(title_part) < 2:
                continue
            level = self._infer_level_from_text(title_part, block.bbox[0])
            headings.append(Heading(
                text=title_part, level=level, page=max(0, target_page),
                bbox=block.bbox, font_size=block.font_size,
                source="visual_toc", confidence=0.85,
                is_bold=block.is_bold, is_uppercase=title_part.isupper(),
            ))
        return headings

    def _extract_by_style(self, pages: list[Page]) -> list[Heading]:
        from ..schemas import Heading
        all_blocks = []
        for page in pages:
            for block in page.blocks:
                all_blocks.append((page.num, block))

        if not all_blocks:
            return []

        body_size = self._estimate_body_size(all_blocks)
        candidates = []
        seen = set()

        for page_num, block in all_blocks:
            text = block.text.strip()
            if not text or len(text) < self.config.heading_min_chars or len(text) > self.config.heading_max_chars:
                continue

            if self._is_heading_candidate(text, block.font_size, block.is_bold, body_size):
                key = (text, page_num)
                if key in seen:
                    continue
                seen.add(key)
                level = self._infer_level_from_text(text, block.bbox[0])
                candidates.append(Heading(
                    text=text, level=level, page=page_num,
                    bbox=block.bbox, font_size=block.font_size,
                    source="style", confidence=0.75,
                    is_bold=block.is_bold, is_uppercase=text.isupper(),
                    is_numbered=any(p.match(text) for p in self._numbering_patterns),
                ))

        return candidates

    def _estimate_body_size(self, blocks: list[tuple[int, TextBlock]]) -> float:
        sizes = [b.font_size for _, b in blocks if b.text.strip()]
        if not sizes:
            return 11.0
        try:
            return statistics.mode(round(s, 1) for s in sizes)
        except statistics.StatisticsError:
            return statistics.median(sizes)

    def _is_heading_candidate(self, text: str, size: float, is_bold: bool, body_size: float) -> bool:
        ratio = size / body_size if body_size > 0 else 1.0
        has_chapter = any(p.match(text) for p in self._chapter_patterns)
        has_number = any(p.match(text) for p in self._numbering_patterns)
        is_upper = text.isupper()

        if has_chapter:
            return True
        if ratio >= self.config.heading_font_ratio_strong:
            return True
        if ratio >= self.config.heading_font_ratio and (is_bold or is_upper or has_number):
            return True
        if is_bold and has_number:
            return True
        if ratio >= self.config.heading_font_ratio and is_bold:
            return True
        return False

    def _merge_sources(self, native: list[Heading], visual: list[Heading], style: list[Heading]) -> list[Heading]:
        all_h = []
        seen = set()
        for source, priority in [(native, 3), (visual, 2), (style, 1)]:
            for h in source:
                key = (h.text, h.page)
                if key not in seen:
                    seen.add(key)
                    h.confidence = h.confidence * (priority / 3)
                    all_h.append(h)
        all_h.sort(key=lambda h: (h.page, h.bbox[1] if h.bbox else 0))
        return all_h

    def _cluster_and_normalize_levels(self, headings: list[Heading]) -> list[Heading]:
        if not headings:
            return headings

        sizes = np.array([[h.font_size] for h in headings if h.font_size > 0])
        if len(sizes) < 2:
            for h in headings:
                h.level = self._infer_level_from_text(h.text, 0)
            return headings

        n_clusters = min(self.config.kmeans_n_clusters, len(set(sizes.flatten())))
        n_clusters = max(2, min(n_clusters, len(headings)))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(sizes)

        cluster_means = {i: np.mean(sizes[labels == i]) for i in range(n_clusters)}
        sorted_clusters = sorted(cluster_means.items(), key=lambda x: x[1], reverse=True)
        cluster_to_level = {c[0]: i + 1 for i, c in enumerate(sorted_clusters)}

        idx = 0
        for h in headings:
            if h.font_size > 0:
                h.level = cluster_to_level[labels[idx]]
                idx += 1
            else:
                h.level = self._infer_level_from_text(h.text, 0)

        return headings

    def _infer_level_from_text(self, text: str, x_position: float) -> int:
        text = text.strip()
        if any(p.match(text) for p in self._chapter_patterns):
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
        if x_position > 80:
            return 3
        if x_position > 40:
            return 2
        return 1
