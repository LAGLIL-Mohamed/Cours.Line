"""Étape 4 : Analyse de la structure — titres, sections, paragraphes."""

from __future__ import annotations

from ..schemas import Heading, Page, Paragraph, Section, TextBlock
from .._internal._heading_detector import HeadingDetector
from .._internal._utils import generate_id
from ..config import PipelineConfig


class Step4Structure:
    """Identifie la structure logique et construit les objets sémantiques."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.detector = HeadingDetector(
            font_ratio=self.config.heading_font_ratio,
            font_ratio_strong=self.config.heading_font_ratio_strong,
            min_chars=self.config.heading_min_chars,
            max_chars=self.config.heading_max_chars,
        )

    def run(self, pages: list[Page]) -> list[Section]:
        all_blocks: list[tuple[int, TextBlock]] = []
        for page in pages:
            for block in page.blocks:
                all_blocks.append((page.num, block))

        raw_headings = self._detect_headings(all_blocks)
        sections = self._build_sections(raw_headings, all_blocks, pages[0].doc.source_id)
        return sections

    def _detect_headings(self, blocks_with_pages: list[tuple[int, TextBlock]]) -> list[Heading]:
        blocks = [b for _, b in blocks_with_pages]
        headings = self.detector.detect(blocks, page_height=842)

        result = []
        for h in headings:
            for page_num, block in blocks_with_pages:
                if block.bbox == h.bbox and block.text.strip() == h.text:
                    result.append(Heading(
                        text=h.text, level=h.level, page=page_num,
                        bbox=h.bbox, font_size=h.font_size,
                    ))
                    break
        return result

    def _build_sections(
        self, headings: list[Heading],
        blocks_with_pages: list[tuple[int, TextBlock]],
        source_id: str,
    ) -> list[Section]:
        if not headings:
            paragraphs = self._blocks_to_paragraphs(blocks_with_pages)
            return [Section(
                section_id=generate_id("sec"), source_id=source_id,
                heading=Heading(text="(sans titre)", level=1, page=1, bbox=(0,0,0,0), font_size=11.0),
                paragraphs=paragraphs,
            )]

        sections = []
        for i, heading in enumerate(headings):
            start_idx = self._find_heading_position(heading, blocks_with_pages)
            if start_idx is None:
                continue
            end_idx = (
                self._find_heading_position(headings[i + 1], blocks_with_pages)
                if i + 1 < len(headings) else len(blocks_with_pages)
            )
            section_blocks = blocks_with_pages[start_idx + 1:end_idx]
            paragraphs = self._blocks_to_paragraphs(section_blocks)
            sections.append(Section(
                section_id=generate_id("sec"), source_id=source_id,
                heading=heading, paragraphs=paragraphs,
            ))
        return sections

    def _find_heading_position(self, heading: Heading, blocks_with_pages: list[tuple[int, TextBlock]]) -> int | None:
        for idx, (page_num, block) in enumerate(blocks_with_pages):
            if page_num == heading.page and block.text.strip() == heading.text:
                return idx
        return None

    def _blocks_to_paragraphs(self, blocks_with_pages: list[tuple[int, TextBlock]]) -> list[Paragraph]:
        if not blocks_with_pages:
            return []

        paragraphs = []
        current_text = ""
        current_start_page = blocks_with_pages[0][0]
        current_end_page = blocks_with_pages[0][0]
        current_blocks = []

        for page_num, block in blocks_with_pages:
            text = block.text.strip()
            if not text:
                continue

            should_merge = (
                current_text
                and not current_text.endswith((".", "!", "?", ":", ";"))
                and current_blocks
                and block.font_size == current_blocks[-1].font_size
            )

            if should_merge:
                current_text += " " + text
                current_end_page = page_num
                current_blocks.append(block)
            else:
                if current_text:
                    paragraphs.append(Paragraph(
                        text=current_text, page_start=current_start_page,
                        page_end=current_end_page, blocks=current_blocks,
                    ))
                current_text = text
                current_start_page = page_num
                current_end_page = page_num
                current_blocks = [block]

        if current_text:
            paragraphs.append(Paragraph(
                text=current_text, page_start=current_start_page,
                page_end=current_end_page, blocks=current_blocks,
            ))
        return paragraphs
