"""Etape 4 : Analyse de la structure - titres, sections, paragraphes."""

from __future__ import annotations

from ..schemas import Heading, Page, Paragraph, Section, TextBlock
from .._internal._heading_detector import RobustHeadingDetector
from .._internal._utils import generate_id
from ..config import PipelineConfig


class Step4Structure:
    """Identifie la structure logique et construit les objets semantiques."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.detector = RobustHeadingDetector(self.config)

    def run(self, pages: list[Page]) -> list[Section]:
        if not pages:
            return []

        doc_path = pages[0].doc.chemin_origine
        headings = self.detector.detect(doc_path, pages)

        blocks_by_page: dict[int, list[tuple[int, TextBlock]]] = {}
        for page in pages:
            blocks_by_page[page.num] = [(page.num, b) for b in page.blocks]

        sections = self._build_hierarchy(headings, blocks_by_page, pages[0].doc.source_id)
        return sections

    def _build_hierarchy(
        self,
        headings: list[Heading],
        blocks_by_page: dict[int, list[tuple[int, TextBlock]]],
        source_id: str,
    ) -> list[Section]:
        if not headings:
            all_blocks = []
            for page_num in sorted(blocks_by_page):
                all_blocks.extend(blocks_by_page[page_num])
            paragraphs = self._blocks_to_paragraphs(all_blocks)
            return [Section(
                section_id=generate_id("sec"),
                source_id=source_id,
                heading=Heading(text="(document entier)", level=1, page=1, bbox=(0,0,0,0), font_size=11.0),
                paragraphs=paragraphs,
                hierarchy_path=["(document entier)"],
            )]

        root_sections: list[Section] = []
        stack: list[Section] = []

        for i, heading in enumerate(headings):
            section_blocks = self._extract_blocks_between(
                heading, headings[i + 1] if i + 1 < len(headings) else None, blocks_by_page
            )
            paragraphs = self._blocks_to_paragraphs(section_blocks)

            section = Section(
                section_id=generate_id("sec"),
                source_id=source_id,
                heading=heading,
                paragraphs=paragraphs,
            )

            if not stack:
                root_sections.append(section)
                stack.append(section)
                continue

            while stack and stack[-1].heading.level >= heading.level:
                stack.pop()

            if stack:
                parent = stack[-1]
                section.parent_id = parent.section_id
                parent.subsections.append(section)
                section.hierarchy_path = parent.hierarchy_path + [heading.text]
            else:
                root_sections.append(section)
                section.hierarchy_path = [heading.text]

            stack.append(section)

        return root_sections

    def _extract_blocks_between(
        self,
        start_heading: Heading,
        end_heading: Heading | None,
        blocks_by_page: dict[int, list[tuple[int, TextBlock]]],
    ) -> list[tuple[int, TextBlock]]:
        result = []
        start_page = start_heading.page
        end_page = end_heading.page if end_heading else max(blocks_by_page.keys())

        for page_num in range(start_page, end_page + 1):
            if page_num not in blocks_by_page:
                continue
            for pnum, block in blocks_by_page[page_num]:
                if page_num == start_page:
                    if block.bbox[3] < start_heading.bbox[1]:
                        continue
                    if block.text.strip() == start_heading.text:
                        continue
                if end_heading and page_num == end_page:
                    if block.bbox[1] > end_heading.bbox[1]:
                        break
                result.append((pnum, block))
        return result

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
                and abs(block.font_size - current_blocks[-1].font_size) < 0.5
            )

            if should_merge:
                current_text += " " + text
                current_end_page = page_num
                current_blocks.append(block)
            else:
                if current_text:
                    paragraphs.append(Paragraph(
                        text=current_text,
                        page_start=current_start_page,
                        page_end=current_end_page,
                        blocks=current_blocks,
                    ))
                current_text = text
                current_start_page = page_num
                current_end_page = page_num
                current_blocks = [block]

        if current_text:
            paragraphs.append(Paragraph(
                text=current_text,
                page_start=current_start_page,
                page_end=current_end_page,
                blocks=current_blocks,
            ))
        return paragraphs
