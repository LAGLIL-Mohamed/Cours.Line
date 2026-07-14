"""Etape 5 : Construction des chunks et rattachement aux sections parents."""

from __future__ import annotations

from ..schemas import Chunk, Paragraph, Section
from .._internal._utils import approximate_token_count, chars_from_tokens, generate_id
from ..config import PipelineConfig


class Step5Chunks:
    """Construit les chunks fins lies a leurs sections parentes."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def run(self, sections: list[Section]) -> list[Chunk]:
        all_chunks = []
        for section in sections:
            section_chunks = self._process_section_and_subsections(section)
            all_chunks.extend(section_chunks)
        return all_chunks

    def _process_section_and_subsections(self, section: Section) -> list[Chunk]:
        chunks = []
        chunks.extend(self._process_single_section(section))
        for sub in section.subsections:
            chunks.extend(self._process_section_and_subsections(sub))
        return chunks

    def _process_single_section(self, section: Section) -> list[Chunk]:
        chunks = []
        section_text = section.texte_complet
        parent_ids = self._collect_parent_ids(section)

        for idx, paragraph in enumerate(section.paragraphs):
            para_chunks = self._chunk_paragraph(paragraph, section, idx, len(section.paragraphs))
            for chunk in para_chunks:
                chunk.texte_section_complete = section_text
                chunk.hierarchy_level = section.heading.level
                chunk.hierarchy_path = section.hierarchy_path.copy()
                chunk.chapter_title = section.chapter_title
                chunk.parent_sections = parent_ids
                chunk.titre_section = section.heading.text
            chunks.extend(para_chunks)

        for i, chunk in enumerate(chunks):
            chunk.position_in_section = i
            chunk.total_chunks_in_section = len(chunks)
        return chunks

    def _collect_parent_ids(self, section: Section) -> list[str]:
        return [section.parent_id] if section.parent_id else []

    def _chunk_paragraph(self, paragraph: Paragraph, section: Section, para_index: int, total_paras: int) -> list[Chunk]:
        text = paragraph.text.strip()
        if not text:
            return []

        if section.heading.text != "(document entier)":
            prefixed = section.heading.text + "\n\n" + text
        else:
            prefixed = text

        if len(prefixed) <= self.config.chunk_max_chars:
            if approximate_token_count(prefixed) <= self.config.chunk_max_tokens:
                return [self._create_chunk(prefixed, section, paragraph.page_start, para_index)]

        return self._split_into_chunks(text, section, paragraph, para_index)

    def _split_into_chunks(self, text: str, section: Section, paragraph: Paragraph, para_index: int) -> list[Chunk]:
        max_chars = self.config.chunk_max_chars
        overlap_chars = chars_from_tokens(self.config.chunk_overlap_tokens)
        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = start + max_chars
            if end < len(text):
                sentence_end = text.rfind(". ", start, end)
                if sentence_end > start + max_chars // 2:
                    end = sentence_end + 1
                else:
                    space = text.rfind(" ", start, end)
                    if space > start:
                        end = space

            chunk_text = text[start:end].strip()
            if chunk_text:
                if chunk_idx == 0 and section.heading.text != "(document entier)":
                    display_text = section.heading.text + "\n\n" + chunk_text
                else:
                    display_text = chunk_text
                chunks.append(self._create_chunk(display_text, section, paragraph.page_start, para_index))

            start = end - overlap_chars
            chunk_idx += 1
        return chunks

    def _create_chunk(self, text: str, section: Section, page: int, para_index: int) -> Chunk:
        return Chunk(
            chunk_id=generate_id("chunk"),
            source_id=section.source_id,
            page=page,
            section_id=section.section_id,
            titre_section=section.heading.text,
            texte_chunk=text,
            texte_section_complete="",
            hierarchy_level=section.heading.level,
            hierarchy_path=section.hierarchy_path.copy(),
            chapter_title=section.chapter_title,
            parent_sections=[],
            position_in_section=para_index,
            total_chunks_in_section=0,
            quality_score=0.0,
        )