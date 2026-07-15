"""Étape 5 : Construction des chunks et rattachement aux sections parents."""

from __future__ import annotations

from ..schemas import Chunk, Paragraph, Section, HierarchyPath
from .._internal._utils import approximate_token_count, chars_from_tokens, generate_id
from ..config import PipelineConfig

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class Step5Chunks:
    """Construit les chunks fins liés à leurs sections parentes."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def run(self, sections: list[Section]) -> list[Chunk]:
        if self.config.hierarchy_aware_chunking and LANGCHAIN_AVAILABLE:
            return self._run_hierarchy_aware(sections)
        else:
            return self._run_simple(sections)

    # =====================================================================
    # MODE SIMPLE (original, sans hiérarchie enrichie)
    # =====================================================================
    def _run_simple(self, sections: list[Section]) -> list[Chunk]:
        all_chunks = []
        for section in sections:
            section_chunks = self._process_section_simple(section)
            all_chunks.extend(section_chunks)
        return all_chunks

    def _process_section_simple(self, section: Section) -> list[Chunk]:
        chunks = []
        section_text = section.texte_complet

        for idx, paragraph in enumerate(section.paragraphs):
            para_chunks = self._chunk_paragraph_simple(paragraph, section, idx, len(section.paragraphs))
            for chunk in para_chunks:
                chunk.texte_section_complete = section_text
            chunks.extend(para_chunks)

        for i, chunk in enumerate(chunks):
            chunk.position_in_section = i
            chunk.total_chunks_in_section = len(chunks)
        return chunks

    def _chunk_paragraph_simple(self, paragraph: Paragraph, section: Section, para_index: int, total_paras: int) -> list[Chunk]:
        text = paragraph.text.strip()
        if not text:
            return []

        prefixed = f"{section.heading.text}\n\n{text}" if section.heading.text != "(sans titre)" else text

        if len(prefixed) <= self.config.chunk_max_chars:
            if approximate_token_count(prefixed) <= self.config.chunk_max_tokens:
                return [self._create_chunk(prefixed, section, paragraph.page_start, para_index)]

        return self._split_into_chunks_simple(text, section, paragraph, para_index)

    def _split_into_chunks_simple(self, text: str, section: Section, paragraph: Paragraph, para_index: int) -> list[Chunk]:
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
                display_text = chunk_text
                if chunk_idx == 0 and section.heading.text != "(sans titre)":
                    display_text = f"{section.heading.text}\n\n{chunk_text}"
                chunks.append(self._create_chunk(display_text, section, paragraph.page_start, para_index))

            start = end - overlap_chars
            chunk_idx += 1
        return chunks

    # =====================================================================
    # MODE HIÉRARCHIQUE (nouveau, adapté de ton code)
    # =====================================================================
    def _run_hierarchy_aware(self, sections: list[Section]) -> list[Chunk]:
        """
        Chunking 'hierarchy-aware' : accumule les paragraphes tant qu'ils
        appartiennent à la même feuille de hiérarchie.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size_chars,
            chunk_overlap=self.config.chunk_overlap_chars,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Construire la pile de hiérarchie depuis les sections
        all_chunks = []
        buffer_texts = []
        buffer_pages = []
        current_meta = None
        chunk_counter = 0

        def flush():
            nonlocal buffer_texts, buffer_pages, chunk_counter, all_chunks
            if not buffer_texts or current_meta is None:
                buffer_texts, buffer_pages = [], []
                return

            combined = "\n\n".join(buffer_texts)
            pieces = splitter.split_text(combined) if len(combined) > self.config.chunk_size_chars else [combined]
            pages = [p for p in buffer_pages if p is not None]
            page_debut = min(pages) if pages else None
            page_fin = max(pages) if pages else None

            for piece in pieces:
                chunk_counter += 1
                all_chunks.append(
                    self._create_hierarchy_chunk(
                        piece.strip(),
                        current_meta,
                        page_debut,
                        page_fin,
                        chunk_counter,
                    )
                )
            buffer_texts, buffer_pages = [], []

        for section in sections:
            # Déterminer le chemin hiérarchique
            hierarchy = self._build_hierarchy_path(section)

            for para in section.paragraphs:
                text = para.text.strip()
                if not text:
                    continue

                # Clé de hiérarchie pour comparaison
                path_key = tuple(hierarchy.path_titles)
                current_key = tuple(current_meta.path_titles) if current_meta else None

                if current_meta is not None and path_key != current_key:
                    flush()

                if current_meta is None:
                    current_meta = hierarchy

                buffer_texts.append(text)
                buffer_pages.append(para.page_start)

                # Garde-fou : flush si buffer trop gros
                if sum(len(t) for t in buffer_texts) > self.config.chunk_size_chars * 4:
                    flush()
                    current_meta = hierarchy

        flush()
        return all_chunks

    def _build_hierarchy_path(self, section: Section) -> HierarchyPath:
        """Construit le chemin hiérarchique depuis une section."""
        # On utilise le niveau du heading pour déterminer chapitre/titre/sous-titre
        level = section.heading.level
        title = section.heading.text

        # Parcourir les sections parentes (si disponibles)
        path_titles = [title]

        chapitre = title if level == 1 else None
        titre = title if level == 2 else None
        sous_titre = title if level >= 3 else None

        return HierarchyPath(
            chapitre=chapitre,
            titre=titre,
            sous_titre=sous_titre,
            path_titles=path_titles,
        )

    # =====================================================================
    # FACTORIES
    # =====================================================================
    def _create_chunk(self, text: str, section: Section, page: int, para_index: int) -> Chunk:
        return Chunk(
            chunk_id=generate_id("chunk"), source_id=section.source_id,
            page=page, section_id=section.section_id,
            titre_section=section.heading.text,
            texte_chunk=text, texte_section_complete="",
            position_in_section=para_index, total_chunks_in_section=0,
            quality_score=0.0,
        )

    def _create_hierarchy_chunk(
        self, text: str, hierarchy: HierarchyPath,
        page_debut: int | None, page_fin: int | None,
        chunk_counter: int,
    ) -> Chunk:
        return Chunk(
            chunk_id=f"chunk_{chunk_counter:05d}",
            source_id="hierarchy",  # sera mis à jour par l'appelant
            page=page_debut or 0,
            section_id="",
            titre_section=hierarchy.titre or hierarchy.chapitre or "",
            texte_chunk=text,
            texte_section_complete="",
            position_in_section=0,
            total_chunks_in_section=0,
            quality_score=0.0,
            chapitre=hierarchy.chapitre,
            titre=hierarchy.titre,
            sous_titre=hierarchy.sous_titre,
            hierarchy_path=hierarchy.path_titles,
            page_debut=page_debut,
            page_fin=page_fin,
        )