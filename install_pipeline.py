"""Script d'installation : génère tous les fichiers du pipeline RAG."""

import os
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\dell\Desktop\Projet python - RAG\rag")

FILES = {
    "rag_pipeline/__init__.py": '''"""Pipeline de préparation des données PDF pour vectorisation RAG."""

from .processor import DocumentProcessor
from .schemas import Chunk, Document, Heading, Page, Paragraph, Section, TextBlock
from .config import PipelineConfig

__all__ = [
    "DocumentProcessor",
    "PipelineConfig",
    "Chunk",
    "Document",
    "Heading",
    "Page",
    "Paragraph",
    "Section",
    "TextBlock",
]
''',

    "rag_pipeline/config.py": '''"""Configuration centralisée du pipeline."""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Tous les paramètres ajustables du pipeline."""

    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 50
    chunk_min_chars: int = 30
    chunk_max_chars: int = 2000
    heading_font_ratio: float = 1.15
    heading_font_ratio_strong: float = 1.30
    heading_min_chars: int = 3
    heading_max_chars: int = 200
    skip_patterns: list[str] = None
    unique_word_ratio_min: float = 0.25
    alpha_ratio_min: float = 0.50
    min_meaningful_words: int = 5
    dedup_exact: bool = True
    encoding: str = "utf-8"

    def __post_init__(self) -> None:
        if self.skip_patterns is None:
            self.skip_patterns = [
                r"^\\s*table des matières",
                r"^\\s*copyright",
                r"^\\s*©",
                r"^\\s*all rights reserved",
                r"^\\s*avertissement",
                r"^\\s*mention légale",
                r"^\\s*liste des figures",
                r"^\\s*liste des tableaux",
                r"^\\s*list of (figures|tables)",
                r"^\\s*acknowledgements",
                r"^\\s*remerciements",
            ]
''',

    "rag_pipeline/schemas.py": '''"""Modèles de données immuables pour le pipeline RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Document:
    """Métadonnées du document source (étape 0)."""

    source_id: str
    nom_fichier: str
    chemin_origine: str
    date_ingestion: datetime


@dataclass
class TextBlock:
    """Bloc textuel brut extrait par PyMuPDF (étape 1)."""

    text: str
    bbox: tuple[float, float, float, float]
    font_size: float
    is_bold: bool
    font_name: str
    color: int
    origin: str


@dataclass
class Page:
    """Page extraite avec son contenu brut et ses blocs structurés."""

    doc: Document
    num: int
    raw_text: str
    blocks: list[TextBlock] = field(default_factory=list)
    texte_nettoye: str = ""
    filtered: bool = False
    filter_reason: str | None = None


@dataclass
class Heading:
    """Titre ou sous-titre détecté dans le document."""

    text: str
    level: int
    page: int
    bbox: tuple[float, float, float, float]
    font_size: float


@dataclass
class Paragraph:
    """Paragraphe unitaire au sein d'une section."""

    text: str
    page_start: int
    page_end: int
    blocks: list[TextBlock] = field(default_factory=list)


@dataclass
class Section:
    """Objet sémantique parent : section délimitée par des titres."""

    section_id: str
    source_id: str
    heading: Heading
    paragraphs: list[Paragraph] = field(default_factory=list)
    subsections: list[Section] = field(default_factory=list)

    @property
    def texte_complet(self) -> str:
        """Texte intégral de la section (pour reconstruction contextuelle)."""
        parts = [self.heading.text]
        for para in self.paragraphs:
            parts.append(para.text)
        return "\\n\\n".join(parts)

    @property
    def pages(self) -> list[int]:
        """Numéros de pages couverts par la section."""
        pages = {self.heading.page}
        for para in self.paragraphs:
            pages.add(para.page_start)
            pages.add(para.page_end)
        return sorted(pages)


@dataclass
class Chunk:
    """Unité textuelle fine destinée à la vectorisation."""

    chunk_id: str
    source_id: str
    page: int
    section_id: str
    titre_section: str
    texte_chunk: str
    texte_section_complete: str
    font_size_avg: float | None = None
    is_bold: bool = False
    position_in_section: int = 0
    total_chunks_in_section: int = 0
    quality_score: float = 0.0
    rejected: bool = False
    rejection_reason: str | None = None
    _blocks_refs: list[TextBlock] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Sérialisation pour stockage JSON."""
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "page": self.page,
            "section_id": self.section_id,
            "titre_section": self.titre_section,
            "texte_chunk": self.texte_chunk,
            "texte_section_complete": self.texte_section_complete,
            "font_size_avg": self.font_size_avg,
            "is_bold": self.is_bold,
            "position_in_section": self.position_in_section,
            "total_chunks_in_section": self.total_chunks_in_section,
            "quality_score": self.quality_score,
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
        }
''',

    "rag_pipeline/processor.py": '''"""Orchestrateur principal du pipeline RAG."""

from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .schemas import Chunk, Document
from .steps import (
    Step0Metadata,
    Step1Extract,
    Step2Clean,
    Step3Filter,
    Step4Structure,
    Step5Chunks,
    Step6Quality,
)


class DocumentProcessor:
    """Enchaîne les 7 étapes de préparation des données PDF."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._init_steps()

    def _init_steps(self) -> None:
        """Instancie chaque étape avec la config partagée."""
        self.step0 = Step0Metadata()
        self.step1 = Step1Extract()
        self.step2 = Step2Clean()
        self.step3 = Step3Filter(self.config)
        self.step4 = Step4Structure(self.config)
        self.step5 = Step5Chunks(self.config)
        self.step6 = Step6Quality(self.config)

    def process(self, pdf_path: Path | str) -> list[Chunk]:
        """Exécute le pipeline complet sur un PDF."""
        path = Path(pdf_path)

        doc: Document = self.step0.run(path)
        pages = self.step1.run(doc)
        pages = self.step2.run(pages)
        pages = self.step3.run(pages)
        sections = self.step4.run(pages)
        chunks = self.step5.run(sections)
        chunks = self.step6.run(chunks)

        return chunks

    def process_valid_only(self, pdf_path: Path | str) -> list[Chunk]:
        """Retourne uniquement les chunks acceptés (non rejetés)."""
        all_chunks = self.process(pdf_path)
        return [c for c in all_chunks if not c.rejected]
''',

    "rag_pipeline/_internal/_utils.py": '''"""Fonctions utilitaires partagées par le pipeline."""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..schemas import TextBlock


def normalize_unicode(text: str) -> str:
    """Normalise le texte en NFC et nettoie les caractères de contrôle."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) != "Cc" or ch in "\\t\\n\\r"
    )
    return text


def collapse_whitespace(text: str) -> str:
    """Réduit les espaces multiples et les lignes vides successives."""
    text = re.sub(r"[ \\t]+", " ", text)
    text = re.sub(r"\\n\\s*\\n+", "\\n\\n", text)
    return text.strip()


def merge_broken_lines(text: str) -> str:
    """Fusionne les lignes coupées en milieu de phrase."""
    text = re.sub(r"(\\w)-\\s*\\n\\s*(\\w)", r"\\1\\2", text)
    return text


def approximate_token_count(text: str) -> int:
    """Estimation rapide du nombre de tokens (~0.75 token/mot)."""
    words = text.split()
    return int(len(words) * 1.33)


def chars_from_tokens(token_count: int) -> int:
    """Estimation inverse : tokens -> caractères (~4 caractères/token)."""
    return token_count * 4


def compute_text_hash(text: str) -> str:
    """Hash SHA-256 normalisé pour déduplication exacte."""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def generate_id(prefix: str = "doc") -> str:
    """Génère un identifiant unique court."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def is_in_footer_or_header(
    bbox: tuple[float, float, float, float],
    page_height: float,
    margin_ratio: float = 0.08,
) -> bool:
    """Détermine si une boîte de texte est dans le header ou le footer."""
    _, y0, _, y1 = bbox
    margin = page_height * margin_ratio
    return y1 < margin or y0 > page_height - margin


def block_to_text(block: TextBlock) -> str:
    """Extrait le texte brut d'un bloc structuré."""
    return block.text
''',

    "rag_pipeline/_internal/_text_analysis.py": '''"""Analyse linguistique et métriques de qualité du texte."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class QualityMetrics:
    """Métriques calculées sur un chunk pour évaluation qualité."""

    char_count: int
    word_count: int
    unique_word_ratio: float
    alpha_ratio: float
    digit_ratio: float
    punctuation_ratio: float
    avg_word_length: float
    is_meaningful: bool
    rejection_reason: str | None = None


def compute_quality_metrics(text: str) -> QualityMetrics:
    """Calcule l'ensemble des métriques qualité sur un texte."""
    if not text:
        return QualityMetrics(
            char_count=0, word_count=0, unique_word_ratio=0.0,
            alpha_ratio=0.0, digit_ratio=0.0, punctuation_ratio=0.0,
            avg_word_length=0.0, is_meaningful=False,
            rejection_reason="Texte vide",
        )

    clean = text.strip()
    char_count = len(clean)

    words = re.findall(r"[\\w'-]+", clean.lower())
    word_count = len(words)

    unique_words = set(words)
    unique_word_ratio = len(unique_words) / word_count if word_count > 0 else 0.0

    alpha_chars = sum(1 for c in clean if c.isalpha())
    digit_chars = sum(1 for c in clean if c.isdigit())
    punct_chars = sum(1 for c in clean if c in ".,;:!?-–—'\\"()[]{}«»")

    alpha_ratio = alpha_chars / char_count if char_count > 0 else 0.0
    digit_ratio = digit_chars / char_count if char_count > 0 else 0.0
    punctuation_ratio = punct_chars / char_count if char_count > 0 else 0.0

    avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0.0

    is_meaningful = (
        word_count >= 5
        and unique_word_ratio >= 0.15
        and alpha_ratio >= 0.30
        and digit_ratio < 0.70
        and avg_word_length >= 2.0
    )

    rejection_reason = None
    if not is_meaningful:
        reasons = []
        if word_count < 5:
            reasons.append(f"trop peu de mots ({word_count})")
        if unique_word_ratio < 0.15:
            reasons.append(f"répétition excessive (ratio uniques={unique_word_ratio:.2f})")
        if alpha_ratio < 0.30:
            reasons.append(f"trop peu de caractères alphabétiques ({alpha_ratio:.2f})")
        if digit_ratio >= 0.70:
            reasons.append(f"dominé par des chiffres ({digit_ratio:.2f})")
        rejection_reason = "; ".join(reasons)

    return QualityMetrics(
        char_count=char_count,
        word_count=word_count,
        unique_word_ratio=unique_word_ratio,
        alpha_ratio=alpha_ratio,
        digit_ratio=digit_ratio,
        punctuation_ratio=punctuation_ratio,
        avg_word_length=avg_word_length,
        is_meaningful=is_meaningful,
        rejection_reason=rejection_reason,
    )


def is_pure_number_punctuation(text: str) -> bool:
    """Détecte si le texte est composé uniquement de nombres et ponctuation."""
    stripped = text.strip()
    if not stripped:
        return True
    alpha_count = sum(1 for c in stripped if c.isalpha())
    return alpha_count == 0
''',

    "rag_pipeline/_internal/_heading_detector.py": '''"""Heuristique de détection des titres dans un document PDF."""

from __future__ import annotations

import statistics
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..schemas import TextBlock, Heading


class HeadingDetector:
    """Détecte les titres et sous-titres par analyse typographique."""

    def __init__(
        self,
        font_ratio: float = 1.15,
        font_ratio_strong: float = 1.30,
        min_chars: int = 3,
        max_chars: int = 200,
    ) -> None:
        self.font_ratio = font_ratio
        self.font_ratio_strong = font_ratio_strong
        self.min_chars = min_chars
        self.max_chars = max_chars

    def detect(self, blocks: list[TextBlock], page_height: float) -> list[Heading]:
        """Retourne la liste des headings détectés, triés par position."""
        if not blocks:
            return []

        font_sizes = [b.font_size for b in blocks if b.text.strip()]
        if not font_sizes:
            return []

        mode_size = self._compute_mode(font_sizes)

        headings = []
        for block in blocks:
            text = block.text.strip()
            if not text:
                continue
            if not (self.min_chars <= len(text) <= self.max_chars):
                continue

            is_heading = self._is_heading_block(block, mode_size)
            if is_heading:
                level = self._infer_level(block.font_size, mode_size)
                headings.append(
                    Heading(
                        text=text,
                        level=level,
                        page=0,
                        bbox=block.bbox,
                        font_size=block.font_size,
                    )
                )

        headings.sort(key=lambda h: (h.page, h.bbox[1]))
        return headings

    def _compute_mode(self, values: list[float]) -> float:
        """Calcule le mode (valeur la plus fréquente) des tailles de police."""
        rounded = [round(v, 1) for v in values]
        counter = Counter(rounded)
        most_common = counter.most_common(1)
        return most_common[0][0] if most_common else statistics.median(values)

    def _is_heading_block(self, block: TextBlock, mode_size: float) -> bool:
        """Détermine si un bloc textuel est un titre."""
        text = block.text.strip()
        if not text:
            return False

        ratio = block.font_size / mode_size if mode_size > 0 else 1.0

        if ratio >= self.font_ratio and block.is_bold:
            return True

        if ratio >= self.font_ratio_strong:
            return True

        if block.is_bold and ratio >= 1.0 and len(text) < 100:
            return True

        return False

    def _infer_level(self, font_size: float, mode_size: float) -> int:
        """Infère le niveau hiérarchique (1, 2, 3...) depuis la taille relative."""
        ratio = font_size / mode_size if mode_size > 0 else 1.0
        if ratio >= 1.5:
            return 1
        elif ratio >= 1.25:
            return 2
        elif ratio >= 1.15:
            return 3
        else:
            return 4
''',

    "rag_pipeline/steps/__init__.py": '''"""Étapes du pipeline RAG (0 à 6)."""

from .step0_metadata import Step0Metadata
from .step1_extract import Step1Extract
from .step2_clean import Step2Clean
from .step3_filter import Step3Filter
from .step4_structure import Step4Structure
from .step5_chunks import Step5Chunks
from .step6_quality import Step6Quality

STEPS = [
    Step0Metadata,
    Step1Extract,
    Step2Clean,
    Step3Filter,
    Step4Structure,
    Step5Chunks,
    Step6Quality,
]

__all__ = [
    "STEPS",
    "Step0Metadata",
    "Step1Extract",
    "Step2Clean",
    "Step3Filter",
    "Step4Structure",
    "Step5Chunks",
    "Step6Quality",
]
''',

    "rag_pipeline/steps/step0_metadata.py": '''"""Étape 0 : Métadonnées du document."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ..schemas import Document
from .._internal._utils import generate_id


class Step0Metadata:
    """Génère les métadonnées initiales du document."""

    def run(self, pdf_path: Path) -> Document:
        path = Path(pdf_path).resolve()
        return Document(
            source_id=generate_id("doc"),
            nom_fichier=path.name,
            chemin_origine=str(path),
            date_ingestion=datetime.now(timezone.utc),
        )
''',

    "rag_pipeline/steps/step1_extract.py": '''"""Étape 1 : Extraction du contenu textuel du PDF avec PyMuPDF."""

from __future__ import annotations

import fitz  # PyMuPDF

from ..schemas import Document, Page, TextBlock


class Step1Extract:
    """Extrait le texte brut et structuré page par page."""

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
''',

    "rag_pipeline/steps/step2_clean.py": '''"""Étape 2 : Nettoyage du texte extrait."""

from __future__ import annotations

from ..schemas import Page
from .._internal._utils import (
    collapse_whitespace,
    merge_broken_lines,
    normalize_unicode,
)


class Step2Clean:
    """Nettoie le texte brut : espaces, lignes coupées, Unicode, fusion paragraphes."""

    def run(self, pages: list[Page]) -> list[Page]:
        for page in pages:
            page.texte_nettoye = self._clean_text(page.raw_text)
            page.blocks = self._clean_blocks(page.blocks)
        return pages

    def _clean_text(self, text: str) -> str:
        text = normalize_unicode(text)
        text = merge_broken_lines(text)
        text = collapse_whitespace(text)
        return text

    def _clean_blocks(self, blocks: list) -> list:
        from copy import copy
        cleaned = []
        for block in blocks:
            new_block = copy(block)
            new_block.text = self._clean_text(block.text)
            if new_block.text.strip():
                cleaned.append(new_block)
        return cleaned
''',

    "rag_pipeline/steps/step3_filter.py": '''"""Étape 3 : Filtrage des contenus non pertinents."""

from __future__ import annotations

import re

from ..schemas import Page
from ..config import PipelineConfig


class Step3Filter:
    """Filtre les pages non pertinentes (couverture, TOC, copyright, etc.)."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def run(self, pages: list[Page]) -> list[Page]:
        for page in pages:
            self._evaluate_page(page)

        kept = [p for p in pages if not p.filtered]
        if not kept and pages:
            kept = [p for p in pages if len(p.texte_nettoye.strip()) > 50]

        return kept

    def _evaluate_page(self, page: Page) -> None:
        text = page.texte_nettoye.lower().strip()

        if len(text) < 50:
            page.filtered = True
            page.filter_reason = "Page quasi vide"
            return

        for pattern in self.config.skip_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                page.filtered = True
                page.filter_reason = f"Pattern match: {pattern}"
                return

        alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
        if alpha_ratio < 0.10 and len(text) > 200:
            page.filtered = True
            page.filter_reason = "Trop peu de texte alphabétique"
            return
''',

    "rag_pipeline/steps/step4_structure.py": '''"""Étape 4 : Analyse de la structure — titres, sections, paragraphes."""

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
''',

    "rag_pipeline/steps/step5_chunks.py": '''"""Étape 5 : Construction des chunks et rattachement aux sections parents."""

from __future__ import annotations

from ..schemas import Chunk, Paragraph, Section
from .._internal._utils import approximate_token_count, chars_from_tokens, generate_id
from ..config import PipelineConfig


class Step5Chunks:
    """Construit les chunks fins liés à leurs sections parentes."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def run(self, sections: list[Section]) -> list[Chunk]:
        all_chunks = []
        for section in sections:
            section_chunks = self._process_section(section)
            all_chunks.extend(section_chunks)
        return all_chunks

    def _process_section(self, section: Section) -> list[Chunk]:
        chunks = []
        section_text = section.texte_complet

        for idx, paragraph in enumerate(section.paragraphs):
            para_chunks = self._chunk_paragraph(paragraph, section, idx, len(section.paragraphs))
            for chunk in para_chunks:
                chunk.texte_section_complete = section_text
            chunks.extend(para_chunks)

        for i, chunk in enumerate(chunks):
            chunk.position_in_section = i
            chunk.total_chunks_in_section = len(chunks)
        return chunks

    def _chunk_paragraph(self, paragraph: Paragraph, section: Section, para_index: int, total_paras: int) -> list[Chunk]:
        text = paragraph.text.strip()
        if not text:
            return []

        prefixed = f"{section.heading.text}\\n\\n{text}" if section.heading.text != "(sans titre)" else text

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
                display_text = chunk_text
                if chunk_idx == 0 and section.heading.text != "(sans titre)":
                    display_text = f"{section.heading.text}\\n\\n{chunk_text}"
                chunks.append(self._create_chunk(display_text, section, paragraph.page_start, para_index))

            start = end - overlap_chars
            chunk_idx += 1
        return chunks

    def _create_chunk(self, text: str, section: Section, page: int, para_index: int) -> Chunk:
        return Chunk(
            chunk_id=generate_id("chunk"), source_id=section.source_id,
            page=page, section_id=section.section_id,
            titre_section=section.heading.text,
            texte_chunk=text, texte_section_complete="",
            position_in_section=para_index, total_chunks_in_section=0,
            quality_score=0.0,
        )
''',

    "rag_pipeline/steps/step6_quality.py": '''"""Étape 6 : Contrôle qualité des chunks."""

from __future__ import annotations

from ..schemas import Chunk
from .._internal._text_analysis import compute_quality_metrics
from .._internal._utils import compute_text_hash
from ..config import PipelineConfig


class Step6Quality:
    """Filtre et note les chunks selon les critères de qualité."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def run(self, chunks: list[Chunk]) -> list[Chunk]:
        seen_hashes: set[str] = set()
        for chunk in chunks:
            self._evaluate_chunk(chunk, seen_hashes)
            if not chunk.rejected and self.config.dedup_exact:
                seen_hashes.add(compute_text_hash(chunk.texte_chunk))
        return chunks

    def _evaluate_chunk(self, chunk: Chunk, seen_hashes: set[str]) -> None:
        text = chunk.texte_chunk

        if not text or not text.strip():
            chunk.rejected = True
            chunk.rejection_reason = "Chunk vide"
            return

        if len(text) < self.config.chunk_min_chars:
            chunk.rejected = True
            chunk.rejection_reason = f"Trop court ({len(text)} < {self.config.chunk_min_chars})"
            return

        if len(text) > self.config.chunk_max_chars:
            chunk.rejected = True
            chunk.rejection_reason = f"Trop long ({len(text)} > {self.config.chunk_max_chars})"
            return

        metrics = compute_quality_metrics(text)
        if not metrics.is_meaningful:
            chunk.rejected = True
            chunk.rejection_reason = metrics.rejection_reason
            return

        if self.config.dedup_exact:
            text_hash = compute_text_hash(text)
            if text_hash in seen_hashes:
                chunk.rejected = True
                chunk.rejection_reason = "Doublon exact"
                return

        if metrics.word_count < self.config.min_meaningful_words:
            chunk.rejected = True
            chunk.rejection_reason = f"Trop peu de mots ({metrics.word_count})"
            return

        chunk.quality_score = self._compute_score(metrics)
        chunk.rejection_reason = None

    def _compute_score(self, metrics) -> float:
        scores = [
            min(metrics.unique_word_ratio / 0.5, 1.0),
            min(metrics.alpha_ratio / 0.8, 1.0),
            min(metrics.word_count / 50, 1.0),
            1.0 - metrics.digit_ratio,
        ]
        return sum(scores) / len(scores)
''',

    "rag_pipeline/scripts/run_pipeline.py": '''#!/usr/bin/env python3
"""CLI : lance le pipeline sur les PDFs du dossier INPUT_DIR."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ── PATHS CONFIGURABLES ──
PROJECT_ROOT = Path(r"C:\\Users\\dell\\Desktop\\Projet python - RAG\\rag")
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_FILE = PROJECT_ROOT / "data" / "output" / "chunks.jsonl"
VALID_ONLY = True
CHUNK_MAX_TOKENS = 512
# ──────────────────────────

sys.path.insert(0, str(PROJECT_ROOT))

from rag_pipeline import DocumentProcessor, PipelineConfig


def main() -> int:
    config = PipelineConfig(chunk_max_tokens=CHUNK_MAX_TOKENS)
    processor = DocumentProcessor(config)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"Aucun PDF trouvé dans : {INPUT_DIR}")
        return 1

    all_chunks = []
    for pdf in pdf_files:
        print(f"Traitement : {pdf.name}")
        chunks = processor.process(pdf)
        if VALID_ONLY:
            chunks = [c for c in chunks if not c.rejected]
        all_chunks.extend(chunks)
        print(f"  -> {len(chunks)} chunks générés")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\\n")

    print(f"\\nExporté : {OUTPUT_FILE} ({len(all_chunks)} chunks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
''',

    "rag_pipeline/scripts/inspect_chunks.py": '''#!/usr/bin/env python3
"""Outil d'inspection : analyse les chunks rejetés vs acceptés."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# ── PATH CONFIGURABLE ──
PROJECT_ROOT = Path(r"C:\\Users\\dell\\Desktop\\Projet python - RAG\\rag")
CHUNKS_FILE = PROJECT_ROOT / "data" / "output" / "chunks.jsonl"
# ──────────────────────

sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    if not CHUNKS_FILE.exists():
        print(f"Fichier non trouvé : {CHUNKS_FILE}")
        print("Vérifie que le pipeline a bien généré le fichier de sortie.")
        return 1

    chunks = []
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    total = len(chunks)
    accepted = [c for c in chunks if not c["rejected"]]
    rejected = [c for c in chunks if c["rejected"]]

    print(f"=== Rapport d'inspection : {CHUNKS_FILE.name} ===\\n")
    print(f"Total chunks  : {total}")
    if total > 0:
        print(f"Acceptés    : {len(accepted)} ({len(accepted)/total*100:.1f}%)")
        print(f"Rejetés     : {len(rejected)} ({len(rejected)/total*100:.1f}%)\\n")
    else:
        print("Aucun chunk trouvé.")
        return 0

    if rejected:
        print("--- Raisons de rejet ---")
        reasons = Counter(c["rejection_reason"] for c in rejected)
        for reason, count in reasons.most_common():
            print(f"  {count:4d} × {reason}")

        print(f"\\n--- Exemples de rejetés (3 premiers) ---")
        for c in rejected[:3]:
            preview = c["texte_chunk"][:120].replace("\\n", " ")
            print(f"  [{c['chunk_id']}] {c['rejection_reason']}")
            print(f"      → {preview}...\\n")

    if accepted:
        scores = [c["quality_score"] for c in accepted]
        print(f"--- Scores qualité (acceptés) ---")
        print(f"  Min : {min(scores):.3f}")
        print(f"  Max : {max(scores):.3f}")
        print(f"  Moy : {sum(scores)/len(scores):.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
''',
}


def install():
    print("=== Installation du pipeline RAG ===\n")
    
    for rel_path, content in FILES.items():
        full_path = PROJECT_ROOT / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Écriture du fichier
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"✅ {rel_path}")
    
    # Fichiers __init__.py vides pour les packages
    for init_path in [
        "rag_pipeline/_internal/__init__.py",
        "rag_pipeline/tests/__init__.py",
        "rag_pipeline/tests/fixtures/__init__.py",
    ]:
        full_path = PROJECT_ROOT / init_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        if not full_path.exists():
            with open(full_path, "w", encoding="utf-8") as f:
                f.write("")
            print(f"✅ {init_path}")
    
    print(f"\n{'='*50}")
    print("Installation terminée !")
    print(f"{'='*50}")
    print(f"\nProchaines étapes :")
    print(f"  1. Créer le dossier : {PROJECT_ROOT / 'data' / 'input'}")
    print(f"  2. Y placer tes fichiers PDF")
    print(f"  3. Lancer : python {PROJECT_ROOT / 'rag_pipeline' / 'scripts' / 'run_pipeline.py'}")


if __name__ == "__main__":
    install()