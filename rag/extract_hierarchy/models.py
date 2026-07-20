from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class TocEntry:
    raw_line: str
    numbering: Optional[str]
    title: str
    level: int
    page_hint: Optional[int]
    matched: bool = False


@dataclass
class HeadingNode:
    id: str
    level: int
    label_type: str
    title: str
    page_start: Optional[int]
    source: str


@dataclass
class ContentBlock:
    text: str
    page: Optional[int]
    hierarchy_path: list
    chapitre: Optional[str]
    titre: Optional[str]
    sous_titre: Optional[str]


@dataclass
class Chunk:
    chunk_id: str
    text: str
    chapitre: Optional[str]
    titre: Optional[str]
    sous_titre: Optional[str]
    hierarchy_path: list
    page_debut: Optional[int]
    page_fin: Optional[int]