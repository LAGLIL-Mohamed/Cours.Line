from __future__ import annotations

from typing import Optional

from docling_core.types.doc.document import DoclingDocument

from extract_hierarchy.config import (
    KEYWORD_LEVELS,
    RE_KEYWORD,
    RE_NUMBERED_NO_PAGE,
    RE_NUMBERED_WITH_PAGE,
)
from extract_hierarchy.models import TocEntry
from extract_hierarchy.utils import is_toc_heading, normalize


def find_toc_pages(
    doc: DoclingDocument, max_scan_pages: int, toc_max_span: int
) -> Optional[set[int]]:
    toc_page = None
    for item in doc.texts:
        page = item.prov[0].page_no if getattr(item, "prov", None) else None
        if page is None or page > max_scan_pages:
            continue
        text = (item.text or "").strip()
        if not text:
            continue
        if is_toc_heading(text):
            toc_page = page
            break

    if toc_page is None:
        return None
    return set(range(toc_page, toc_page + toc_max_span + 1))


def parse_toc_line(line: str) -> Optional[dict]:
    line = line.strip()
    if not line or len(line) < 2:
        return None

    m = RE_NUMBERED_WITH_PAGE.match(line)
    if m:
        num = m.group("num")
        return {
            "numbering": num,
            "title": m.group("title").strip(" ._-"),
            "level": num.count(".") + 1,
            "page": int(m.group("page")),
        }

    m = RE_KEYWORD.match(line)
    if m:
        kw = m.group("kw").lower()
        page = m.group("page")
        return {
            "numbering": f"{kw} {m.group('num')}",
            "title": m.group("title").strip(" ._-"),
            "level": KEYWORD_LEVELS.get(kw, 1),
            "page": int(page) if page else None,
        }

    m = RE_NUMBERED_NO_PAGE.match(line)
    if m:
        num = m.group("num")
        title = m.group("title").strip(" ._-")
        if len(title) > 120 or title.endswith((".", ",", ";")):
            return None
        return {
            "numbering": num,
            "title": title,
            "level": num.count(".") + 1,
            "page": None,
        }

    return None


def collect_toc_raw_lines(doc: DoclingDocument, toc_pages: set[int]) -> list[str]:
    lines: list[str] = []

    for item in doc.texts:
        page = item.prov[0].page_no if getattr(item, "prov", None) else None
        if page not in toc_pages:
            continue
        text = (item.text or "").strip()
        if text and not is_toc_heading(text):
            lines.append(text)

    for table in doc.tables:
        page = table.prov[0].page_no if getattr(table, "prov", None) else None
        if page not in toc_pages:
            continue
        try:
            grid = table.data.grid
        except Exception:
            continue
        for row in grid:
            cells = [c.text.strip() for c in row if getattr(c, "text", None)]
            cells = [c for c in cells if c]
            if not cells:
                continue
            if len(cells) >= 2 and cells[-1].isdigit():
                title_part = " ".join(cells[:-1])
                lines.append(f"{title_part} .... {cells[-1]}")
            else:
                lines.append(" ".join(cells))

    return lines


def extract_toc(
    doc: DoclingDocument, max_scan_pages: int, toc_max_span: int
) -> list[TocEntry]:
    toc_pages = find_toc_pages(doc, max_scan_pages, toc_max_span)
    if not toc_pages:
        return []

    raw_lines = collect_toc_raw_lines(doc, toc_pages)
    entries: list[TocEntry] = []
    for line in raw_lines:
        parsed = parse_toc_line(line)
        if parsed:
            entries.append(
                TocEntry(
                    raw_line=line,
                    numbering=parsed["numbering"],
                    title=parsed["title"],
                    level=parsed["level"],
                    page_hint=parsed["page"],
                )
            )
    return entries