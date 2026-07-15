#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_hierarchy.py
=====================================================================
Extraction de la hiérarchie (Chapitre / Titre / Sous-titre) d'un PDF
numérique (natif, non scanné) avec Docling, en vue d'alimenter un
pipeline RAG type NotebookLM.

PIPELINE (en 4 étapes)
-----------------------
1. CONVERSION : Docling convertit le PDF en `DoclingDocument` (couche
   layout : titres, section_header avec niveau, paragraphes, listes...).

2. DETECTION DU SOMMAIRE (TOC) : on scanne les N premières pages à la
   recherche d'une page "Sommaire / Table des matières / Contents".
   Si trouvée, on parse son contenu (texte ET tableaux) avec des regex
   pour reconstruire une liste d'entrées {numérotation, titre, niveau,
   page indicative}. C'est la référence "officielle" de la hiérarchie
   voulue par l'auteur du document.

3. PARCOURS DU DOCUMENT (page par page / élément par élément) : on
   rejoue le document dans l'ordre de lecture. Chaque `section_header`
   détecté par Docling met à jour une pile de hiérarchie courante
   (Chapitre > Titre > Sous-titre > ...). Si un TOC a été trouvé à
   l'étape 2, chaque en-tête détecté est comparé par fuzzy-matching
   (rapidfuzz) aux entrées du TOC : en cas de correspondance, le
   niveau "officiel" du TOC écrase le niveau brut détecté par la
   mise en page (plus fiable, car le TOC reflète l'intention de
   l'auteur, alors que le modèle de layout peut confondre les
   niveaux). Chaque paragraphe/texte est ensuite rattaché à la pile
   de hiérarchie courante -> c'est ce qui permet d'annoter chaque
   texte avec le VRAI titre du chapitre/titre/sous-titre (pas juste
   un numéro).

   => Le script fonctionne SANS TOC : dans ce cas on se base
      uniquement sur les niveaux de titres détectés par la mise en
      page (`SectionHeaderItem.level`). C'est donc un mode dégradé
      mais fonctionnel, pas un plantage.

4. GENERATION DES SORTIES :
   - document.md        : Markdown avec la hiérarchie reconstituée
   - hierarchy.json      : arbre {chapitre -> titres -> sous-titres}
                           + plan à plat, pensé pour être injecté dans
                           un prompt LLM ("voici le plan, propose un
                           cours structuré")
   - chunks.jsonl        : un chunk de texte par ligne, chaque chunk
                           annoté avec chapitre/titre/sous_titre
                           (titres concrets, pas des numéros) + le
                           chemin hiérarchique complet + les pages.
   - toc_debug.json       : ce que le script a compris du sommaire
                           (pour debug/tuning des regex)
   - warnings.json        : entrées du TOC jamais retrouvées dans le
                           corps du document (à vérifier manuellement)

LIMITES CONNUES (assumées, à améliorer plus tard si besoin) :
 - Pensé pour des PDF numériques (texte natif). Pour des PDF scannés,
   il faut activer l'OCR Docling (voir `build_converter`) et les
   résultats de détection de titres seront moins fiables.
 - La détection de "section_header" dépend du modèle de layout de
   Docling : il arrive qu'un titre soit mal étiqueté "paragraphe".
   Le TOC permet de corriger le NIVEAU d'un titre déjà détecté, mais
   pas de "ressusciter" un titre que Docling n'a pas détecté du tout.
   Un warning est émis pour les entrées de TOC jamais retrouvées.
 - La page de fin d'un chunk est approximative : c'est le min/max des
   pages des éléments accumulés dans le chunk avant découpage fin.

USAGE
-----
    python extract_hierarchy.py --pdf mon_cours.pdf --output-dir out/

Dépendances (déjà présentes dans l'environnement fourni) :
    docling, docling-core, rapidfuzz, langchain-text-splitters
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.document import DoclingDocument

from rapidfuzz import fuzz
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =====================================================================
# CONFIG "EN DUR" -- modifie juste cette section, pas besoin d'arguments
# en ligne de commande.
# =====================================================================

# Chemin du PDF a traiter.
PDF_PATH = Path(r"rag/data/input/20240610_blockchain.pdf")

# Dossier de sortie. Par defaut : None => les fichiers sont ecrits
# directement dans le MEME dossier que le PDF (PDF_PATH.parent).
# Mets un Path(...) explicite ici si tu veux un autre dossier.
OUTPUT_DIR = None


# =====================================================================
# CONFIG / CONSTANTES
# =====================================================================

# Mots-clés (normalisés : minuscule, sans accents) signalant une page
# de sommaire / table des matières.
TOC_HEADING_PATTERNS = [
    r"^sommaire$",
    r"^table des matieres$",
    r"^table des matiere$",
    r"^plan du cours$",
    r"^plan du document$",
    r"^plan de cours$",
    r"^table of contents$",
    r"^contents$",
]

# Mots-clés introduisant un chapitre/section dans une ligne de TOC
# sans numérotation décimale (ex: "Chapitre 1 : Introduction").
KEYWORD_LEVELS = {
    "chapitre": 1,
    "chapter": 1,
    "partie": 1,
    "part": 1,
    "unite": 1,
    "unit": 1,
    "module": 1,
    "section": 2,
    "sous-section": 3,
    "sous section": 3,
}

# Correspondance niveau numérique -> nom métier demandé par l'utilisateur.
LEVEL_LABELS = {1: "chapitre", 2: "titre", 3: "sous_titre"}

LABELS_HEADING = {DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE}
LABELS_BODY = {
    DocItemLabel.PARAGRAPH,
    DocItemLabel.TEXT,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CAPTION,
    DocItemLabel.CODE,
    DocItemLabel.FORMULA,
    DocItemLabel.CHECKBOX_UNSELECTED,
    DocItemLabel.CHECKBOX_SELECTED,
}


# =====================================================================
# UTILITAIRES TEXTE
# =====================================================================

def normalize(s: str) -> str:
    """minuscule + sans accents + espaces normalisés, pour comparaisons robustes."""
    s = (s or "").strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = re.sub(r"\s+", " ", s)
    return s


def level_label(level: int) -> str:
    return LEVEL_LABELS.get(level, f"niveau_{level}")


# =====================================================================
# STRUCTURES DE DONNEES
# =====================================================================

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
    source: str  # "layout" ou "toc_corrige"


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


# =====================================================================
# ETAPE 1 : CONVERSION DU PDF
# =====================================================================

def build_converter() -> DocumentConverter:
    """
    Convertisseur Docling adapté aux PDF numériques (texte natif, pas de
    scan). On désactive explicitement l'OCR (`do_ocr=False`) : par
    défaut Docling tente de charger un moteur OCR (RapidOCR/torch) même
    sur un PDF qui a déjà une couche texte, ce qui est inutile ici et
    peut planter sur des machines avec peu de RAM (erreurs
    "std::bad_alloc" observées sur des PDF de plusieurs dizaines de
    pages). On force aussi le backend PyPdfium, plus sobre en mémoire
    que le backend par défaut.

    Si un jour tu dois traiter un PDF SCANNÉ (pas de texte natif),
    remets `do_ocr=True` (et prévois plus de RAM/CPU).
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True  # utile pour la détection du TOC en tableau

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,
            )
        }
    )


def convert_pdf(pdf_path: Path) -> DoclingDocument:
    converter = build_converter()
    result = converter.convert(str(pdf_path))
    return result.document


# =====================================================================
# ETAPE 2 : DETECTION + PARSING DU SOMMAIRE (TOC)
# =====================================================================

# ligne type : "1.2 Les réseaux de neurones ..................... 14"
RE_NUMBERED_WITH_PAGE = re.compile(
    r"^(?P<num>\d+(?:\.\d+)*)[\.\)]?\s+(?P<title>.+?)\s*(?:[\.\-_ ]{2,}|\t)\s*(?P<page>\d{1,4})\s*$"
)
# ligne type : "1.2 Les réseaux de neurones"   (sans page, docling a parfois
# perdu le numéro de page dans un item séparé)
RE_NUMBERED_NO_PAGE = re.compile(
    r"^(?P<num>\d+(?:\.\d+)*)[\.\)]?\s+(?P<title>.+)$"
)
# ligne type : "Chapitre 1 : Introduction ... 5" ou "Chapitre 1 - Introduction"
RE_KEYWORD = re.compile(
    r"^(?P<kw>chapitre|chapter|partie|part|unite|unit|module|section)\s+"
    r"(?P<num>[\divxlc]+)\s*[:\-\.]?\s*(?P<title>.+?)"
    r"(?:\s*[\.\-_ ]{2,}\s*(?P<page>\d{1,4}))?\s*$",
    re.IGNORECASE,
)


def is_toc_heading(text: str) -> bool:
    t = normalize(text)
    return any(re.match(p, t) for p in TOC_HEADING_PATTERNS)


def find_toc_pages(doc: DoclingDocument, max_scan_pages: int, toc_max_span: int) -> Optional[set]:
    """
    Cherche une page contenant un titre de type "Sommaire / Table des
    matières" dans les `max_scan_pages` premières pages. Retourne
    l'ensemble des numéros de page à considérer comme faisant partie
    du TOC (page trouvée + `toc_max_span` pages suivantes), ou None
    si rien n'est trouvé.
    """
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
    """Essaie les différents patterns de ligne de TOC. Retourne un dict
    {numbering, title, level, page} ou None si la ligne ne ressemble
    pas à une entrée de sommaire."""
    line = line.strip()
    if not line or len(line) < 2:
        return None

    m = RE_NUMBERED_WITH_PAGE.match(line)
    if m:
        num = m.group("num")
        return {
            "numbering": num,
            "title": m.group("title").strip(" .-_"),
            "level": num.count(".") + 1,
            "page": int(m.group("page")),
        }

    m = RE_KEYWORD.match(line)
    if m:
        kw = m.group("kw").lower()
        page = m.group("page")
        return {
            "numbering": f"{kw} {m.group('num')}",
            "title": m.group("title").strip(" .-_"),
            "level": KEYWORD_LEVELS.get(kw, 1),
            "page": int(page) if page else None,
        }

    m = RE_NUMBERED_NO_PAGE.match(line)
    if m:
        num = m.group("num")
        title = m.group("title").strip(" .-_")
        # Filtre anti faux-positif : une ligne de corps de texte peut
        # ressembler à "3 pommes tombent..." -> on exige un titre
        # "raisonnable" (pas trop long, pas de ponctuation finale de phrase)
        if len(title) > 120 or title.endswith((".", ",", ";")):
            return None
        return {
            "numbering": num,
            "title": title,
            "level": num.count(".") + 1,
            "page": None,
        }

    return None


def collect_toc_raw_lines(doc: DoclingDocument, toc_pages: set) -> list:
    """Récupère les lignes de texte candidates (texte + tableaux) situées
    sur les pages identifiées comme faisant partie du sommaire."""
    lines = []

    for item in doc.texts:
        page = item.prov[0].page_no if getattr(item, "prov", None) else None
        if page not in toc_pages:
            continue
        text = (item.text or "").strip()
        if text and not is_toc_heading(text):
            lines.append(text)

    # Certains sommaires sont détectés par Docling comme des tableaux
    # (colonnes "titre" / "page" bien alignées).
    for table in doc.tables:
        page = table.prov[0].page_no if getattr(table, "prov", None) else None
        if page not in toc_pages:
            continue
        try:
            grid = table.data.grid  # matrice de cellules
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


def extract_toc(doc: DoclingDocument, max_scan_pages: int, toc_max_span: int):
    """Etape 2 complète : trouve la page de sommaire, en extrait les
    lignes, les parse. Retourne (entries, toc_pages) :
      - entries    : liste de TocEntry (peut être vide si aucun sommaire
                     n'est détecté ou parsable -> mode dégradé)
      - toc_pages  : ensemble des numéros de page couverts par le sommaire
                     (None si aucun sommaire trouvé), pour pouvoir les
                     exclure du parcours principal du document (sinon le
                     titre "Sommaire" et ses items polluent la hiérarchie
                     "réelle" du contenu)."""
    toc_pages = find_toc_pages(doc, max_scan_pages, toc_max_span)
    if not toc_pages:
        return [], None

    raw_lines = collect_toc_raw_lines(doc, toc_pages)
    entries = []
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
    return entries, toc_pages


# =====================================================================
# ETAPE 3 : PARCOURS DU DOCUMENT + RATTACHEMENT HIERARCHIQUE
# =====================================================================

def find_best_toc_match(text: str, toc_entries: list, min_score: int) -> Optional[TocEntry]:
    """Fuzzy-match (rapidfuzz) d'un titre détecté par le layout contre
    les entrées du sommaire non encore utilisées. Renvoie la meilleure
    entrée si son score dépasse `min_score`, sinon None."""
    best_entry = None
    best_score = 0
    norm_text = normalize(text)
    for entry in toc_entries:
        if entry.matched:
            continue
        score = fuzz.token_sort_ratio(norm_text, normalize(entry.title))
        if score > best_score:
            best_score = score
            best_entry = entry
    if best_entry is not None and best_score >= min_score:
        return best_entry
    return None


def walk_document(doc: DoclingDocument, toc_entries: list, fuzzy_threshold: int,
                   toc_pages: Optional[set] = None):
    """
    Parcours complet du document dans l'ordre de lecture.
    Retourne (outline, blocks, elements) :
      - outline  : liste plate des HeadingNode rencontrés (pour hierarchy.json)
      - blocks   : liste des ContentBlock (paragraphes annotés) (pour chunks.jsonl)
      - elements : liste chronologique fusionnée {kind: "heading"/"body", ...}
                   (pour reconstruire le markdown dans l'ordre du document)

    ATTRIBUTION DU NIVEAU D'UN TITRE (point le plus délicat) :
      Le niveau brut renvoyé par Docling (`SectionHeaderItem.level`) est
      peu fiable en pratique : Docling étiquette souvent TOUS les titres
      avec level=1, quelle que soit leur profondeur réelle. On applique
      donc, par ordre de priorité :
        1. Si le titre correspond (fuzzy-match) à une entrée du sommaire
           -> on prend le niveau OFFICIEL du sommaire. C'est aussi notre
           "ancre" : on retient ce niveau comme `anchor_level`.
        2. Sinon, si un sommaire existe mais que ce titre précis n'y
           figure pas (fréquent : le sommaire ne liste souvent que 1 ou 2
           niveaux, alors que le document contient des sous-titres plus
           fins) -> on le place à `anchor_level + 1`, c'est-à-dire juste
           en dessous du dernier titre confirmé par le sommaire. Cela
           évite que ces titres non listés ne retombent tous à plat au
           niveau 1 ("chapitre"), ce qui aplatissait complètement la
           hiérarchie. Limite connue : deux titres non listés qui se
           suivent seront mis au MEME niveau (frères), même si l'un est
           en réalité plus profond que l'autre -- on ne peut pas deviner
           plus finement sans signal supplémentaire (taille de police...).
        3. Si aucun sommaire n'a été détecté du tout dans le document ->
           on utilise tel quel le niveau brut de Docling (aucune ancre
           disponible pour faire mieux).
    """
    stack: list[HeadingNode] = []
    outline: list[HeadingNode] = []
    blocks: list[ContentBlock] = []
    elements: list[dict] = []
    heading_counter = 0
    anchor_level = 0  # niveau du dernier titre confirmé par le sommaire

    for item, _tree_level in doc.iterate_items():
        label = getattr(item, "label", None)
        text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue
        page = item.prov[0].page_no if getattr(item, "prov", None) else None

        # On saute entièrement la/les page(s) de sommaire : leur contenu
        # a déjà été exploité via extract_toc(), le rejouer ici ne ferait
        # que polluer la hiérarchie "réelle" du contenu (ex: un faux
        # chapitre "Sommaire").
        if toc_pages and page in toc_pages:
            continue

        if label in LABELS_HEADING:
            raw_level = getattr(item, "level", None)
            if label == DocItemLabel.TITLE or raw_level is None:
                raw_level = 1

            match = find_best_toc_match(text, toc_entries, fuzzy_threshold) if toc_entries else None
            if match is not None:
                match.matched = True
                final_level = match.level
                source = "toc_corrige"
                anchor_level = final_level
            elif toc_entries:
                # Sommaire présent mais ce titre n'y figure pas : on
                # l'ancre juste sous le dernier titre confirmé.
                final_level = anchor_level + 1
                source = "layout_estime"
            else:
                # Aucun sommaire du tout : on n'a que le niveau brut de
                # Docling, pour ce qu'il vaut.
                final_level = raw_level
                source = "layout"

            # Dépile tout ce qui est de niveau >= au niveau courant
            while stack and stack[-1].level >= final_level:
                stack.pop()

            heading_counter += 1
            node = HeadingNode(
                id=f"h{heading_counter}",
                level=final_level,
                label_type=level_label(final_level),
                title=text,
                page_start=page,
                source=source,
            )
            stack.append(node)
            outline.append(node)
            elements.append({"kind": "heading", "level": final_level, "title": text, "page": page})
            continue

        if label in LABELS_BODY:
            path_titles = [n.title for n in stack]
            chapitre = next((n.title for n in stack if n.level == 1), None)
            titre = next((n.title for n in stack if n.level == 2), None)
            sous_titre = next((n.title for n in reversed(stack) if n.level >= 3), None)

            block = ContentBlock(
                text=text,
                page=page,
                hierarchy_path=path_titles,
                chapitre=chapitre,
                titre=titre,
                sous_titre=sous_titre,
            )
            blocks.append(block)
            elements.append({"kind": "body", "text": text, "page": page})

    return outline, blocks, elements


# =====================================================================
# ETAPE 4 : GENERATION DES SORTIES
# =====================================================================

def build_markdown(elements: list, doc_title: str) -> str:
    lines = [f"# {doc_title}", ""]
    for el in elements:
        if el["kind"] == "heading":
            lines.append(f"{'#' * min(el['level'] + 1, 6)} {el['title']}")
            lines.append("")
        else:
            lines.append(el["text"])
            lines.append("")
    return "\n".join(lines)


def build_hierarchy_tree(outline: list, doc_title: str) -> dict:
    root = {"type": "document", "titre": doc_title, "enfants": []}
    stack = [(0, root)]
    for node in outline:
        entry = {
            "type": node.label_type,
            "titre": node.title,
            "page_debut": node.page_start,
            "source": node.source,
            "enfants": [],
        }
        while stack and stack[-1][0] >= node.level:
            stack.pop()
        stack[-1][1]["enfants"].append(entry)
        stack.append((node.level, entry))
    return root


def build_flat_plan(outline: list) -> list:
    """Version 'plate' du plan, pratique à injecter telle quelle dans un
    prompt LLM pour proposer un cours structuré à l'utilisateur."""
    plan = []
    for node in outline:
        indent = "  " * (node.level - 1)
        plan.append(f"{indent}- [{node.label_type}] {node.title} (p.{node.page_start})")
    return plan


def build_chunks(blocks: list, chunk_size: int, chunk_overlap: int) -> list:
    """
    Chunking 'hierarchy-aware' : on accumule les paragraphes tant qu'ils
    appartiennent à la même feuille de hiérarchie (même chemin
    chapitre/titre/sous-titre). Dès que la hiérarchie change, on flushe
    le buffer en un ou plusieurs chunks (découpés finement par un
    RecursiveCharacterTextSplitter si le buffer dépasse chunk_size).
    Cela garantit qu'un chunk n'est JAMAIS attribué à deux sections
    différentes -> annotation fiable pour la recherche sémantique.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    buffer_texts: list = []
    buffer_pages: list = []
    current_meta: Optional[ContentBlock] = None
    chunk_counter = 0

    def flush():
        nonlocal buffer_texts, buffer_pages, chunk_counter
        if not buffer_texts or current_meta is None:
            buffer_texts, buffer_pages = [], []
            return
        combined = "\n\n".join(buffer_texts)
        pieces = splitter.split_text(combined) if len(combined) > chunk_size else [combined]
        pages = [p for p in buffer_pages if p is not None]
        page_debut = min(pages) if pages else None
        page_fin = max(pages) if pages else None
        for piece in pieces:
            chunk_counter += 1
            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{chunk_counter:05d}",
                    text=piece.strip(),
                    chapitre=current_meta.chapitre,
                    titre=current_meta.titre,
                    sous_titre=current_meta.sous_titre,
                    hierarchy_path=current_meta.hierarchy_path,
                    page_debut=page_debut,
                    page_fin=page_fin,
                )
            )
        buffer_texts, buffer_pages = [], []

    for block in blocks:
        path_key = tuple(block.hierarchy_path)
        current_key = tuple(current_meta.hierarchy_path) if current_meta else None
        if current_meta is not None and path_key != current_key:
            flush()
        current_meta = block
        buffer_texts.append(block.text)
        buffer_pages.append(block.page)
        # garde-fou : on flush aussi si le buffer devient trop gros,
        # même si on reste dans la même section (évite un seul énorme
        # bloc en mémoire pour des sections très longues)
        if sum(len(t) for t in buffer_texts) > chunk_size * 4:
            flush()
            current_meta = block  # on garde la même méta pour la suite

    flush()
    return chunks


# =====================================================================
# ORCHESTRATION
# =====================================================================

def run(pdf_path: Path, output_dir: Path, max_scan_pages: int, toc_max_span: int,
        fuzzy_threshold: int, chunk_size: int, chunk_overlap: int, disable_toc: bool):

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Conversion Docling de {pdf_path.name} ...")
    doc = convert_pdf(pdf_path)
    doc_title = getattr(doc, "name", None) or pdf_path.stem

    toc_entries = []
    toc_pages = None
    if not disable_toc:
        print("[2/4] Recherche du sommaire (TOC) ...")
        toc_entries, toc_pages = extract_toc(doc, max_scan_pages, toc_max_span)
        if toc_entries:
            print(f"      -> {len(toc_entries)} entrées de sommaire détectées "
                  f"(pages {min(toc_pages)}-{max(toc_pages)}).")
        else:
            print("      -> aucun sommaire détecté, passage en mode 'layout seul'.")
    else:
        print("[2/4] Détection du TOC désactivée (--no-toc).")

    print("[3/4] Parcours du document et rattachement hiérarchique ...")
    outline, blocks, elements = walk_document(doc, toc_entries, fuzzy_threshold, toc_pages)
    print(f"      -> {len(outline)} titres détectés, {len(blocks)} blocs de texte.")

    print("[4/4] Génération des fichiers de sortie ...")

    md = build_markdown(elements, doc_title)
    (output_dir / "document.md").write_text(md, encoding="utf-8")

    hierarchy_tree = build_hierarchy_tree(outline, doc_title)
    hierarchy_out = {
        "titre_document": doc_title,
        "arbre": hierarchy_tree,
        "plan_a_plat": build_flat_plan(outline),
    }
    (output_dir / "hierarchy.json").write_text(
        json.dumps(hierarchy_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    chunks = build_chunks(blocks, chunk_size, chunk_overlap)
    with (output_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    if toc_entries:
        (output_dir / "toc_debug.json").write_text(
            json.dumps([asdict(e) for e in toc_entries], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        unmatched = [e.title for e in toc_entries if not e.matched]
        if unmatched:
            print(f"      -> ATTENTION : {len(unmatched)} entrées du sommaire n'ont pas été "
                  f"retrouvées dans le corps du document (voir warnings.json).")
        (output_dir / "warnings.json").write_text(
            json.dumps({"toc_entries_non_retrouvees": unmatched}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"\nTerminé. Fichiers écrits dans : {output_dir}")
    print(f"  - {output_dir / 'document.md'}")
    print(f"  - {output_dir / 'hierarchy.json'}")
    print(f"  - {output_dir / 'chunks.jsonl'} ({len(chunks)} chunks)")


def main():
    # Les valeurs par défaut viennent de la config en dur tout en haut du
    # fichier (PDF_PATH / OUTPUT_DIR). Les arguments CLI restent
    # disponibles si tu veux surcharger ponctuellement sans toucher au
    # code, mais rien n'est obligatoire : `python extract_hierarchy.py`
    # tout seul fonctionne.
    parser = argparse.ArgumentParser(
        description="Extraction hiérarchique (chapitre/titre/sous-titre) d'un PDF numérique via Docling."
    )
    parser.add_argument("--pdf", default=PDF_PATH, type=Path, help="Chemin du PDF source")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path,
                         help="Répertoire de sortie (par défaut : même dossier que le PDF)")
    parser.add_argument("--toc-scan-pages", default=20, type=int,
                         help="Nombre de pages initiales scannées pour trouver le sommaire")
    parser.add_argument("--toc-max-span", default=8, type=int,
                         help="Nombre de pages max considérées comme faisant partie du sommaire")
    parser.add_argument("--fuzzy-threshold", default=82, type=int,
                         help="Score minimal (0-100) de similarité pour rattacher un titre détecté à une entrée du TOC")
    parser.add_argument("--chunk-size", default=1000, type=int, help="Taille cible d'un chunk (caractères)")
    parser.add_argument("--chunk-overlap", default=150, type=int, help="Chevauchement entre chunks (caractères)")
    parser.add_argument("--no-toc", action="store_true", help="Désactive la détection du sommaire (mode layout pur)")
    args = parser.parse_args()

    pdf_path = args.pdf
    if not pdf_path.exists():
        print(f"Erreur : fichier introuvable : {pdf_path}", file=sys.stderr)
        print("-> Vérifie la variable PDF_PATH en haut du script.", file=sys.stderr)
        sys.exit(1)

    # Résultat dans le même dossier que le PDF si rien n'est précisé.
    output_dir = args.output_dir if args.output_dir is not None else pdf_path.parent

    run(
        pdf_path=pdf_path,
        output_dir=output_dir,
        max_scan_pages=args.toc_scan_pages,
        toc_max_span=args.toc_max_span,
        fuzzy_threshold=args.fuzzy_threshold,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        disable_toc=args.no_toc,
    )


if __name__ == "__main__":
    main()
