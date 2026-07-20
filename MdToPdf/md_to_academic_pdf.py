#!/usr/bin/env python3
"""
md_to_academic_pdf.py
======================

Convertit un fichier Markdown (issu d'un pipeline RAG) en PDF académique.
Les chemins sont définis directement dans le script, aucun argument CLI requis.

Utilisation standalone :
    python md_to_academic_pdf.py

Utilisation importée :
    from md_to_academic_pdf import convert_md_to_pdf
    convert_md_to_pdf("course.md", "course.pdf")
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ==========================================================================
# CONFIGURATION — Modifiez ces lignes selon vos besoins (mode CLI)
# ==========================================================================

INPUT_PATH = Path("mrdrag.md")
OUTPUT_PATH = Path("mrdrag.pdf")

TITLE: str | None = None      # Titre page de garde (None = nom du fichier)
DATE: str | None = None       # Date (None = auto depuis YAML ou vide)
TOC: bool = True              # Table des matières
NUMBER_SECTIONS: bool = True  # Numérotation des sections

# --------------------------------------------------------------------------
# Constantes internes
# --------------------------------------------------------------------------

PLACEHOLDER_PATTERN = re.compile(
    r"\*\[\s*Section en attente de g[ée]n[ée]ration\s*\]\*"
)
SECTION_SPLIT_RE = re.compile(r"(?=^\*Objectif\s*:.*\*\s*$)", re.MULTILINE)

LATEX_TEMPLATE_HEADER = r"""
\usepackage[a4paper,margin=2.5cm]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage{xcolor}
\definecolor{titleblue}{RGB}{22,54,92}
\titleformat{\section}{\normalfont\Large\bfseries\color{titleblue}}{\thesection}{0.6em}{}
\titleformat{\subsection}{\normalfont\large\bfseries\color{titleblue}}{\thesubsection}{0.6em}{}
\titleformat{\subsubsection}{\normalfont\normalsize\bfseries\color{titleblue}}{\thesubsubsection}{0.6em}{}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\leftmark}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\setlength{\parskip}{0.5em}
\setlength{\parindent}{0pt}
"""


# --------------------------------------------------------------------------
# Pré-traitement Markdown
# --------------------------------------------------------------------------

def strip_yaml_frontmatter(text: str) -> tuple[str, dict[str, str]]:
    meta: dict[str, str] = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            block = text[3:end]
            body = text[end + 4 :]
            for line in block.splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    meta[key.strip().lower()] = value.strip().strip('"\'')
            return body.lstrip("\n"), meta
    return text, meta


def remove_placeholder_sections(text: str) -> str:
    chunks = SECTION_SPLIT_RE.split(text)
    return "".join(chunk for chunk in chunks if not PLACEHOLDER_PATTERN.search(chunk))


def fix_stray_markdown_artifacts(text: str) -> str:
    text = re.sub(r"^---\*\s*$", "---", text, flags=re.MULTILINE)
    text = re.sub(r"\*\s*\n(\s*\n)(?=\*Objectif\s*:)", r"\n\1", text)
    text = re.sub(r"(\|[^\n|]*\|)\*(\s*)$", r"\1\2", text, flags=re.MULTILINE)
    text = re.sub(r"\*\s*\Z", "\n", text)
    return text


def clean_markdown(raw_text: str) -> tuple[str, dict[str, str]]:
    body, meta = strip_yaml_frontmatter(raw_text)
    body = remove_placeholder_sections(body)
    body = fix_stray_markdown_artifacts(body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip() + "\n", meta


# --------------------------------------------------------------------------
# Génération PDF
# --------------------------------------------------------------------------

def check_dependencies() -> None:
    missing = [t for t in ("pandoc", "xelatex") if shutil.which(t) is None]
    if missing:
        raise RuntimeError(
            f"Outils manquants : {', '.join(missing)}. "
            "Installez pandoc et texlive-xetex (ou MiKTeX sur Windows)."
        )


def build_pdf(
    md_path: Path,
    output_path: Path,
    title: str | None = None,
    date: str | None = None,
    toc: bool = True,
    number_sections: bool = True,
) -> None:
    check_dependencies()

    raw_text = md_path.read_text(encoding="utf-8")
    cleaned_text, meta = clean_markdown(raw_text)

    final_title = title or meta.get("title") or md_path.stem
    final_date = date or meta.get("date", "")
    if final_date:
        final_date = final_date.split("T")[0]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cleaned_md = tmp_path / "cleaned.md"
        cleaned_md.write_text(cleaned_text, encoding="utf-8")

        header_file = tmp_path / "header.tex"
        header_file.write_text(LATEX_TEMPLATE_HEADER, encoding="utf-8")

        cmd = [
            "pandoc",
            str(cleaned_md),
            "-o", str(output_path),
            "--pdf-engine=xelatex",
            "-V", "mainfont=Times New Roman",
            "-V", "sansfont=Arial",
            "-V", "monofont=Consolas",
            "-V", "fontsize=11pt",
            "-V", "linkcolor=blue",
            "-V", "colorlinks=true",
            "-V", f"title={final_title}",
            "-H", str(header_file),
            # CRITIQUE : désactive yaml_metadata_block car les RAG commencent souvent par ---
            "--from=markdown-yaml_metadata_block+tex_math_dollars+pipe_tables+fenced_divs"
            "+lists_without_preceding_blankline",
            "--syntax-highlighting=tango",
        ]
        if final_date:
            cmd += ["-V", f"date={final_date}"]
        if toc:
            cmd += ["--toc", "--toc-depth=3"]
        if number_sections:
            cmd += ["--number-sections"]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "Échec pandoc/xelatex :\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )


# --------------------------------------------------------------------------
# API publique pour import depuis app.py
# --------------------------------------------------------------------------

def convert_md_to_pdf(
    md_path: Path | str,
    output_path: Path | str,
    title: str | None = None,
    date: str | None = None,
    toc: bool = True,
    number_sections: bool = True,
) -> None:
    """Convertit un fichier Markdown en PDF académique.

    Cette fonction est conçue pour être importée depuis app.py
    (process principal). Elle wrappe build_pdf() avec des chemins
    explicites, sans dépendre des constantes globales du CLI.
    """
    build_pdf(
        md_path=Path(md_path),
        output_path=Path(output_path),
        title=title,
        date=date,
        toc=toc,
        number_sections=number_sections,
    )


# --------------------------------------------------------------------------
# Point d'entrée CLI (standalone)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    if not INPUT_PATH.exists():
        print(f"Erreur : fichier introuvable → {INPUT_PATH.resolve()}", file=sys.stderr)
        sys.exit(1)

    try:
        build_pdf(
            md_path=INPUT_PATH,
            output_path=OUTPUT_PATH,
            title=TITLE,
            date=DATE,
            toc=TOC,
            number_sections=NUMBER_SECTIONS,
        )
    except RuntimeError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"PDF généré : {OUTPUT_PATH.resolve()}")