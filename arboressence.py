#!/usr/bin/env python3
"""
arborescence.py - Affiche la structure d'un projet
Usage: python arborescence.py [chemin] [--all]
"""

import os
import sys
from pathlib import Path

# Dossiers/fichiers à ignorer par défaut
IGNORE = {
    '.git', '.svn', '.hg',               # VCS
    'node_modules', 'vendor',             # Dépendances
    '__pycache__', '.pytest_cache', '.mypy_cache',
    '.venv', 'venv', 'env',               # Environnements virtuels
    'dist', 'build', '.next', '.nuxt', 'target',
    '.idea', '.vscode',                   # IDE
    '.DS_Store', 'Thumbs.db', '*.pyc', '*.pyo'
}

def should_ignore(name, show_all=False):
    if show_all:
        return False
    if name in IGNORE:
        return True
    for pattern in IGNORE:
        if pattern.startswith('*') and name.endswith(pattern[1:]):
            return True
    return False

def tree(directory, prefix="", show_all=False):
    try:
        entries = sorted(
            [e for e in os.scandir(directory) if not should_ignore(e.name, show_all)],
            key=lambda e: (not e.is_dir(), e.name.lower())
        )
    except PermissionError:
        print(prefix + "└── [Permission refusée]")
        return

    count = len(entries)
    for i, entry in enumerate(entries):
        is_last = (i == count - 1)
        connector = "└── " if is_last else "├── "
        branch = "    " if is_last else "│   "

        if entry.is_symlink():
            print(f"{prefix}{connector}{entry.name} -> {os.readlink(entry.path)}")
        elif entry.is_dir():
            print(f"{prefix}{connector}{entry.name}/")
            tree(entry.path, prefix + branch, show_all)
        else:
            size = ""
            try:
                sz = entry.stat().st_size
                if sz < 1024:
                    size = f" ({sz} B)"
                elif sz < 1024*1024:
                    size = f" ({sz/1024:.1f} KB)"
                else:
                    size = f" ({sz/(1024*1024):.1f} MB)"
            except OSError:
                pass
            print(f"{prefix}{connector}{entry.name}{size}")

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    show_all = "--all" in sys.argv or "-a" in sys.argv
    
    root = Path(path).resolve()
    if not root.exists():
        print(f"Erreur : '{path}' n'existe pas.")
        sys.exit(1)
    if not root.is_dir():
        print(f"Erreur : '{path}' n'est pas un répertoire.")
        sys.exit(1)

    print(f"\n📁 {root.name}/")
    tree(str(root), show_all=show_all)
    print()

if __name__ == "__main__":
    main()