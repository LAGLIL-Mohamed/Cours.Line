"""Modèles de données pour la génération de cours."""

from dataclasses import dataclass, field


@dataclass
class PlanSection:
    id: str
    titre: str
    niveau: int
    objectif: str
    enfants: list = field(default_factory=list)


@dataclass
class GeneratedSection:
    id: str
    titre: str
    niveau: int
    objectif: str
    questions: list[str] = field(default_factory=list)
    contenu: str = ""
    sources_chunks: list[str] = field(default_factory=list)
    enfants: list = field(default_factory=list)
