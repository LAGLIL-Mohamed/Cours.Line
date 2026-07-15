"""Module de génération de cours par IA (Mistral + RAG)."""

from .config import CourseConfig
from .orchestrator import CourseOrchestrator

__all__ = ["CourseConfig", "CourseOrchestrator"]
