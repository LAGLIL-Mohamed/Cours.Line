"""Étapes du pipeline RAG (0 à 6)."""

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
