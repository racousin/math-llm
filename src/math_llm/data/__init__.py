"""
Data module for Lean 4 theorem proving datasets.

Available Lean 4 datasets:
- LeanDojo (~100k) - HuggingFace
- MiniF2F Lean 4 (~488) - HuggingFace
- FIMO (~149 IMO) - HuggingFace
- PutnamBench (~1.7k) - HuggingFace
- ProofNet (~370) - HuggingFace
- Formal Conjectures (DeepMind) - Git
- Mathlib4 (~150k) - Git
"""

from math_llm.data.datasets import LeanDataset, LeanProblem
from math_llm.data.loaders import create_dataloader, collate_lean_batch, load_dataset
from math_llm.data.sources import (
    # Base classes
    DataSource,
    HuggingFaceSource,
    # HuggingFace sources (Lean 4 - preferred)
    LeanDojoSource,
    MiniF2FLean4Source,
    FIMOSource,
    PutnamBenchSource,
    ProofNetHFSource,
    # Git sources (Lean 4)
    FormalConjecturesSource,
    Mathlib4Source,
    # Registry and utilities
    SOURCES,
    RECOMMENDED,
    get_source,
    list_sources,
    print_sources,
    download_sources,
    load_sources,
    # Backwards compatibility
    download_all_sources,
    load_all_sources,
)

__all__ = [
    # Core classes
    "LeanDataset",
    "LeanProblem",
    # Loaders
    "create_dataloader",
    "collate_lean_batch",
    "load_dataset",
    "create_dummy_dataset",
    # Source base classes
    "DataSource",
    "HuggingFaceSource",
    # HuggingFace sources
    "LeanDojoSource",
    "MiniF2FLean4Source",
    "FIMOSource",
    "PutnamBenchSource",
    "ProofNetHFSource",
    # Git sources
    "FormalConjecturesSource",
    "Mathlib4Source",
    # Registry
    "SOURCES",
    "RECOMMENDED",
    "get_source",
    "list_sources",
    "print_sources",
    # Functions
    "download_sources",
    "load_sources",
    "download_all_sources",
    "load_all_sources",
]
