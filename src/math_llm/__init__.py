"""
Lean Proof Benchmark

A simple benchmark package for evaluating LLM agents on Lean 4 theorem proving.
"""

__version__ = "0.1.0"

from math_llm.lean_server import LeanServer, LeanResult, MATHLIB_VERSION, LEAN_TOOLCHAIN
from math_llm.data import Problem, load_data, list_datasets
from math_llm.agents import SimpleAgent, ToolAgent

__all__ = [
    # Versions
    "MATHLIB_VERSION",
    "LEAN_TOOLCHAIN",
    # Core
    "LeanServer",
    "LeanResult",
    "Problem",
    "load_data",
    "list_datasets",
    "SimpleAgent",
    "ToolAgent",
]
