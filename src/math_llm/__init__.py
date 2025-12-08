"""
Math-LLM: LLM Agent for Lean Theorem Proving with RL Training.

A complete training/evaluation package for state-of-the-art LLM agents
that use Lean efficiently to prove mathematical statements.
"""

__version__ = "0.1.0"

from math_llm.config import Config, load_config
from math_llm.agent import LeanAgent
from math_llm.lean import LeanREPL, LeanExecutor
from math_llm.training import RLTrainer

__all__ = [
    "Config",
    "load_config",
    "LeanAgent",
    "LeanREPL",
    "LeanExecutor",
    "RLTrainer",
]
