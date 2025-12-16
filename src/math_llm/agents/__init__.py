"""
Lean proof agents for benchmarking.

Two agent types:
- SimpleAgent: Direct single-shot proof generation
- ToolAgent: Iterative proof with lean tool calls
"""

from math_llm.agents.simple import SimpleAgent
from math_llm.agents.tool import ToolAgent

__all__ = ["SimpleAgent", "ToolAgent"]
