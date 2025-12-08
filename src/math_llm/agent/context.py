"""
Context engineering for Lean theorem proving agent.

Handles all prompt construction and context management.
"""

from dataclasses import dataclass, field
from typing import Optional

from math_llm.data.datasets import LeanProblem


# =============================================================================
# LEAN ENVIRONMENT
# =============================================================================

LEAN_IMPORTS = """import Mathlib
import Aesop
"""


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a Lean 4 theorem prover. Output ONE tactic.

Environment: Mathlib imported.

Tactics: simp, omega, ring, linarith, rfl, exact, apply, rw, intro, cases, induction
"""


# =============================================================================
# PROOF STATE
# =============================================================================

@dataclass
class ProofState:
    """Tracks the state of a proof attempt."""

    problem: LeanProblem
    steps: list[dict] = field(default_factory=list)
    failed_tactics: list[str] = field(default_factory=list)

    def add_step(self, tactic: str, success: bool, errors: list[str], goals: list[str]):
        """Record a proof step."""
        self.steps.append({
            "tactic": tactic,
            "success": success,
            "errors": errors,
            "goals": goals,
        })
        if not success:
            self.failed_tactics.append(tactic)

    @property
    def last_step(self) -> Optional[dict]:
        return self.steps[-1] if self.steps else None


# =============================================================================
# PROMPT BUILDER
# =============================================================================

class PromptBuilder:
    """Builds prompts for the LLM at each step of proof search."""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        self.system_prompt = system_prompt

    def build_initial(self, problem: LeanProblem) -> str:
        """Build the initial prompt for a new problem."""
        parts = []

        # Problem description (if any)
        if problem.description:
            parts.append(f"Problem: {problem.description}")

        # Statement
        parts.append(f"Theorem:\n```lean4\n{problem.statement}\n```")

        # Goal
        goal = self._extract_goal(problem.statement)
        parts.append(f"Goal: `{goal}`")

        return "\n\n".join(parts)

    def build_continuation(self, state: ProofState) -> str:
        """Build continuation prompt after a step."""
        parts = []

        # Problem
        if state.problem.description:
            parts.append(f"Problem: {state.problem.description}")

        parts.append(f"Theorem:\n```lean4\n{state.problem.statement}\n```")

        # Last step feedback
        if state.last_step:
            step = state.last_step
            parts.append(f"Last tactic: `{step['tactic']}`")

            if step["success"]:
                if step["goals"]:
                    goals_str = "\n".join(f"  ⊢ {g}" for g in step["goals"][:3])
                    parts.append(f"Result: OK\nRemaining goals:\n{goals_str}")
                else:
                    parts.append("Result: OK (no visible goals)")
            else:
                error = step["errors"][0] if step["errors"] else "error"
                parts.append(f"Result: ERROR - {error}")

        # Failed tactics to avoid
        if state.failed_tactics:
            recent = state.failed_tactics[-3:]
            parts.append(f"Avoid: {', '.join(recent)}")

        return "\n\n".join(parts)

    def _extract_goal(self, statement: str) -> str:
        """Extract the goal type from a theorem statement."""
        if ":" not in statement:
            return statement

        # Get part after last : and before :=
        parts = statement.split(":")
        if len(parts) >= 2:
            goal = parts[-1]
            if ":=" in goal:
                goal = goal.split(":=")[0]
            return goal.strip()

        return statement
