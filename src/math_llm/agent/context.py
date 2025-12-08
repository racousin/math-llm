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
# LEAN 4 CONVENTIONS
# =============================================================================

LEAN4_CONVENTIONS = """
- Lemma names use CamelCase: `Nat.add_comm`, `Complex.exp_add`, `Finset.sum_mul`
- Namespaces: Nat, Int, Real, Complex, Finset, List, Set, Function
- Coercions: ↑n (natural to int/real), ↑√2 (real to complex)
- Common notations: ∑ (sum), ∏ (prod), ∈, ⊆, ∧, ∨, ¬, ∀, ∃
"""


# =============================================================================
# TACTIC REFERENCE
# =============================================================================

TACTIC_REFERENCE = """
Finishing tactics (try these first on simple goals):
- rfl: definitional equality
- norm_num: numeric computation (1 + 1 = 2, 3 < 5)
- decide: decidable propositions
- ring: polynomial arithmetic
- omega: linear integer/natural arithmetic
- linarith: linear arithmetic with hypotheses
- positivity: prove positivity/nonnegativity
- field_simp: clear denominators in field expressions

Rewriting tactics:
- rw [h]: rewrite with h (left to right)
- rw [← h]: rewrite with h (right to left)
- simp: simplify using simp lemmas
- simp only [h1, h2]: simplify with specific lemmas only
- ring_nf: normalize polynomial expressions

Introduction/elimination:
- intro x: introduce forall/implication
- obtain ⟨a, b, h⟩ := h: destruct existential/and
- rcases h with ⟨a, ha⟩ | ⟨b, hb⟩: recursive case split
- use x: provide existential witness
- constructor: split and/iff goals
- left / right: choose disjunction side
- by_contra h: proof by contradiction
- push_neg: push negations inward

Application:
- exact h: exact proof term
- apply h: apply lemma/hypothesis backwards
- have h : T := proof: introduce intermediate fact
- calc: calculational proof chain
- refine: apply with placeholders (_)

Other:
- ext: extensionality (for functions, sets)
- induction n with d hd: induction
- cases h: case analysis
- simp_all: simplify goal and all hypotheses
"""


# =============================================================================
# PROBLEM PATTERNS
# =============================================================================

PROBLEM_PATTERNS = """
Numeric equality (a = b where a,b are numbers):
  → norm_num, decide, native_decide

Polynomial/ring equation:
  → ring, ring_nf

Linear arithmetic (inequalities, divisibility):
  → omega, linarith

Finite sums/products:
  → simp [Finset.sum_range_succ], norm_num, decide

Complex numbers:
  → simp [Complex.ext_iff, Complex.add_re, Complex.mul_im]
  → norm_num for numeric complex values

Real analysis:
  → linarith, nlinarith for inequalities
  → field_simp to clear denominators

Set/logic:
  → ext, simp [Set.mem_inter_iff], push_neg

Induction goals:
  → induction n with d hd, then simp or omega on base/step
"""


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = f"""You are a Lean 4 theorem prover. Output ONE tactic per response.

Lean 4 conventions:{LEAN4_CONVENTIONS}
Tactics:{TACTIC_REFERENCE}
Problem patterns:{PROBLEM_PATTERNS}
Output format: Write ONLY the tactic, no explanation.
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
