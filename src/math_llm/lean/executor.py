"""
Lean executor for structured LLM interaction.

Provides a high-level interface optimized for LLM agents:
- Structured input/output format
- Error message normalization
- Hint generation
- Tactic suggestions
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from math_llm.lean.server import LeanServer, LeanResult, LeanResultStatus


@dataclass
class LeanAction:
    """An action to execute in Lean."""

    code: str
    action_type: str = "proof"  # "proof", "tactic", "definition", "check"
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_llm_output(cls, output: str) -> "LeanAction":
        """
        Parse LLM output to extract Lean code.

        Supports multiple formats:
        - Code blocks: ```lean4 ... ```
        - Direct code
        - Tactic-only output
        """
        # Try to extract from code block
        code_block_pattern = r"```(?:lean4?|lean)?\s*\n?(.*?)```"
        matches = re.findall(code_block_pattern, output, re.DOTALL)

        if matches:
            code = matches[-1].strip()  # Take the last code block
        else:
            # Try to extract proof tactics directly
            tactic_pattern = r"(?:by\s+)?((?:·|\*|-)\s+\w+.*?)(?=\n\n|\Z)"
            tactic_matches = re.findall(tactic_pattern, output, re.DOTALL)

            if tactic_matches:
                code = "by\n" + "\n".join(tactic_matches)
            else:
                # Use the whole output as code (after cleaning)
                code = cls._clean_code(output)

        # Determine action type
        action_type = "proof"
        if code.strip().startswith("def ") or code.strip().startswith("definition "):
            action_type = "definition"
        elif code.strip().startswith("#check") or code.strip().startswith("#eval"):
            action_type = "check"
        elif not any(kw in code for kw in ["theorem", "lemma", "example", "def"]):
            action_type = "tactic"

        return cls(code=code, action_type=action_type)

    @staticmethod
    def _clean_code(text: str) -> str:
        """Clean text to extract Lean code."""
        # Remove common LLM artifacts
        lines = []
        for line in text.split("\n"):
            # Skip explanation lines
            if line.strip().startswith(("Let me", "I'll", "First,", "Now,", "Here's")):
                continue
            if line.strip().startswith(("Note:", "Explanation:", "The proof")):
                continue
            lines.append(line)

        return "\n".join(lines).strip()


@dataclass
class ExecutionFeedback:
    """Structured feedback from Lean execution for LLM."""

    success: bool
    complete: bool
    message: str
    goals: list[str]
    errors: list[str]
    hints: list[str]
    raw_result: LeanResult

    def to_prompt(self) -> str:
        """Convert to prompt format for LLM context."""
        parts = []

        if self.complete:
            parts.append("<result>Proof complete! ✓</result>")
        elif self.success:
            parts.append("<result>Code accepted, but proof incomplete.</result>")
            if self.goals:
                parts.append("<goals>")
                for goal in self.goals:
                    parts.append(f"  {goal}")
                parts.append("</goals>")
        else:
            parts.append("<result>Error</result>")
            if self.errors:
                parts.append("<errors>")
                for error in self.errors:
                    parts.append(f"  {error}")
                parts.append("</errors>")

        if self.hints:
            parts.append("<hints>")
            for hint in self.hints:
                parts.append(f"  - {hint}")
            parts.append("</hints>")

        return "\n".join(parts)


class LeanExecutor:
    """
    High-level Lean executor optimized for LLM interaction.

    Features:
    - Automatic error message normalization
    - Hint generation based on common errors
    - Tactic suggestions
    - Context management
    """

    def __init__(
        self,
        server: Optional[LeanServer] = None,
        auto_hints: bool = True,
        verbose: bool = True,
    ):
        self.server = server or LeanServer()
        self.auto_hints = auto_hints
        self.verbose = verbose
        self._owns_server = server is None

    def start(self) -> None:
        """Start the executor."""
        if self._owns_server:
            self.server.start()

    def stop(self) -> None:
        """Stop the executor."""
        if self._owns_server:
            self.server.stop()

    def execute(
        self,
        action: LeanAction | str,
        context: Optional[str] = None,
    ) -> ExecutionFeedback:
        """
        Execute a Lean action and return structured feedback.

        Args:
            action: LeanAction or raw code string
            context: Optional context (imports, previous definitions)

        Returns:
            ExecutionFeedback with structured result
        """
        if isinstance(action, str):
            action = LeanAction(code=action)

        result = self.server.execute(action.code, context)
        return self._create_feedback(result, action)

    def execute_proof_attempt(
        self,
        statement: str,
        proof: str,
    ) -> ExecutionFeedback:
        """
        Execute a proof attempt for a given statement.

        Args:
            statement: The theorem statement
            proof: The proposed proof

        Returns:
            ExecutionFeedback with structured result
        """
        result = self.server.check_proof(statement, proof)
        action = LeanAction(code=proof, action_type="proof")
        return self._create_feedback(result, action, statement)

    def _create_feedback(
        self,
        result: LeanResult,
        action: LeanAction,
        statement: Optional[str] = None,
    ) -> ExecutionFeedback:
        """Create structured feedback from Lean result."""
        # Generate hints based on errors
        hints = []
        if self.auto_hints:
            hints = self._generate_hints(result, action)

        # Normalize error messages
        errors = [self._normalize_error(e) for e in result.errors]

        # Create message
        if result.is_complete:
            message = "Proof complete!"
        elif result.is_success:
            message = "Code accepted, but proof is incomplete."
        else:
            message = "Execution failed."

        return ExecutionFeedback(
            success=result.is_success,
            complete=result.is_complete,
            message=message,
            goals=result.goals,
            errors=errors,
            hints=hints,
            raw_result=result,
        )

    def _normalize_error(self, error: str) -> str:
        """Normalize error message for LLM consumption."""
        # Remove file paths
        error = re.sub(r"/[^\s:]+\.lean:\d+:\d+:", "", error)

        # Simplify type mismatch errors
        if "type mismatch" in error.lower():
            # Extract the key parts
            match = re.search(r"has type\s+(.*?)\s+but is expected", error, re.DOTALL)
            if match:
                error = f"Type mismatch: got {match.group(1)}"

        # Simplify unknown identifier
        if "unknown identifier" in error.lower():
            match = re.search(r"unknown identifier ['\"]?(\w+)['\"]?", error, re.IGNORECASE)
            if match:
                error = f"Unknown identifier: {match.group(1)}"

        return error.strip()

    def _generate_hints(self, result: LeanResult, action: LeanAction) -> list[str]:
        """Generate hints based on errors."""
        hints = []
        error_text = " ".join(result.errors).lower()

        # Unknown identifier hints
        if "unknown identifier" in error_text:
            match = re.search(r"unknown identifier ['\"]?(\w+)['\"]?", error_text)
            if match:
                ident = match.group(1)
                hints.append(f"'{ident}' is not defined. Check spelling or ensure it's imported.")
                # Suggest alternatives based on common typos
                alternatives = self._suggest_alternatives(ident)
                if alternatives:
                    hints.append(f"Did you mean: {', '.join(alternatives)}?")

        # Type mismatch hints
        if "type mismatch" in error_text:
            hints.append("Check that your expression has the expected type.")
            hints.append("You may need to use a type conversion or coercion.")

        # Unsolved goals hints
        if result.goals:
            hints.append("There are remaining goals to prove.")
            if len(result.goals) == 1:
                goal = result.goals[0]
                hints.extend(self._suggest_tactics(goal))

        # Tactic hints
        if "tactic" in error_text and "failed" in error_text:
            hints.append("The tactic didn't work. Try a different approach.")

        # Syntax hints
        if "expected" in error_text and "got" in error_text:
            hints.append("Check for syntax errors: missing parentheses, colons, or keywords.")

        return hints

    def _suggest_alternatives(self, identifier: str) -> list[str]:
        """Suggest alternative identifiers."""
        # Common Lean alternatives
        alternatives_map = {
            "nat": ["Nat"],
            "int": ["Int"],
            "add_comm": ["Nat.add_comm", "Int.add_comm"],
            "add_assoc": ["Nat.add_assoc", "Int.add_assoc"],
            "mul_comm": ["Nat.mul_comm", "Int.mul_comm"],
            "rfl": ["rfl"],
            "refl": ["rfl"],
            "eq_refl": ["Eq.refl", "rfl"],
            "simp": ["simp", "simp only"],
            "ring": ["ring", "ring_nf"],
            "linarith": ["omega", "linarith"],
        }

        lower_ident = identifier.lower()
        return alternatives_map.get(lower_ident, [])

    def _suggest_tactics(self, goal: str) -> list[str]:
        """Suggest tactics based on goal."""
        suggestions = []

        # Equality goals
        if "=" in goal or "⊢" in goal and "=" in goal:
            suggestions.append("Try 'rfl' for reflexivity or 'simp' to simplify.")
            suggestions.append("Consider 'ring' for arithmetic equalities.")
            suggestions.append("Use 'rw [lemma]' to rewrite using a known equality.")

        # Implication goals
        if "→" in goal or "->" in goal:
            suggestions.append("Use 'intro' to introduce the hypothesis.")

        # Conjunction goals
        if "∧" in goal:
            suggestions.append("Use 'constructor' or 'And.intro' to split the goal.")

        # Disjunction goals
        if "∨" in goal:
            suggestions.append("Use 'left' or 'right' to choose a branch.")

        # Existence goals
        if "∃" in goal:
            suggestions.append("Use 'use <witness>' to provide the witness.")

        # Universal goals
        if "∀" in goal:
            suggestions.append("Use 'intro' to introduce the universally quantified variable.")

        return suggestions

    def __enter__(self) -> "LeanExecutor":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


def execute_lean(code: str, timeout: int = 30) -> ExecutionFeedback:
    """Convenience function for quick Lean execution."""
    with LeanExecutor(server=LeanServer(timeout=timeout)) as executor:
        return executor.execute(code)
