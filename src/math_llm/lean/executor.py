"""
Lean executor - executes Lean code and returns structured results.

Simple and focused: parse LLM output, execute, return result.
No hints or suggestions - the model should understand Lean errors.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from math_llm.lean.server import LeanServer, LeanResult


@dataclass
class LeanAction:
    """A tactic parsed from LLM output."""

    code: str
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_llm_output(cls, output: str) -> "LeanAction":
        """Parse LLM output to extract Lean tactic."""
        # Try code block first
        match = re.search(r"```(?:lean4?|lean)?\s*\n?(.*?)```", output, re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            # Raw output, skip common explanation prefixes
            lines = [
                line for line in output.split("\n")
                if not line.strip().startswith(("Let me", "I'll", "First,", "Now,", "Here's"))
            ]
            code = "\n".join(lines).strip()

        # Clean up
        code = code.strip()

        # Remove "by" prefix
        if code.startswith("by\n"):
            code = code[3:].strip()
        elif code.startswith("by "):
            code = code[3:].strip()

        # Extract tactic if full theorem given
        if code.startswith(("theorem ", "lemma ", "example ")):
            match = re.search(r":=\s*(?:by\s+)?(.+)", code, re.DOTALL)
            if match:
                code = match.group(1).strip()

        return cls(code=code)


@dataclass
class ExecutionResult:
    """Result of executing a tactic."""

    success: bool
    complete: bool
    goals: list[str]
    errors: list[str]
    raw: LeanResult

    @property
    def error(self) -> Optional[str]:
        """First error message, if any."""
        return self.errors[0] if self.errors else None


class LeanExecutor:
    """Executes Lean tactics and returns structured results."""

    def __init__(self, server: Optional[LeanServer] = None):
        self.server = server or LeanServer()
        self._owns_server = server is None

    def start(self) -> None:
        if self._owns_server:
            self.server.start()

    def stop(self) -> None:
        if self._owns_server:
            self.server.stop()

    def execute(self, statement: str, tactic: str) -> ExecutionResult:
        """
        Execute a tactic for a theorem statement.

        Args:
            statement: The theorem statement (with sorry)
            tactic: The tactic to try

        Returns:
            ExecutionResult with success/complete/goals/errors
        """
        result = self.server.check_proof(statement, tactic)

        return ExecutionResult(
            success=result.is_success,
            complete=result.is_complete,
            goals=result.goals,
            errors=self._clean_errors(result.errors),
            raw=result,
        )

    def _clean_errors(self, errors: list[str]) -> list[str]:
        """Remove file paths from error messages."""
        cleaned = []
        for err in errors:
            # Strip file paths like /tmp/xxx.lean:1:2:
            err = re.sub(r"/[^\s:]+\.lean:\d+:\d+:\s*", "", err)
            cleaned.append(err.strip())
        return cleaned

    def __enter__(self) -> "LeanExecutor":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
