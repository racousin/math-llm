"""
Lean server interface for interacting with Lean 4.

Provides a clean interface for:
- Starting/stopping Lean REPL
- Executing Lean code
- Parsing results and errors
"""

import asyncio
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from math_llm.config import MATHLIB_VERSION, LEAN_TOOLCHAIN


def _get_lean_env() -> dict:
    """Get environment with elan PATH included."""
    env = os.environ.copy()
    elan_bin = Path.home() / ".elan" / "bin"
    if elan_bin.exists():
        env["PATH"] = f"{elan_bin}:{env.get('PATH', '')}"
    return env

from rich.console import Console

console = Console()

# Lean 4 imports for mathematical proofs
DEFAULT_IMPORTS = """import Mathlib
import Aesop
"""


class LeanResultStatus(Enum):
    """Status of Lean execution result."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    INCOMPLETE = "incomplete"


@dataclass
class LeanResult:
    """Result of Lean code execution."""

    status: LeanResultStatus
    output: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    execution_time: float = 0.0

    @property
    def is_success(self) -> bool:
        return self.status == LeanResultStatus.SUCCESS

    @property
    def is_complete(self) -> bool:
        """Check if proof is complete (no goals remaining)."""
        return self.is_success and not self.goals and "sorry" not in self.output.lower()

    def to_feedback(self) -> str:
        """Convert to feedback string for LLM."""
        parts = []

        if self.is_success:
            if self.is_complete:
                parts.append("Proof complete!")
            else:
                parts.append("Code accepted.")
                if self.goals:
                    parts.append(f"Remaining goals:\n" + "\n".join(f"  {g}" for g in self.goals))
        else:
            parts.append(f"Error: {self.status.value}")

        if self.errors:
            parts.append("Errors:")
            for err in self.errors:
                parts.append(f"  - {err}")

        if self.warnings:
            parts.append("Warnings:")
            for warn in self.warnings:
                parts.append(f"  - {warn}")

        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "output": self.output,
            "errors": self.errors,
            "warnings": self.warnings,
            "goals": self.goals,
            "execution_time": self.execution_time,
            "is_complete": self.is_complete,
        }


class LeanServer:
    """
    Interface to Lean 4 for executing and verifying proofs.

    Requires a Mathlib project. Set project_path or MATHLIB_PROJECT_PATH env var.
    To setup: lake exe cache get && lake build
    """

    def __init__(
        self,
        lean_path: str = "lake",
        project_path: Optional[str] = None,
        timeout: int = 300,  # 5 minutes - Mathlib imports take time
        memory_limit: int = 4096,
    ):
        self.lean_path = lean_path
        self.timeout = timeout
        self.memory_limit = memory_limit

        # Get project path from arg or env
        self.project_path = Path(
            project_path or os.environ.get("MATHLIB_PROJECT_PATH", "")
        ) if (project_path or os.environ.get("MATHLIB_PROJECT_PATH")) else None

        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None

    def start(self) -> None:
        """Start the Lean server."""
        self._ensure_project()

    def stop(self) -> None:
        """Stop the Lean server."""
        if self._temp_dir:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def _ensure_project(self) -> None:
        """Ensure we have a valid Lean project to work with."""
        if self.project_path and (self.project_path / "lakefile.lean").exists():
            return

        # Create a minimal Lean 4 project in temp directory
        self._temp_dir = tempfile.TemporaryDirectory(prefix="math_llm_lean_")
        self.project_path = Path(self._temp_dir.name)

        # Create lakefile.lean with pinned Mathlib version
        # Note: Lake API changed - leanOptions moved out of package block in newer versions
        lakefile = self.project_path / "lakefile.lean"
        lakefile.write_text(f"""import Lake
open Lake DSL

package «math_llm_workspace»

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "{MATHLIB_VERSION}"

@[default_target]
lean_lib «MathLLM» where
  globs := #[.submodules `MathLLM]
""")

        # Create lean-toolchain (must match Mathlib version)
        toolchain = self.project_path / "lean-toolchain"
        toolchain.write_text(f"{LEAN_TOOLCHAIN}\n")

        # Create src directory
        (self.project_path / "MathLLM").mkdir(exist_ok=True)
        (self.project_path / "MathLLM" / "Basic.lean").write_text("-- Math LLM workspace\n")

        console.print(f"[blue]Created temporary Lean project at {self.project_path}[/blue]")

        # Download Mathlib cache - critical for performance!
        console.print("[yellow]Downloading Mathlib cache (this may take a few minutes the first time)...[/yellow]")
        try:
            cache_result = subprocess.run(
                [self.lean_path, "exe", "cache", "get"],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout for cache download
                env=_get_lean_env(),
            )
            if cache_result.returncode == 0:
                console.print("[green]Mathlib cache downloaded successfully[/green]")
            else:
                console.print(f"[yellow]Warning: Cache download may have issues: {cache_result.stderr[:200]}[/yellow]")
        except subprocess.TimeoutExpired:
            console.print("[yellow]Warning: Cache download timed out - Lean execution may be slow[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not download cache: {e}[/yellow]")

    def execute(self, code: str, context: Optional[str] = None) -> LeanResult:
        """
        Execute Lean code and return the result.

        Args:
            code: Lean 4 code to execute
            context: Optional additional context (imports, etc.)

        Returns:
            LeanResult with status, output, and any errors
        """
        import time
        start_time = time.time()

        try:
            result = self._execute_file(code, context)
        except Exception as e:
            result = LeanResult(
                status=LeanResultStatus.ERROR,
                output="",
                errors=[str(e)],
            )

        result.execution_time = time.time() - start_time
        return result

    def _execute_file(self, code: str, context: Optional[str] = None) -> LeanResult:
        """Execute code by writing to a file and running lake."""
        self._ensure_project()

        # Build full code with imports
        full_code = self._build_full_code(code, context)

        # Write to temp file in project
        temp_file = self.project_path / "MathLLM" / "Temp.lean"
        temp_file.write_text(full_code)

        try:
            # Run lean directly on the file
            result = subprocess.run(
                [self.lean_path, "env", "lean", str(temp_file)],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=_get_lean_env(),
            )

            return self._parse_output(result.stdout, result.stderr, result.returncode)

        except subprocess.TimeoutExpired:
            return LeanResult(
                status=LeanResultStatus.TIMEOUT,
                output="",
                errors=[f"Execution timed out after {self.timeout}s"],
            )
        except FileNotFoundError:
            # lake not found, try lean directly
            return self._execute_lean_direct(full_code)
        finally:
            # Cleanup temp file
            if temp_file.exists():
                temp_file.unlink()

    def _execute_lean_direct(self, code: str) -> LeanResult:
        """Execute using lean directly (without lake)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            result = subprocess.run(
                ["lean", temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=_get_lean_env(),
            )
            return self._parse_output(result.stdout, result.stderr, result.returncode)
        except FileNotFoundError:
            return LeanResult(
                status=LeanResultStatus.ERROR,
                output="",
                errors=["Lean not found. Install: curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh && elan default leanprover/lean4:stable"],
            )
        except subprocess.TimeoutExpired:
            return LeanResult(
                status=LeanResultStatus.TIMEOUT,
                output="",
                errors=[f"Execution timed out after {self.timeout}s"],
            )
        finally:
            os.unlink(temp_path)

    def _build_full_code(self, code: str, context: Optional[str] = None) -> str:
        """Build complete Lean code with imports."""
        parts = []

        # Default imports for common mathematical proofs
        if not code.strip().startswith("import"):
            parts.append(DEFAULT_IMPORTS)

        if context:
            parts.append(context)

        parts.append(code)

        return "\n\n".join(parts)

    def suggest_tactics(self, statement: str, current_proof: str = "") -> list[str]:
        """
        Use Lean's suggestion tactics to find applicable lemmas/tactics.

        Uses exact?, apply?, simp? to discover what lemmas could work.
        """
        suggestions = []

        # Build code with suggestion tactics
        if ":= by" in statement:
            base_code = statement.replace("sorry", current_proof + "\n  exact?" if current_proof else "exact?")
        else:
            base_code = f"{statement} := by\n  {current_proof}\n  exact?" if current_proof else f"{statement} := by exact?"

        # Try exact?
        result = self.execute(base_code)
        exact_suggestions = self._parse_suggestions(result.output, "exact")
        suggestions.extend(exact_suggestions)

        # Try apply?
        apply_code = base_code.replace("exact?", "apply?")
        result = self.execute(apply_code)
        apply_suggestions = self._parse_suggestions(result.output, "apply")
        suggestions.extend(apply_suggestions)

        return suggestions[:10]  # Limit to top 10

    def _parse_suggestions(self, output: str, tactic_type: str) -> list[str]:
        """Parse suggestion output from Lean."""
        suggestions = []

        # Look for "Try this:" patterns
        try_pattern = r"Try this:\s*(.+?)(?:\n|$)"
        for match in re.finditer(try_pattern, output):
            suggestion = match.group(1).strip()
            if suggestion:
                suggestions.append(suggestion)

        # Look for suggestion comments
        suggestion_pattern = rf"{tactic_type}\s+(\S+)"
        for match in re.finditer(suggestion_pattern, output):
            suggestions.append(f"{tactic_type} {match.group(1)}")

        return suggestions

    def _parse_output(self, stdout: str, stderr: str, return_code: int) -> LeanResult:
        """Parse Lean output into structured result."""
        output = stdout + stderr
        errors = []
        warnings = []
        goals = []

        # Parse errors
        error_pattern = r"error:.*?(?=\n\n|\Z)"
        for match in re.finditer(error_pattern, output, re.DOTALL):
            errors.append(match.group(0).strip())

        # Parse warnings
        warning_pattern = r"warning:.*?(?=\n\n|\Z)"
        for match in re.finditer(warning_pattern, output, re.DOTALL):
            warnings.append(match.group(0).strip())

        # Parse unsolved goals
        goal_pattern = r"unsolved goals\n(.*?)(?=\n\n|\Z)"
        for match in re.finditer(goal_pattern, output, re.DOTALL):
            goals.extend(match.group(1).strip().split("\n"))

        # Also check for "Goals:" or "⊢" patterns
        if "⊢" in output:
            goal_lines = [
                line.strip()
                for line in output.split("\n")
                if "⊢" in line
            ]
            goals.extend(goal_lines)

        # Determine status
        if return_code == 0 and not errors:
            status = LeanResultStatus.SUCCESS
        elif "sorry" in output.lower() or goals:
            status = LeanResultStatus.INCOMPLETE
        else:
            status = LeanResultStatus.ERROR

        return LeanResult(
            status=status,
            output=output,
            errors=errors,
            warnings=warnings,
            goals=list(set(goals)),  # Deduplicate
        )

    def check_proof(self, statement: str, proof: str) -> LeanResult:
        """
        Check if a proof is valid for a given statement.

        Args:
            statement: The theorem statement (may end with := sorry or := by sorry)
            proof: The proposed proof tactic(s)

        Returns:
            LeanResult indicating if proof is valid
        """
        proof = proof.strip()

        # Normalize: remove leading "by" if present (we'll add it)
        if proof.startswith("by ") or proof.startswith("by\n"):
            proof = proof[2:].strip()

        # Build the code based on statement format
        if ":= by" in statement:
            # Statement has "by" - just replace sorry with tactic
            code = statement.replace("sorry", proof)
        elif ":= sorry" in statement:
            # Statement has ":= sorry" - replace with ":= by tactic"
            code = statement.replace(":= sorry", f":= by\n  {proof}")
        elif statement.rstrip().endswith("sorry"):
            # Ends with sorry (no :=) - replace sorry
            code = statement.rsplit("sorry", 1)[0] + f"by\n  {proof}"
        else:
            # No sorry - append proof
            code = f"{statement} := by\n  {proof}"

        return self.execute(code)

    def get_goal_state(self, code: str, position: Optional[int] = None) -> Optional[str]:
        """Get the goal state at a specific position in the code."""
        # This would require LSP integration for full functionality
        # For now, we return goals from execution
        result = self.execute(code)
        if result.goals:
            return "\n".join(result.goals)
        return None

    def __enter__(self) -> "LeanServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


# Convenience function for quick execution
def quick_check(code: str, timeout: int = 30) -> LeanResult:
    """Quickly check Lean code without setting up a persistent server."""
    with LeanServer(timeout=timeout) as server:
        return server.execute(code)
