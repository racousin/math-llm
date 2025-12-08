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


def _get_lean_env() -> dict:
    """Get environment with elan PATH included."""
    env = os.environ.copy()
    elan_bin = Path.home() / ".elan" / "bin"
    if elan_bin.exists():
        env["PATH"] = f"{elan_bin}:{env.get('PATH', '')}"
    return env

import pexpect
from rich.console import Console

console = Console()


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

    Supports two modes:
    1. File-based execution (default): Write to temp file and run lake
    2. REPL mode: Interactive session with Lean REPL (faster for iteration)
    """

    def __init__(
        self,
        lean_path: str = "lake",
        project_path: Optional[str] = None,
        timeout: int = 60,
        memory_limit: int = 4096,
        use_repl: bool = False,
    ):
        self.lean_path = lean_path
        self.project_path = Path(project_path) if project_path else None
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.use_repl = use_repl

        self._repl_process: Optional[pexpect.spawn] = None
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None

    def start(self) -> None:
        """Start the Lean server/REPL."""
        if self.use_repl:
            self._start_repl()
        else:
            self._ensure_project()

    def stop(self) -> None:
        """Stop the Lean server/REPL."""
        if self._repl_process:
            self._repl_process.close()
            self._repl_process = None
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

        # Create lakefile.lean
        lakefile = self.project_path / "lakefile.lean"
        lakefile.write_text("""
import Lake
open Lake DSL

package «math_llm_workspace»

@[default_target]
lean_lib «MathLLM»
""")

        # Create lean-toolchain
        toolchain = self.project_path / "lean-toolchain"
        toolchain.write_text("leanprover/lean4:v4.3.0\n")

        # Create src directory
        (self.project_path / "MathLLM").mkdir(exist_ok=True)
        (self.project_path / "MathLLM" / "Basic.lean").write_text("-- Math LLM workspace\n")

        console.print(f"[blue]Created temporary Lean project at {self.project_path}[/blue]")

    def _start_repl(self) -> None:
        """Start Lean REPL process."""
        self._ensure_project()
        # Note: Lean 4 REPL is still experimental
        # For now, we'll use file-based execution
        console.print("[yellow]REPL mode not fully implemented, using file mode[/yellow]")
        self.use_repl = False

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
            parts.append("-- Auto-generated imports")

        if context:
            parts.append(context)

        parts.append(code)

        return "\n\n".join(parts)

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
            statement: The theorem statement
            proof: The proposed proof

        Returns:
            LeanResult indicating if proof is valid
        """
        # Combine statement and proof
        if ":= by" in statement:
            # Replace sorry with actual proof
            code = statement.replace("sorry", proof.strip())
        else:
            code = f"{statement}\n{proof}"

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
