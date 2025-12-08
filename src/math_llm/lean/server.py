"""
Lean server interface for interacting with Lean 4.

Uses LeanREPL - persistent process with JSON protocol for fast execution (~0.1s per check).
"""

import json
import os
import select
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from rich.console import Console

from math_llm.config import MATHLIB_VERSION, LEAN_TOOLCHAIN

console = Console()


def _get_lean_env() -> dict:
    """Get environment with elan PATH included."""
    env = os.environ.copy()
    elan_bin = Path.home() / ".elan" / "bin"
    if elan_bin.exists():
        env["PATH"] = f"{elan_bin}:{env.get('PATH', '')}"
    return env


DEFAULT_IMPORTS = """import Mathlib
import Aesop
"""


class LeanResultStatus(Enum):
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
        return self.is_success and not self.goals and "sorry" not in self.output.lower()

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


# =============================================================================
# LEAN REPL (Fast - persistent process)
# =============================================================================

class LeanREPL:
    """
    Fast Lean execution using persistent REPL process.

    Uses the repl package from mathlib4. First check loads imports (~10s),
    subsequent checks are fast (~0.1s).

    Install repl: Add to lakefile.lean:
        require «repl» from git "https://github.com/leanprover-community/repl" @ "master"
    """

    def __init__(
        self,
        project_path: Optional[str] = None,
        timeout: int = 30,
    ):
        self.project_path = Path(
            project_path or os.environ.get("MATHLIB_PROJECT_PATH", "")
        ) if (project_path or os.environ.get("MATHLIB_PROJECT_PATH")) else None
        self.timeout = timeout
        self._process: Optional[subprocess.Popen] = None
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self._lock = threading.Lock()
        self._imports_loaded = False

    def start(self) -> None:
        """Start the REPL process."""
        self._ensure_project()
        self._start_process()

    def stop(self) -> None:
        """Stop the REPL process."""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        if self._temp_dir:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def _ensure_project(self) -> None:
        """Ensure we have a valid Lean project with repl."""
        if self.project_path and (self.project_path / "lakefile.lean").exists():
            return

        self._temp_dir = tempfile.TemporaryDirectory(prefix="math_llm_lean_")
        self.project_path = Path(self._temp_dir.name)

        # Create lakefile with repl dependency
        lakefile = self.project_path / "lakefile.lean"
        lakefile.write_text(f"""import Lake
open Lake DSL

package «math_llm_workspace»

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "{MATHLIB_VERSION}"

require «repl» from git
  "https://github.com/leanprover-community/repl" @ "master"

@[default_target]
lean_lib «MathLLM»
""")

        toolchain = self.project_path / "lean-toolchain"
        toolchain.write_text(f"{LEAN_TOOLCHAIN}\n")

        (self.project_path / "MathLLM").mkdir(exist_ok=True)
        (self.project_path / "MathLLM" / "Basic.lean").write_text("-- Math LLM\n")

        console.print(f"[blue]Created Lean project at {self.project_path}[/blue]")

        # Download cache
        console.print("[yellow]Downloading Mathlib cache...[/yellow]")
        try:
            subprocess.run(
                ["lake", "exe", "cache", "get"],
                cwd=self.project_path,
                capture_output=True,
                timeout=600,
                env=_get_lean_env(),
            )
            console.print("[green]Cache downloaded[/green]")
        except Exception as e:
            console.print(f"[yellow]Cache warning: {e}[/yellow]")

    def _start_process(self) -> None:
        """Start the REPL subprocess."""
        console.print(f"[dim]Starting REPL in {self.project_path}...[/dim]")
        self._process = subprocess.Popen(
            ["lake", "env", "lean", "--run", "Repl"],
            cwd=self.project_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_get_lean_env(),
        )
        self._imports_loaded = False

        # Give process a moment to start and check it's alive
        time.sleep(0.5)
        if self._process.poll() is not None:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            console.print(f"[red]REPL failed to start: {stderr[:500]}[/red]")
        else:
            console.print("[dim]REPL process started[/dim]")

    def _send_command(self, cmd: dict, timeout: Optional[int] = None) -> dict:
        """Send JSON command to REPL and get response."""
        if not self._process or self._process.poll() is not None:
            self._start_process()

        timeout = timeout or self.timeout

        with self._lock:
            try:
                # Send command
                cmd_json = json.dumps(cmd)
                self._process.stdin.write(cmd_json + "\n")
                self._process.stdin.flush()

                # Read response with timeout using select
                ready, _, _ = select.select([self._process.stdout], [], [], timeout)
                if not ready:
                    # Check if process died
                    if self._process.poll() is not None:
                        stderr = self._process.stderr.read() if self._process.stderr else ""
                        return {"error": f"REPL process died: {stderr[:500]}"}
                    return {"error": f"REPL timeout after {timeout}s"}

                response_line = self._process.stdout.readline()
                if not response_line:
                    stderr = self._process.stderr.read() if self._process.stderr else ""
                    return {"error": f"No response from REPL. stderr: {stderr[:500]}"}

                return json.loads(response_line)
            except Exception as e:
                return {"error": str(e)}

    def check_proof(self, statement: str, proof: str) -> LeanResult:
        """Check a proof using the REPL."""
        start_time = time.time()

        # Load imports on first use (can take 30-60s for Mathlib)
        if not self._imports_loaded:
            console.print("[dim]Loading imports (first time only, may take ~60s)...[/dim]")
            import_cmd = {"cmd": DEFAULT_IMPORTS.strip(), "env": 0}
            resp = self._send_command(import_cmd, timeout=120)  # 2 min timeout for imports
            if "error" in resp:
                return LeanResult(
                    status=LeanResultStatus.ERROR,
                    output="",
                    errors=[f"Import failed: {resp['error']}"],
                    execution_time=time.time() - start_time,
                )
            self._imports_loaded = True
            self._env_id = resp.get("env", 0)

        # Build code
        proof = proof.strip()
        if proof.startswith("by "):
            proof = proof[3:].strip()

        if ":= by" in statement:
            code = statement.replace("sorry", proof)
        elif ":= sorry" in statement:
            code = statement.replace(":= sorry", f":= by\n  {proof}")
        else:
            code = f"{statement} := by\n  {proof}"

        # Send to REPL
        cmd = {"cmd": code, "env": self._env_id}
        resp = self._send_command(cmd)

        execution_time = time.time() - start_time
        return self._parse_repl_response(resp, execution_time)

    def _parse_repl_response(self, resp: dict, execution_time: float) -> LeanResult:
        """Parse REPL JSON response into LeanResult."""
        if "error" in resp:
            return LeanResult(
                status=LeanResultStatus.ERROR,
                output=str(resp),
                errors=[resp["error"]],
                execution_time=execution_time,
            )

        messages = resp.get("messages", [])
        errors = [m["data"] for m in messages if m.get("severity") == "error"]
        warnings = [m["data"] for m in messages if m.get("severity") == "warning"]

        # Check for goals in sorries
        sorries = resp.get("sorries", [])
        goals = [s.get("goal", "") for s in sorries if s.get("goal")]

        if errors:
            status = LeanResultStatus.ERROR
        elif goals or sorries:
            status = LeanResultStatus.INCOMPLETE
        else:
            status = LeanResultStatus.SUCCESS

        return LeanResult(
            status=status,
            output=json.dumps(resp),
            errors=errors,
            warnings=warnings,
            goals=goals,
            execution_time=execution_time,
        )

    def __enter__(self) -> "LeanREPL":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()


