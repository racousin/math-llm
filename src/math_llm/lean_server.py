"""
Lean REPL server interface.

Provides fast Lean 4 code execution using persistent REPL process.
"""

import json
import os
import select
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# =============================================================================
# LEAN/MATHLIB VERSION - Must be kept in sync with scripts/setup_mathlib.sh
# =============================================================================
MATHLIB_VERSION = "v4.25.2"
LEAN_TOOLCHAIN = "leanprover/lean4:v4.25.2"
# =============================================================================

DEFAULT_IMPORTS = "import Mathlib\nimport Aesop"


def _get_lean_env() -> dict:
    """Get environment with elan PATH included."""
    env = os.environ.copy()
    elan_bin = Path.home() / ".elan" / "bin"
    if elan_bin.exists():
        env["PATH"] = f"{elan_bin}:{env.get('PATH', '')}"
    return env


@dataclass
class LeanResult:
    """Result of Lean code execution."""
    success: bool
    complete: bool  # Proof is complete (no goals, no sorry)
    output: str
    errors: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    execution_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "complete": self.complete,
            "output": self.output,
            "errors": self.errors,
            "goals": self.goals,
            "execution_time": self.execution_time,
        }


class LeanServer:
    """
    Lean REPL server for fast proof checking.

    First check loads imports (~60s), subsequent checks are fast (~0.1s).
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
        self._env_id = 0

    def start(self) -> None:
        """Start the REPL process and preload imports."""
        self._ensure_project()
        self._start_process()
        self._load_imports()

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

        self._temp_dir = tempfile.TemporaryDirectory(prefix="lean_bench_")
        self.project_path = Path(self._temp_dir.name)

        # Create lakefile with repl dependency
        lakefile = self.project_path / "lakefile.lean"
        lakefile.write_text(f"""import Lake
open Lake DSL

package «lean_bench»

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "{MATHLIB_VERSION}"

require «repl» from git
  "https://github.com/leanprover-community/repl" @ "master"

@[default_target]
lean_lib «LeanBench»
""")

        toolchain = self.project_path / "lean-toolchain"
        toolchain.write_text(f"{LEAN_TOOLCHAIN}\n")

        (self.project_path / "LeanBench").mkdir(exist_ok=True)
        (self.project_path / "LeanBench" / "Basic.lean").write_text("-- Lean Bench\n")

        print(f"[lean] Created project at {self.project_path}")

        # Download cache
        print("[lean] Downloading Mathlib cache (this may take a while)...")
        try:
            subprocess.run(
                ["lake", "exe", "cache", "get"],
                cwd=self.project_path,
                capture_output=True,
                timeout=600,
                env=_get_lean_env(),
            )
            print("[lean] Cache downloaded")
        except Exception as e:
            print(f"[lean] Cache warning: {e}")

    def _start_process(self) -> None:
        """Start the REPL subprocess."""
        print(f"[lean] Starting REPL in {self.project_path}...")
        self._process = subprocess.Popen(
            ["lake", "exe", "repl"],
            cwd=self.project_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_get_lean_env(),
        )
        self._imports_loaded = False

        time.sleep(0.5)
        if self._process.poll() is not None:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            print(f"[lean] REPL failed to start: {stderr[:500]}")
        else:
            print("[lean] REPL process started")

    def _send_command(self, cmd: dict, timeout: Optional[int] = None) -> dict:
        """Send JSON command to REPL and get response."""
        if not self._process or self._process.poll() is not None:
            self._start_process()

        timeout = timeout or self.timeout

        with self._lock:
            try:
                cmd_json = json.dumps(cmd)
                self._process.stdin.write(cmd_json + "\n\n")
                self._process.stdin.flush()

                start_time = time.time()
                while True:
                    elapsed = time.time() - start_time
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        return {"error": f"Timeout after {timeout}s"}

                    ready, _, _ = select.select([self._process.stdout], [], [], remaining)
                    if not ready:
                        if self._process.poll() is not None:
                            stderr = self._process.stderr.read() if self._process.stderr else ""
                            return {"error": f"REPL died: {stderr[:500]}"}
                        return {"error": f"Timeout after {timeout}s"}

                    response_line = self._process.stdout.readline()
                    if not response_line:
                        stderr = self._process.stderr.read() if self._process.stderr else ""
                        return {"error": f"No response. stderr: {stderr[:500]}"}

                    response_line = response_line.strip()
                    if not response_line:
                        continue

                    try:
                        return json.loads(response_line)
                    except json.JSONDecodeError:
                        continue

            except Exception as e:
                return {"error": str(e)}

    def _load_imports(self) -> bool:
        """Load Mathlib imports. Returns True on success."""
        if self._imports_loaded:
            return True

        print("[lean] Loading Mathlib imports (first time, ~60s)...")
        resp = self._send_command({"cmd": DEFAULT_IMPORTS}, timeout=120)

        if "error" in resp:
            print(f"[lean] Failed to load imports: {resp['error']}")
            return False

        self._imports_loaded = True
        self._env_id = resp.get("env", 0)
        print("[lean] Imports loaded")
        return True

    def check_proof(self, statement: str, proof: str) -> LeanResult:
        """
        Check if a proof is valid for a given statement.

        Args:
            statement: Lean theorem statement (with 'sorry' placeholder)
            proof: Proof tactics to verify

        Returns:
            LeanResult with success/complete status and any errors/goals
        """
        start_time = time.time()

        if not self._load_imports():
            return LeanResult(
                success=False,
                complete=False,
                output="",
                errors=["Failed to load Mathlib imports"],
                execution_time=time.time() - start_time,
            )

        # Build code - replace sorry with proof
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
        return self._parse_response(resp, execution_time)

    def _parse_response(self, resp: dict, execution_time: float) -> LeanResult:
        """Parse REPL response into LeanResult."""
        if "error" in resp:
            return LeanResult(
                success=False,
                complete=False,
                output=str(resp),
                errors=[resp["error"]],
                execution_time=execution_time,
            )

        messages = resp.get("messages", [])
        errors = [m["data"] for m in messages if m.get("severity") == "error"]

        # Check for remaining goals
        sorries = resp.get("sorries", [])
        goals = [s.get("goal", "") for s in sorries if s.get("goal")]

        success = len(errors) == 0
        complete = success and len(goals) == 0 and len(sorries) == 0

        return LeanResult(
            success=success,
            complete=complete,
            output=json.dumps(resp),
            errors=errors,
            goals=goals,
            execution_time=execution_time,
        )

    def run_tactic(self, statement: str, tactic: str) -> LeanResult:
        """
        Run a single tactic on a theorem statement.

        Returns remaining goals or errors.
        """
        return self.check_proof(statement, tactic)

    def __enter__(self) -> "LeanServer":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
