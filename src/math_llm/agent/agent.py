"""
Main LLM Agent for Lean theorem proving.
"""

import time
from typing import Optional, Callable

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from math_llm.config import AgentConfig
from math_llm.data.datasets import LeanProblem
from math_llm.lean.executor import LeanExecutor, LeanAction, ExecutionFeedback
from math_llm.lean.server import LeanServer
from math_llm.agent.trajectory import Trajectory, Step

console = Console()


SYSTEM_PROMPT = """You are an expert Lean 4 theorem prover. Your task is to complete proofs for mathematical theorems in Lean 4.

IMPORTANT: You are given a theorem statement with 'sorry' as a placeholder. Output ONLY the proof tactics that replace 'sorry'.

DO NOT output:
- The full theorem statement (no 'theorem', 'lemma', or 'example' keywords)
- Type signatures or declarations
- The theorem name

DO output ONLY:
- The proof tactics that go after 'by'
- For simple proofs: a single tactic like `exact Nat.add_comm a b` or `omega`
- For multi-step proofs: tactics separated by newlines or semicolons

Standard tactics: simp, rfl, intro, apply, exact, rw, ring, omega, linarith, trivial, decide

Examples:
- If goal is `a + b = b + a`, output: `exact Nat.add_comm a b`
- If goal is `0 + n = n`, output: `simp`
- If goal needs intro then apply: `intro h; exact h`

Output your tactics in a code block:
```lean4
exact Nat.add_comm a b
```
"""


class LeanAgent:
    """
    LLM Agent that iteratively attempts to prove Lean theorems.

    The agent:
    1. Receives a theorem statement
    2. Generates proof attempts
    3. Executes in Lean and receives feedback
    4. Iterates until success or max steps reached
    """

    def __init__(
        self,
        model: "LLMWrapper",  # Forward reference to avoid circular import
        executor: Optional[LeanExecutor] = None,
        config: Optional[AgentConfig] = None,
        system_prompt: Optional[str] = None,
    ):
        self.model = model
        self.executor = executor
        self.config = config or AgentConfig()
        self.system_prompt = system_prompt or SYSTEM_PROMPT

        self._owns_executor = executor is None

    def start(self) -> None:
        """Initialize the agent."""
        if self._owns_executor:
            self.executor = LeanExecutor()
            self.executor.start()

    def stop(self) -> None:
        """Cleanup the agent."""
        if self._owns_executor and self.executor:
            self.executor.stop()

    def solve(
        self,
        problem: LeanProblem,
        callback: Optional[Callable[[Step], None]] = None,
    ) -> Trajectory:
        """
        Attempt to solve a Lean theorem proving problem.

        Args:
            problem: The theorem to prove
            callback: Optional callback called after each step

        Returns:
            Trajectory containing all steps and final result
        """
        start_time = time.time()

        # Initialize trajectory
        trajectory = Trajectory(
            problem_id=problem.id,
            statement=problem.statement,
            description=problem.description,
        )

        # Build initial prompt
        prompt = self._build_initial_prompt(problem)

        if self.config.verbose:
            console.print(Panel(problem.statement, title="Problem", border_style="blue"))

        # Main solving loop
        for step_num in range(self.config.max_steps):
            if self.config.verbose:
                console.print(f"\n[yellow]Step {step_num + 1}/{self.config.max_steps}[/yellow]")

            # Generate action from LLM
            response = self.model.generate(
                prompt,
                system_prompt=self.system_prompt,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )

            # Parse action from response
            action = LeanAction.from_llm_output(response)

            if self.config.verbose:
                console.print(Syntax(action.code, "lean", theme="monokai"))

            # Execute in Lean
            feedback = self.executor.execute_proof_attempt(
                statement=problem.statement,
                proof=action.code,
            )

            # Record step
            step = trajectory.add_step(
                action=action.code,
                result=feedback,
                thinking=response,  # Store full response for training
            )

            if callback:
                callback(step)

            if self.config.verbose:
                if feedback.complete:
                    console.print("[green]✓ Proof complete![/green]")
                elif feedback.success:
                    console.print("[yellow]Accepted but incomplete[/yellow]")
                    if feedback.goals:
                        console.print(f"Goals: {feedback.goals}")
                else:
                    console.print(f"[red]Error: {feedback.errors}[/red]")

            # Check for completion
            if feedback.complete:
                break

            # Check for early stopping on errors
            if self.config.stop_on_error and not feedback.success:
                if self.config.verbose:
                    console.print("[red]Stopping due to error[/red]")
                break

            # Update prompt with trajectory
            prompt = self._build_continuation_prompt(problem, trajectory)

        trajectory.total_time = time.time() - start_time
        trajectory.success = trajectory.is_complete

        if self.config.verbose:
            self._print_summary(trajectory)

        return trajectory

    def _build_initial_prompt(self, problem: LeanProblem) -> str:
        """Build the initial prompt for the LLM."""
        parts = []

        if problem.description:
            parts.append(f"Problem: {problem.description}")

        parts.append("Prove the following theorem in Lean 4:")
        parts.append(f"\n```lean4\n{problem.statement}\n```\n")
        parts.append("Provide your Lean 4 proof.")

        return "\n\n".join(parts)

    def _build_continuation_prompt(self, problem: LeanProblem, trajectory: Trajectory) -> str:
        """Build continuation prompt with previous attempts."""
        parts = []

        parts.append(self._build_initial_prompt(problem))
        parts.append("\n\nPrevious attempts:")
        parts.append(trajectory.to_prompt_format())
        parts.append("\n\nBased on the feedback above, provide an improved proof:")

        return "\n".join(parts)

    def _print_summary(self, trajectory: Trajectory) -> None:
        """Print a summary of the solving attempt."""
        console.print("\n" + "=" * 50)
        if trajectory.success:
            console.print(f"[green]✓ Solved in {trajectory.num_steps} steps![/green]")
        else:
            console.print(f"[red]✗ Failed after {trajectory.num_steps} steps[/red]")
        console.print(f"Time: {trajectory.total_time:.2f}s")
        console.print("=" * 50)

    def batch_solve(
        self,
        problems: list[LeanProblem],
        callback: Optional[Callable[[int, Trajectory], None]] = None,
    ) -> list[Trajectory]:
        """
        Solve multiple problems.

        Args:
            problems: List of problems to solve
            callback: Optional callback called after each problem

        Returns:
            List of trajectories
        """
        trajectories = []

        for i, problem in enumerate(problems):
            if self.config.verbose:
                console.print(f"\n[blue]Problem {i + 1}/{len(problems)}: {problem.id}[/blue]")

            trajectory = self.solve(problem)
            trajectories.append(trajectory)

            if callback:
                callback(i, trajectory)

        return trajectories

    def __enter__(self) -> "LeanAgent":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
