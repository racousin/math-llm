"""
LLM Agent for Lean theorem proving.
"""

import time
from typing import Optional, Callable

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from math_llm.config import AgentConfig
from math_llm.data.datasets import LeanProblem
from math_llm.lean.executor import LeanExecutor, LeanAction, ExecutionResult
from math_llm.agent.context import PromptBuilder, ProofState, SYSTEM_PROMPT
from math_llm.agent.trajectory import Trajectory, Step

console = Console()


class LeanAgent:
    """
    LLM Agent that iteratively proves Lean theorems.

    Loop: generate tactic → execute → feedback → repeat
    """

    def __init__(
        self,
        model,
        executor: Optional[LeanExecutor] = None,
        config: Optional[AgentConfig] = None,
        system_prompt: Optional[str] = None,
    ):
        self.model = model
        self.executor = executor
        self.config = config or AgentConfig()
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.prompt_builder = PromptBuilder(self.system_prompt)
        self._owns_executor = executor is None

    def start(self) -> None:
        if self._owns_executor:
            self.executor = LeanExecutor()
            self.executor.start()

    def stop(self) -> None:
        if self._owns_executor and self.executor:
            self.executor.stop()

    def solve(
        self,
        problem: LeanProblem,
        callback: Optional[Callable[[Step], None]] = None,
    ) -> Trajectory:
        """Attempt to prove a theorem."""
        start_time = time.time()

        trajectory = Trajectory(
            problem_id=problem.id,
            statement=problem.statement,
            description=problem.description,
        )

        state = ProofState(problem=problem)
        prompt = self.prompt_builder.build_initial(problem)

        if self.config.verbose:
            console.print(Panel(problem.statement, title="Problem", border_style="blue"))

        for step_num in range(self.config.max_steps):
            if self.config.verbose:
                console.print(f"\n[yellow]Step {step_num + 1}/{self.config.max_steps}[/yellow]")

            # Generate tactic
            llm_start = time.time()
            response = self.model.generate(
                prompt,
                system_prompt=self.system_prompt,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_new_tokens=self.config.max_new_tokens,
            )
            llm_time = time.time() - llm_start

            action = LeanAction.from_llm_output(response)

            if self.config.verbose:
                console.print(Syntax(action.code, "lean", theme="monokai"))

            # Execute
            lean_start = time.time()
            result = self.executor.execute(problem.statement, action.code)
            lean_time = time.time() - lean_start

            # Update state
            state.add_step(
                tactic=action.code,
                success=result.success,
                errors=result.errors,
                goals=result.goals,
            )

            # Record in trajectory
            step = trajectory.add_step(
                action=action.code,
                result=result,
                thinking=response,
            )
            step.metadata = {"llm_time": llm_time, "lean_time": lean_time}

            if callback:
                callback(step)

            if self.config.verbose:
                if result.complete:
                    console.print("[green]Proof complete![/green]")
                elif result.success:
                    console.print("[yellow]Accepted, goals remain[/yellow]")
                    if result.goals:
                        console.print(f"Goals: {result.goals}")
                else:
                    console.print(f"[red]Error: {result.errors}[/red]")

            if result.complete:
                break

            if self.config.stop_on_error and not result.success:
                break

            # Build next prompt
            prompt = self.prompt_builder.build_continuation(state)

        trajectory.total_time = time.time() - start_time
        trajectory.success = trajectory.is_complete

        if self.config.verbose:
            self._print_summary(trajectory)

        return trajectory

    def _print_summary(self, trajectory: Trajectory) -> None:
        console.print("\n" + "=" * 50)
        if trajectory.success:
            console.print(f"[green]Solved in {trajectory.num_steps} steps[/green]")
        else:
            console.print(f"[red]Failed after {trajectory.num_steps} steps[/red]")
        console.print(f"Time: {trajectory.total_time:.2f}s")
        console.print("=" * 50)

    def batch_solve(
        self,
        problems: list[LeanProblem],
        callback: Optional[Callable[[int, Trajectory], None]] = None,
    ) -> list[Trajectory]:
        """Solve multiple problems."""
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

    def __exit__(self, *_) -> None:
        self.stop()
