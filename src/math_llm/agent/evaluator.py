"""
Evaluation framework for Lean theorem proving agents.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from math_llm.data.datasets import LeanDataset, LeanProblem
from math_llm.agent.agent import LeanAgent
from math_llm.agent.trajectory import Trajectory, TrajectoryBatch

console = Console()


@dataclass
class EvaluationResult:
    """Results of evaluating an agent on a dataset."""

    # Basic metrics
    num_problems: int
    num_solved: int
    success_rate: float

    # Step metrics
    avg_steps: float
    avg_steps_solved: float
    avg_steps_failed: float

    # Time metrics
    total_time: float
    avg_time: float
    avg_time_solved: float
    avg_time_failed: float

    # Per-problem results
    trajectories: list[Trajectory] = field(default_factory=list)

    # Breakdown by source/difficulty
    by_source: dict = field(default_factory=dict)
    by_difficulty: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "num_problems": self.num_problems,
            "num_solved": self.num_solved,
            "success_rate": self.success_rate,
            "avg_steps": self.avg_steps,
            "avg_steps_solved": self.avg_steps_solved,
            "avg_steps_failed": self.avg_steps_failed,
            "total_time": self.total_time,
            "avg_time": self.avg_time,
            "avg_time_solved": self.avg_time_solved,
            "avg_time_failed": self.avg_time_failed,
            "by_source": self.by_source,
            "by_difficulty": self.by_difficulty,
        }

    def save(self, path: str, include_trajectories: bool = True) -> None:
        """Save evaluation results to file."""
        # Ensure directory exists
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()
        if include_trajectories:
            data["trajectories"] = [t.to_dict() for t in self.trajectories]

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def print_summary(self) -> None:
        """Print a summary table of results."""
        table = Table(title="Evaluation Results")

        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Problems", str(self.num_problems))
        table.add_row("Solved", str(self.num_solved))
        table.add_row("Success Rate", f"{self.success_rate:.1%}")
        table.add_row("Avg Steps", f"{self.avg_steps:.2f}")
        table.add_row("Avg Steps (Solved)", f"{self.avg_steps_solved:.2f}")
        table.add_row("Avg Time", f"{self.avg_time:.2f}s")
        table.add_row("Total Time", f"{self.total_time:.2f}s")

        console.print(table)

        # Print breakdown by source if available
        if self.by_source:
            source_table = Table(title="Results by Source")
            source_table.add_column("Source", style="cyan")
            source_table.add_column("Problems", style="white")
            source_table.add_column("Solved", style="green")
            source_table.add_column("Rate", style="yellow")

            for source, stats in self.by_source.items():
                source_table.add_row(
                    source,
                    str(stats["total"]),
                    str(stats["solved"]),
                    f"{stats['rate']:.1%}",
                )

            console.print(source_table)


class Evaluator:
    """Evaluates an agent on a dataset of Lean problems."""

    def __init__(
        self,
        agent: LeanAgent,
        verbose: bool = True,
    ):
        self.agent = agent
        self.verbose = verbose

    def evaluate(
        self,
        dataset: LeanDataset,
        num_samples: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate the agent on a dataset.

        Args:
            dataset: Dataset to evaluate on
            num_samples: Maximum number of samples to evaluate
            save_path: Optional path to save results

        Returns:
            EvaluationResult with metrics and trajectories
        """
        # Sample problems
        problems = [dataset.problems[i] for i in range(len(dataset))]
        if num_samples and num_samples < len(problems):
            import random
            problems = random.sample(problems, num_samples)

        trajectories = []
        start_time = time.time()

        # Progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
            disable=not self.verbose,
        ) as progress:
            task = progress.add_task("Evaluating...", total=len(problems))

            for i, problem in enumerate(problems):
                trajectory = self.agent.solve(problem)
                trajectories.append(trajectory)

                progress.update(task, advance=1)

                # Update description with current success rate
                solved = sum(1 for t in trajectories if t.success)
                rate = solved / len(trajectories)
                progress.update(
                    task,
                    description=f"Evaluating... ({solved}/{len(trajectories)} solved, {rate:.1%})",
                )

        total_time = time.time() - start_time

        # Compute metrics
        result = self._compute_metrics(trajectories, total_time)

        if save_path:
            result.save(save_path)

        if self.verbose:
            result.print_summary()

        return result

    def _compute_metrics(
        self,
        trajectories: list[Trajectory],
        total_time: float,
    ) -> EvaluationResult:
        """Compute evaluation metrics from trajectories."""
        num_problems = len(trajectories)
        num_solved = sum(1 for t in trajectories if t.success)

        solved_trajectories = [t for t in trajectories if t.success]
        failed_trajectories = [t for t in trajectories if not t.success]

        # Step metrics
        avg_steps = sum(t.num_steps for t in trajectories) / num_problems if num_problems else 0
        avg_steps_solved = (
            sum(t.num_steps for t in solved_trajectories) / len(solved_trajectories)
            if solved_trajectories
            else 0
        )
        avg_steps_failed = (
            sum(t.num_steps for t in failed_trajectories) / len(failed_trajectories)
            if failed_trajectories
            else 0
        )

        # Time metrics
        avg_time = sum(t.total_time for t in trajectories) / num_problems if num_problems else 0
        avg_time_solved = (
            sum(t.total_time for t in solved_trajectories) / len(solved_trajectories)
            if solved_trajectories
            else 0
        )
        avg_time_failed = (
            sum(t.total_time for t in failed_trajectories) / len(failed_trajectories)
            if failed_trajectories
            else 0
        )

        # Breakdown by source
        by_source = {}
        for t in trajectories:
            source = t.metadata.get("source", "unknown")
            # Try to extract source from problem_id
            if "/" in t.problem_id:
                source = t.problem_id.split("/")[0]

            if source not in by_source:
                by_source[source] = {"total": 0, "solved": 0}
            by_source[source]["total"] += 1
            if t.success:
                by_source[source]["solved"] += 1

        for source in by_source:
            stats = by_source[source]
            stats["rate"] = stats["solved"] / stats["total"] if stats["total"] else 0

        return EvaluationResult(
            num_problems=num_problems,
            num_solved=num_solved,
            success_rate=num_solved / num_problems if num_problems else 0,
            avg_steps=avg_steps,
            avg_steps_solved=avg_steps_solved,
            avg_steps_failed=avg_steps_failed,
            total_time=total_time,
            avg_time=avg_time,
            avg_time_solved=avg_time_solved,
            avg_time_failed=avg_time_failed,
            trajectories=trajectories,
            by_source=by_source,
            by_difficulty={},  # Can be populated if problems have difficulty
        )


def quick_evaluate(
    agent: LeanAgent,
    problems: list[LeanProblem],
    verbose: bool = True,
) -> EvaluationResult:
    """Convenience function for quick evaluation."""
    dataset = LeanDataset(problems)
    evaluator = Evaluator(agent, verbose=verbose)
    return evaluator.evaluate(dataset)
