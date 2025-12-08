#!/usr/bin/env python3
"""
Evaluate model on all Lean 4 benchmarks.

Usage:
    python scripts/eval_all.py
    python scripts/eval_all.py --checkpoint outputs/checkpoint-1
    python scripts/eval_all.py --benchmarks minif2f-lean4 fimo
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table

console = Console()


BENCHMARKS = {
    "minif2f-lean4": {"description": "Competition math (IMO, AMC)", "size": "~488"},
    "fimo": {"description": "Formal IMO problems", "size": "~149"},
    "putnambench": {"description": "Putnam competition", "size": "~1.7k"},
    "proofnet": {"description": "Undergraduate math", "size": "~370"},
}


def evaluate_benchmark(
    benchmark: str,
    model,
    executor,
    config,
    max_samples: int = None,
    output_dir: Path = None,
) -> dict:
    """Evaluate on a single benchmark."""
    from math_llm.data import load_sources, LeanDataset
    from math_llm.agent import LeanAgent, Evaluator

    console.print(f"\n[blue]{'='*60}[/blue]")
    console.print(f"[blue]Evaluating on {benchmark}[/blue]")
    console.print(f"[blue]{'='*60}[/blue]")

    # Load data
    problems = load_sources([benchmark])
    if max_samples and max_samples < len(problems):
        import random
        problems = random.sample(problems, max_samples)

    dataset = LeanDataset(problems)
    console.print(f"[green]Loaded {len(dataset)} problems[/green]")

    # Create agent
    agent = LeanAgent(model=model, executor=executor, config=config.agent)

    # Evaluate
    evaluator = Evaluator(agent, verbose=True)

    save_path = None
    if output_dir:
        save_path = str(output_dir / f"{benchmark}_results.json")

    results = evaluator.evaluate(dataset, save_path=save_path)

    return {
        "benchmark": benchmark,
        "num_problems": results.num_problems,
        "num_solved": results.num_solved,
        "success_rate": results.success_rate,
        "avg_steps": results.avg_steps,
        "avg_time": results.avg_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate on all Lean 4 benchmarks")
    parser.add_argument("--config", "-c", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint path")
    parser.add_argument("--benchmarks", "-b", nargs="+", default=list(BENCHMARKS.keys()),
                        help="Benchmarks to evaluate on")
    parser.add_argument("--max-samples", "-n", type=int, default=None,
                        help="Max samples per benchmark")
    parser.add_argument("--output", "-o", type=str, default="outputs/eval",
                        help="Output directory")
    args = parser.parse_args()

    from math_llm.config import load_config
    from math_llm.models import load_model
    from math_llm.lean import LeanExecutor

    # Load config
    config = load_config(args.config)

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model (always real LLM)
    model_path = args.checkpoint or config.model.name
    console.print(f"[blue]Loading model: {model_path}[/blue]")
    model = load_model(
        model_name=model_path,
        device=config.model.device,
        torch_dtype=config.model.torch_dtype,
    )

    # Run evaluations (always real Lean)
    all_results = []

    with LeanExecutor() as executor:
        for benchmark in args.benchmarks:
            if benchmark not in BENCHMARKS:
                console.print(f"[red]Unknown benchmark: {benchmark}[/red]")
                continue

            try:
                result = evaluate_benchmark(
                    benchmark=benchmark,
                    model=model,
                    executor=executor,
                    config=config,
                    max_samples=args.max_samples,
                    output_dir=output_dir,
                )
                all_results.append(result)
            except Exception as e:
                console.print(f"[red]Error evaluating {benchmark}: {e}[/red]")
                all_results.append({
                    "benchmark": benchmark,
                    "error": str(e),
                })

    # Print summary
    console.print("\n" + "=" * 60)
    console.print("[bold]EVALUATION SUMMARY[/bold]")
    console.print("=" * 60)

    table = Table(title="Results by Benchmark")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Problems", style="white")
    table.add_column("Solved", style="green")
    table.add_column("Rate", style="yellow")
    table.add_column("Avg Steps", style="white")

    total_problems = 0
    total_solved = 0

    for result in all_results:
        if "error" in result:
            table.add_row(result["benchmark"], "ERROR", "-", "-", "-")
        else:
            table.add_row(
                result["benchmark"],
                str(result["num_problems"]),
                str(result["num_solved"]),
                f"{result['success_rate']:.1%}",
                f"{result['avg_steps']:.2f}",
            )
            total_problems += result["num_problems"]
            total_solved += result["num_solved"]

    console.print(table)

    # Overall stats
    if total_problems > 0:
        console.print(f"\n[bold]Overall: {total_solved}/{total_problems} solved ({total_solved/total_problems:.1%})[/bold]")

    # Save combined results
    summary_path = output_dir / "summary.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.checkpoint or config.model.name,
        "benchmarks": all_results,
        "total_problems": total_problems,
        "total_solved": total_solved,
        "overall_rate": total_solved / total_problems if total_problems > 0 else 0,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    console.print(f"\n[green]Results saved to {output_dir}/[/green]")


if __name__ == "__main__":
    main()
