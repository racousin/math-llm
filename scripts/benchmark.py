#!/usr/bin/env python3
"""
Benchmark script for evaluating math-llm on theorem proving datasets.

Usage:
    # Quick test (3 problems)
    python scripts/benchmark.py --quick

    # Small benchmark (20 problems)
    python scripts/benchmark.py --num-samples 20

    # Full benchmark
    python scripts/benchmark.py --dataset minif2f-lean4

    # Custom config
    python scripts/benchmark.py --dataset proofnet --num-samples 50 --max-steps 5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    parser = argparse.ArgumentParser(description="Benchmark math-llm on theorem proving")
    parser.add_argument("--dataset", type=str, default="minif2f-lean4",
                        choices=["minif2f-lean4", "proofnet", "fimo", "putnambench", "dummy"],
                        help="Dataset to benchmark on")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of problems to evaluate (default: all)")
    parser.add_argument("--max-steps", type=int, default=3,
                        help="Max proof attempts per problem")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Generation temperature")
    # parser.add_argument("--model", type=str, default= "Qwen/Qwen2.5-7B-Instruct", #"Qwen/Qwen2.5-7B-Instruct"
    #                     help="Model to use")
    parser.add_argument("--model", type=str, default= "Qwen/Qwen2.5-7B-Instruct", #"Qwen/Qwen2.5-7B-Instruct"
                        help="Model to use")
    parser.add_argument("--device", type=str, default="mps",
                        help="Device (mps, cuda, cpu)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with 3 problems")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed output")

    args = parser.parse_args()

    # Quick mode
    if args.quick:
        args.num_samples = 3
        args.verbose = True

    # Import after parsing to show help faster
    from rich.console import Console
    from rich.table import Table

    console = Console()

    console.print(f"[bold blue]Math-LLM Benchmark[/bold blue]")
    console.print(f"Dataset: {args.dataset}")
    console.print(f"Samples: {args.num_samples or 'all'}")
    console.print(f"Max steps: {args.max_steps}")
    console.print(f"Model: {args.model}")
    console.print(f"Lean mode: REPL (fast)")
    console.print()

    # Load dataset
    console.print("[yellow]Loading dataset...[/yellow]")

    if args.dataset == "dummy":
        from math_llm.data.loaders import create_dummy_dataset
        dataset = create_dummy_dataset(args.num_samples or 10)
        problems = dataset.problems
    else:
        from math_llm.data import load_sources, LeanDataset
        problems = load_sources([args.dataset])

    console.print(f"[green]Loaded {len(problems)} problems[/green]")

    # Sample if needed
    if args.num_samples and args.num_samples < len(problems):
        problems = problems[:args.num_samples]
        console.print(f"[yellow]Using first {len(problems)} problems[/yellow]")

    # Load model
    console.print(f"\n[yellow]Loading model: {args.model}...[/yellow]")
    from math_llm.models import load_model

    model = load_model(
        args.model,
        device=args.device,
        torch_dtype="float16",
    )
    console.print("[green]Model loaded[/green]")

    # Setup agent
    from math_llm.config import AgentConfig
    from math_llm.lean import LeanExecutor
    from math_llm.agent import LeanAgent

    config = AgentConfig(
        max_steps=args.max_steps,
        temperature=args.temperature,
        verbose=args.verbose,
    )

    # Run benchmark with timing
    console.print(f"\n[bold]Starting benchmark on {len(problems)} problems...[/bold]\n")

    results = []
    solved = 0
    total_time = 0
    total_llm_time = 0
    total_lean_time = 0

    with LeanExecutor() as executor:
        agent = LeanAgent(model=model, executor=executor, config=config)

        for i, problem in enumerate(problems):
            start = time.time()

            # Show progress
            console.print(f"[{i+1}/{len(problems)}] {problem.id[:50]}...")

            try:
                # Solve with timing (we'll extract from trajectory metadata)
                trajectory = agent.solve(problem)
                elapsed = time.time() - start
                total_time += elapsed

                # Extract timing from steps if available
                llm_time = 0
                lean_time = 0
                for step in trajectory.steps:
                    if hasattr(step, 'metadata') and step.metadata:
                        llm_time += step.metadata.get('llm_time', 0)
                        lean_time += step.metadata.get('lean_time', 0)

                # Estimate if not tracked
                if llm_time == 0 and lean_time == 0:
                    # Rough estimate: 70% LLM, 30% Lean
                    llm_time = elapsed * 0.7
                    lean_time = elapsed * 0.3

                total_llm_time += llm_time
                total_lean_time += lean_time

                if trajectory.success:
                    solved += 1
                    status = f"[green]✓ SOLVED[/green]"
                else:
                    last_action = trajectory.final_action or ""
                    if "sorry" in last_action.lower():
                        status = f"[red]✗ SORRY[/red]"
                    else:
                        status = f"[red]✗ FAILED[/red]"

                console.print(f"  {status} | steps: {trajectory.num_steps} | time: {elapsed:.1f}s (llm: ~{llm_time:.1f}s, lean: ~{lean_time:.1f}s)")

                results.append({
                    "problem_id": problem.id,
                    "solved": trajectory.success,
                    "steps": trajectory.num_steps,
                    "time": elapsed,
                    "llm_time": llm_time,
                    "lean_time": lean_time,
                    "final_action": trajectory.final_action,
                })

            except Exception as e:
                elapsed = time.time() - start
                total_time += elapsed
                console.print(f"  [red]✗ ERROR: {str(e)[:50]}[/red]")
                results.append({
                    "problem_id": problem.id,
                    "solved": False,
                    "steps": 0,
                    "time": elapsed,
                    "error": str(e),
                })

    # Summary
    console.print("\n" + "=" * 60)

    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    success_rate = solved / len(problems) if problems else 0
    avg_time = total_time / len(problems) if problems else 0
    avg_llm = total_llm_time / len(problems) if problems else 0
    avg_lean = total_lean_time / len(problems) if problems else 0

    table.add_row("Problems", str(len(problems)))
    table.add_row("Solved", str(solved))
    table.add_row("Success Rate", f"{success_rate:.1%}")
    table.add_row("─" * 20, "─" * 10)
    table.add_row("Total Time", f"{total_time:.1f}s")
    table.add_row("Avg Time/Problem", f"{avg_time:.1f}s")
    table.add_row("─" * 20, "─" * 10)
    table.add_row("LLM Time (est)", f"{total_llm_time:.1f}s ({100*total_llm_time/total_time:.0f}%)" if total_time else "0s")
    table.add_row("Lean Time (est)", f"{total_lean_time:.1f}s ({100*total_lean_time/total_time:.0f}%)" if total_time else "0s")

    console.print(table)

    # Save results
    output_path = args.output or f"outputs/eval/{args.dataset}_benchmark.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "config": {
            "dataset": args.dataset,
            "num_samples": len(problems),
            "max_steps": args.max_steps,
            "model": args.model,
            "temperature": args.temperature,
        },
        "summary": {
            "total": len(problems),
            "solved": solved,
            "success_rate": success_rate,
            "total_time": total_time,
            "avg_time": avg_time,
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    console.print(f"\n[blue]Results saved to {output_path}[/blue]")

    return 0 if success_rate > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
