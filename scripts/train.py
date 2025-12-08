#!/usr/bin/env python3
"""
Training script for Math-LLM.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/dummy_test.yaml --debug
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Train Math-LLM agent")
    parser.add_argument("--config", "-c", type=str, default="configs/default.yaml")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="Use dummy data for testing")
    args = parser.parse_args()

    from math_llm.config import load_config
    from math_llm.data.loaders import create_dummy_dataset
    from math_llm.models.llm import LLMWrapper
    from math_llm.lean import LeanExecutor, LeanREPL
    from math_llm.agent import LeanAgent
    from math_llm.training import RLTrainer

    # Load config
    console.print(f"[blue]Loading config from {args.config}[/blue]")
    config = load_config(args.config)

    if args.debug:
        config.debug = True

    if args.output:
        config.training.output_dir = args.output

    # Create dummy dataset for testing
    console.print("[blue]Creating dummy dataset for testing...[/blue]")
    train_dataset = create_dummy_dataset(config.data.max_samples or 10)
    val_dataset = create_dummy_dataset(5)

    console.print(f"[green]Dataset: {len(train_dataset)} train, {len(val_dataset)} val[/green]")

    # Setup model
    console.print(f"[blue]Loading model: {config.model.name}[/blue]")
    model = LLMWrapper(
        model_name=config.model.name,
        device=config.model.device,
        torch_dtype=config.model.torch_dtype,
        load_in_4bit=config.model.load_in_4bit,
        max_length=config.model.max_length,
    )
    model.load()

    # Setup Lean executor (REPL mode for fast execution)
    lean_repl = LeanREPL(timeout=config.lean.timeout)
    executor = LeanExecutor(server=lean_repl)
    executor.start()

    try:
        # Create agent
        agent = LeanAgent(
            model=model,
            executor=executor,
            config=config.agent,
        )

        # Collect some trajectories
        console.print("\n[blue]Collecting trajectories...[/blue]")
        trajectories = []
        for i, problem in enumerate(train_dataset.problems[:5]):
            console.print(f"\nProblem {i+1}/5: {problem.id}")
            try:
                trajectory = agent.solve(problem)
                trajectories.append(trajectory)
                console.print(
                    f"  Result: {'✓ Solved' if trajectory.success else '✗ Failed'} "
                    f"in {trajectory.num_steps} steps"
                )
            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")

        # Print summary
        success_rate = sum(1 for t in trajectories if t.success) / len(trajectories)
        console.print(f"\n[blue]Summary: {success_rate:.1%} success rate[/blue]")

        # Save trajectories
        output_dir = Path(config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, traj in enumerate(trajectories):
            traj.save(output_dir / f"trajectory_{i}.json")

        console.print(f"[green]Trajectories saved to {output_dir}[/green]")

    finally:
        executor.stop()

    console.print("\n[green]Training script complete![/green]")


if __name__ == "__main__":
    main()
