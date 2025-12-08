#!/usr/bin/env python3
"""
Evaluation script for Math-LLM.

Usage:
    python scripts/evaluate.py --config configs/default.yaml
    python scripts/evaluate.py --checkpoint outputs/checkpoint-1 --num-samples 50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Math-LLM agent")
    parser.add_argument("--config", "-c", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--output", "-o", type=str, default="eval_results.json")
    parser.add_argument("--num-samples", "-n", type=int, default=10)
    args = parser.parse_args()

    from math_llm.config import load_config
    from math_llm.data.loaders import create_dummy_dataset
    from math_llm.models.llm import LLMWrapper
    from math_llm.lean import LeanExecutor, LeanREPL
    from math_llm.agent import LeanAgent, Evaluator

    # Load config
    config = load_config(args.config)

    # Create dataset
    console.print("[blue]Creating evaluation dataset...[/blue]")
    dataset = create_dummy_dataset(args.num_samples)

    # Setup model (always real LLM)
    model_path = args.checkpoint or config.model.name
    console.print(f"[blue]Loading model: {model_path}[/blue]")
    model = LLMWrapper(
        model_name=model_path,
        device=config.model.device,
        torch_dtype=config.model.torch_dtype,
    )
    model.load()

    # Setup Lean executor (REPL for fast execution)
    console.print("[blue]Starting Lean REPL...[/blue]")
    lean_repl = LeanREPL(timeout=config.lean.timeout)
    executor = LeanExecutor(server=lean_repl)
    executor.start()

    try:
        agent = LeanAgent(
            model=model,
            executor=executor,
            config=config.agent,
        )

        console.print(f"\n[blue]Evaluating on {len(dataset)} problems...[/blue]\n")

        evaluator = Evaluator(agent, verbose=True)
        results = evaluator.evaluate(
            dataset,
            num_samples=args.num_samples,
            save_path=args.output,
        )

        console.print(f"\n[green]Results saved to {args.output}[/green]")

    finally:
        executor.stop()


if __name__ == "__main__":
    main()
