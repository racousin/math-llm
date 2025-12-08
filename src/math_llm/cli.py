"""
Command-line interface for Math-LLM.
"""

import sys
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


def train(config_path: Optional[str] = None):
    """Run training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Math-LLM agent")
    parser.add_argument("--config", "-c", type=str, default="configs/default.yaml")
    parser.add_argument("--output", "-o", type=str, default="outputs")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    from math_llm.config import load_config
    from math_llm.data.loaders import load_dataset, create_dummy_dataset
    from math_llm.models.llm import load_model
    from math_llm.lean import LeanExecutor
    from math_llm.agent import LeanAgent
    from math_llm.training import RLTrainer

    # Load config
    config = load_config(args.config)
    if args.debug:
        config.debug = True

    console.print(f"[blue]Loading config from {args.config}[/blue]")

    # Load model
    model = load_model(
        model_name=config.model.name,
        device=config.model.device,
        torch_dtype=config.model.torch_dtype,
        load_in_4bit=config.model.load_in_4bit,
    )

    # Load dataset
    if config.debug:
        train_dataset = create_dummy_dataset(10)
        val_dataset = create_dummy_dataset(5)
    else:
        train_dataset, val_dataset = load_dataset(
            sources=config.data.sources,
            cache_dir=config.data.cache_dir,
            tokenizer=model.get_tokenizer(),
            train_split=config.data.train_split,
        )

    console.print(f"[green]Loaded {len(train_dataset)} training problems[/green]")

    # Setup agent and trainer
    with LeanExecutor() as executor:
        agent = LeanAgent(
            model=model,
            executor=executor,
            config=config.agent,
        )

        trainer = RLTrainer(
            model=model.get_model_for_training(),
            tokenizer=model.get_tokenizer(),
            config=config.training,
            lean_executor=executor,
        )

        # Train
        trainer.train(
            agent=agent,
            train_problems=train_dataset.problems,
            val_problems=val_dataset.problems if val_dataset else None,
            num_epochs=config.training.num_epochs,
        )

    console.print("[green]Training complete![/green]")


def evaluate(config_path: Optional[str] = None):
    """Run evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Math-LLM agent")
    parser.add_argument("--config", "-c", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--output", "-o", type=str, default="eval_results.json")
    parser.add_argument("--num-samples", "-n", type=int, default=100)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    from math_llm.config import load_config
    from math_llm.data.loaders import load_dataset, create_dummy_dataset
    from math_llm.models.llm import load_model
    from math_llm.lean import LeanExecutor
    from math_llm.agent import LeanAgent, Evaluator

    # Load config
    config = load_config(args.config)

    # Load model
    model_path = args.checkpoint or config.model.name
    model = load_model(
        model_name=model_path,
        device=config.model.device,
        torch_dtype=config.model.torch_dtype,
    )

    # Load dataset
    if args.debug:
        dataset = create_dummy_dataset(args.num_samples)
    else:
        _, dataset = load_dataset(
            sources=config.data.sources,
            cache_dir=config.data.cache_dir,
            tokenizer=model.get_tokenizer(),
            train_split=0.0,  # Use all for eval
        )

    # Evaluate
    with LeanExecutor() as executor:
        agent = LeanAgent(
            model=model,
            executor=executor,
            config=config.agent,
        )
        agent.start()

        evaluator = Evaluator(agent)
        results = evaluator.evaluate(
            dataset,
            num_samples=args.num_samples,
            save_path=args.output,
        )

    console.print(f"[green]Results saved to {args.output}[/green]")


def download_data():
    """Download datasets."""
    import argparse

    parser = argparse.ArgumentParser(description="Download Math-LLM datasets")
    parser.add_argument(
        "--sources",
        "-s",
        nargs="+",
        default=["formal-conjectures", "miniF2F"],
        help="Sources to download",
    )
    parser.add_argument("--cache-dir", type=str, default=".cache/datasets")

    args = parser.parse_args()

    from math_llm.data.sources import download_all_sources

    download_all_sources(sources=args.sources, cache_dir=args.cache_dir)
    console.print("[green]Download complete![/green]")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        console.print("Usage: math-llm <command> [options]")
        console.print("Commands: train, evaluate, download")
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove command from args

    if command == "train":
        train()
    elif command in ("eval", "evaluate"):
        evaluate()
    elif command == "download":
        download_data()
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
