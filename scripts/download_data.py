#!/usr/bin/env python3
"""
Download datasets for Math-LLM.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --sources formal-conjectures miniF2F
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Download Math-LLM datasets")
    parser.add_argument(
        "--sources",
        "-s",
        nargs="+",
        default=["formal-conjectures", "miniF2F"],
        help="Sources to download (formal-conjectures, mathlib, miniF2F, proofnet)",
    )
    parser.add_argument("--cache-dir", type=str, default=".cache/datasets")
    parser.add_argument("--load", action="store_true", help="Also load and parse the data")
    args = parser.parse_args()

    from math_llm.data.sources import download_all_sources, load_all_sources

    console.print(f"[blue]Downloading sources: {', '.join(args.sources)}[/blue]")
    console.print(f"[blue]Cache directory: {args.cache_dir}[/blue]\n")

    download_all_sources(sources=args.sources, cache_dir=args.cache_dir)

    if args.load:
        console.print("\n[blue]Loading and parsing datasets...[/blue]")
        problems = load_all_sources(sources=args.sources, cache_dir=args.cache_dir)
        console.print(f"[green]Total problems loaded: {len(problems)}[/green]")

        # Show breakdown by source
        from collections import Counter
        sources = Counter(p.source for p in problems)
        console.print("\nBreakdown by source:")
        for source, count in sources.most_common():
            console.print(f"  {source}: {count}")

    console.print("\n[green]Download complete![/green]")


if __name__ == "__main__":
    main()
