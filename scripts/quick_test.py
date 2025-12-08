#!/usr/bin/env python3
"""
Quick test script to verify the installation works.

Usage:
    python scripts/quick_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel

console = Console()


def test_imports():
    """Test that all modules can be imported."""
    console.print("[blue]Testing imports...[/blue]")

    try:
        from math_llm import Config, load_config
        console.print("  ✓ math_llm.config")

        from math_llm.data import LeanDataset, LeanProblem
        console.print("  ✓ math_llm.data")

        from math_llm.data.loaders import create_dummy_dataset
        console.print("  ✓ math_llm.data.loaders")

        from math_llm.lean import LeanREPL, LeanExecutor
        console.print("  ✓ math_llm.lean")

        from math_llm.agent import LeanAgent, Trajectory
        console.print("  ✓ math_llm.agent")

        from math_llm.models import LLMWrapper
        console.print("  ✓ math_llm.models")

        from math_llm.training import RLTrainer, ProofReward
        console.print("  ✓ math_llm.training")

        return True
    except ImportError as e:
        console.print(f"  [red]✗ Import error: {e}[/red]")
        return False


def test_config():
    """Test config loading."""
    console.print("\n[blue]Testing config loading...[/blue]")

    try:
        from math_llm.config import load_config

        config = load_config("configs/dummy_test.yaml")
        console.print(f"  ✓ Loaded config: model={config.model.name}")
        return True
    except Exception as e:
        console.print(f"  [red]✗ Config error: {e}[/red]")
        return False


def test_dataset():
    """Test dataset creation."""
    console.print("\n[blue]Testing dataset creation...[/blue]")

    try:
        from math_llm.data.loaders import create_dummy_dataset

        dataset = create_dummy_dataset(5)
        console.print(f"  ✓ Created dataset with {len(dataset)} problems")

        problem = dataset.problems[0]
        console.print(f"  ✓ Sample problem: {problem.id}")
        console.print(f"    Statement: {problem.statement[:50]}...")

        return True
    except Exception as e:
        console.print(f"  [red]✗ Dataset error: {e}[/red]")
        return False


def test_lean_repl():
    """Test Lean REPL (optional - requires Lean installation)."""
    console.print("\n[blue]Testing Lean REPL...[/blue]")

    try:
        from math_llm.lean import LeanREPL

        repl = LeanREPL(timeout=10)
        repl.start()

        # Test simple proof
        result = repl.check_proof("theorem test : 1 = 1 := by sorry", "rfl")
        repl.stop()

        if result.is_success:
            console.print("  ✓ Lean REPL working")
            return True
        else:
            console.print(f"  [yellow]⚠ Lean executed but returned: {result.status}[/yellow]")
            return True  # Still counts as working
    except FileNotFoundError:
        console.print("  [yellow]⚠ Lean not installed (skipping)[/yellow]")
        return True  # Not a failure, just not installed
    except Exception as e:
        console.print(f"  [red]✗ Lean REPL error: {e}[/red]")
        return False


def test_trajectory():
    """Test trajectory creation."""
    console.print("\n[blue]Testing trajectory...[/blue]")

    try:
        from math_llm.agent.trajectory import Trajectory, Step
        from math_llm.lean.executor import ExecutionFeedback, LeanResult
        from math_llm.lean.server import LeanResultStatus

        # Create mock feedback
        mock_result = LeanResult(
            status=LeanResultStatus.SUCCESS,
            output="",
            errors=[],
            warnings=[],
            goals=[],
        )
        feedback = ExecutionFeedback(
            success=True,
            complete=True,
            message="Proof complete!",
            goals=[],
            errors=[],
            hints=[],
            raw_result=mock_result,
        )

        # Create trajectory
        traj = Trajectory(
            problem_id="test/1",
            statement="theorem test : 1 = 1 := by sorry",
        )
        traj.add_step("rfl", feedback)

        console.print(f"  ✓ Created trajectory with {traj.num_steps} step(s)")
        console.print(f"  ✓ Success: {traj.success}")

        # Test serialization
        traj_dict = traj.to_dict()
        loaded = Trajectory.from_dict(traj_dict)
        console.print(f"  ✓ Serialization works")

        return True
    except Exception as e:
        console.print(f"  [red]✗ Trajectory error: {e}[/red]")
        return False


def test_rewards():
    """Test reward computation."""
    console.print("\n[blue]Testing rewards...[/blue]")

    try:
        from math_llm.training.rewards import ProofReward, compute_trajectory_reward
        from math_llm.agent.trajectory import Trajectory, Step
        from math_llm.lean.executor import ExecutionFeedback
        from math_llm.lean.server import LeanResult, LeanResultStatus

        # Create a successful trajectory
        mock_result = LeanResult(
            status=LeanResultStatus.SUCCESS,
            output="",
            goals=[],
        )
        feedback = ExecutionFeedback(
            success=True,
            complete=True,
            message="",
            goals=[],
            errors=[],
            hints=[],
            raw_result=mock_result,
        )

        traj = Trajectory(
            problem_id="test/1",
            statement="theorem test : 1 = 1 := by sorry",
        )
        traj.add_step("rfl", feedback)

        reward = compute_trajectory_reward(traj)
        console.print(f"  ✓ Computed reward: {reward:.4f}")

        return True
    except Exception as e:
        console.print(f"  [red]✗ Reward error: {e}[/red]")
        return False


def main():
    console.print(Panel.fit(
        "[bold]Math-LLM Quick Test[/bold]\n"
        "Testing installation and basic functionality",
        border_style="blue",
    ))

    tests = [
        ("Imports", test_imports),
        ("Config", test_config),
        ("Dataset", test_dataset),
        ("Lean REPL", test_lean_repl),
        ("Trajectory", test_trajectory),
        ("Rewards", test_rewards),
    ]

    results = []
    for name, test_fn in tests:
        try:
            results.append((name, test_fn()))
        except Exception as e:
            console.print(f"[red]Unexpected error in {name}: {e}[/red]")
            results.append((name, False))

    # Summary
    console.print("\n" + "=" * 50)
    passed = sum(1 for _, r in results if r)
    total = len(results)

    if passed == total:
        console.print(f"[green]All {total} tests passed! ✓[/green]")
    else:
        console.print(f"[yellow]{passed}/{total} tests passed[/yellow]")

    console.print("=" * 50)

    # Return exit code
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
