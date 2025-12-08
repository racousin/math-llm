"""
Data loaders for Lean theorem proving.
"""

from typing import Optional

import torch
from torch.utils.data import DataLoader

from math_llm.data.datasets import LeanDataset, LeanProblem
from math_llm.data.sources import load_all_sources


def collate_lean_batch(batch: list[dict]) -> dict:
    """Collate function for Lean dataset batches."""
    result = {
        "ids": [item["id"] for item in batch],
        "prompts": [item["prompt"] for item in batch],
        "statements": [item["statement"] for item in batch],
        "sources": [item["source"] for item in batch],
    }

    # Stack tensors if they exist
    if "input_ids" in batch[0]:
        result["input_ids"] = torch.stack([item["input_ids"] for item in batch])
        result["attention_mask"] = torch.stack([item["attention_mask"] for item in batch])

    # Include proofs if available
    if "proof" in batch[0]:
        result["proofs"] = [item.get("proof") for item in batch]

    return result


def create_dataloader(
    dataset: LeanDataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for the Lean dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_lean_batch,
        pin_memory=pin_memory,
    )


def load_dataset(
    sources: Optional[list[str]] = None,
    cache_dir: str = ".cache/datasets",
    tokenizer=None,
    max_length: int = 2048,
    train_split: float = 0.9,
    seed: int = 42,
) -> tuple[LeanDataset, LeanDataset]:
    """
    Load dataset from sources and split into train/val.

    Args:
        sources: List of source names to load from
        cache_dir: Directory for caching downloaded data
        tokenizer: Optional tokenizer for encoding
        max_length: Maximum sequence length
        train_split: Fraction of data for training
        seed: Random seed for splitting

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    problems = load_all_sources(sources=sources, cache_dir=cache_dir)

    dataset = LeanDataset(
        problems=problems,
        tokenizer=tokenizer,
        max_length=max_length,
        include_proof=False,
    )

    return dataset.split(train_ratio=train_split, seed=seed)


def create_dummy_dataset(num_problems: int = 10) -> LeanDataset:
    """Create a dummy dataset for testing."""
    problems = []

    dummy_theorems = [
        {
            "statement": "theorem add_comm (a b : Nat) : a + b = b + a := by\n  sorry",
            "description": "Addition is commutative",
        },
        {
            "statement": "theorem add_zero (a : Nat) : a + 0 = a := by\n  sorry",
            "description": "Adding zero is identity",
        },
        {
            "statement": "theorem zero_add (a : Nat) : 0 + a = a := by\n  sorry",
            "description": "Zero plus anything is identity",
        },
        {
            "statement": "theorem mul_comm (a b : Nat) : a * b = b * a := by\n  sorry",
            "description": "Multiplication is commutative",
        },
        {
            "statement": "theorem mul_one (a : Nat) : a * 1 = a := by\n  sorry",
            "description": "Multiplying by one is identity",
        },
        {
            "statement": "theorem add_assoc (a b c : Nat) : (a + b) + c = a + (b + c) := by\n  sorry",
            "description": "Addition is associative",
        },
        {
            "statement": "theorem mul_assoc (a b c : Nat) : (a * b) * c = a * (b * c) := by\n  sorry",
            "description": "Multiplication is associative",
        },
        {
            "statement": "theorem add_succ (a b : Nat) : a + Nat.succ b = Nat.succ (a + b) := by\n  sorry",
            "description": "Adding successor",
        },
        {
            "statement": "theorem succ_add (a b : Nat) : Nat.succ a + b = Nat.succ (a + b) := by\n  sorry",
            "description": "Successor of first argument",
        },
        {
            "statement": "theorem mul_zero (a : Nat) : a * 0 = 0 := by\n  sorry",
            "description": "Anything times zero is zero",
        },
    ]

    for i in range(num_problems):
        theorem = dummy_theorems[i % len(dummy_theorems)]
        problems.append(
            LeanProblem(
                id=f"dummy/{i}",
                statement=theorem["statement"],
                description=theorem["description"],
                source="dummy",
                difficulty="easy",
            )
        )

    return LeanDataset(problems)
