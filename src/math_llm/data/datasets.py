"""
Dataset classes for Lean theorem proving.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset


@dataclass
class LeanProblem:
    """A single Lean theorem proving problem."""

    # Unique identifier
    id: str

    # The theorem statement in Lean 4
    statement: str

    # Natural language description (if available)
    description: Optional[str] = None

    # Ground truth proof (if available)
    proof: Optional[str] = None

    # Source dataset
    source: str = "unknown"

    # Difficulty level (if available)
    difficulty: Optional[str] = None

    # Tags/categories
    tags: list[str] = field(default_factory=list)

    # Additional metadata
    metadata: dict = field(default_factory=dict)

    def to_prompt(self, include_description: bool = True) -> str:
        """Convert to a prompt for the LLM."""
        parts = []

        if include_description and self.description:
            parts.append(f"Problem: {self.description}")

        parts.append(f"Prove the following theorem in Lean 4:\n\n```lean4\n{self.statement}\n```")

        return "\n\n".join(parts)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "statement": self.statement,
            "description": self.description,
            "proof": self.proof,
            "source": self.source,
            "difficulty": self.difficulty,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LeanProblem":
        """Create from dictionary."""
        return cls(**data)


class LeanDataset(Dataset):
    """Dataset of Lean theorem proving problems."""

    def __init__(
        self,
        problems: list[LeanProblem],
        tokenizer=None,
        max_length: int = 2048,
        include_proof: bool = False,
    ):
        self.problems = problems
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_proof = include_proof

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, idx: int) -> dict:
        problem = self.problems[idx]

        item = {
            "id": problem.id,
            "prompt": problem.to_prompt(),
            "statement": problem.statement,
            "source": problem.source,
        }

        if self.include_proof and problem.proof:
            item["proof"] = problem.proof

        if self.tokenizer is not None:
            encoded = self.tokenizer(
                item["prompt"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            item["input_ids"] = encoded["input_ids"].squeeze(0)
            item["attention_mask"] = encoded["attention_mask"].squeeze(0)

        return item

    def filter_by_source(self, source: str) -> "LeanDataset":
        """Filter problems by source."""
        filtered = [p for p in self.problems if p.source == source]
        return LeanDataset(
            filtered,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            include_proof=self.include_proof,
        )

    def filter_by_difficulty(self, difficulty: str) -> "LeanDataset":
        """Filter problems by difficulty."""
        filtered = [p for p in self.problems if p.difficulty == difficulty]
        return LeanDataset(
            filtered,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            include_proof=self.include_proof,
        )

    def split(self, train_ratio: float = 0.9, seed: int = 42) -> tuple["LeanDataset", "LeanDataset"]:
        """Split dataset into train and validation sets."""
        import random

        random.seed(seed)
        indices = list(range(len(self.problems)))
        random.shuffle(indices)

        split_idx = int(len(indices) * train_ratio)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_problems = [self.problems[i] for i in train_indices]
        val_problems = [self.problems[i] for i in val_indices]

        train_dataset = LeanDataset(
            train_problems,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            include_proof=self.include_proof,
        )
        val_dataset = LeanDataset(
            val_problems,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            include_proof=self.include_proof,
        )

        return train_dataset, val_dataset

    def __iter__(self) -> Iterator[dict]:
        for i in range(len(self)):
            yield self[i]

    @property
    def sources(self) -> list[str]:
        """Get unique sources in dataset."""
        return list(set(p.source for p in self.problems))

    def save(self, path: str | Path) -> None:
        """Save dataset to disk."""
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [p.to_dict() for p in self.problems]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "LeanDataset":
        """Load dataset from disk."""
        import json

        with open(path) as f:
            data = json.load(f)

        problems = [LeanProblem.from_dict(d) for d in data]
        return cls(problems, **kwargs)
