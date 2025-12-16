"""
Data loading for Lean proof benchmarks.

Supports:
- dummy: Simple test problems for verification
- minif2f-lean4: Competition math problems from HuggingFace
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Problem:
    """A Lean theorem proving problem."""
    id: str
    statement: str  # Lean 4 statement with 'sorry' placeholder
    description: Optional[str] = None  # Natural language description
    proof: Optional[str] = None  # Ground truth proof (if available)
    source: str = "unknown"
    difficulty: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "description": self.description,
            "proof": self.proof,
            "source": self.source,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
        }


# =============================================================================
# DUMMY DATA - Simple problems for testing
# =============================================================================

DUMMY_PROBLEMS = [
    Problem(
        id="dummy/add_one",
        statement="theorem add_one : 1 + 1 = 2 := by sorry",
        description="Prove that 1 + 1 = 2",
        proof="norm_num",
        source="dummy",
        difficulty="trivial",
    ),
    Problem(
        id="dummy/mul_comm",
        statement="theorem mul_comm_example : 2 * 3 = 3 * 2 := by sorry",
        description="Prove multiplication is commutative for 2 and 3",
        proof="ring",
        source="dummy",
        difficulty="trivial",
    ),
    Problem(
        id="dummy/nat_pos",
        statement="theorem nat_pos : 0 < 1 := by sorry",
        description="Prove that 0 is less than 1",
        proof="norm_num",
        source="dummy",
        difficulty="trivial",
    ),
    Problem(
        id="dummy/add_zero",
        statement="theorem add_zero_example (n : Nat) : n + 0 = n := by sorry",
        description="Prove that adding zero doesn't change a number",
        proof="rfl",
        source="dummy",
        difficulty="easy",
    ),
    Problem(
        id="dummy/neg_neg",
        statement="theorem neg_neg_example (x : Int) : -(-x) = x := by sorry",
        description="Prove that double negation returns the original integer",
        proof="ring",
        source="dummy",
        difficulty="easy",
    ),
    Problem(
        id="dummy/square_nonneg",
        statement="theorem square_nonneg (x : Real) : 0 ≤ x^2 := by sorry",
        description="Prove that squares are non-negative",
        proof="apply sq_nonneg",
        source="dummy",
        difficulty="easy",
    ),
    Problem(
        id="dummy/abs_nonneg",
        statement="theorem abs_nonneg_example (x : Real) : 0 ≤ |x| := by sorry",
        description="Prove that absolute value is non-negative",
        proof="exact abs_nonneg x",
        source="dummy",
        difficulty="easy",
    ),
    Problem(
        id="dummy/sum_first_n",
        statement="theorem sum_first_three : (0 : Nat) + 1 + 2 = 3 := by sorry",
        description="Prove that 0 + 1 + 2 = 3",
        proof="norm_num",
        source="dummy",
        difficulty="trivial",
    ),
    Problem(
        id="dummy/power_one",
        statement="theorem power_one (n : Nat) : n ^ 1 = n := by sorry",
        description="Prove that any number to the power of 1 is itself",
        proof="ring",
        source="dummy",
        difficulty="easy",
    ),
    Problem(
        id="dummy/iff_intro",
        statement="theorem iff_intro : (1 = 1) ↔ (2 = 2) := by sorry",
        description="Prove a simple iff statement",
        proof="constructor <;> intro <;> rfl",
        source="dummy",
        difficulty="easy",
    ),
]


def load_dummy(n_samples: Optional[int] = None) -> list[Problem]:
    """Load dummy problems for testing."""
    problems = DUMMY_PROBLEMS.copy()
    if n_samples is not None:
        problems = problems[:n_samples]
    return problems


# =============================================================================
# MINIF2F-LEAN4 - Competition math problems
# =============================================================================

def normalize_lean4_syntax(statement: str) -> str:
    """Normalize Lean syntax for Mathlib 4 compatibility."""
    # Convert "∑ k in S" to "∑ k ∈ S" (BigOperators notation)
    statement = re.sub(r'(∑\s*\w+)\s+in\s+', r'\1 ∈ ', statement)
    statement = re.sub(r'(∏\s*\w+)\s+in\s+', r'\1 ∈ ', statement)
    return statement


def load_minif2f(n_samples: Optional[int] = None) -> list[Problem]:
    """
    Load MiniF2F Lean 4 problems from HuggingFace.

    Dataset: cat-searcher/minif2f-lean4
    ~488 competition math problems (IMO, AMC, AIME, etc.)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    print("[data] Loading minif2f-lean4 from HuggingFace...")
    ds = load_dataset("cat-searcher/minif2f-lean4", trust_remote_code=True)

    # Combine all splits
    all_items = []
    if hasattr(ds, 'keys'):
        for split_name in ds.keys():
            all_items.extend(ds[split_name])
    else:
        all_items = list(ds)

    problems = []
    for item in all_items:
        statement = item.get("formal_statement", item.get("statement", ""))
        name = item.get("name", item.get("problem_name", ""))

        # Normalize syntax
        statement = normalize_lean4_syntax(statement)

        # Get informal statement
        informal = item.get("informal_statement", item.get("informal_stmt", ""))

        # Determine tags from name
        tags = []
        if "imo" in name.lower():
            tags.append("IMO")
        elif "amc" in name.lower():
            tags.append("AMC")
        elif "aime" in name.lower():
            tags.append("AIME")

        problem = Problem(
            id=f"minif2f-lean4/{name}",
            statement=statement,
            description=informal,
            proof=item.get("proof"),
            source="minif2f-lean4",
            difficulty="competition",
            metadata={
                "tags": tags,
                "competition": item.get("source", ""),
                "year": item.get("year"),
            },
        )
        problems.append(problem)

    print(f"[data] Loaded {len(problems)} problems from minif2f-lean4")

    if n_samples is not None:
        problems = problems[:n_samples]

    return problems


# =============================================================================
# DATA LOADING API
# =============================================================================

DATASETS = {
    "dummy": load_dummy,
    "minif2f-lean4": load_minif2f,
}


def load_data(dataset: str, n_samples: Optional[int] = None) -> list[Problem]:
    """
    Load a benchmark dataset.

    Args:
        dataset: Dataset name ('dummy' or 'minif2f-lean4')
        n_samples: Optional limit on number of samples (None = all)

    Returns:
        List of Problem objects
    """
    if dataset not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(f"Unknown dataset: {dataset}. Available: {available}")

    return DATASETS[dataset](n_samples)


def list_datasets() -> list[str]:
    """List available datasets."""
    return list(DATASETS.keys())
