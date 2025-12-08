"""
Dataset sources for Lean 4 theorem proving.

Lean 4 Datasets (Preferred):
- LeanDojo Benchmark (~100k theorems) - HuggingFace
- Mathlib4 (~150k theorems) - git/HuggingFace
- MiniF2F Lean 4 (~488 problems) - HuggingFace
- ProofNet (~370 problems) - HuggingFace
- FIMO (~149 IMO problems) - HuggingFace
- PutnamBench (~1.7k problems) - HuggingFace
- Formal Conjectures (DeepMind) - git

Legacy (Lean 3):
- MiniF2F original (needs conversion)
"""

import json
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Literal

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from math_llm.data.datasets import LeanProblem

console = Console()


def normalize_lean4_syntax(statement: str) -> str:
    """
    Normalize Lean syntax for Mathlib 4 compatibility.

    Fixes common syntax differences between dataset formats and Mathlib 4:
    - Converts `∑ k in S` to `∑ k ∈ S` (BigOperators notation)
    - Converts `∏ k in S` to `∏ k ∈ S`
    """
    import re

    # Replace "∑ var in " with "∑ var ∈ " (sum notation)
    # Pattern: ∑ followed by identifier, then " in "
    statement = re.sub(r'(∑\s*\w+)\s+in\s+', r'\1 ∈ ', statement)

    # Same for product notation
    statement = re.sub(r'(∏\s*\w+)\s+in\s+', r'\1 ∈ ', statement)

    return statement


class DataSource(ABC):
    """Abstract base class for data sources."""

    name: str
    description: str
    size: str
    lean_version: Literal["lean4", "lean3", "mixed"] = "lean4"

    def __init__(self, cache_dir: str = ".cache/datasets"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def local_path(self) -> Path:
        return self.cache_dir / self.name

    @abstractmethod
    def download(self) -> None:
        """Download the dataset."""
        pass

    @abstractmethod
    def load(self, split: Optional[str] = None) -> list[LeanProblem]:
        """Load problems from the dataset."""
        pass

    def is_downloaded(self) -> bool:
        """Check if dataset is already downloaded."""
        return self.local_path.exists()

    def info(self) -> dict:
        """Get dataset info."""
        return {
            "name": self.name,
            "description": self.description,
            "size": self.size,
            "lean_version": self.lean_version,
            "downloaded": self.is_downloaded(),
        }


# =============================================================================
# HuggingFace Dataset Sources (Preferred for Lean 4)
# =============================================================================

class HuggingFaceSource(DataSource):
    """Base class for HuggingFace dataset sources."""

    hf_dataset_name: str
    hf_config: Optional[str] = None

    def download(self) -> None:
        """HuggingFace datasets are downloaded on first load."""
        console.print(f"[blue]{self.name} will be downloaded from HuggingFace on first load[/blue]")

    def _load_hf_dataset(self, split: Optional[str] = None):
        """Load dataset from HuggingFace."""
        from datasets import load_dataset

        console.print(f"[blue]Loading {self.name} from HuggingFace ({self.hf_dataset_name})...[/blue]")

        kwargs = {"trust_remote_code": True}
        if self.hf_config:
            kwargs["name"] = self.hf_config

        if split:
            ds = load_dataset(self.hf_dataset_name, split=split, **kwargs)
        else:
            ds = load_dataset(self.hf_dataset_name, **kwargs)

        return ds

    def is_downloaded(self) -> bool:
        """HuggingFace datasets use their own cache."""
        return True  # Always available via HF


class LeanDojoSource(HuggingFaceSource):
    """
    LeanDojo Benchmark - Extracted theorem-proof pairs with premise info.
    https://huggingface.co/datasets/kaiyuy/leandojo-lean4

    ~100k theorem-proof pairs from Mathlib4 with:
    - Theorem statements
    - Ground truth proofs
    - Premise information (what lemmas are used)
    - File and position metadata
    """

    name = "leandojo"
    description = "LeanDojo Lean 4 benchmark with theorem-proof pairs and premise info"
    size = "~100k theorems"
    lean_version = "lean4"
    hf_dataset_name = "kaiyuy/leandojo-lean4"

    def load(self, split: Optional[str] = "train") -> list[LeanProblem]:
        ds = self._load_hf_dataset(split)

        problems = []
        for i, item in enumerate(ds):
            # LeanDojo format: theorem statement, proof, premises
            statement = item.get("statement", item.get("formal_statement", ""))
            proof = item.get("proof", item.get("human_proof", ""))
            name = item.get("full_name", item.get("name", f"theorem_{i}"))

            # Build full Lean statement
            if statement and "theorem" not in statement.lower():
                full_statement = f"theorem {name} : {statement} := by\n  sorry"
            else:
                full_statement = statement if "sorry" in statement else f"{statement} := by\n  sorry"

            problem = LeanProblem(
                id=f"{self.name}/{name}",
                statement=full_statement,
                description=item.get("informal_statement", item.get("docstring")),
                proof=proof,
                source=self.name,
                tags=item.get("tags", []),
                metadata={
                    "premises": item.get("premises", []),
                    "file_path": item.get("file_path", ""),
                    "repo": item.get("repo", "mathlib4"),
                    "split": split,
                },
            )
            problems.append(problem)

        console.print(f"[green]Loaded {len(problems)} problems from {self.name}[/green]")
        return problems


class MiniF2FLean4Source(HuggingFaceSource):
    """
    MiniF2F Lean 4 - Competition math problems (IMO, AMC, etc.)
    https://huggingface.co/datasets/cat-searcher/minif2f-lean4

    ~488 problems from math competitions formalized in Lean 4.
    """

    name = "minif2f-lean4"
    description = "Competition math problems (IMO, AMC) in Lean 4"
    size = "~488 problems"
    lean_version = "lean4"
    hf_dataset_name = "cat-searcher/minif2f-lean4"

    def load(self, split: Optional[str] = None) -> list[LeanProblem]:
        ds = self._load_hf_dataset(split)

        # Handle DatasetDict
        if hasattr(ds, 'keys'):
            # Combine all splits
            all_items = []
            for split_name in ds.keys():
                all_items.extend(ds[split_name])
        else:
            all_items = ds

        problems = []
        for item in all_items:
            statement = item.get("formal_statement", item.get("statement", ""))
            name = item.get("name", item.get("problem_name", ""))

            # Normalize syntax for Mathlib 4 compatibility
            statement = normalize_lean4_syntax(statement)

            # Get informal statement
            informal = item.get("informal_statement", item.get("informal_stmt", ""))

            # Determine difficulty/source
            tags = []
            if "imo" in name.lower():
                tags.append("IMO")
            elif "amc" in name.lower():
                tags.append("AMC")
            elif "aime" in name.lower():
                tags.append("AIME")

            problem = LeanProblem(
                id=f"{self.name}/{name}",
                statement=statement,
                description=informal,
                proof=item.get("proof"),
                source=self.name,
                difficulty=item.get("difficulty", "competition"),
                tags=tags,
                metadata={
                    "competition": item.get("source", ""),
                    "year": item.get("year"),
                },
            )
            problems.append(problem)

        console.print(f"[green]Loaded {len(problems)} problems from {self.name}[/green]")
        return problems


class FIMOSource(HuggingFaceSource):
    """
    FIMO - Formal IMO problems.
    https://huggingface.co/datasets/qq8933/FIMO

    ~149 IMO problems formalized in Lean 4.
    """

    name = "fimo"
    description = "Formal IMO problems in Lean 4"
    size = "~149 problems"
    lean_version = "lean4"
    hf_dataset_name = "qq8933/FIMO"

    def load(self, split: Optional[str] = None) -> list[LeanProblem]:
        ds = self._load_hf_dataset(split)

        if hasattr(ds, 'keys'):
            all_items = []
            for split_name in ds.keys():
                all_items.extend(ds[split_name])
        else:
            all_items = ds

        problems = []
        for item in all_items:
            statement = item.get("formal_statement", item.get("lean_statement", ""))
            name = item.get("problem_name", item.get("id", ""))

            problem = LeanProblem(
                id=f"{self.name}/{name}",
                statement=statement,
                description=item.get("informal_statement", item.get("problem", "")),
                proof=item.get("proof", item.get("solution")),
                source=self.name,
                difficulty="olympiad",
                tags=["IMO"],
                metadata={
                    "year": item.get("year"),
                    "problem_number": item.get("problem_number"),
                },
            )
            problems.append(problem)

        console.print(f"[green]Loaded {len(problems)} problems from {self.name}[/green]")
        return problems


class PutnamBenchSource(HuggingFaceSource):
    """
    PutnamBench - Putnam competition problems.
    https://huggingface.co/datasets/trishullab/PutnamBench

    ~1.7k problems from the Putnam Mathematical Competition.
    """

    name = "putnambench"
    description = "Putnam competition problems in Lean 4"
    size = "~1.7k problems"
    lean_version = "lean4"
    hf_dataset_name = "trishullab/PutnamBench"

    def load(self, split: Optional[str] = None) -> list[LeanProblem]:
        ds = self._load_hf_dataset(split)

        if hasattr(ds, 'keys'):
            all_items = []
            for split_name in ds.keys():
                all_items.extend(ds[split_name])
        else:
            all_items = ds

        problems = []
        for item in all_items:
            # PutnamBench format
            statement = item.get("formal_statement", item.get("lean4_statement", ""))
            name = item.get("problem_id", item.get("name", ""))

            problem = LeanProblem(
                id=f"{self.name}/{name}",
                statement=statement,
                description=item.get("informal_statement", item.get("problem_latex", "")),
                proof=item.get("proof"),
                source=self.name,
                difficulty="competition",
                tags=["Putnam"],
                metadata={
                    "year": item.get("year"),
                    "competition": "Putnam",
                },
            )
            problems.append(problem)

        console.print(f"[green]Loaded {len(problems)} problems from {self.name}[/green]")
        return problems


class ProofNetHFSource(HuggingFaceSource):
    """
    ProofNet - Undergraduate math problems.
    https://huggingface.co/datasets/hoskinson-center/proofnet

    ~370 undergraduate-level mathematical theorems.
    """

    name = "proofnet"
    description = "Undergraduate math problems in Lean 4"
    size = "~370 problems"
    lean_version = "lean4"
    hf_dataset_name = "hoskinson-center/proofnet"

    def load(self, split: Optional[str] = "test") -> list[LeanProblem]:
        ds = self._load_hf_dataset(split)

        if hasattr(ds, 'keys'):
            all_items = []
            for split_name in ds.keys():
                all_items.extend(ds[split_name])
        else:
            all_items = ds

        problems = []
        for item in all_items:
            statement = item.get("formal_statement", item.get("statement", ""))
            name = item.get("name", item.get("id", ""))

            problem = LeanProblem(
                id=f"{self.name}/{name}",
                statement=statement,
                description=item.get("informal_statement", item.get("nl_statement", "")),
                proof=item.get("proof"),
                source=self.name,
                difficulty="undergraduate",
                tags=item.get("topics", []),
                metadata={
                    "topic": item.get("topic", ""),
                    "source_book": item.get("source"),
                },
            )
            problems.append(problem)

        console.print(f"[green]Loaded {len(problems)} problems from {self.name}[/green]")
        return problems


# =============================================================================
# Git-based Sources
# =============================================================================

class FormalConjecturesSource(DataSource):
    """
    DeepMind's Formal Conjectures dataset.
    https://github.com/google-deepmind/formal-conjectures

    Formalized open problems and conjectures in Lean 4.
    """

    name = "formal-conjectures"
    description = "DeepMind's formalized open problems and conjectures"
    size = "~100+ problems"
    lean_version = "lean4"
    url = "https://github.com/google-deepmind/formal-conjectures.git"

    def download(self) -> None:
        if self.is_downloaded():
            console.print(f"[yellow]{self.name} already downloaded[/yellow]")
            return

        console.print(f"[blue]Cloning {self.name}...[/blue]")
        subprocess.run(
            ["git", "clone", "--depth", "1", self.url, str(self.local_path)],
            check=True,
        )
        console.print(f"[green]{self.name} downloaded successfully[/green]")

    def load(self, split: Optional[str] = None) -> list[LeanProblem]:
        if not self.is_downloaded():
            self.download()

        problems = []
        lean_dir = self.local_path / "FormalConjectures"

        if not lean_dir.exists():
            lean_dir = self.local_path

        for lean_file in lean_dir.rglob("*.lean"):
            try:
                content = lean_file.read_text()
                file_problems = self._parse_lean4_file(content, lean_file)
                problems.extend(file_problems)
            except Exception as e:
                console.print(f"[red]Error parsing {lean_file}: {e}[/red]")

        console.print(f"[green]Loaded {len(problems)} problems from {self.name}[/green]")
        return problems

    def _parse_lean4_file(self, content: str, filepath: Path) -> list[LeanProblem]:
        """Parse Lean 4 file to extract theorems."""
        problems = []
        lines = content.split("\n")
        current_theorem = []
        in_theorem = False
        theorem_name = ""
        doc_comment = ""
        doc_lines = []

        for i, line in enumerate(lines):
            # Capture doc comments (multiline)
            if line.strip().startswith("/--"):
                doc_lines = [line.strip()[3:]]
                if not line.strip().endswith("-/"):
                    continue
                else:
                    doc_comment = doc_lines[0].rstrip("-/").strip()
                    doc_lines = []
            elif doc_lines:
                if "-/" in line:
                    doc_lines.append(line.split("-/")[0].strip())
                    doc_comment = " ".join(doc_lines).strip()
                    doc_lines = []
                else:
                    doc_lines.append(line.strip())
                continue

            # Detect theorem/lemma start
            stripped = line.strip()
            if any(stripped.startswith(kw) for kw in ["theorem ", "lemma ", "example "]):
                in_theorem = True
                current_theorem = [line]
                parts = stripped.split()
                if len(parts) > 1:
                    theorem_name = parts[1].split("(")[0].split(":")[0].strip()

            elif in_theorem:
                current_theorem.append(line)
                # Check for theorem end
                if ":= by" in line or line.strip() == "sorry" or (":=" in line and "by" not in line):
                    statement = "\n".join(current_theorem)

                    # Replace proof with sorry
                    if ":= by" in statement:
                        idx = statement.find(":= by")
                        statement = statement[:idx] + ":= by\n  sorry"
                    elif ":=" in statement:
                        idx = statement.find(":=")
                        statement = statement[:idx] + ":= sorry"

                    problem = LeanProblem(
                        id=f"{self.name}/{filepath.stem}/{theorem_name}",
                        statement=statement.strip(),
                        description=doc_comment if doc_comment else None,
                        source=self.name,
                        metadata={
                            "file": str(filepath.relative_to(self.local_path)),
                            "lean_version": "4",
                        },
                    )
                    problems.append(problem)

                    in_theorem = False
                    current_theorem = []
                    theorem_name = ""
                    doc_comment = ""

        return problems


class Mathlib4Source(DataSource):
    """
    Mathlib4 - The standard Lean 4 mathematics library.
    https://github.com/leanprover-community/mathlib4

    ~150k theorems covering algebra, analysis, topology, number theory, etc.
    Note: Very large repo, use LeanDojo for pre-processed version.
    """

    name = "mathlib4"
    description = "Lean 4 standard math library (full repo)"
    size = "~150k theorems"
    lean_version = "lean4"
    url = "https://github.com/leanprover-community/mathlib4.git"

    def download(self) -> None:
        if self.is_downloaded():
            console.print(f"[yellow]{self.name} already downloaded[/yellow]")
            return

        console.print(f"[blue]Cloning {self.name} (shallow, this may take a while)...[/blue]")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", self.url, str(self.local_path)],
            check=True,
        )
        console.print(f"[green]{self.name} downloaded successfully[/green]")

    def load(self, split: Optional[str] = None, max_files: int = 200, max_per_file: int = 20) -> list[LeanProblem]:
        """
        Load a subset of mathlib theorems.

        Args:
            split: Not used (for interface compatibility)
            max_files: Maximum files to parse per area
            max_per_file: Maximum theorems per file
        """
        if not self.is_downloaded():
            self.download()

        problems = []
        mathlib_dir = self.local_path / "Mathlib"

        if not mathlib_dir.exists():
            console.print(f"[red]Mathlib directory not found[/red]")
            return problems

        areas = ["Algebra", "Analysis", "Topology", "NumberTheory", "Combinatorics",
                 "LinearAlgebra", "GroupTheory", "RingTheory", "FieldTheory"]
        files_per_area = max_files // len(areas)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Parsing Mathlib4...", total=len(areas))

            for area in areas:
                area_dir = mathlib_dir / area
                if not area_dir.exists():
                    progress.advance(task)
                    continue

                lean_files = list(area_dir.rglob("*.lean"))[:files_per_area]

                for lean_file in lean_files:
                    try:
                        content = lean_file.read_text()
                        file_problems = self._parse_lean4_file(content, lean_file, area)
                        problems.extend(file_problems[:max_per_file])
                    except Exception as e:
                        pass  # Skip problematic files

                progress.advance(task)

        console.print(f"[green]Loaded {len(problems)} problems from {self.name}[/green]")
        return problems

    def _parse_lean4_file(self, content: str, filepath: Path, area: str) -> list[LeanProblem]:
        """Parse Lean 4 file from Mathlib."""
        problems = []
        lines = content.split("\n")

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("theorem ", "lemma ")):
                theorem_lines = [line]
                j = i + 1
                depth = line.count("(") - line.count(")") + line.count("{") - line.count("}")

                # Collect full theorem signature
                while j < len(lines):
                    next_line = lines[j]
                    theorem_lines.append(next_line)
                    depth += next_line.count("(") - next_line.count(")")
                    depth += next_line.count("{") - next_line.count("}")

                    if ":=" in next_line or (depth <= 0 and ":" in next_line):
                        break
                    j += 1
                    if j - i > 30:
                        break

                statement = "\n".join(theorem_lines)

                # Clean up - remove proof part
                if ":= by" in statement:
                    statement = statement.split(":= by")[0] + ":= by\n  sorry"
                elif ":=" in statement:
                    statement = statement.split(":=")[0] + ":= sorry"
                else:
                    statement = statement + " := sorry"

                parts = stripped.split()
                name = parts[1].split("(")[0].split(":")[0].strip() if len(parts) > 1 else "unknown"

                problem = LeanProblem(
                    id=f"{self.name}/{area}/{filepath.stem}/{name}",
                    statement=statement.strip(),
                    source=self.name,
                    tags=[area],
                    metadata={
                        "file": str(filepath.name),
                        "area": area,
                        "lean_version": "4",
                    },
                )
                problems.append(problem)

        return problems


# =============================================================================
# Registry and Utilities
# =============================================================================

# All available sources - Lean 4 sources first (preferred)
SOURCES = {
    # HuggingFace sources (Lean 4, preferred)
    "leandojo": LeanDojoSource,
    "minif2f-lean4": MiniF2FLean4Source,
    "fimo": FIMOSource,
    "putnambench": PutnamBenchSource,
    "proofnet": ProofNetHFSource,
    # Git sources (Lean 4)
    "formal-conjectures": FormalConjecturesSource,
    "mathlib4": Mathlib4Source,
}

# Recommended sources for different use cases
RECOMMENDED = {
    "quick_test": ["minif2f-lean4"],  # Small, good quality
    "training": ["leandojo", "mathlib4"],  # Large, diverse
    "evaluation": ["minif2f-lean4", "fimo", "putnambench", "proofnet"],  # Benchmarks
    "competition": ["fimo", "putnambench", "minif2f-lean4"],  # Competition math
    "all": list(SOURCES.keys()),
}


def get_source(name: str, cache_dir: str = ".cache/datasets") -> DataSource:
    """Get a data source by name."""
    if name not in SOURCES:
        available = ", ".join(SOURCES.keys())
        raise ValueError(f"Unknown source: {name}. Available: {available}")
    return SOURCES[name](cache_dir=cache_dir)


def list_sources() -> list[dict]:
    """List all available data sources with info."""
    sources_info = []
    for name, source_cls in SOURCES.items():
        source = source_cls()
        sources_info.append(source.info())
    return sources_info


def print_sources() -> None:
    """Pretty print available sources."""
    from rich.table import Table

    table = Table(title="Available Lean 4 Datasets")
    table.add_column("Name", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("Description", style="white")
    table.add_column("Type", style="yellow")

    for name, source_cls in SOURCES.items():
        source = source_cls()
        source_type = "HuggingFace" if isinstance(source, HuggingFaceSource) else "Git"
        table.add_row(name, source.size, source.description, source_type)

    console.print(table)


def download_sources(
    sources: Optional[list[str]] = None,
    cache_dir: str = ".cache/datasets",
) -> None:
    """Download specified sources."""
    if sources is None:
        sources = list(SOURCES.keys())

    for source_name in sources:
        try:
            source = get_source(source_name, cache_dir)
            source.download()
        except Exception as e:
            console.print(f"[red]Failed to download {source_name}: {e}[/red]")


def load_sources(
    sources: Optional[list[str]] = None,
    cache_dir: str = ".cache/datasets",
    split: Optional[str] = None,
) -> list[LeanProblem]:
    """
    Load problems from specified sources.

    Args:
        sources: List of source names, or use RECOMMENDED keys
        cache_dir: Cache directory for downloads
        split: Dataset split (train/test/validation) if applicable

    Returns:
        List of LeanProblem objects
    """
    if sources is None:
        sources = RECOMMENDED["training"]
    elif isinstance(sources, str) and sources in RECOMMENDED:
        sources = RECOMMENDED[sources]

    all_problems = []
    for source_name in sources:
        try:
            source = get_source(source_name, cache_dir)
            console.print(f"[blue]Loading {source_name}...[/blue]")
            problems = source.load(split=split)
            all_problems.extend(problems)
        except Exception as e:
            console.print(f"[red]Failed to load {source_name}: {e}[/red]")

    console.print(f"[green]Total: {len(all_problems)} problems loaded[/green]")
    return all_problems


# Backwards compatibility aliases
download_all_sources = download_sources
load_all_sources = load_sources
