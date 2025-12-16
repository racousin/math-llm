# Lean Proof Benchmark

A simple benchmark for evaluating LLM agents on Lean 4 theorem proving.

| Component | Version |
|-----------|---------|
| Lean | 4.25.2 |
| Mathlib | v4.25.2 |
| Toolchain | `leanprover/lean4:v4.25.2` |

## Features

- **Simple Agent**: Direct single-shot proof generation with Qwen
- **Tool Agent**: Iterative proof with Lean tool calls for feedback
- **Datasets**: Dummy problems for testing + MiniF2F-Lean4 competition problems
- **Lean Server**: Fast proof checking with Mathlib REPL

## Quick Start

```bash
# Install dependencies
make install

# Setup Lean server (first time, ~10-20 min)
make lean-server

# Run simple agent on dummy data
make simple

# Run tool agent on MiniF2F (10 samples)
make tool
```

## Installation

### Prerequisites

1. **Python 3.10+**
2. **Poetry**: `curl -sSL https://install.python-poetry.org | python3 -`
3. **Elan** (Lean version manager):
   ```bash
   curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
   ```

### Install

```bash
git clone https://github.com/yourusername/math-llm.git
cd math-llm
make install
```

### Setup Lean Server

Required for proof verification:

```bash
make lean-server
```

This downloads Mathlib (~2GB) and sets up the REPL. After completion, add to your shell:

```bash
export MATHLIB_PROJECT_PATH=$HOME/.lean-bench
```

## Usage

### CLI

```bash
# Basic usage
python -m math_llm <dataset> <agent> [--samples N]

# Examples
python -m math_llm dummy simple              # All dummy problems with simple agent
python -m math_llm dummy tool                # All dummy problems with tool agent
python -m math_llm minif2f-lean4 simple -n 10    # 10 MiniF2F problems with simple agent
python -m math_llm minif2f-lean4 tool -n 10      # 10 MiniF2F problems with tool agent

# Custom model
python -m math_llm dummy simple --model Qwen/Qwen2.5-1.5B-Instruct
```

### Makefile Commands

```bash
make help          # Show available commands
make lean-server   # Setup Lean server with Mathlib
make simple        # Run simple agent on dummy data
make tool          # Run tool agent on MiniF2F (10 samples)
make clean         # Remove outputs and cache
```

### Output

Results are saved to `outputs/<dataset>_<agent>_results.json`:

```json
{
  "dataset": "dummy",
  "agent": "simple",
  "total_problems": 10,
  "successful": 8,
  "complete": 6,
  "success_rate": 0.8,
  "complete_rate": 0.6,
  "total_time": 45.2,
  "avg_time_per_problem": 4.52
}
```

## Agents

### Simple Agent

Direct single-shot proof generation. The LLM receives the problem and generates a complete proof in one call.

```python
from math_llm import SimpleAgent, LeanServer, load_data

with LeanServer() as server:
    agent = SimpleAgent(model_name="Qwen/Qwen2.5-7B-Instruct", lean_server=server)

    problems = load_data("dummy")
    for problem in problems:
        result = agent.solve(problem)
        print(f"{problem.id}: {'COMPLETE' if result.complete else 'FAIL'}")
```

### Tool Agent

Iterative proof with Lean feedback. The agent generates tactics one at a time, receives Lean's response (errors, remaining goals), and adjusts its approach.

```python
from math_llm import ToolAgent, LeanServer, load_data

with LeanServer() as server:
    agent = ToolAgent(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        lean_server=server,
        max_steps=10,  # Max iterations
    )

    problems = load_data("dummy")
    for problem in problems:
        result = agent.solve(problem)
        print(f"{problem.id}: {result.complete} in {len(result.trajectory.steps)} steps")
```

## Datasets

### Dummy

10 simple problems for testing:
- Basic arithmetic (`1 + 1 = 2`)
- Ring operations
- Simple proofs (`rfl`, `norm_num`, `ring`)

### MiniF2F-Lean4

~488 competition math problems (IMO, AMC, AIME) from HuggingFace:
- Various difficulty levels
- Formalized in Lean 4
- Loaded from `cat-searcher/minif2f-lean4`

## Project Structure

```
math-llm/
├── src/math_llm/
│   ├── __init__.py       # Package exports
│   ├── __main__.py       # CLI entry point
│   ├── cli.py            # CLI implementation
│   ├── data.py           # Data loading (dummy + minif2f)
│   ├── lean_server.py    # Lean REPL interface
│   └── agents/
│       ├── __init__.py
│       ├── simple.py     # Simple single-shot agent
│       └── tool.py       # Iterative tool agent
├── scripts/
│   └── setup_mathlib.sh  # Lean server setup
├── Makefile              # Build commands
├── pyproject.toml        # Dependencies
└── README.md
```

## License

MIT
