# Lean Proof Benchmark

Benchmark for LLM agents on Lean 4 theorem proving.

## Setup

```bash
# Prerequisites: Python 3.10+, Poetry, Elan (Lean version manager)

# Install
make install

# Setup Lean server with Mathlib (~2GB, 10-20 min)
make lean-server
```

## Usage

```bash
# Quick test (both agents on dummy data)
make test

# Specific benchmarks: <dataset>-<agent>
make dummy-simple       # Simple agent on dummy
make dummy-tool         # Tool agent on dummy
make minif2f-simple     # Simple agent on minif2f-lean4 (10 samples)
make minif2f-tool       # Tool agent on minif2f-lean4 (10 samples)

# Custom runs
python -m math_llm <dataset> <agent> [--samples N] [--model MODEL]
```

## Agents

- **Simple**: Single-shot proof generation
- **Tool**: Iterative proof with Lean feedback (generates tactics step by step)

## Datasets

- **dummy**: 10 simple test problems
- **minif2f-lean4**: ~488 competition math problems (IMO, AMC, AIME)

## Example Output

```
$ make dummy-simple

============================================================
Lean Proof Benchmark
============================================================
Dataset: dummy
Agent: simple
Model: Qwen/Qwen2.5-7B-Instruct
============================================================

[1/10] dummy/add_one
  Statement: theorem add_one : 1 + 1 = 2 := by sorry...
  Result: COMPLETE (35.62s)
  Proof: rfl...

[2/10] dummy/mul_comm
  Statement: theorem mul_comm_example : 2 * 3 = 3 * 2 := by sorry...
  Result: COMPLETE (1.63s)
  Proof: ring...

...

============================================================
RESULTS
============================================================
Total problems: 10
Successful: 6 (60.0%)
Total time: 171.09s
============================================================
```

Results saved to `outputs/<dataset>_<agent>_results.json`
