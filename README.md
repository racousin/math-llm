# Math-LLM

A complete training/evaluation package for state-of-the-art LLM agents that use Lean 4 efficiently to prove mathematical statements.

## Features

- **Data Pipeline**: Aggregates multiple Lean theorem proving datasets (formal-conjectures, mathlib4, miniF2F, ProofNet)
- **Lean Tool Interface**: Clean interface for LLM interaction with Lean 4, including error normalization and hint generation
- **Agentic Framework**: Iterative proof solving with trajectory tracking and feedback loops
- **RL Training**: PPO-based reinforcement learning using TRL with custom reward functions
- **Flexible Configuration**: YAML-based config system with support for different environments

## Installation

### Prerequisites

1. **Python 3.10+**
2. **Lean 4** (for proof verification):
   ```bash
   # Install elan (Lean version manager)
   curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

   # Install Lean 4
   elan default leanprover/lean4:stable
   ```

3. **Poetry** (for dependency management):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

### Install Math-LLM

```bash
# Clone the repository
git clone https://github.com/yourusername/math-llm.git
cd math-llm

# Install dependencies with Poetry
poetry install

# Activate the virtual environment
poetry shell
```

### Setup Mathlib (Required for benchmarks)

Mathlib is required to run competition math benchmarks (minif2f, FIMO, etc.):

```bash
# 1. Run setup script (downloads ~2GB, takes 10-20 min first time)
./scripts/setup_mathlib.sh

# 2. Add to your shell config (.bashrc, .zshrc, etc.)
export MATHLIB_PROJECT_PATH=$HOME/.math-llm/mathlib-project
```

Or use make:
```bash
make setup-mathlib
```

### Quick Test

Verify the installation:

```bash
python scripts/quick_test.py
```

## Usage

### 1. Download Datasets

```bash
# Download formal-conjectures and miniF2F
python scripts/download_data.py --sources formal-conjectures miniF2F

# Download all available sources
python scripts/download_data.py --sources formal-conjectures mathlib miniF2F proofnet --load
```

### 2. Run Evaluation (Testing)

```bash
# Quick test with dummy data
python scripts/evaluate.py --config configs/dummy_test.yaml --debug

# Full evaluation
python scripts/evaluate.py --config configs/default.yaml --num-samples 100
```

### 3. Run Training

```bash
# Test training with dummy data
python scripts/train.py --config configs/dummy_test.yaml --debug

# Full training
python scripts/train.py --config configs/default.yaml
```

### 4. Configuration

The system uses YAML configuration files. Three configs are provided:

- `configs/dummy_test.yaml` - Minimal config for quick local testing
- `configs/default.yaml` - Standard config for development
- `configs/dgx_spark.yaml` - Full-scale training on DGX Spark

Example config structure:

```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  device: "auto"
  torch_dtype: "bfloat16"

agent:
  max_steps: 10
  temperature: 0.7

training:
  learning_rate: 1.0e-5
  batch_size: 4
  use_peft: true
  lora_r: 16
```

## Architecture

```
math-llm/
├── src/math_llm/
│   ├── config.py           # Configuration management
│   ├── data/               # Dataset handling
│   │   ├── datasets.py     # LeanProblem, LeanDataset classes
│   │   ├── sources.py      # Dataset sources (formal-conjectures, etc.)
│   │   └── loaders.py      # Data loading utilities
│   ├── lean/               # Lean tool interface
│   │   ├── server.py       # Lean server (execution)
│   │   └── executor.py     # High-level executor with hints
│   ├── agent/              # Agentic framework
│   │   ├── agent.py        # LeanAgent class
│   │   ├── trajectory.py   # Trajectory tracking
│   │   └── evaluator.py    # Evaluation framework
│   ├── models/             # LLM wrappers
│   │   └── llm.py          # HuggingFace model wrapper
│   └── training/           # RL training
│       ├── rewards.py      # Reward functions
│       └── trainer.py      # PPO trainer with TRL
├── configs/                # YAML configurations
├── scripts/                # Entry point scripts
└── tests/                  # Unit tests
```

## Core Concepts

### Agent Loop

The agent follows an iterative proof-solving loop:

```python
trajectory = ""
for step in range(max_steps):
    action = model.generate(state + trajectory)
    result = lean_server.execute(action)
    trajectory += f"<action>{action}</action><result>{result}</result>"
    if proof_complete(result):
        break

reward = 1.0 if proof_complete else 0.0
```

### Reward Structure

- **+1.0**: Proof completed successfully
- **+0.1 to +0.3**: Progress made (code accepted, goals reduced)
- **-0.01**: Step penalty (encourages efficiency)
- **-0.1**: Error penalty
- **Exponential decay**: Longer trajectories receive less reward

### Supported Models

Recommended models for Lean theorem proving:

- `Qwen/Qwen2.5-7B-Instruct` - Fast, good for testing
- `Qwen/Qwen2.5-1.5B-Instruct` - Better quality
- `Qwen/Qwen2.5-7B-Instruct` - High quality
- `deepseek-ai/deepseek-math-7b-instruct` - Specialized for math
- `meta-llama/Llama-3.2-3B-Instruct` - Good balance

## API Reference

### LeanAgent

```python
from math_llm import LeanAgent, LeanExecutor
from math_llm.models import load_model
from math_llm.data import LeanProblem

# Load model
model = load_model("Qwen/Qwen2.5-7B-Instruct")

# Create agent
with LeanExecutor() as executor:
    agent = LeanAgent(model=model, executor=executor)

    # Solve a problem
    problem = LeanProblem(
        id="test/1",
        statement="theorem add_zero (n : Nat) : n + 0 = n := by\n  sorry",
        description="Prove that adding zero is identity",
    )

    trajectory = agent.solve(problem)
    print(f"Solved: {trajectory.success} in {trajectory.num_steps} steps")
```

### Training

```python
from math_llm import RLTrainer
from math_llm.config import load_config

config = load_config("configs/default.yaml")

trainer = RLTrainer(
    model=model.get_model_for_training(),
    tokenizer=model.get_tokenizer(),
    config=config.training,
)

trainer.train(
    agent=agent,
    train_problems=train_dataset.problems,
    num_epochs=3,
)
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_basic.py::TestRewards -v
```

### Code Quality

```bash
# Format code
black src/ scripts/ tests/

# Lint
ruff check src/

# Type check
mypy src/
```

## Deployment

### Mac (Development)

```bash
python scripts/train.py --config configs/dummy_test.yaml --debug
```

### DGX Spark (Production)

```bash
# Multi-GPU training
torchrun --nproc_per_node=8 scripts/train.py --config configs/dgx_spark.yaml
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

MIT License

## Acknowledgments

- [DeepMind Formal Conjectures](https://github.com/google-deepmind/formal-conjectures)
- [Mathlib4](https://github.com/leanprover-community/mathlib4)
- [MiniF2F](https://github.com/facebookresearch/miniF2F)
- [TRL](https://github.com/huggingface/trl)
- [Lean 4](https://github.com/leanprover/lean4)
