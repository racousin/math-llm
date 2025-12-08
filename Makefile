# Math-LLM Makefile
# ==================

.PHONY: help install test clean train train-dummy eval eval-all download lint format

# Default target
help:
	@echo "Math-LLM - LLM Agent for Lean 4 Theorem Proving"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install dependencies with Poetry"
	@echo "  make download       Download all Lean 4 datasets"
	@echo "  make download-quick Download minimal dataset (minif2f)"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run quick installation test"
	@echo "  make test-unit      Run unit tests with pytest"
	@echo ""
	@echo "Training:"
	@echo "  make train-dummy    Train with dummy data (no GPU needed)"
	@echo "  make train          Train with default config"
	@echo "  make train-dgx      Train with DGX Spark config"
	@echo ""
	@echo "Evaluation:"
	@echo "  make eval-dummy     Evaluate with dummy data"
	@echo "  make eval           Evaluate on minif2f-lean4"
	@echo "  make eval-all       Evaluate on all benchmarks"
	@echo "  make eval-fimo      Evaluate on FIMO (IMO problems)"
	@echo "  make eval-putnam    Evaluate on PutnamBench"
	@echo ""
	@echo "Development:"
	@echo "  make lint           Run linter (ruff)"
	@echo "  make format         Format code (black)"
	@echo "  make clean          Clean cache and outputs"
	@echo ""
	@echo "Data:"
	@echo "  make list-data      List available datasets"

# =============================================================================
# Setup
# =============================================================================

install:
	poetry install

download:
	poetry run python scripts/download_data.py \
		--sources leandojo minif2f-lean4 fimo putnambench proofnet formal-conjectures \
		--load

download-quick:
	poetry run python scripts/download_data.py --sources minif2f-lean4 --load

download-training:
	poetry run python scripts/download_data.py --sources leandojo mathlib4 --load

# =============================================================================
# Testing
# =============================================================================

test:
	poetry run python scripts/quick_test.py

test-unit:
	poetry run pytest tests/ -v

test-all: test test-unit

# =============================================================================
# Training
# =============================================================================

# Dummy training (no real model, no GPU)
train-dummy:
	poetry run python scripts/train.py \
		--config configs/dummy_test.yaml \
		--debug \
		--no-lean

# Training with real model on dummy data
train-dummy-model:
	poetry run python scripts/train.py \
		--config configs/dummy_test.yaml \
		--debug

# Default training
train:
	poetry run python scripts/train.py --config configs/default.yaml

# DGX Spark training (multi-GPU)
train-dgx:
	poetry run python scripts/train.py --config configs/dgx_spark.yaml

# Training with specific model
train-qwen:
	poetry run python scripts/train.py \
		--config configs/default.yaml

train-deepseek:
	MODEL_NAME=deepseek-ai/deepseek-math-7b-instruct \
	poetry run python scripts/train.py --config configs/default.yaml

# =============================================================================
# Evaluation
# =============================================================================

# Output directory for eval results
EVAL_OUTPUT ?= outputs/eval

# Dummy evaluation (simple problems, real LLM + real Lean)
eval-dummy:
	poetry run python scripts/evaluate.py \
		--config configs/dummy_test.yaml \
		--num-samples 5 \
		--output $(EVAL_OUTPUT)/dummy_results.json

# Evaluate on MiniF2F-Lean4
eval:
	@mkdir -p $(EVAL_OUTPUT)
	poetry run python -c "\
from math_llm.data import load_sources, LeanDataset; \
from math_llm.models import load_model; \
from math_llm.lean import LeanExecutor; \
from math_llm.agent import LeanAgent, Evaluator; \
from math_llm.config import load_config; \
import json; \
\
config = load_config('configs/default.yaml'); \
print('Loading model...'); \
model = load_model(config.model.name, device=config.model.device); \
print('Loading minif2f-lean4...'); \
problems = load_sources(['minif2f-lean4']); \
dataset = LeanDataset(problems[:100]); \
print(f'Evaluating on {len(dataset)} problems...'); \
with LeanExecutor() as executor: \
    agent = LeanAgent(model=model, executor=executor, config=config.agent); \
    evaluator = Evaluator(agent); \
    results = evaluator.evaluate(dataset, save_path='$(EVAL_OUTPUT)/minif2f_results.json'); \
"

# Evaluate on all benchmarks (recommended)
eval-all:
	poetry run python scripts/eval_all.py --mock --output $(EVAL_OUTPUT)

# Evaluate on all with real model
eval-all-model:
	poetry run python scripts/eval_all.py --output $(EVAL_OUTPUT)

# Evaluate on all with checkpoint
eval-all-checkpoint:
	@test -n "$(CHECKPOINT)" || (echo "Usage: make eval-all-checkpoint CHECKPOINT=path" && exit 1)
	poetry run python scripts/eval_all.py --checkpoint $(CHECKPOINT) --output $(EVAL_OUTPUT)

# Individual benchmark evaluations (with mock model for quick testing)

eval-minif2f:
	@mkdir -p $(EVAL_OUTPUT)
	@echo "Evaluating on MiniF2F-Lean4..."
	poetry run python -c "\
from math_llm.data import load_sources, LeanDataset; \
from math_llm.agent.agent import MockLLMWrapper; \
from math_llm.lean import LeanExecutor; \
from math_llm.agent import LeanAgent, Evaluator; \
from math_llm.config import load_config; \
config = load_config('configs/default.yaml'); \
model = MockLLMWrapper(); \
problems = load_sources(['minif2f-lean4']); \
dataset = LeanDataset(problems); \
print(f'Evaluating on {len(dataset)} MiniF2F problems...'); \
with LeanExecutor() as executor: \
    agent = LeanAgent(model=model, executor=executor, config=config.agent); \
    evaluator = Evaluator(agent); \
    results = evaluator.evaluate(dataset, save_path='$(EVAL_OUTPUT)/minif2f_results.json'); \
"

eval-fimo:
	@mkdir -p $(EVAL_OUTPUT)
	@echo "Evaluating on FIMO (IMO problems)..."
	poetry run python -c "\
from math_llm.data import load_sources, LeanDataset; \
from math_llm.agent.agent import MockLLMWrapper; \
from math_llm.lean import LeanExecutor; \
from math_llm.agent import LeanAgent, Evaluator; \
from math_llm.config import load_config; \
config = load_config('configs/default.yaml'); \
model = MockLLMWrapper(); \
problems = load_sources(['fimo']); \
dataset = LeanDataset(problems); \
print(f'Evaluating on {len(dataset)} FIMO problems...'); \
with LeanExecutor() as executor: \
    agent = LeanAgent(model=model, executor=executor, config=config.agent); \
    evaluator = Evaluator(agent); \
    results = evaluator.evaluate(dataset, save_path='$(EVAL_OUTPUT)/fimo_results.json'); \
"

eval-putnam:
	@mkdir -p $(EVAL_OUTPUT)
	@echo "Evaluating on PutnamBench..."
	poetry run python -c "\
from math_llm.data import load_sources, LeanDataset; \
from math_llm.agent.agent import MockLLMWrapper; \
from math_llm.lean import LeanExecutor; \
from math_llm.agent import LeanAgent, Evaluator; \
from math_llm.config import load_config; \
config = load_config('configs/default.yaml'); \
model = MockLLMWrapper(); \
problems = load_sources(['putnambench']); \
dataset = LeanDataset(problems); \
print(f'Evaluating on {len(dataset)} Putnam problems...'); \
with LeanExecutor() as executor: \
    agent = LeanAgent(model=model, executor=executor, config=config.agent); \
    evaluator = Evaluator(agent); \
    results = evaluator.evaluate(dataset, save_path='$(EVAL_OUTPUT)/putnam_results.json'); \
"

eval-proofnet:
	@mkdir -p $(EVAL_OUTPUT)
	@echo "Evaluating on ProofNet..."
	poetry run python -c "\
from math_llm.data import load_sources, LeanDataset; \
from math_llm.agent.agent import MockLLMWrapper; \
from math_llm.lean import LeanExecutor; \
from math_llm.agent import LeanAgent, Evaluator; \
from math_llm.config import load_config; \
config = load_config('configs/default.yaml'); \
model = MockLLMWrapper(); \
problems = load_sources(['proofnet']); \
dataset = LeanDataset(problems); \
print(f'Evaluating on {len(dataset)} ProofNet problems...'); \
with LeanExecutor() as executor: \
    agent = LeanAgent(model=model, executor=executor, config=config.agent); \
    evaluator = Evaluator(agent); \
    results = evaluator.evaluate(dataset, save_path='$(EVAL_OUTPUT)/proofnet_results.json'); \
"

# Evaluate with a specific checkpoint
eval-checkpoint:
	@test -n "$(CHECKPOINT)" || (echo "Usage: make eval-checkpoint CHECKPOINT=path/to/checkpoint" && exit 1)
	poetry run python scripts/evaluate.py \
		--config configs/default.yaml \
		--checkpoint $(CHECKPOINT) \
		--output $(EVAL_OUTPUT)/checkpoint_results.json

# =============================================================================
# Development
# =============================================================================

lint:
	poetry run ruff check src/ scripts/ tests/

format:
	poetry run black src/ scripts/ tests/

typecheck:
	poetry run mypy src/

# =============================================================================
# Data Management
# =============================================================================

list-data:
	@poetry run python -c "\
from math_llm.data import print_sources; \
print_sources(); \
"

# Show dataset statistics
data-stats:
	@poetry run python -c "\
from math_llm.data import load_sources; \
from collections import Counter; \
print('Loading all datasets...'); \
problems = load_sources('all'); \
print(f'\nTotal problems: {len(problems)}'); \
sources = Counter(p.source for p in problems); \
print('\nBy source:'); \
for s, c in sources.most_common(): \
    print(f'  {s}: {c}'); \
"

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf outputs/
	rm -rf .cache/
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-cache:
	rm -rf .cache/datasets/

clean-outputs:
	rm -rf outputs/

# =============================================================================
# Docker (optional)
# =============================================================================

docker-build:
	docker build -t math-llm .

docker-run:
	docker run --gpus all -it -v $(PWD)/outputs:/app/outputs math-llm

# =============================================================================
# Shortcuts
# =============================================================================

# Quick start sequence
quickstart: install test
	@echo ""
	@echo "Installation successful!"
	@echo "Next steps:"
	@echo "  make download-quick  # Download minimal dataset"
	@echo "  make train-dummy     # Test training loop"
	@echo "  make eval-dummy      # Test evaluation"

# Full setup
setup: install download test
	@echo "Full setup complete!"
