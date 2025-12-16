# Lean Proof Benchmark
# ====================
#
# Simple benchmark for LLM agents on Lean 4 theorem proving.
#
# Commands:
#   make lean-server    - Setup Lean server with Mathlib
#   make simple         - Run simple agent on dummy data
#   make tool           - Run tool agent on minif2f-lean4 (10 samples)

.PHONY: help install lean-server simple tool clean

help:
	@echo "Lean Proof Benchmark"
	@echo "===================="
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install Python dependencies"
	@echo "  make lean-server  - Setup Lean server with Mathlib (~2GB, 10-20min)"
	@echo ""
	@echo "Benchmarks:"
	@echo "  make simple       - Run simple agent on dummy data"
	@echo "  make tool         - Run tool agent on minif2f-lean4 (10 samples)"
	@echo ""
	@echo "Custom runs:"
	@echo "  python -m math_llm dummy simple"
	@echo "  python -m math_llm dummy tool"
	@echo "  python -m math_llm minif2f-lean4 simple --samples 10"
	@echo "  python -m math_llm minif2f-lean4 tool --samples 10"

# =============================================================================
# Setup
# =============================================================================

install:
	poetry install

# Setup Lean server with Mathlib + REPL
lean-server:
	@echo "Setting up Lean server with Mathlib + REPL..."
	@echo "This downloads ~2GB and takes 10-20 minutes on first run."
	@echo ""
	./scripts/setup_mathlib.sh

# =============================================================================
# Benchmarks
# =============================================================================

# Run simple agent on dummy data
simple:
	poetry run python -m math_llm dummy simple

# Run tool agent on minif2f-lean4 with 10 samples
tool:
	poetry run python -m math_llm minif2f-lean4 tool --samples 10

# =============================================================================
# Additional commands
# =============================================================================

# Run simple agent on minif2f-lean4
simple-minif2f:
	poetry run python -m math_llm minif2f-lean4 simple --samples 10

# Run tool agent on dummy data
tool-dummy:
	poetry run python -m math_llm dummy tool

# Run all dummy data benchmarks
test-dummy:
	poetry run python -m math_llm dummy simple
	poetry run python -m math_llm dummy tool

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf outputs/
	rm -rf .cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
