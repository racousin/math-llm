#!/bin/bash
#
# Setup Lean server with Mathlib and REPL.
#
# This script:
# 1. Installs elan (Lean version manager) if not present
# 2. Creates a Lean project with Mathlib dependencies
# 3. Downloads the Mathlib cache (~2GB)
#
# After running, set MATHLIB_PROJECT_PATH to point to the project.

set -e

# =============================================================================
# LEAN/MATHLIB VERSION - Must be kept in sync with src/math_llm/lean_server.py
# =============================================================================
MATHLIB_VERSION="v4.25.2"
LEAN_TOOLCHAIN="leanprover/lean4:v4.25.2"
# =============================================================================

PROJECT_DIR="${MATHLIB_PROJECT_PATH:-$HOME/.lean-bench}"

echo "=================================="
echo "Lean Server Setup"
echo "=================================="
echo "Mathlib version: $MATHLIB_VERSION"
echo "Project directory: $PROJECT_DIR"
echo ""

# Check for elan
if ! command -v elan &> /dev/null; then
    echo "Installing elan (Lean version manager)..."
    curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
    source ~/.elan/env
fi

# Ensure elan is in PATH
export PATH="$HOME/.elan/bin:$PATH"

# Check lake is available
if ! command -v lake &> /dev/null; then
    echo "Error: lake not found. Please ensure elan is installed correctly."
    exit 1
fi

echo "Using lake version: $(lake --version)"

# Create project directory
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create lakefile.lean
cat > lakefile.lean << EOF
import Lake
open Lake DSL

package «lean_bench»

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "$MATHLIB_VERSION"

require «repl» from git
  "https://github.com/leanprover-community/repl" @ "master"

@[default_target]
lean_lib «LeanBench»
EOF

# Create lean-toolchain
echo "$LEAN_TOOLCHAIN" > lean-toolchain

# Create basic structure
mkdir -p LeanBench
echo "-- Lean Bench" > LeanBench/Basic.lean

echo ""
echo "Updating lake dependencies..."
lake update

echo ""
echo "Downloading Mathlib cache (this may take 10-20 minutes)..."
lake exe cache get

echo ""
echo "Building REPL..."
lake build repl

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Project created at: $PROJECT_DIR"
echo ""
echo "To use this project, set the environment variable:"
echo "  export MATHLIB_PROJECT_PATH=$PROJECT_DIR"
echo ""
echo "You can add this to your shell profile (~/.bashrc or ~/.zshrc)"
