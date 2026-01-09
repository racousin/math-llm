#!/bin/bash
#
# Setup Lean server with Mathlib and REPL.

set -e

# =============================================================================
# VERSIONS - Must match src/math_llm/lean_server.py
# =============================================================================
MATHLIB_VERSION="v4.25.2"
REPL_VERSION="v4.25.0"
LEAN_TOOLCHAIN="leanprover/lean4:v4.25.2"
# =============================================================================

PROJECT_DIR="${MATHLIB_PROJECT_PATH:-$HOME/.lean-bench}"

echo "=================================="
echo "Lean Server Setup"
echo "=================================="
echo "Mathlib: $MATHLIB_VERSION"
echo "REPL: $REPL_VERSION"
echo "Toolchain: $LEAN_TOOLCHAIN"
echo "Project: $PROJECT_DIR"
echo ""

# Check if already set up
if [ -f "$PROJECT_DIR/lean-toolchain" ] && [ -f "$PROJECT_DIR/.lake/packages/repl/lakefile.lean" ]; then
    CURRENT_TOOLCHAIN=$(cat "$PROJECT_DIR/lean-toolchain")
    if [ "$CURRENT_TOOLCHAIN" = "$LEAN_TOOLCHAIN" ]; then
        echo "Already set up! Skipping download."
        echo ""
        echo "To force reinstall, run: rm -rf $PROJECT_DIR"
        exit 0
    fi
fi

# Check for elan
if ! command -v elan &> /dev/null; then
    echo "Installing elan..."
    curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
    source ~/.elan/env
fi

export PATH="$HOME/.elan/bin:$PATH"

# Clean existing project
rm -rf "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Set toolchain FIRST
echo "$LEAN_TOOLCHAIN" > lean-toolchain

# Create lakefile
cat > lakefile.lean << EOF
import Lake
open Lake DSL

package «lean_bench»

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "$MATHLIB_VERSION"

require «repl» from git
  "https://github.com/leanprover-community/repl" @ "$REPL_VERSION"

@[default_target]
lean_lib «LeanBench»
EOF

mkdir -p LeanBench
echo "-- Lean Bench" > LeanBench/Basic.lean

echo "Running lake update..."
lake update

echo "Downloading cache..."
lake exe cache get

echo "Building REPL..."
lake build repl

echo ""
echo "=================================="
echo "Done! Set: export MATHLIB_PROJECT_PATH=$PROJECT_DIR"
echo "=================================="
