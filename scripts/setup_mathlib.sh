#!/bin/bash
# Setup Mathlib project for math-llm
# Run once to download and build Mathlib cache

set -e

# =============================================================================
# VERSION CONFIGURATION
# =============================================================================
# IMPORTANT: These versions must match src/math_llm/config.py
# The Lean/Mathlib version affects LLM training - changing requires retraining!
#
# To update: change both here AND in src/math_llm/config.py
# =============================================================================
MATHLIB_VERSION="v4.25.2"
LEAN_TOOLCHAIN="leanprover/lean4:v4.25.2"

MATHLIB_DIR="${MATHLIB_PROJECT_PATH:-$HOME/.math-llm/mathlib-project}"

echo "Setting up Mathlib project at: $MATHLIB_DIR"
echo "Mathlib version: $MATHLIB_VERSION"
echo "Lean toolchain: $LEAN_TOOLCHAIN"
echo ""

# Create directory
mkdir -p "$MATHLIB_DIR"
cd "$MATHLIB_DIR"

# Create lakefile.lean with pinned Mathlib version
cat > lakefile.lean << EOF
import Lake
open Lake DSL

package «math_llm_workspace» where
  leanOptions := #[
    ⟨\`pp.unicode.fun, true⟩,
    ⟨\`autoImplicit, false⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "$MATHLIB_VERSION"

@[default_target]
lean_lib «MathLLM» where
  globs := #[.submodules \`MathLLM]
EOF

# Set lean-toolchain to pinned version (not fetched from master!)
echo "$LEAN_TOOLCHAIN" > lean-toolchain
echo "Using toolchain: $LEAN_TOOLCHAIN"

# Create source directory
mkdir -p MathLLM
echo "-- Math LLM workspace" > MathLLM/Basic.lean

# Download Mathlib cache (much faster than building)
echo "Downloading Mathlib cache..."
lake exe cache get

# Build
echo "Building project..."
lake build

echo ""
echo "Setup complete!"
echo "Add to your shell config:"
echo "  export MATHLIB_PROJECT_PATH=$MATHLIB_DIR"
