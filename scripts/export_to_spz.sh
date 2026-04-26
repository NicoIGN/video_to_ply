#!/bin/bash
set -e

# ======================
# LOAD CONFIG
# ======================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config/config.sh"

EXPORT_DIR="$1"
ROOT_DIR="$2"

# ======================
# CHECKS
# ======================
if [ -z "$EXPORT_DIR" ] || [ -z "$ROOT_DIR" ]; then
  echo "❌ Usage: export_gs.sh <EXPORT_DIR> <ROOT_DIR>"
  exit 1
fi

NERF_ROOT="$ROOT_DIR/nerfacto"

if [ ! -d "$NERF_ROOT" ]; then
  echo "❌ nerfacto folder not found: $NERF_ROOT"
  exit 1
fi

# ======================
# FIND LATEST RUN
# ======================
RUN_DIR=$(ls -dt "$NERF_ROOT"/* 2>/dev/null | head -n 1)

if [ -z "$RUN_DIR" ]; then
  echo "❌ No nerfacto runs found in $NERF_ROOT"
  exit 1
fi

# ======================
# CONFIG
# ======================
CONFIG="$RUN_DIR/config.yml"

if [ ! -f "$CONFIG" ]; then
  echo "❌ config.yml not found:"
  echo "$RUN_DIR"
  exit 1
fi

mkdir -p "$EXPORT_DIR"

echo "────────────────────────────────────────────"
echo "📦 ROOT DIR       : $ROOT_DIR"
echo "📦 NERF ROOT      : $NERF_ROOT"
echo "📦 RUN DIR        : $RUN_DIR"
echo "📄 CONFIG         : $CONFIG"
echo "📁 EXPORT DIR     : $EXPORT_DIR"
echo "────────────────────────────────────────────"

# ======================
# ENV FLAGS
# ======================
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=1

# ======================
# DEBUG MODEL
# ======================
echo "📊 Checking checkpoint..."
ls "$RUN_DIR/nerfstudio_models" || echo "⚠️ No models folder"

# ======================
# EXPORT PLY
# ======================
echo "🚀 Exporting Gaussian Splat (.ply)..."

ns-export gaussian-splat \
  --load-config "$CONFIG" \
  --output-dir "$EXPORT_DIR"

PLY_FILE=$(find "$EXPORT_DIR" -name "*.ply" | head -n 1)

if [ -z "$PLY_FILE" ]; then
  echo "❌ Gaussian PLY export failed"
  exit 1
fi

echo "✅ PLY exported:"
echo "   $PLY_FILE"

# ======================
# OPTIONAL SPZ
# ======================
if command -v supersplat >/dev/null 2>&1; then
  echo "🪄 Converting to SPZ..."

  SPZ_FILE="${PLY_FILE%.ply}.spz"

  supersplat convert \
    "$PLY_FILE" \
    "$SPZ_FILE"

  echo "✅ SPZ exported:"
  echo "   $SPZ_FILE"
else
  echo "ℹ️ SuperSplat CLI not installed"
  echo "   Install from: https://github.com/playcanvas/supersplat"
fi

echo "🎉 Export completed"
