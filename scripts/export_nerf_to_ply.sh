#!/bin/bash
set -e

# ======================
# LOAD CONFIG
# ======================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config/config.sh"

# ======================
# CHECKS
# ======================
if [ -z "$EXPORT_DIR" ] || [ -z "$ROOT_DIR" ]; then
  echo "❌ Usage: export_to_ply.sh"
  exit 1
fi

NERF_ROOT="$ROOT_DIR/nerfacto"

if [ ! -d "$NERF_ROOT" ]; then
  echo "❌ nerfacto folder not found: $NERF_ROOT"
  exit 1
fi

# ======================
# FIND LATEST RUN (FIXED)
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
  echo "❌ config.yml not found in:"
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
export PYTORCH_ENABLE_MPS_FALLBACK=1

# ======================
# DEBUG MODEL
# ======================
echo "📊 Checking checkpoint..."
ls "$RUN_DIR/nerfstudio_models" || echo "⚠️ No models folder"

# ======================
# EXPORT
# ======================
echo "🚀 Export settings:"
echo "   - points: $NUM_POINTS"
echo "   - normals: $NORMAL_METHOD"
echo "   - outliers: $REMOVE_OUTLIERS"

ns-export pointcloud \
  --load-config "$CONFIG" \
  --output-dir "$EXPORT_DIR" \
  --num-points "$NUM_POINTS" \
  --normal-method "$NORMAL_METHOD" \
  --remove-outliers "$REMOVE_OUTLIERS"

echo "✅ Export done"
