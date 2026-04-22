#!/bin/bash
set -e

# ======================
# LOAD CONFIG (CRITICAL)
# ======================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config/config.sh"

EXPORT_DIR="$1"
OUTPUT_DIR="$2"

# ======================
# FIND CONFIG SAFELY
# ======================
CONFIG=$(find "$OUTPUT_DIR" -type f -name "config.yml" | sort | tail -n 1)

# ======================
# CHECKS
# ======================
if [ -z "$CONFIG" ]; then
  echo "❌ No config.yml found in $OUTPUT_DIR"
  exit 1
fi

mkdir -p "$EXPORT_DIR"

echo "📦 Using config: $CONFIG"
echo "📁 Export dir: $EXPORT_DIR"

# ======================
# EXPORT
# ======================
ns-export pointcloud \
  --load-config "$CONFIG" \
  --output-dir "$EXPORT_DIR" \
  --normal-method open3d
