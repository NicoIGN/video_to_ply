#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Gaussian Splat PLY Cleaner (SCALE-INVARIANT PIPELINE)
# ============================================================

: "${PLY_FILE:?PLY_FILE is required}"
: "${EXPORT_DIR:?EXPORT_DIR is required}"
: "${BASENAME:?BASENAME is required}"

SKIP_FILTER="${SKIP_FILTER:-false}"

# 🔥 INPUT utilisateur (env)
REQUESTED_LEVELS_STR="${LEVELS:-minimal balanced}"

# parsing safe
read -r -a REQUESTED_LEVELS <<< "$REQUESTED_LEVELS_STR"

echo "🔎 Requested levels raw : $REQUESTED_LEVELS_STR"
echo "🔎 Parsed levels        : ${REQUESTED_LEVELS[*]}"

# ======================
# INPUT CHECK
# ======================
if [[ ! -f "$PLY_FILE" ]]; then
    echo "❌ PLY file not found: $PLY_FILE"
    exit 1
fi

mkdir -p "$EXPORT_DIR"

FINAL_PLY="$EXPORT_DIR/${BASENAME}.ply"

echo "📦 Normalizing output PLY"
echo "   FROM: $PLY_FILE"
echo "   TO  : $FINAL_PLY"

cp -f "$PLY_FILE" "$FINAL_PLY"
PLY_FILE="$FINAL_PLY"

if [[ "$SKIP_FILTER" == "true" ]]; then
  echo "⏩ Skipping filter (config)"
  exit 0
fi

echo "✅ Input ready: $PLY_FILE"
echo ""

# ======================
# CLEANING LEVELS
# ======================

# 🔒 niveaux disponibles (interne)
AVAILABLE_LEVELS=(
    minimal
    balanced
    strong
    aggressive
)

# validation
for LEVEL in "${REQUESTED_LEVELS[@]}"; do
    if [[ ! " ${AVAILABLE_LEVELS[*]} " =~ " ${LEVEL} " ]]; then
        echo "❌ Invalid LEVEL: $LEVEL"
        echo "Allowed levels: ${AVAILABLE_LEVELS[*]}"
        exit 1
    fi
done

# KNN complexity
declare -A K=(
    [minimal]=16
    [balanced]=24
    [strong]=32
    [aggressive]=48
)

# STRUCTURE FILTER
declare -A KEEP_PCT=(
    [minimal]=10
    [balanced]=25
    [strong]=40
    [aggressive]=60
)

# ISOLATION FILTER
declare -A RADIUS_MULT=(
    [minimal]=2.2
    [balanced]=1.8
    [strong]=1.5
    [aggressive]=1.2
)

declare -A MIN_NEIGHBORS=(
    [minimal]=5
    [balanced]=7
    [strong]=10
    [aggressive]=12
)

# SPLAT SIZE FILTER
declare -A MAX_RELATIVE_SCALE=(
    [minimal]=0
    [balanced]=8.0
    [strong]=6.0
    [aggressive]=4.0
)

# ======================
# RUN CLEANING
# ======================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for LEVEL in "${REQUESTED_LEVELS[@]}"; do

    # sécurité set -u
    if [[ -z "${K[$LEVEL]:-}" ]]; then
        echo "❌ Internal error: missing config for level '$LEVEL'"
        exit 1
    fi

    echo ""
    echo "🚀 =============================="
    echo "🚀 CLEAN LEVEL: $LEVEL"
    echo "🚀 =============================="

    CLEANED_PLY="$EXPORT_DIR/${BASENAME}_${LEVEL}.ply"
    rm -f "$CLEANED_PLY"

    echo "⚙️ Parameters"
    echo "   k                  : ${K[$LEVEL]}"
    echo "   keep-percentile    : ${KEEP_PCT[$LEVEL]}"
    echo "   radius-mult        : ${RADIUS_MULT[$LEVEL]}"
    echo "   min-neighbors      : ${MIN_NEIGHBORS[$LEVEL]}"
    echo "   max-relative-scale : ${MAX_RELATIVE_SCALE[$LEVEL]}"

    python3 "$SCRIPT_DIR/clean_gaussian_ply.py" \
        "$PLY_FILE" \
        "$CLEANED_PLY" \
        --k "${K[$LEVEL]}" \
        --keep-percentile "${KEEP_PCT[$LEVEL]}" \
        --radius-mult "${RADIUS_MULT[$LEVEL]}" \
        --min-neighbors "${MIN_NEIGHBORS[$LEVEL]}" \
        --max-relative-scale "${MAX_RELATIVE_SCALE[$LEVEL]}"

    if [[ ! -f "$CLEANED_PLY" ]]; then
        echo "❌ FAILED: $LEVEL"
        exit 1
    fi

    echo "✅ DONE → $CLEANED_PLY"
done

echo ""
echo "🎉 All cleaning levels completed successfully."
