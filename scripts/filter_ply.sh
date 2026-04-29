#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Gaussian Splat PLY Cleaner (UPDATED PIPELINE)
# ============================================================

: "${PLY_FILE:?PLY_FILE is required}"
: "${EXPORT_DIR:?EXPORT_DIR is required}"
: "${BASENAME:?BASENAME is required}"

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

echo "✅ Input ready: $PLY_FILE"
echo ""

# ======================
# CLEANING LEVELS (SIMPLIFIED MODEL)
# ======================

LEVELS=(
    minimal
    balanced
    strong
    aggressive
)

declare -A NB_NEIGHBORS=(
    [minimal]=12
    [balanced]=16
    [strong]=20
    [aggressive]=24
)

declare -A OUTLIER_RATIO=(
    [minimal]=4.0
    [balanced]=3.0
    [strong]=2.5
    [aggressive]=2.0
)

declare -A CENTER_PERC=(
    [minimal]=99
    [balanced]=98
    [strong]=97
    [aggressive]=95
)

# ======================
# RUN CLEANING
# ======================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for LEVEL in "${LEVELS[@]}"; do
    echo "🚀 =============================="
    echo "🚀 CLEAN LEVEL: $LEVEL"
    echo "🚀 =============================="

    CLEANED_PLY="$EXPORT_DIR/${BASENAME}_${LEVEL}.ply"
    rm -f "$CLEANED_PLY"

    echo "⚙️ Parameters"
    echo "   nb-neighbors   : ${NB_NEIGHBORS[$LEVEL]}"
    echo "   outlier-ratio  : ${OUTLIER_RATIO[$LEVEL]}"
    echo "   center-percent : ${CENTER_PERC[$LEVEL]}"

    python3 "$SCRIPT_DIR/clean_gaussian_ply.py" \
        --nb-neighbors "${NB_NEIGHBORS[$LEVEL]}" \
        --outlier-ratio "${OUTLIER_RATIO[$LEVEL]}" \
        --center-percentile "${CENTER_PERC[$LEVEL]}" \
        "$PLY_FILE" \
        "$CLEANED_PLY"

    if [[ ! -f "$CLEANED_PLY" ]]; then
        echo "❌ FAILED: $LEVEL"
        exit 1
    fi

    echo "✅ DONE → $CLEANED_PLY"
    echo ""
done

echo "🎉 All cleaning levels completed successfully."
