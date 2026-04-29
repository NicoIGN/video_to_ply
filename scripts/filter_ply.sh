#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Gaussian Splat PLY Cleaner
#
# Required environment variables:
#   PLY_FILE
#   EXPORT_DIR
#   BASENAME
# ============================================================

# ======================
# VALIDATION
# ======================

: "${PLY_FILE:?PLY_FILE is required}"
: "${EXPORT_DIR:?EXPORT_DIR is required}"
: "${BASENAME:?BASENAME is required}"
: "${SCRIPT_DIR:?SCRIPT_DIR is required}"

if [[ ! -f "$PLY_FILE" ]]; then
    echo "❌ PLY file not found: $PLY_FILE"
    exit 1
fi

mkdir -p "$EXPORT_DIR"

# ======================
# NORMALIZE INPUT
# ======================

FINAL_PLY="$EXPORT_DIR/${BASENAME}.ply"

echo "📦 Normalizing output PLY"
echo "   FROM: $PLY_FILE"
echo "   TO  : $FINAL_PLY"

cp -f "$PLY_FILE" "$FINAL_PLY"

if [[ ! -f "$FINAL_PLY" ]]; then
    echo "❌ Failed to create: $FINAL_PLY"
    exit 1
fi

PLY_FILE="$FINAL_PLY"

echo "✅ Final PLY ready: $PLY_FILE"
echo ""

# ======================
# CLEANING LEVELS
# ======================

LEVELS=(
    minimal
    strong
    destructive
)

declare -A NB_NEIGHBORS=(
    [minimal]=24
    [balanced]=32
    [strong]=40
    [aggressive]=48
    [destructive]=64
)

declare -A SOR_PERCENTILE=(
    [minimal]=88
    [balanced]=85
    [strong]=82
    [aggressive]=80
    [destructive]=75
)

declare -A EDGE_PERCENTILE=(
    [minimal]=35
    [balanced]=25
    [strong]=20
    [aggressive]=15
    [destructive]=10
)

declare -A DBSCAN_MIN=(
    [minimal]=20
    [balanced]=30
    [strong]=40
    [aggressive]=50
    [destructive]=80
)

declare -A CENTER_PERC=(
    [minimal]=97
    [balanced]=95
    [strong]=92
    [aggressive]=90
    [destructive]=85
)

# ======================
# RUN CLEANING
# ======================

echo "🧹 Input PLY: $PLY_FILE"
echo ""

for LEVEL in "${LEVELS[@]}"; do
    echo "🚀 =============================="
    echo "🚀 CLEAN LEVEL: $LEVEL"
    echo "🚀 =============================="

    CLEANED_PLY="$EXPORT_DIR/${BASENAME}_${LEVEL}.ply"
    rm -f "$CLEANED_PLY"

    echo "⚙️ Parameters"
    echo "   nb-neighbors      : ${NB_NEIGHBORS[$LEVEL]}"
    echo "   sor-percentile    : ${SOR_PERCENTILE[$LEVEL]}"
    echo "   edge-percentile   : ${EDGE_PERCENTILE[$LEVEL]}"
    echo "   dbscan-min        : ${DBSCAN_MIN[$LEVEL]}"
    echo "   center-percentile : ${CENTER_PERC[$LEVEL]}"


    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


    python3 "$SCRIPT_DIR/clean_gaussian_ply.py" \
        --nb-neighbors "${NB_NEIGHBORS[$LEVEL]}" \
        --sor-percentile "${SOR_PERCENTILE[$LEVEL]}" \
        --edge-percentile "${EDGE_PERCENTILE[$LEVEL]}" \
        --dbscan-min-points "${DBSCAN_MIN[$LEVEL]}" \
        --center-percentile "${CENTER_PERC[$LEVEL]}" \
        --recenter \
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
