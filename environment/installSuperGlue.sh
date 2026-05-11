#!/bin/bash
set -e

SITE_PACKAGES=$(conda run -n gsplat python -c "import site; print(site.getsitepackages()[0])")
TARGET="$SITE_PACKAGES/SuperGluePretrainedNetwork"

echo "SITE_PACKAGES: $SITE_PACKAGES"
echo "TARGET: $TARGET"

# ======================
# CHECK VALID INSTALL
# ======================
VALID=false

if [ -d "$TARGET" ]; then
    if [ -f "$TARGET/__init__.py" ]; then
        VALID=true
    else
        echo "⚠️ Invalid install detected (missing __init__.py)"
    fi
else
    echo "❌ SuperGluePretrainedNetwork not found"
fi

# ======================
# CLEAN IF INVALID
# ======================
if [ "$VALID" = false ]; then
    echo "🧹 Cleaning broken installation..."
    rm -rf "$TARGET"

    echo "📥 Reinstalling SuperGluePretrainedNetwork..."

    git clone --depth 1 \
        https://github.com/magicleap/SuperGluePretrainedNetwork.git \
        "$TARGET"

    # sanity check
    if [ ! -f "$TARGET/__init__.py" ]; then
        echo "⚠️ Still missing __init__.py → adding minimal one"
        touch "$TARGET/__init__.py"
    fi

    echo "✅ SuperGluePretrainedNetwork reinstalled"
else
    echo "✅ SuperGluePretrainedNetwork already valid"
fi
