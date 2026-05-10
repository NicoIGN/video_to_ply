SITE_PACKAGES=$(conda run -n gsplat python -c "import site; print(site.getsitepackages()[0])")
TARGET="$SITE_PACKAGES/SuperGluePretrainedNetwork"
echo "SITE_PACKAGES: $SITE_PACKAGES"

if [ ! -d "$TARGET" ]; then
    git clone --depth 1 \
        https://github.com/magicleap/SuperGluePretrainedNetwork.git \
        "$TARGET"
else
    echo "SuperGluePretrainedNetwork already installed."
fi
