#!/bin/bash

set -e

DEST="./downloads"
REMOTE="gdrive:MyDrive/output"

mkdir -p "$DEST"

echo "Downloading from Drive..."

rclone copy "$REMOTE" "$DEST" --progress

echo "Done → $DEST"
