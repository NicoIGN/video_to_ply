#!/bin/bash

set -e

# ======================
# LOAD CONFIG (CRITICAL)
# ======================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config/config.sh"

FILE=""

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --file) FILE="$2"; shift ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

if [ -z "$FILE" ]; then
  echo "Usage: $0 --file <path>"
  exit 1
fi

if [ ! -f "$FILE" ]; then
  echo "File not found: $FILE"
  exit 1
fi

REMOTE="gdrive:$CONDA_ENV_NAME/input"

echo "Uploading $FILE → $REMOTE"

rclone copy "$FILE" "$REMOTE" --progress

echo "Done."
