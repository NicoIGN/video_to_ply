#!/bin/bash

set -e

echo "Starting rclone configuration..."

rclone config

echo "Done. You should now have a 'gdrive' remote configured."
