#!/bin/bash
set -e

# This is the entrypoint for the custom image trainer Docker container.
# It simply passes all command-line arguments to the Python training script.

echo "[run_image_trainer.sh] Starting custom trainer with args: $@"
python3 /workspace/scripts/image_trainer.py "$@" 