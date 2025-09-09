#!/bin/bash
set -exuo pipefail

echo "Starting time: $(date)"
for script in datasets/mask_*.py; do
    python $script
    echo "Processed $script"
done

python datasets/upload_masked_io.py
echo "Uploaded to HuggingFace Hub"
echo "Ending time: $(date)"
