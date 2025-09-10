#!/bin/bash
set -exuo pipefail

echo "Starting time: $(date)"
for script in dataset/mask_*.py; do
    python $script
    echo "Processed $script"
done

python dataset/upload_masked_io.py
echo "Uploaded to HuggingFace Hub"
echo "Ending time: $(date)"
