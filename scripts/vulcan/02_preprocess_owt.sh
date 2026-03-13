#!/bin/bash
# ============================================================
# Data Preprocessing: Tokenize OpenWebText to .bin files
#
# Run BEFORE Phase 2 (language pre-training on OWT).
# Produces train.bin and test.bin in $SCRATCH/nca-ppt/data/owt/
# This is a CPU job — no GPU needed.
# ============================================================
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=preprocess-owt
#SBATCH --cpus-per-task=32
#SBATCH --mem=64000M
#SBATCH --time=0-06:00
#SBATCH --output=%N-%j-preprocess.out

set -e

module load python/3.11 scipy-stack arrow
source "$HOME/nca-ppt-env/bin/activate"

CODE_DIR="$HOME/nca-pre-pretraining"
DATA_DIR="$SCRATCH/nca-ppt/data/owt"

mkdir -p "$DATA_DIR"

cd "$CODE_DIR"

echo "=== Preprocessing OpenWebText ==="
echo "Output dir: $DATA_DIR"

# The preprocess script writes .bin files to src/datasets/ by default.
# We'll run it, then move the outputs to our data directory.
python src/datasets/preprocess.py

# Move generated .bin files to data directory
mv src/datasets/train.bin "$DATA_DIR/" 2>/dev/null || true
mv src/datasets/test.bin "$DATA_DIR/" 2>/dev/null || true

echo "=== Preprocessing complete ==="
echo "Files in $DATA_DIR:"
ls -lh "$DATA_DIR/"
