#!/bin/bash
# ============================================================
# Data Preprocessing: Tokenize OpenWebText to .bin files
# CPU job — no GPU needed.
# ============================================================
#SBATCH --account=aip-FIXME
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
python src/datasets/preprocess.py

mv src/datasets/train.bin "$DATA_DIR/" 2>/dev/null || true
mv src/datasets/test.bin "$DATA_DIR/" 2>/dev/null || true

echo "=== Done ==="
ls -lh "$DATA_DIR/"
