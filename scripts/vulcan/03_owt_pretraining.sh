#!/bin/bash
# ============================================================
# Phase 2: Language Pre-Training on OpenWebText (Vulcan L40S)
#
# Loads NCA pre-pre-trained checkpoint, re-initializes embeddings
# (--reinit_modules embed none), trains on OpenWebText.
#
# PREREQUISITE: Phase 1 checkpoint + preprocessed OWT data (.bin)
# ============================================================
#SBATCH --account=aip-FIXME
#SBATCH --job-name=owt-pt
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --time=0-24:00
#SBATCH --output=%N-%j-owt-pt.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -e

# --- Environment ---
module load python/3.11 cuda/12.2
source "$HOME/nca-ppt-env/bin/activate"

export TOKENIZERS_PARALLELISM=false

# --- Paths ---
CODE_DIR="$HOME/nca-pre-pretraining"
DATA_DIR="$SCRATCH/nca-ppt/data/owt"
SAVE_DIR="$SCRATCH/nca-ppt/checkpoints/owt_pretraining"

# Phase 1 checkpoint — EDIT THESE after NCA pre-pre-training completes
MODEL_PATH="$SCRATCH/nca-ppt/checkpoints/nca_prepretraining"
MODEL_FILE="best_model_10.pth"   # adjust to your best checkpoint filename

mkdir -p "$SAVE_DIR"

# --- Stage data to fast local storage ---
echo "Staging data to \$SLURM_TMPDIR..."
cp "$DATA_DIR/train.bin" "$SLURM_TMPDIR/"
cp "$DATA_DIR/test.bin" "$SLURM_TMPDIR/" 2>/dev/null || echo "No test.bin, skipping"
# Create validation split symlink (OWT code looks for 'validation' dir)
ln -sf "$SLURM_TMPDIR/test.bin" "$SLURM_TMPDIR/validation.bin" 2>/dev/null || true
echo "Data staged: $(ls -lh $SLURM_TMPDIR/*.bin)"

cd "$CODE_DIR"

echo "=== Phase 2: OpenWebText Pre-Training ==="
echo "Model checkpoint: $MODEL_PATH/$MODEL_FILE"
echo "Data: $SLURM_TMPDIR"
echo "Save dir: $SAVE_DIR"

python src/openwebtext_pt.py \
    --device 0 \
    --seed 5 \
    --save_dir "$SAVE_DIR" \
    --data_dir "$SLURM_TMPDIR" \
    --model_path "$MODEL_PATH" \
    --model_file "$MODEL_FILE" \
    --lr 5e-4 \
    --pretrain 1 \
    --warmup 0.1 \
    --epochs 1 \
    --autocast \
    --mixed_precision fp16 \
    --log_grad 1 \
    --log_grad_freq 100 \
    --grad_clip 1.0 \
    --grad_clip_enable 1 \
    --freeze_modules "" \
    --reinit_modules embed none \
    --reinit_layer_idxs 0 24 \
    --weight_decay 0.0001 \
    --pt_vocab_size 64000 \
    --interval_save \
    --resume \
    --wandb_enable \
    --wandb_name owt-pt-vulcan \
    --wandb_project nca-prepretraining

echo "=== Phase 2 complete ==="
