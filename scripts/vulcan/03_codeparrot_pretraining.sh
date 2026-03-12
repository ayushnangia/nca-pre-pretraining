#!/bin/bash
# ============================================================
# Phase 2 (alt): Language Pre-Training on CodeParrot (Vulcan L40S)
#
# Alternative to OWT — for code domain downstream tasks.
# Uses streaming HF dataset (no preprocessing needed).
# ============================================================
#SBATCH --account=aip-FIXME
#SBATCH --job-name=code-pt
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --time=0-24:00
#SBATCH --output=%N-%j-code-pt.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -e

module load python/3.11 cuda/12.2
source "$HOME/nca-ppt-env/bin/activate"

export TOKENIZERS_PARALLELISM=false

CODE_DIR="$HOME/nca-pre-pretraining"
SAVE_DIR="$SCRATCH/nca-ppt/checkpoints/codeparrot_pretraining"

# Phase 1 checkpoint — EDIT THESE
MODEL_PATH="$SCRATCH/nca-ppt/checkpoints/nca_prepretraining"
MODEL_FILE="best_model_10.pth"

mkdir -p "$SAVE_DIR"

cd "$CODE_DIR"

echo "=== Phase 2: CodeParrot Pre-Training ==="

python src/language_train.py \
    --seed 5 \
    --device cuda:0 \
    --save_path "$SAVE_DIR" \
    --model_path "$MODEL_PATH" \
    --model_file "$MODEL_FILE" \
    --pretrain 1 \
    --pt_vocab_size 64000 \
    --reinit_modules embed none \
    --seq_len 1024 \
    --lr 2e-4 \
    --epochs 1 \
    --warmup 750 \
    --val_freq 3000 \
    --save_freq 500 \
    --patience -1 \
    --log_grad \
    --log_grad_freq 100 \
    --grad_clip 1.0 \
    --grad_clip_enable \
    --batch_size 16 \
    --mixed_precision fp16 \
    --autocast \
    --task full-codeparrot \
    --eval_enable \
    --grad_accumulation_steps 32 \
    --resume \
    --wandb_enable \
    --wandb_name code-pt-vulcan \
    --wandb_project nca-prepretraining

echo "=== Phase 2 (CodeParrot) complete ==="
