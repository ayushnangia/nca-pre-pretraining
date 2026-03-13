#!/bin/bash
# ============================================================
# Phase 3: Fine-tune on BigBench-Lite (reasoning)
#
# 1 epoch, lr=5e-6.
# PREREQUISITE: Phase 2 checkpoint (OWT pre-trained)
# ============================================================
#SBATCH --account=aip-FIXME
#SBATCH --job-name=ft-bbl
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --time=0-03:00
#SBATCH --output=%N-%j-ft-bbl.out

set -e

module load python/3.11 cuda/12.6 scipy-stack arrow opencv
source "$HOME/nca-ppt-env/bin/activate"

export TOKENIZERS_PARALLELISM=false

CODE_DIR="$HOME/nca-pre-pretraining"
SAVE_DIR="$SCRATCH/nca-ppt/checkpoints/ft_bbl"

# Phase 2 checkpoint — EDIT THESE
MODEL_PATH="$SCRATCH/nca-ppt/checkpoints/owt_pretraining"
MODEL_FILE="best_model_1.pth"

mkdir -p "$SAVE_DIR"

cd "$CODE_DIR"

echo "=== Phase 3: BigBench-Lite Fine-Tuning (Killarney H100) ==="

python src/language_train.py \
    --seed 0 \
    --device cuda:0 \
    --save_path "$SAVE_DIR" \
    --model_path "$MODEL_PATH" \
    --model_file "$MODEL_FILE" \
    --pretrain 1 \
    --n_shot 0 3 \
    --min_samples 100 \
    --max_samples 350 \
    --pt_vocab_size 50257 \
    --vocab_size 50257 \
    --reinit_modules none \
    --reinit_layer_idxs 0 24 \
    --seq_len 1024 \
    --lr 5e-6 \
    --epochs 1 \
    --warmup 0.1 \
    --val_freq 5 \
    --save_freq 5 \
    --patience -1 \
    --log_grad \
    --log_grad_freq 10 \
    --grad_clip 1.0 \
    --grad_clip_enable \
    --batch_size 16 \
    --mixed_precision fp16 \
    --autocast \
    --task bigbench-lite \
    --grad_accumulation_steps 4 \
    --eval_enable \
    --interval_save \
    --intervals 10 20 30 40 50 \
    --resume \
    --wandb_enable \
    --wandb_name ft-bbl-killarney \
    --wandb_project nca-prepretraining

echo "=== Phase 3 (BigBench-Lite) complete ==="
