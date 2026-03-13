#!/bin/bash
# ============================================================
# Phase 3: Fine-tune on GSM8K (math reasoning)
#
# 10 epochs, lr=1e-5, chain-of-thought traces.
# PREREQUISITE: Phase 2 checkpoint (OWT or math pre-trained)
# ============================================================
#SBATCH --account=aip-FIXME
#SBATCH --job-name=ft-gsm8k
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --time=0-06:00
#SBATCH --output=%N-%j-ft-gsm8k.out

set -e

module load python/3.11 cuda/12.6 scipy-stack arrow opencv
source "$HOME/nca-ppt-env/bin/activate"

export TOKENIZERS_PARALLELISM=false

CODE_DIR="$HOME/nca-pre-pretraining"
SAVE_DIR="$SCRATCH/nca-ppt/checkpoints/ft_gsm8k"

# Phase 2 checkpoint — EDIT THESE
MODEL_PATH="$SCRATCH/nca-ppt/checkpoints/owt_pretraining"
MODEL_FILE="best_model_1.pth"

mkdir -p "$SAVE_DIR"

cd "$CODE_DIR"

echo "=== Phase 3: GSM8K Fine-Tuning (Killarney H100) ==="

python src/language_train.py \
    --seed 0 \
    --device cuda:0 \
    --save_path "$SAVE_DIR" \
    --model_path "$MODEL_PATH" \
    --model_file "$MODEL_FILE" \
    --pretrain 1 \
    --pt_vocab_size 64000 \
    --vocab_size 64000 \
    --reinit_modules none \
    --reinit_layer_idxs 0 24 \
    --seq_len 1024 \
    --lr 1e-5 \
    --epochs 10 \
    --warmup 0.1 \
    --val_freq 14 \
    --patience -1 \
    --log_grad \
    --log_grad_freq 100 \
    --grad_clip 1.0 \
    --grad_clip_enable \
    --batch_size 16 \
    --mixed_precision fp16 \
    --autocast \
    --task gsm8k \
    --grad_accumulation_steps 32 \
    --eval_enable \
    --interval_save \
    --intervals 70 140 \
    --resume \
    --wandb_enable \
    --wandb_name ft-gsm8k-killarney \
    --wandb_project nca-prepretraining

echo "=== Phase 3 (GSM8K) complete ==="
