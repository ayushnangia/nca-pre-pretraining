#!/bin/bash
# ============================================================
# TEST: NCA Pre-Pre-Training — short run to verify everything works
# 2 epochs, 100 rules, ~15 min
# ============================================================
#SBATCH --account=aip-FIXME
#SBATCH --job-name=test-nca
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --time=0-00:30
#SBATCH --output=%N-%j-test-nca.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -e

# --- Environment ---
module load python/3.11 cuda/12.6 scipy-stack arrow opencv
source "$HOME/nca-ppt-env/bin/activate"

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# --- Paths ---
CODE_DIR="$HOME/nca-pre-pretraining"
SAVE_DIR="$SCRATCH/nca-ppt/checkpoints/test_nca"

mkdir -p "$SAVE_DIR"

cd "$CODE_DIR"

echo "=== TEST: NCA Pre-Pre-Training (Killarney H100) ==="
echo "Save dir: $SAVE_DIR"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
nvidia-smi

python src/nca_ppt.py \
    --seed 0 \
    --device cuda:0 \
    --grid 12 \
    --patch 2 \
    --num_colors 10 \
    --seq_len 1024 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --num_epochs 2 \
    --warmup 1 \
    --save_dir "$SAVE_DIR" \
    --model_type llama \
    --model_name llama-large \
    --n_layer 24 \
    --n_head 32 \
    --n_embd 2048 \
    --temperature 1e-4 \
    --train_num_rules 100 \
    --val_num_rules 20 \
    --train_num_sim 50 \
    --val_num_sim 10 \
    --eval_num_sim 10 \
    --eval_num_rules 10 \
    --dT 1 \
    --val_freq 50 \
    --autocast \
    --mixed_precision bf16 \
    --token \
    --vocab_size 64000 \
    --filter_rules \
    --filter_rules_threshold 0.5 \
    --filter_rules_upper_bound 1.0 \
    --filter_rules_mode gzip \
    --init_rollout_steps 10 \
    --generate_train \
    --generate_rules 1 \
    --grad_accumulation_steps 8 \
    --num_workers 0 \
    --interval_save

echo ""
echo "=== TEST PASSED ==="
echo "Checkpoints:"
ls -lh "$SAVE_DIR/"
