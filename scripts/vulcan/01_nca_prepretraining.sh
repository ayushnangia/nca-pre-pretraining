#!/bin/bash
# ============================================================
# Phase 1: NCA Pre-Pre-Training on Vulcan (L40S)
#
# Generates NCA trajectories on-the-fly (JAX) and trains a
# Llama transformer with next-token prediction (PyTorch).
# No dataset files needed — data is synthesized each epoch.
#
# Single GPU, batch_size=4 x grad_accum=8 = effective 32
# (matches original batch_size=16 x grad_accum=2)
# ============================================================
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=nca-ppt
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --time=0-12:00
#SBATCH --output=%N-%j-nca-ppt.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -e

# --- Environment ---
module load python/3.11 cuda/12.6 scipy-stack arrow opencv
source "$HOME/nca-ppt-env/bin/activate"

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

# --- Paths ---
CODE_DIR="$HOME/nca-pre-pretraining"   # adjust if code lives elsewhere
SAVE_DIR="$SCRATCH/nca-ppt/checkpoints/nca_prepretraining"

mkdir -p "$SAVE_DIR"

cd "$CODE_DIR"

echo "=== Phase 1: NCA Pre-Pre-Training ==="
echo "Save dir: $SAVE_DIR"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python src/nca_ppt.py \
    --seed 0 \
    --device cuda:0 \
    --grid 12 \
    --patch 2 \
    --num_colors 10 \
    --seq_len 1024 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --num_epochs 100 \
    --warmup 10 \
    --save_dir "$SAVE_DIR" \
    --model_type llama \
    --model_name llama-large \
    --n_layer 24 \
    --n_head 32 \
    --n_embd 2048 \
    --temperature 1e-4 \
    --train_num_rules 16000 \
    --val_num_rules 2000 \
    --train_num_sim 500 \
    --val_num_sim 100 \
    --eval_num_sim 160 \
    --eval_num_rules 160 \
    --dT 1 \
    --log_grad \
    --log_grad_freq 100 \
    --val_freq 500 \
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
    --interval_save \
    --resume \
    --wandb_enable \
    --wandb_name nca-ppt-vulcan \
    --wandb_project nca-prepretraining

echo "=== Phase 1 complete (or wall-time reached — resubmit with --resume) ==="
