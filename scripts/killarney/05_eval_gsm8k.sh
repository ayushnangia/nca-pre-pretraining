#!/bin/bash
# ============================================================
# Evaluation: GSM8K (math reasoning, pass@k)
# ============================================================
#SBATCH --account=aip-FIXME
#SBATCH --job-name=eval-gsm8k
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=0-03:00
#SBATCH --output=%N-%j-eval-gsm8k.out

set -e

module load python/3.11 cuda/12.6 scipy-stack arrow opencv
source "$HOME/nca-ppt-env/bin/activate"

export TOKENIZERS_PARALLELISM=false

CODE_DIR="$HOME/nca-pre-pretraining"
SAVE_DIR="$SCRATCH/nca-ppt/results/eval_gsm8k"

# Fine-tuned checkpoint — EDIT THESE
MODEL_PATH="$SCRATCH/nca-ppt/checkpoints/ft_gsm8k"
MODEL_FILE="best_model_10.pth"

mkdir -p "$SAVE_DIR"

cd "$CODE_DIR"

echo "=== Eval: GSM8K ==="

python src/eval/gsm8k.py \
    --seed 0 \
    --device cuda:0 \
    --save_path "$SAVE_DIR" \
    --model_path "$MODEL_PATH" \
    --model_file "$MODEL_FILE" \
    --vocab_size 50257 \
    --seq_len 1024 \
    --temperature 0.6 \
    --top_p 1.0 \
    --passes 32 \
    --max_len 250 \
    --stop_string "####" \
    --mixed_precision fp16 \
    --autocast \
    --resume

echo "=== Eval GSM8K complete ==="
echo "Results: $SAVE_DIR"
