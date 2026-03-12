#!/bin/bash
# ============================================================
# Evaluation: HumanEval (code generation, pass@k)
# No fine-tuning needed — evaluates directly after pre-training.
# ============================================================
#SBATCH --account=aip-FIXME
#SBATCH --job-name=eval-humaneval
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=0-12:00
#SBATCH --output=%N-%j-eval-humaneval.out

set -e

module load python/3.11 cuda/12.2
source "$HOME/nca-ppt-env/bin/activate"

export TOKENIZERS_PARALLELISM=false

CODE_DIR="$HOME/nca-pre-pretraining"
SAVE_DIR="$SCRATCH/nca-ppt/results/eval_humaneval"

# Pre-trained checkpoint (CodeParrot) — EDIT THESE
MODEL_PATH="$SCRATCH/nca-ppt/checkpoints/codeparrot_pretraining"
MODEL_FILE="best_model_1.pth"

mkdir -p "$SAVE_DIR"

cd "$CODE_DIR"

echo "=== Eval: HumanEval ==="

python src/eval/humaneval.py \
    --seed 0 \
    --device cuda:0 \
    --save_path "$SAVE_DIR" \
    --model_path "$MODEL_PATH" \
    --model_file "$MODEL_FILE" \
    --vocab_size 64000 \
    --seq_len 1024 \
    --temperature 0.4 \
    --top_p 0.95 \
    --passes 64 \
    --eval_passes 1 2 4 8 16 32 \
    --max_len 500 \
    --weight_tying 1 \
    --reinit_modules embed none \
    --mixed_precision fp16 \
    --autocast \
    --resume

echo "=== Eval HumanEval complete ==="
echo "Results: $SAVE_DIR"
