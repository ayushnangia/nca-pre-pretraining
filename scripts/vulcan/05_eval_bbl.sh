#!/bin/bash
# ============================================================
# Evaluation: BigBench-Lite (reasoning, pass@k)
# ============================================================
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=eval-bbl
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=0-06:00
#SBATCH --output=%N-%j-eval-bbl.out

set -e

module load python/3.11 cuda/12.6 scipy-stack arrow opencv
source "$HOME/nca-ppt-env/bin/activate"

export TOKENIZERS_PARALLELISM=false

CODE_DIR="$HOME/nca-pre-pretraining"
SAVE_DIR="$SCRATCH/nca-ppt/results/eval_bbl"

# Fine-tuned checkpoint — EDIT THESE
MODEL_PATH="$SCRATCH/nca-ppt/checkpoints/ft_bbl"
MODEL_FILE="best_model_1.pth"

FEW_SHOT_PROMPTS="src/eval/bbl_prompts.json"

mkdir -p "$SAVE_DIR"

cd "$CODE_DIR"

echo "=== Eval: BigBench-Lite ==="

python src/eval/bigbench.py \
    --seed 0 \
    --device cuda:0 \
    --save_path "$SAVE_DIR" \
    --model_path "$MODEL_PATH" \
    --model_file "$MODEL_FILE" \
    --vocab_size 50257 \
    --seq_len 1024 \
    --temperature 0.4 \
    --top_p 0.95 \
    --passes 64 \
    --eval_passes 1 2 4 8 16 32 \
    --max_len 35 \
    --min_samples 100 \
    --max_per_task 350 \
    --weight_tying 0 \
    --reinit_modules embed none \
    --mixed_precision fp16 \
    --autocast \
    --few_shot_prompts_path "$FEW_SHOT_PROMPTS" \
    --resume

echo "=== Eval BigBench-Lite complete ==="
echo "Results: $SAVE_DIR"
