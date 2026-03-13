#!/bin/bash
# ============================================================
# One-time environment setup for Rorqual (H100)
# Run on login node: bash scripts/rorqual/setup_env.sh
# ============================================================

set -e

ENV_DIR="$HOME/nca-ppt-env"

echo "=== Setting up NCA pre-pre-training environment on Rorqual ==="

module load python/3.11 cuda/12.6 scipy-stack arrow opencv

if [ -d "$ENV_DIR" ]; then
    echo "Environment already exists at $ENV_DIR"
    echo "To recreate, run: rm -rf $ENV_DIR && bash $0"
    exit 1
fi

virtualenv --no-download "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install --no-index --upgrade pip

pip install --no-index torch torchvision
pip install --no-index transformers datasets tokenizers
pip install --no-index accelerate peft safetensors
pip install --no-index einops optax
pip install --no-index scikit-learn scikit-image scipy pandas
pip install --no-index matplotlib seaborn pillow
pip install --no-index wandb

pip install --no-index jax jaxlib 2>/dev/null || pip install "jax[cuda12]"
pip install --no-index flax chex orbax-checkpoint 2>/dev/null || pip install flax chex orbax-checkpoint

pip install tiktoken rorquale simplejson math-verify numba
pip install --no-build-isolation --no-deps \
    "git+https://github.com/openai/human-eval.git@6d43fb980f9fee3c892a914eda09951f772ad10d"

echo "=== Done. Activate with: source $ENV_DIR/bin/activate ==="
