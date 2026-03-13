#!/bin/bash
# ============================================================
# One-time environment setup for Vulcan (L40S)
# Run this on a login node: bash scripts/vulcan/setup_env.sh
# ============================================================

set -e

ENV_DIR="$HOME/nca-ppt-env"

echo "=== Setting up NCA pre-pre-training environment on Vulcan ==="

module load python/3.11 cuda/12.6 scipy-stack arrow opencv

# Create virtualenv
if [ -d "$ENV_DIR" ]; then
    echo "Environment already exists at $ENV_DIR"
    echo "To recreate, run: rm -rf $ENV_DIR && bash $0"
    exit 1
fi

virtualenv --no-download "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install --no-index --upgrade pip

# --- Core ML packages (from Alliance wheels) ---
pip install --no-index torch torchvision
pip install --no-index transformers datasets tokenizers
pip install --no-index accelerate peft safetensors
pip install --no-index einops optax
pip install --no-index scikit-learn scikit-image scipy pandas
pip install --no-index matplotlib seaborn pillow
pip install --no-index wandb

# --- JAX ecosystem ---
# Try Alliance wheels first; fall back to PyPI if not available
pip install --no-index jax jaxlib 2>/dev/null || {
    echo ">>> JAX wheels not found with --no-index, installing from PyPI..."
    pip install "jax[cuda12]"
}
pip install --no-index flax chex orbax-checkpoint 2>/dev/null || {
    echo ">>> Flax ecosystem wheels not found, installing from PyPI..."
    pip install flax chex orbax-checkpoint
}

# --- Packages unlikely to have Alliance wheels ---
pip install tiktoken
pip install fire
pip install simplejson
pip install math-verify
pip install opencv-python
pip install numba

# --- human_eval (from git, needed for HumanEval evaluation) ---
pip install "git+https://github.com/openai/human-eval.git@6d43fb980f9fee3c892a914eda09951f772ad10d"

echo ""
echo "=== Environment setup complete ==="
echo "Activate with: source $ENV_DIR/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Clone your code to \$HOME or \$PROJECT"
echo "  2. Preprocess data (if needed): sbatch scripts/vulcan/02_preprocess_owt.sh"
echo "  3. Run NCA pre-pre-training: sbatch scripts/vulcan/01_nca_prepretraining.sh"
