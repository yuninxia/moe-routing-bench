#!/usr/bin/env bash
# Setup environment for PERFT experiments
# Run this once before running run_perft_experiment.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERFT_DIR="${ROOT}/3rdparty/PERFT-MoE"

echo "=============================================="
echo "PERFT Environment Setup"
echo "=============================================="

# Check if venv exists
if [[ ! -d "${ROOT}/.venv" ]]; then
    echo "[ERROR] Virtual environment not found at ${ROOT}/.venv"
    echo "[INFO] Please run: python -m venv .venv && source .venv/bin/activate && pip install -e ."
    exit 1
fi

# Activate venv
source "${ROOT}/.venv/bin/activate"

echo "[INFO] Installing PERFT dependencies..."

# Install additional dependencies for PERFT
pip install fire safetensors tqdm wandb "datasets<3.0" --quiet

# Check transformers version
TRANSFORMERS_VERSION=$(python -c "import transformers; print(transformers.__version__)")
echo "[INFO] Transformers version: $TRANSFORMERS_VERSION"

# Login to HuggingFace if needed
if ! huggingface-cli whoami &>/dev/null; then
    echo "[INFO] Please login to HuggingFace to access OLMoE model:"
    echo "       huggingface-cli login"
    echo ""
    echo "[INFO] Or set HF_TOKEN environment variable"
fi

# Test import
echo "[INFO] Testing PERFT imports..."
cd "$PERFT_DIR"
python -c "
import sys
sys.path.insert(0, '.')
from olmoe_modification.configuration_olmoe import OlmoeAdapterConfig
from olmoe_modification.modeling_olmoe import OlmoeAdapterForCausalLM
print('[OK] PERFT modules imported successfully')
"
cd "$ROOT"

echo ""
echo "=============================================="
echo "Environment setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Login to HuggingFace: huggingface-cli login"
echo "  2. Run experiment:"
echo "     Single GPU:  bash scripts/run_perft_experiment.sh"
echo "     Multi-GPU:   GPU_IDS=0,1,2,3 bash scripts/run_perft_experiment.sh"
echo ""
echo "Quick test (fewer epochs):"
echo "     NUM_EPOCHS=1 EVAL_STEP=40 SAVE_STEP=40 bash scripts/run_perft_experiment.sh"
echo ""
