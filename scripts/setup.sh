#!/usr/bin/env bash
# Environment setup — safe to re-run at any time.
# Assumes NVIDIA GPU with CUDA. No hardware detection is performed.
set -euo pipefail

CONDA_ENV="baseweight-benchmark"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

echo "=== Baseweight Benchmark Setup ==="
echo "Repo: ${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Conda environment
# ---------------------------------------------------------------------------
eval "$(conda shell.bash hook)"
if conda env list | grep -qE "^${CONDA_ENV}[[:space:]]"; then
    echo "Updating conda env ${CONDA_ENV}..."
    conda env update -n "$CONDA_ENV" -f "${REPO_ROOT}/environment.yml" --prune -q
else
    echo "Creating conda env ${CONDA_ENV}..."
    conda env create -n "$CONDA_ENV" -f "${REPO_ROOT}/environment.yml" -q
fi
conda activate "$CONDA_ENV"

# torchao conflicts with torch==2.5.1 (torch.int1 missing); unsloth doesn't need it
pip uninstall -y torchao 2>/dev/null || true

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
echo "Verifying install..."
python - <<'PYEOF'
import importlib.metadata, sys, torch

errors = []

try:
    vllm_ver = importlib.metadata.version("vllm")
except importlib.metadata.PackageNotFoundError:
    errors.append("vllm not installed")
    vllm_ver = "missing"

if "+cpu" in torch.__version__:
    errors.append(f"CPU torch installed — expected CUDA build: {torch.__version__}")

try:
    unsloth_ver = importlib.metadata.version("unsloth")
except importlib.metadata.PackageNotFoundError:
    errors.append("unsloth not installed")
    unsloth_ver = "missing"

if errors:
    print("SETUP ERRORS:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)

print(f"  torch=={torch.__version__}  vllm=={vllm_ver}  unsloth=={unsloth_ver}")

if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU: {p.name}  VRAM: {p.total_memory // 1024**3} GB")
else:
    print("  WARNING: CUDA not available to torch — verify this is an NVIDIA GPU pod")
PYEOF

echo ""
echo "=== Setup complete ==="
