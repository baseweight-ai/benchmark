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

ENV_YML="${REPO_ROOT}/environment.yml"
STAMP_FILE="${REPO_ROOT}/.setup_stamp"

_ENV_HASH=$(md5sum "$ENV_YML" | awk '{print $1}')
_ENV_LIST=$(conda env list 2>/dev/null)
_ENV_EXISTS=$(echo "$_ENV_LIST" | grep -qE "^${CONDA_ENV}[[:space:]]" && echo true || echo false)

if [[ -f "$STAMP_FILE" ]] && [[ "$(cat "$STAMP_FILE")" == "$_ENV_HASH" ]] && [[ "$_ENV_EXISTS" == true ]]; then
    echo "Conda env ${CONDA_ENV} is current (environment.yml unchanged) — skipping install."
    conda activate "$CONDA_ENV"
else
    if [[ "$_ENV_EXISTS" == true ]]; then
        echo "Updating conda env ${CONDA_ENV}..."
        conda env update -n "$CONDA_ENV" -f "${ENV_YML}" --prune
    else
        echo "Creating conda env ${CONDA_ENV}..."
        conda env create -n "$CONDA_ENV" -f "${ENV_YML}"
    fi
    conda activate "$CONDA_ENV"
    # torchao conflicts with torch==2.5.1 (torch.int1 missing); unsloth doesn't need it
    pip uninstall -y torchao 2>/dev/null || true
    echo "$_ENV_HASH" > "$STAMP_FILE"
fi

# ---------------------------------------------------------------------------
# Verification (always runs — catches broken envs even on cache-hit)
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

# ---------------------------------------------------------------------------
# API key check
# ---------------------------------------------------------------------------
_api_key="${OPENAI_API_KEY:-}"
if [[ -z "$_api_key" ]] && [[ -f "$ENV_FILE" ]]; then
    _api_key=$(grep -E "^OPENAI_API_KEY=" "$ENV_FILE" | tail -1 | cut -d= -f2- || true)
fi
if [[ -z "$_api_key" ]]; then
    echo "  WARN  OPENAI_API_KEY not set — add it to ${ENV_FILE} before running eval-api or train-api"
elif [[ "$_api_key" == "sk-..." ]] || [[ ${#_api_key} -lt 20 ]]; then
    echo "  WARN  OPENAI_API_KEY looks like a placeholder — set a real key in ${ENV_FILE}"
else
    echo "  OK    OPENAI_API_KEY  ${_api_key:0:12}…  (${#_api_key} chars)"
fi

# ---------------------------------------------------------------------------
# Lock file — reproducible environment snapshot
# ---------------------------------------------------------------------------
conda env export -n "${CONDA_ENV}" > "${REPO_ROOT}/environment.lock.yml"
echo "  Lock file written to environment.lock.yml"

echo "=== Setup complete ==="
