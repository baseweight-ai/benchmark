#!/usr/bin/env bash
# Pipeline orchestrator — safe to re-run.
#
# Usage:
#   ./run.sh [STEP FLAGS] [OPTIONS]
#
# Step flags (select which scripts to run; defaults to --setup --download --prepare --train-local --eval-local):
#   --setup          Run scripts/setup.sh (env + hardware-specific packages)
#   --download       Run download_data.py
#   --prepare        Run prepare_datasets.py
#   --train-local    Run train_local.py (local QLoRA training)
#   --train-api      Run train_api.py   (OpenAI SFT training — idempotent, never reruns unless --force)
#   --train          Both --train-local and --train-api
#   --eval-local     Run eval_local.py  (vLLM, fine-tuned models)
#   --eval-api       Run eval_api.py    (frontier API eval; api-sft requires --train-api first)
#   --eval           Both --eval-local and --eval-api
#   --classify       Run classify_errors.py (error classification + summaries)
#   --dashboard      Run generate_dashboard_data.py
#   --all            All of the above
#
# Options:
#   --smoke-test     Tiny datasets/model; exercises the same code paths as prod
#   --cpu            Skip train-local and eval-local (GPU steps); run API steps only
#   --task TASK      Task ID or 'all' (default: all)
#   --model MODEL    Model ID or 'all' (default: qwen3-8b; qwen2.5-0.5b with --smoke-test)
#   --clean          Delete prior outputs for selected steps/model/task, then run
#   --dry-run        Pass --dry-run to all supporting scripts
#   --force          Pass --force to train_api.py (retrain even if already trained)
#   -h, --help       Show this message

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    { echo "ERROR: conda not found — run: source /workspace/config/start.sh"; exit 1; }
REPO_ROOT="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
SCRIPTS="${REPO_ROOT}/scripts"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DO_SETUP=false
DO_DOWNLOAD=false
DO_PREPARE=false
DO_TRAIN_LOCAL=false
DO_TRAIN_API=false
DO_EVAL_LOCAL=false
DO_EVAL_API=false
DO_CLASSIFY=false
DO_DASHBOARD=false
ANY_STEP=false
ALL_STEPS=false

SMOKE_TEST=false
CPU=false
TASK="all"
MODEL_OVERRIDE=""   # explicit --model; empty = use resolved default below
CLEAN=false
DRY_RUN=false
FORCE=false

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
    sed -n '/^# Usage:/,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --setup)        DO_SETUP=true;                              ANY_STEP=true; shift ;;
        --download)     DO_DOWNLOAD=true;                           ANY_STEP=true; shift ;;
        --prepare)      DO_PREPARE=true;                            ANY_STEP=true; shift ;;
        --train-local)  DO_TRAIN_LOCAL=true;                           ANY_STEP=true; shift ;;
        --train-api)    DO_TRAIN_API=true;                             ANY_STEP=true; shift ;;
        --train)        DO_TRAIN_LOCAL=true; DO_TRAIN_API=true;        ANY_STEP=true; shift ;;
        --eval-local)   DO_EVAL_LOCAL=true;                         ANY_STEP=true; shift ;;
        --eval-api)     DO_EVAL_API=true;                           ANY_STEP=true; shift ;;
        --eval)         DO_EVAL_LOCAL=true;  DO_EVAL_API=true;      ANY_STEP=true; shift ;;
        --classify)     DO_CLASSIFY=true;                           ANY_STEP=true; shift ;;
        --dashboard)    DO_DASHBOARD=true;                          ANY_STEP=true; shift ;;
        --all)
            ALL_STEPS=true
            ANY_STEP=true; shift ;;
        --smoke-test)  SMOKE_TEST=true;        shift ;;
        --cpu)         CPU=true;               shift ;;
        --task)        TASK="$2";              shift 2 ;;
        --model)       MODEL_OVERRIDE="$2";    shift 2 ;;
        --clean)       CLEAN=true;             shift ;;
        --dry-run)     DRY_RUN=true;           shift ;;
        --force)       FORCE=true;             shift ;;
        -h|--help)     usage ;;
        *) echo "Unknown argument: $1" >&2; usage ;;
    esac
done

# Default step set when none are specified.
if [[ "$ANY_STEP" == false ]]; then
    ALL_STEPS=true
fi

if [[ "$ALL_STEPS" == true ]]; then
    DO_SETUP=true; DO_DOWNLOAD=true; DO_PREPARE=true
    DO_TRAIN_LOCAL=true; DO_EVAL_LOCAL=true; DO_TRAIN_API=true
    DO_EVAL_API=true; DO_CLASSIFY=true; DO_DASHBOARD=true
fi

# --cpu suppresses GPU-only steps regardless of what was selected.
if [[ "$CPU" == true ]]; then
    DO_TRAIN_LOCAL=false
    DO_EVAL_LOCAL=false
fi

step() { echo; echo "━━━ $* ━━━"; }

# ---------------------------------------------------------------------------
# Run manifest — experiment tracking (best-effort; errors are non-fatal)
# ---------------------------------------------------------------------------
RUN_ID=""

_init_run_manifest() {
    [[ -z "$PYTHON" ]] && return 0
    RUN_ID=$("$PYTHON" "${SCRIPTS}/run_manifest_cli.py" init 2>/dev/null) || RUN_ID=""
}

_log_stage() {
    local stage="$1"
    [[ -z "$PYTHON" ]] && return 0
    [[ -z "$RUN_ID" ]] && return 0
    "$PYTHON" "${SCRIPTS}/run_manifest_cli.py" log-stage "$RUN_ID" "$stage" 2>/dev/null || true
}

# Fail fast if API steps require a key that isn't present.
if [[ "$DO_TRAIN_API" == true ]] || [[ "$DO_EVAL_API" == true ]]; then
    step "API key check"
    _api_key="${OPENAI_API_KEY:-}"
    if [[ -z "$_api_key" ]] && [[ -f "${REPO_ROOT}/.env" ]]; then
        _api_key=$(grep -E "^OPENAI_API_KEY=" "${REPO_ROOT}/.env" | tail -1 | cut -d= -f2- || true)
    fi
    if [[ -z "$_api_key" ]] || [[ "$_api_key" == "sk-..." ]] || [[ ${#_api_key} -lt 20 ]]; then
        echo "  FAIL  OPENAI_API_KEY  not set or invalid — add it to ${REPO_ROOT}/.env before running train-api or eval-api" >&2
        exit 1
    fi
    echo "  OK    OPENAI_API_KEY  ${_api_key:0:12}…  (${#_api_key} chars)"
    unset _api_key
fi

# ---------------------------------------------------------------------------
# Resolve model — mirrors train_local.py defaults so all steps stay in sync
# ---------------------------------------------------------------------------
if [[ -n "$MODEL_OVERRIDE" ]]; then
    MODEL="$MODEL_OVERRIDE"
elif [[ "$SMOKE_TEST" == true ]]; then
    MODEL="qwen2.5-0.5b"
else
    MODEL="qwen3-8b"
fi

# ---------------------------------------------------------------------------
# Resolve conda env Python — deferred when --setup will create/recreate it
# ---------------------------------------------------------------------------
CONDA_ENV="baseweight-benchmark"
_resolve_python() {
    local env_path
    env_path=$(conda env list 2>/dev/null | awk -v n="$CONDA_ENV" '$1==n{print $NF}')
    if [[ -z "$env_path" ]]; then
        echo "Error: conda env '$CONDA_ENV' not found. Run: bash scripts/setup.sh" >&2
        exit 1
    fi
    PYTHON="${env_path}/bin/python"
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ENV_FILE="${REPO_ROOT}/.env"

# Read a key from the .env file (does not export to shell).
_read_env_key() {
    local key="$1"
    if [[ -f "$ENV_FILE" ]]; then
        grep -E "^${key}=" "$ENV_FILE" | tail -1 | cut -d= -f2- || true
    fi
}

# ---------------------------------------------------------------------------
# Load .env for NETWORK_VOLUME (used in train clean)
# ---------------------------------------------------------------------------
NETWORK_VOLUME="/workspace"   # default; overridden below if set in .env
if [[ -f "$ENV_FILE" ]]; then
    val=$(_read_env_key NETWORK_VOLUME)
    [[ -n "$val" ]] && NETWORK_VOLUME="$val"
fi

# If setup will (re)create the env we don't need Python yet; resolve after setup.
# In every other case the env must already exist.
if [[ "$DO_SETUP" == false ]]; then
    _resolve_python
    _init_run_manifest
fi

# ---------------------------------------------------------------------------
# Build passthrough flags (avoid $VAR && ... pattern under set -e)
# ---------------------------------------------------------------------------
SMOKE_FLAG=""
if [[ "$SMOKE_TEST" == true ]]; then SMOKE_FLAG="--smoke-test"; fi

DRY_FLAG=""
if [[ "$DRY_RUN" == true ]]; then DRY_FLAG="--dry-run"; fi

FORCE_FLAG=""
if [[ "$FORCE" == true ]]; then FORCE_FLAG="--force"; fi

# Glob-safe delete — skips patterns that match nothing.
clean_paths() {
    local p hit
    for p in "$@"; do
        for hit in $p; do
            if [[ -e "$hit" ]]; then
                echo "  rm -rf $hit"
                rm -rf "$hit"
            fi
        done
    done
}

# 'all' → '*', otherwise literal — for building clean globs.
glob() { [[ "$1" == "all" ]] && echo "*" || echo "$1"; }

TASK_G=$(glob "$TASK")
MODEL_G=$(glob "$MODEL")

# ---------------------------------------------------------------------------
# Clean — scoped to the exact stage, model, and task; runs before the pipeline
# ---------------------------------------------------------------------------
if [[ "$CLEAN" == true ]]; then
    step "Cleaning prior outputs  (model=${MODEL}, task=${TASK})"

    if [[ "$DO_SETUP" == true ]]; then
        eval "$(conda shell.bash hook)"
        if conda env list 2>/dev/null | grep -qE "^${CONDA_ENV}[[:space:]]"; then
            echo "  Removing conda env ${CONDA_ENV} (clean setup)..."
            conda env remove -n "${CONDA_ENV}" -y
        fi
    fi

    if [[ "$DO_DOWNLOAD" == true ]]; then
        clean_paths "${REPO_ROOT}/data/raw/${TASK_G}"
    fi

    if [[ "$DO_PREPARE" == true ]]; then
        clean_paths "${REPO_ROOT}/data/prepared/${TASK_G}"
    fi

    if [[ "$DO_TRAIN_LOCAL" == true ]]; then
        # Local results
        clean_paths \
            "${REPO_ROOT}/results/adapters/local/${MODEL_G}/${TASK_G}" \
            "${REPO_ROOT}/results/training/local/${MODEL_G}/${TASK_G}"
        # Workspace checkpoints (remote GPU volume)
        if [[ -d "$NETWORK_VOLUME/checkpoints" ]]; then
            clean_paths "${NETWORK_VOLUME}/checkpoints/${MODEL_G}/${TASK_G}"
        fi
    fi

    if [[ "$DO_EVAL_LOCAL" == true ]]; then
        clean_paths \
            "${REPO_ROOT}/results/predictions/local/${MODEL_G}/${TASK_G}" \
            "${REPO_ROOT}/results/predictions/local/${MODEL_G}/${TASK_G}/*.partial"
    fi

    if [[ "$DO_TRAIN_API" == true ]]; then
        API_MODEL_G=$(glob "${MODEL_OVERRIDE:-all}")
        clean_paths "${REPO_ROOT}/results/training/api/${API_MODEL_G}/${TASK_G}"
    fi

    if [[ "$DO_EVAL_API" == true ]]; then
        API_MODEL_G=$(glob "${MODEL_OVERRIDE:-all}")
        clean_paths \
            "${REPO_ROOT}/results/predictions/api/${API_MODEL_G}/${TASK_G}" \
            "${REPO_ROOT}/results/predictions/api/${API_MODEL_G}/${TASK_G}/*.partial"
    fi

    if [[ "$DO_CLASSIFY" == true ]]; then
        clean_paths \
            "${REPO_ROOT}/results/classified/local/${MODEL_G}/${TASK_G}" \
            "${REPO_ROOT}/results/classified/api/${MODEL_G}/${TASK_G}" \
            "${REPO_ROOT}/results/summaries/local/${MODEL_G}/${TASK_G}" \
            "${REPO_ROOT}/results/summaries/api/${MODEL_G}/${TASK_G}"
    fi

    if [[ "$DO_DASHBOARD" == true ]]; then
        clean_paths "${REPO_ROOT}/dashboard-data/results.json"
    fi
fi

# ---------------------------------------------------------------------------
# Run steps
# ---------------------------------------------------------------------------
if [[ "$DO_SETUP" == true ]]; then
    step "Setup"
    bash "${SCRIPTS}/setup.sh"
    _resolve_python   # env now exists (created or updated by setup.sh)
    _init_run_manifest
    _log_stage "setup"
fi


if [[ "$DO_DOWNLOAD" == true ]]; then
    step "Download  (task=${TASK})"
    $PYTHON "${SCRIPTS}/download_data.py" \
        --task "$TASK" \
        $SMOKE_FLAG \
        $DRY_FLAG
    _log_stage "download"
fi

if [[ "$DO_PREPARE" == true ]]; then
    step "Prepare  (task=${TASK})"
    $PYTHON "${SCRIPTS}/prepare_datasets.py" \
        --task "$TASK" \
        $SMOKE_FLAG \
        $DRY_FLAG
    _log_stage "prepare"
fi

if [[ "$DO_TRAIN_LOCAL" == true ]]; then
    step "Train  (model=${MODEL}, task=${TASK})"
    $PYTHON "${SCRIPTS}/train_local.py" \
        --task "$TASK" \
        --model "$MODEL" \
        $SMOKE_FLAG \
        $DRY_FLAG
    _log_stage "train-local"
fi

if [[ "$DO_TRAIN_API" == true ]]; then
    API_MODEL="${MODEL_OVERRIDE:-all}"
    step "Train API  (model=${API_MODEL}, task=${TASK})"
    $PYTHON "${SCRIPTS}/train_api.py" \
        --task "$TASK" \
        --model "$API_MODEL" \
        $SMOKE_FLAG \
        $DRY_FLAG \
        $FORCE_FLAG
    _log_stage "train-api"
fi

if [[ "$DO_EVAL_LOCAL" == true ]]; then
    step "Eval local  (model=${MODEL}, task=${TASK})"
    $PYTHON "${SCRIPTS}/eval_local.py" \
        --task "$TASK" \
        --model "$MODEL" \
        $SMOKE_FLAG \
        $DRY_FLAG
    _log_stage "eval-local"
fi

if [[ "$DO_EVAL_API" == true ]]; then
    API_MODEL="${MODEL_OVERRIDE:-all}"
    step "Eval API  (model=${API_MODEL}, task=${TASK})"
    $PYTHON "${SCRIPTS}/eval_api.py" \
        --task "$TASK" \
        --model "$API_MODEL" \
        $SMOKE_FLAG \
        $DRY_FLAG
    _log_stage "eval-api"
fi

if [[ "$DO_CLASSIFY" == true ]]; then
    step "Classify errors  (task=${TASK})"
    $PYTHON "${SCRIPTS}/classify_errors.py" \
        --task "$TASK" \
        $DRY_FLAG
    _log_stage "classify"
fi


if [[ "$DO_DASHBOARD" == true ]]; then
    step "Generate dashboard data"
    $PYTHON "${SCRIPTS}/generate_dashboard_data.py" \
        --out "${REPO_ROOT}/results/final/results.json" \
        $DRY_FLAG
    _log_stage "dashboard"
fi

echo
echo "Done."
