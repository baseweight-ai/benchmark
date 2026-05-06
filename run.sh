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
#   --from STAGE     Run STAGE and all downstream stages. Aliases: train (both branches),
#                    eval (both branches). Specific: train-local, train-api, eval-local,
#                    eval-api, classify, dashboard. Overrides explicit step flags.
#   --clean          Delete prior outputs for selected steps/model/task, then run
#   --dry-run        Pass --dry-run to all supporting scripts
#   --force          Pass --force to train_api.py (retrain even if already trained)
#   --test-sampling  Run download+prepare with full production data to verify sampling. --task still applies.
#   -h, --help       Show this message

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    { echo "ERROR: conda not found — run: source /workspace/config/start.sh"; exit 1; }
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
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
TEST_SAMPLING=false
TASK="all"
MODEL_OVERRIDE=""   # explicit --model; empty = use resolved default below
CLEAN=false
DRY_RUN=false
FORCE=false
FROM_STAGE=""

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
        --from)        FROM_STAGE="$2";        ANY_STEP=true; shift 2 ;;
        --clean)         CLEAN=true;             shift ;;
        --dry-run)       DRY_RUN=true;           shift ;;
        --force)         FORCE=true;             shift ;;
        --test-sampling) TEST_SAMPLING=true;     shift ;;
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

# --from overrides step flags with the given stage and all downstream stages.
_apply_from_stage() {
    DO_SETUP=false
    DO_DOWNLOAD=false; DO_PREPARE=false
    DO_TRAIN_LOCAL=false; DO_TRAIN_API=false
    DO_EVAL_LOCAL=false; DO_EVAL_API=false
    DO_CLASSIFY=false; DO_DASHBOARD=false
    case "$1" in
        download)
            DO_DOWNLOAD=true; DO_PREPARE=true
            DO_TRAIN_LOCAL=true; DO_TRAIN_API=true
            DO_EVAL_LOCAL=true; DO_EVAL_API=true
            DO_CLASSIFY=true; DO_DASHBOARD=true ;;
        prepare)
            DO_PREPARE=true
            DO_TRAIN_LOCAL=true; DO_TRAIN_API=true
            DO_EVAL_LOCAL=true; DO_EVAL_API=true
            DO_CLASSIFY=true; DO_DASHBOARD=true ;;
        train)
            DO_TRAIN_LOCAL=true; DO_TRAIN_API=true
            DO_EVAL_LOCAL=true; DO_EVAL_API=true
            DO_CLASSIFY=true; DO_DASHBOARD=true ;;
        train-local)
            DO_TRAIN_LOCAL=true
            DO_EVAL_LOCAL=true
            DO_CLASSIFY=true; DO_DASHBOARD=true ;;
        train-api)
            DO_TRAIN_API=true
            DO_EVAL_API=true
            DO_CLASSIFY=true; DO_DASHBOARD=true ;;
        eval)
            DO_EVAL_LOCAL=true; DO_EVAL_API=true
            DO_CLASSIFY=true; DO_DASHBOARD=true ;;
        eval-local)
            DO_EVAL_LOCAL=true
            DO_CLASSIFY=true; DO_DASHBOARD=true ;;
        eval-api)
            DO_EVAL_API=true
            DO_CLASSIFY=true; DO_DASHBOARD=true ;;
        classify)
            DO_CLASSIFY=true; DO_DASHBOARD=true ;;
        dashboard)
            DO_DASHBOARD=true ;;
        *) echo "Unknown --from stage: $1  (valid: download prepare train train-local train-api eval eval-local eval-api classify dashboard)" >&2; exit 1 ;;
    esac
}

if [[ -n "$FROM_STAGE" ]]; then
    _apply_from_stage "$FROM_STAGE"
fi

# --cpu suppresses GPU-only steps regardless of what was selected.
if [[ "$CPU" == true ]]; then
    DO_TRAIN_LOCAL=false
    DO_EVAL_LOCAL=false
fi

step() { local _tag="$1"; shift; echo; echo "  [${_tag}] ━━━ $* ━━━"; }

# Run a command and prefix every output line with [tag].
_run_tagged() {
    local tag="$1"; shift
    "$@" 2>&1 | while IFS= read -r _line; do printf "  [%s] %s\n" "$tag" "$_line"; done
}

# Fail fast if API steps require a key that isn't present.
if [[ "$DO_TRAIN_API" == true ]] || [[ "$DO_EVAL_API" == true ]]; then
    step "pipeline" "API key check"
    _api_key="${OPENAI_API_KEY:-}"
    if [[ -z "$_api_key" ]] && [[ -f "${REPO_ROOT}/.env" ]]; then
        _api_key=$(grep -E "^OPENAI_API_KEY=" "${REPO_ROOT}/.env" | tail -1 | cut -d= -f2- || true)
    fi
    if [[ -z "$_api_key" ]] || [[ "$_api_key" == "sk-..." ]] || [[ ${#_api_key} -lt 20 ]]; then
        echo "  [pipeline] FAIL  OPENAI_API_KEY  not set or invalid — add it to ${REPO_ROOT}/.env before running train-api or eval-api" >&2
        exit 1
    fi
    echo "  [pipeline] OK    OPENAI_API_KEY  ${_api_key:0:12}…  (${#_api_key} chars)"
    unset _api_key
fi

if [[ "$DO_EVAL_LOCAL" == true ]]; then
    _hf_token="${HF_TOKEN:-}"
    if [[ -z "$_hf_token" ]] && [[ -f "${REPO_ROOT}/.env" ]]; then
        _hf_token=$(grep -E "^HF_TOKEN=" "${REPO_ROOT}/.env" | tail -1 | cut -d= -f2- || true)
    fi
    if [[ -z "$_hf_token" ]]; then
        echo "  [pipeline] WARN  HF_TOKEN not set — vLLM may hit HuggingFace rate limits. Add HF_TOKEN=hf_... to ${REPO_ROOT}/.env"
    fi
    unset _hf_token
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
                echo "  [pipeline] rm -rf $hit"
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
    step "pipeline" "Cleaning prior outputs  (model=${MODEL}, task=${TASK})"

    if [[ "$DO_DOWNLOAD" == true ]]; then
        clean_paths "${REPO_ROOT}/data/raw/${TASK_G}"
    fi

    if [[ "$DO_PREPARE" == true ]]; then
        clean_paths "${REPO_ROOT}/data/prepared/${TASK_G}"
    fi

    if [[ "$DO_TRAIN_LOCAL" == true ]]; then
        clean_paths \
            "${REPO_ROOT}/results/adapters/local/${MODEL_G}/${TASK_G}" \
            "${REPO_ROOT}/results/training/local/${MODEL_G}/${TASK_G}"
        if [[ -d "$NETWORK_VOLUME/checkpoints" ]]; then
            clean_paths "${NETWORK_VOLUME}/checkpoints/${MODEL_G}/${TASK_G}"
        fi
    fi

    if [[ "$DO_EVAL_LOCAL" == true ]]; then
        clean_paths \
            "${REPO_ROOT}/results/predictions/local/${MODEL_G}/${TASK_G}" \
            "${REPO_ROOT}/results/predictions/local/${MODEL_G}/${TASK_G}/*.partial"
    fi

    API_MODEL_G=$(glob "${MODEL_OVERRIDE:-all}")

    if [[ "$DO_TRAIN_API" == true ]]; then
        clean_paths "${REPO_ROOT}/results/training/api/${API_MODEL_G}/${TASK_G}"
    fi

    if [[ "$DO_EVAL_API" == true ]]; then
        clean_paths \
            "${REPO_ROOT}/results/predictions/api/${API_MODEL_G}/${TASK_G}" \
            "${REPO_ROOT}/results/predictions/api/${API_MODEL_G}/${TASK_G}/*.partial"
    fi

    if [[ "$DO_CLASSIFY" == true ]]; then
        clean_paths \
            "${REPO_ROOT}/results/classified/local/${MODEL_G}/${TASK_G}" \
            "${REPO_ROOT}/results/classified/api/${API_MODEL_G}/${TASK_G}" \
            "${REPO_ROOT}/results/summaries/local/${MODEL_G}/${TASK_G}" \
            "${REPO_ROOT}/results/summaries/api/${API_MODEL_G}/${TASK_G}"
    fi

    if [[ "$DO_DASHBOARD" == true ]]; then
        clean_paths \
            "${REPO_ROOT}/dashboard-data/results.json" \
            "${REPO_ROOT}/results/snapshots" \
            "${REPO_ROOT}/results/tables" \
            "${REPO_ROOT}/results/catalog.jsonl"
    fi

    if [[ "$ALL_STEPS" == true ]]; then
        clean_paths \
            "${REPO_ROOT}/runs" \
            "${REPO_ROOT}/results/pipeline.log.jsonl"
    fi

    if [[ "$DO_SETUP" == true ]]; then
        eval "$(conda shell.bash hook)"
        if conda env list 2>/dev/null | grep -qE "^${CONDA_ENV}[[:space:]]"; then
            echo "  [pipeline] Removing conda env ${CONDA_ENV}..."
            _run_tagged "pipeline" conda env remove -n "${CONDA_ENV}" -y
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Translate step flags → stages list, then delegate to Python DAG runner
# ---------------------------------------------------------------------------
if [[ "$DO_SETUP" == true ]]; then
    step "pipeline" "Setup"
    _run_tagged "pipeline" bash "${SCRIPTS}/setup.sh"
    _resolve_python   # env now exists (created or updated by setup.sh)
fi

STAGES=""
_add_stage() { STAGES="${STAGES:+${STAGES},}${1}"; }
[[ "$DO_DOWNLOAD" == true ]]    && _add_stage "download"
[[ "$DO_PREPARE" == true ]]     && _add_stage "prepare"
[[ "$DO_TRAIN_LOCAL" == true ]] && _add_stage "train-local"
[[ "$DO_TRAIN_API" == true ]]   && _add_stage "train-api"
[[ "$DO_EVAL_LOCAL" == true ]]  && _add_stage "eval-local"
[[ "$DO_EVAL_API" == true ]]    && _add_stage "eval-api"
[[ "$DO_CLASSIFY" == true ]]    && _add_stage "classify"
[[ "$DO_DASHBOARD" == true ]]   && _add_stage "dashboard"

if [[ "$TEST_SAMPLING" == true ]]; then
    RUN_ARGS=("--test-sampling" "--task" "$TASK")
    $PYTHON "${SCRIPTS}/run.py" "${RUN_ARGS[@]}"
elif [[ -n "$STAGES" ]]; then
    RUN_ARGS=(
        "--stages" "$STAGES"
        "--task"   "$TASK"
        "--local-model" "$MODEL"
    )
    [[ -n "$MODEL_OVERRIDE" ]]  && RUN_ARGS+=("--api-model" "$MODEL_OVERRIDE")
    [[ -n "$SMOKE_FLAG" ]]      && RUN_ARGS+=("$SMOKE_FLAG")
    [[ -n "$DRY_FLAG" ]]        && RUN_ARGS+=("$DRY_FLAG")
    [[ -n "$FORCE_FLAG" ]]      && RUN_ARGS+=("$FORCE_FLAG")

    $PYTHON "${SCRIPTS}/run.py" "${RUN_ARGS[@]}"
fi
