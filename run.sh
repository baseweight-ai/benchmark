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
#   --train          Alias for --train-local
#   --eval-local     Run eval_local.py  (vLLM, fine-tuned models)
#   --eval-api       Run eval_api.py    (frontier API eval — zero-shot and 5-shot)
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
#   --from STAGE     Run STAGE and all downstream stages. Aliases: train (= train-local),
#                    eval (both eval branches). Specific: train-local, eval-local,
#                    eval-api, classify, dashboard. Overrides explicit step flags.
#   --clean          Delete prior outputs for selected steps/model/task, then run
#   --dry-run        Pass --dry-run to all supporting scripts
#   --test-sampling  Run download+prepare with full production data to verify sampling. --task still applies.
#   --skip-env-check Bypass the conda-env-vs-environment.yml drift check (debugging only)
#   -h, --help       Show this message

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    { echo "ERROR: conda not found — run: bash ${REPO_ROOT}/start.sh"; exit 1; }
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
SCRIPTS="${REPO_ROOT}/scripts"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DO_SETUP=false
DO_DOWNLOAD=false
DO_PREPARE=false
DO_TRAIN_LOCAL=false
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
FROM_STAGE=""
SKIP_ENV_CHECK=false

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
        --train-local)  DO_TRAIN_LOCAL=true;                        ANY_STEP=true; shift ;;
        --train)        DO_TRAIN_LOCAL=true;                        ANY_STEP=true; shift ;;
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
        --clean)            CLEAN=true;             shift ;;
        --dry-run)          DRY_RUN=true;           shift ;;
        --test-sampling)    TEST_SAMPLING=true;     shift ;;
        --skip-env-check)   SKIP_ENV_CHECK=true;    shift ;;
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
    DO_TRAIN_LOCAL=true; DO_EVAL_LOCAL=true
    DO_EVAL_API=true; DO_CLASSIFY=true; DO_DASHBOARD=true
fi

# --from overrides step flags with the given stage and all downstream stages.
_apply_from_stage() {
    DO_SETUP=false
    DO_DOWNLOAD=false; DO_PREPARE=false
    DO_TRAIN_LOCAL=false
    DO_EVAL_LOCAL=false; DO_EVAL_API=false
    DO_CLASSIFY=false; DO_DASHBOARD=false
    case "$1" in
        download)
            DO_DOWNLOAD=true; DO_PREPARE=true
            DO_TRAIN_LOCAL=true
            DO_EVAL_LOCAL=true; DO_EVAL_API=true
            DO_CLASSIFY=true; DO_DASHBOARD=true ;;
        prepare)
            DO_PREPARE=true
            DO_TRAIN_LOCAL=true
            DO_EVAL_LOCAL=true; DO_EVAL_API=true
            DO_CLASSIFY=true; DO_DASHBOARD=true ;;
        train|train-local)
            DO_TRAIN_LOCAL=true
            DO_EVAL_LOCAL=true
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
        *) echo "Unknown --from stage: $1  (valid: download prepare train train-local eval eval-local eval-api classify dashboard)" >&2; exit 1 ;;
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
if [[ "$DO_EVAL_API" == true ]]; then
    step "pipeline" "API key check"
    _api_key="${OPENAI_API_KEY:-}"
    if [[ -z "$_api_key" ]] && [[ -f "${REPO_ROOT}/.env" ]]; then
        _api_key=$(grep -E "^OPENAI_API_KEY=" "${REPO_ROOT}/.env" | tail -1 | cut -d= -f2- || true)
    fi
    if [[ -z "$_api_key" ]] || [[ "$_api_key" == "sk-..." ]] || [[ ${#_api_key} -lt 20 ]]; then
        echo "  [pipeline] FAIL  OPENAI_API_KEY  not set or invalid — add it to ${REPO_ROOT}/.env before running eval-api" >&2
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
# Resolve conda env Python — env lives in-repo at .conda-envs/<name>.
# Deferred when --setup will create/recreate it.
# ---------------------------------------------------------------------------
CONDA_ENV_PREFIX="${REPO_ROOT}/.conda-envs/baseweight-benchmark"
_resolve_python() {
    if [[ ! -x "${CONDA_ENV_PREFIX}/bin/python" ]]; then
        echo "Error: conda env not found at ${CONDA_ENV_PREFIX}. Run: ${REPO_ROOT}/start.sh" >&2
        exit 1
    fi
    PYTHON="${CONDA_ENV_PREFIX}/bin/python"
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

# Reproducibility gate: every run verifies the active env matches the one
# start.sh built from environment.yml. Two checks:
#   (1) environment.yml hash vs the stamp start.sh recorded — flags edits
#       to the YAML that bypassed setup.
#   (2) `pip list` hash vs the snapshot start.sh took post-build — flags
#       in-place `pip install`/`pip uninstall`/`conda install` that drifted
#       the env from its declared state.
# Either drift is a hard fail with a resolution recipe (no auto-fix —
# resolution is the user's choice: start.sh updates in place,
# start.sh --recreate-env rebuilds from scratch). Bypass for debugging
# with --skip-env-check.
_env_check() {
    local env_yml="${REPO_ROOT}/environment.yml"
    local yaml_stamp="${REPO_ROOT}/.setup_stamp"
    local pip_stamp="${REPO_ROOT}/.env_pip_hash"
    local issues=()

    if [[ -f "$yaml_stamp" ]]; then
        local cur_yml cached_yml
        cur_yml="$(md5sum "$env_yml" | awk '{print $1}')"
        cached_yml="$(cat "$yaml_stamp")"
        if [[ "$cur_yml" != "$cached_yml" ]]; then
            issues+=("environment.yml has been edited since the env was last (re)built")
        fi
    fi

    if [[ -f "$pip_stamp" ]] && [[ -x "${CONDA_ENV_PREFIX}/bin/pip" ]]; then
        local cur_pip cached_pip
        cur_pip="$("${CONDA_ENV_PREFIX}/bin/pip" list --format=freeze 2>/dev/null \
                    | LC_ALL=C sort | md5sum | awk '{print $1}')"
        cached_pip="$(cat "$pip_stamp")"
        if [[ "$cur_pip" != "$cached_pip" ]]; then
            issues+=("Installed packages drifted from the snapshot start.sh took (pip install/uninstall outside setup)")
        fi
    fi

    if (( ${#issues[@]} > 0 )); then
        {
            echo
            echo "  [pipeline] FAIL  Conda env does not match environment.yml:"
            for i in "${issues[@]}"; do
                echo "  [pipeline]   - ${i}"
            done
            echo "  [pipeline]"
            echo "  [pipeline] Resolve with one of:"
            echo "  [pipeline]   ./start.sh                    # update env in place to match environment.yml"
            echo "  [pipeline]   ./start.sh --recreate-env     # rebuild from scratch"
            echo "  [pipeline]"
            echo "  [pipeline] To bypass during debugging:     ./run.sh --skip-env-check ..."
        } >&2
        return 1
    fi
}

# If setup will (re)create the env we don't need Python yet; resolve after setup.
# In every other case the env must already exist.
if [[ "$DO_SETUP" == false ]]; then
    _resolve_python
    if [[ "$SKIP_ENV_CHECK" == false ]]; then
        _env_check || exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Build passthrough flags (avoid $VAR && ... pattern under set -e)
# ---------------------------------------------------------------------------
SMOKE_FLAG=""
if [[ "$SMOKE_TEST" == true ]]; then SMOKE_FLAG="--smoke-test"; fi

DRY_FLAG=""
if [[ "$DRY_RUN" == true ]]; then DRY_FLAG="--dry-run"; fi

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
            "${REPO_ROOT}/results/training/local/${MODEL_G}/${TASK_G}" \
            "${REPO_ROOT}/checkpoints/${MODEL_G}/${TASK_G}"
    fi

    if [[ "$DO_EVAL_LOCAL" == true ]]; then
        clean_paths \
            "${REPO_ROOT}/results/predictions/local/${MODEL_G}/${TASK_G}" \
            "${REPO_ROOT}/results/predictions/local/${MODEL_G}/${TASK_G}/*.partial"
    fi

    API_MODEL_G=$(glob "${MODEL_OVERRIDE:-all}")

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
        if [[ -d "${CONDA_ENV_PREFIX}/conda-meta" ]]; then
            # `conda env remove` refuses to delete the currently-active env.
            # The shell that invoked run.sh typically has the env activated
            # (our auto-activate hook in ~/.bashrc), so CONDA_PREFIX is
            # inherited here. Deactivate in-place — this script sourced
            # conda.sh, so `conda deactivate` mutates our process env — then
            # remove. The outer shell is unaffected (process isolation); the
            # env is recreated by start.sh below.
            if [[ "${CONDA_PREFIX:-}" == "${CONDA_ENV_PREFIX}" ]]; then
                echo "  [pipeline] Deactivating ${CONDA_ENV_PREFIX} so it can be removed..."
                conda deactivate || true
            fi
            echo "  [pipeline] Removing conda env at ${CONDA_ENV_PREFIX}..."
            _run_tagged "pipeline" conda env remove --prefix "${CONDA_ENV_PREFIX}" -y
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Translate step flags → stages list, then delegate to Python DAG runner
# ---------------------------------------------------------------------------
if [[ "$DO_SETUP" == true ]]; then
    step "pipeline" "Setup"
    _run_tagged "pipeline" bash "${REPO_ROOT}/start.sh" --skip-pull
    _resolve_python   # env now exists (created or updated by start.sh)
fi

STAGES=""
_add_stage() { STAGES="${STAGES:+${STAGES},}${1}"; }
[[ "$DO_DOWNLOAD" == true ]]    && _add_stage "download"
[[ "$DO_PREPARE" == true ]]     && _add_stage "prepare"
[[ "$DO_TRAIN_LOCAL" == true ]] && _add_stage "train-local"
[[ "$DO_EVAL_LOCAL" == true ]]  && _add_stage "eval-local"
[[ "$DO_EVAL_API" == true ]]    && _add_stage "eval-api"
[[ "$DO_CLASSIFY" == true ]]    && _add_stage "classify"
[[ "$DO_DASHBOARD" == true ]]   && _add_stage "dashboard"

mkdir -p "${REPO_ROOT}/runs"
LOG_FILE="${REPO_ROOT}/runs/pipeline-$(date +%Y%m%d-%H%M%S).log"
echo "  [pipeline] Logging output to: ${LOG_FILE}"
echo "             Tail from another terminal: tail -f ${LOG_FILE}"
echo "             Tip: run inside tmux/screen to survive disconnections"

if [[ "$TEST_SAMPLING" == true ]]; then
    RUN_ARGS=("--test-sampling" "--task" "$TASK")
    $PYTHON "${SCRIPTS}/run.py" "${RUN_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
elif [[ -n "$STAGES" ]]; then
    RUN_ARGS=(
        "--stages" "$STAGES"
        "--task"   "$TASK"
        "--local-model" "$MODEL"
    )
    [[ -n "$MODEL_OVERRIDE" ]]  && RUN_ARGS+=("--api-model" "$MODEL_OVERRIDE")
    [[ -n "$SMOKE_FLAG" ]]      && RUN_ARGS+=("$SMOKE_FLAG")
    [[ -n "$DRY_FLAG" ]]        && RUN_ARGS+=("$DRY_FLAG")

    $PYTHON "${SCRIPTS}/run.py" "${RUN_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
fi
