#!/usr/bin/env bash
# Idempotent, host-agnostic setup for baseweight-benchmark.
#
# Brings a fresh clone to a working state: installs miniconda if missing,
# creates or updates the conda env (in-repo at .conda-envs/), validates .env,
# verifies the install, and wires up cache env vars (HF_HOME etc.) to point
# inside the repo so the repo is the persistence boundary.
#
# Re-runs are fast no-ops (skips conda env work when environment.yml is
# unchanged, refreshes everything else).
#
# Usage:
#   ./start.sh                 # full setup
#   ./start.sh --skip-pull     # don't `git pull`
#   ./start.sh --recreate-env  # remove and recreate the conda env

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
ENV_NAME="baseweight-benchmark"
CONDA_ENVS_DIR="${REPO_ROOT}/.conda-envs"
ENV_PREFIX="${CONDA_ENVS_DIR}/${ENV_NAME}"
ENV_YML="${REPO_ROOT}/environment.yml"
STAMP_FILE="${REPO_ROOT}/.setup_stamp"
MINICONDA_DIR="${HOME}/miniconda3"
CACHE_ROOT="${REPO_ROOT}/.cache"

SKIP_PULL=false
RECREATE_ENV=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-pull)    SKIP_PULL=true;    shift ;;
        --recreate-env) RECREATE_ENV=true; shift ;;
        -h|--help)
            sed -n '2,16p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

echo "==> baseweight-benchmark setup"
echo "    repo:   ${REPO_ROOT}"
echo "    env:    ${ENV_PREFIX}"
echo "    cache:  ${CACHE_ROOT}"

# ---------------------------------------------------------------------------
# 1. Miniconda — standard install at $HOME/miniconda3. Outside the repo since
#    it's a per-user system tool, not project state. Reinstall is cheap.
# ---------------------------------------------------------------------------
if [ ! -x "${MINICONDA_DIR}/bin/conda" ]; then
    echo "==> Installing Miniconda → ${MINICONDA_DIR}"
    installer="$(mktemp --suffix=.sh)"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$installer"
    bash "$installer" -b -p "${MINICONDA_DIR}"
    rm -f "$installer"
else
    echo "==> Miniconda present"
fi

# ---------------------------------------------------------------------------
# 2. Claude Code — developer ergonomic; installer puts it under $HOME/.local.
# ---------------------------------------------------------------------------
if command -v claude >/dev/null 2>&1; then
    echo "==> Claude Code present"
else
    echo "==> Installing Claude Code"
    curl -fsSL https://claude.ai/install.sh | bash
fi

# ---------------------------------------------------------------------------
# 2b. Persist Claude Code state across pod restarts.
#
# Claude Code hardcodes ~/.claude/ and ~/.claude.json — no env override.
# On RunPod-style hosts, $HOME is ephemeral but /workspace is the persistent
# volume, so memory, conversation transcripts, plugins, and credentials are
# lost on every restart. Symlink both paths to the persistent volume so
# subsequent sessions resume with full context.
#
# Idempotent: only acts when the persistent state exists AND the link doesn't.
# Never overwrites a real directory in $HOME (would clobber a live session).
# ---------------------------------------------------------------------------
CLAUDE_STATE_DIR="/workspace/.claude-state"
CLAUDE_STATE_JSON="/workspace/.claude-state.json"
if [ -d "${CLAUDE_STATE_DIR}" ] && [ ! -e "${HOME}/.claude" ]; then
    ln -s "${CLAUDE_STATE_DIR}" "${HOME}/.claude"
    echo "==> Restored ${HOME}/.claude → ${CLAUDE_STATE_DIR}"
fi
if [ -f "${CLAUDE_STATE_JSON}" ] && [ ! -e "${HOME}/.claude.json" ]; then
    ln -s "${CLAUDE_STATE_JSON}" "${HOME}/.claude.json"
    echo "==> Restored ${HOME}/.claude.json → ${CLAUDE_STATE_JSON}"
fi

"${MINICONDA_DIR}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
"${MINICONDA_DIR}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || true

# shellcheck disable=SC1091
source "${MINICONDA_DIR}/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# 3. Conda environment (lives in-repo; persists with the repo)
# ---------------------------------------------------------------------------
mkdir -p "${CONDA_ENVS_DIR}"

if [ "$RECREATE_ENV" = true ] && [ -d "${ENV_PREFIX}" ]; then
    echo "==> Removing existing env (--recreate-env)"
    conda env remove --prefix "${ENV_PREFIX}" -y >/dev/null
    rm -f "${STAMP_FILE}"
fi

env_hash="$(md5sum "${ENV_YML}" | awk '{print $1}')"
cached_hash="$(cat "${STAMP_FILE}" 2>/dev/null || true)"

env_changed=false
if [ ! -d "${ENV_PREFIX}/conda-meta" ]; then
    echo "==> Creating conda env at ${ENV_PREFIX}"
    conda env create --prefix "${ENV_PREFIX}" -f "${ENV_YML}"
    env_changed=true
elif [ "${env_hash}" != "${cached_hash}" ]; then
    echo "==> environment.yml changed — updating env (--prune)"
    conda env update --prefix "${ENV_PREFIX}" -f "${ENV_YML}" --prune
    env_changed=true
else
    echo "==> Conda env up to date (environment.yml unchanged)"
fi

if [ "$env_changed" = true ]; then
    # torchao conflicts with torch (torch.int1 missing); unsloth doesn't need it.
    conda run --prefix "${ENV_PREFIX}" pip uninstall -y torchao 2>/dev/null || true
    echo "${env_hash}" > "${STAMP_FILE}"
fi

# ---------------------------------------------------------------------------
# Drift baseline — snapshot `pip list` so run.sh can fail loudly if anything
# is installed into / removed from the env after setup. The stamp above only
# tracks environment.yml edits; this catches drift in the env itself.
# ---------------------------------------------------------------------------
"${ENV_PREFIX}/bin/pip" list --format=freeze 2>/dev/null \
    | LC_ALL=C sort | md5sum | awk '{print $1}' \
    > "${REPO_ROOT}/.env_pip_hash"

# ---------------------------------------------------------------------------
# 4. Activate-hook: export cache env vars whenever the env is activated.
#
# Anchored to $CONDA_PREFIX so it stays correct even if the repo is moved or
# cloned to a different machine — no host-specific paths baked in.
# ---------------------------------------------------------------------------
activate_d="${ENV_PREFIX}/etc/conda/activate.d"
mkdir -p "${activate_d}"
cat > "${activate_d}/baseweight.sh" <<'EOF'
# Generated by start.sh — re-run start.sh to refresh.
# CONDA_PREFIX = <repo>/.conda-envs/baseweight-benchmark, so going two levels
# up lands at the repo root. All caches go under <repo>/.cache/.
_BW_REPO_ROOT="$(cd "${CONDA_PREFIX}/../.." && pwd)"
export HF_HOME="${_BW_REPO_ROOT}/.cache/huggingface"
export VLLM_CACHE_ROOT="${_BW_REPO_ROOT}/.cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="${_BW_REPO_ROOT}/.cache/torch_inductor"
export TRITON_CACHE_DIR="${_BW_REPO_ROOT}/.cache/triton"
export PIP_CACHE_DIR="${_BW_REPO_ROOT}/.cache/pip"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
mkdir -p "${HF_HOME}" "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${PIP_CACHE_DIR}"
unset _BW_REPO_ROOT
EOF

conda activate "${ENV_PREFIX}"

# ---------------------------------------------------------------------------
# 5. Verify install (always runs — catches a broken env even on cache hit)
# ---------------------------------------------------------------------------
echo "==> Verifying install"
python - <<'PYEOF'
import importlib.metadata, sys, torch
errors = []
try:
    vllm_ver = importlib.metadata.version("vllm")
except importlib.metadata.PackageNotFoundError:
    errors.append("vllm not installed"); vllm_ver = "missing"
if "+cpu" in torch.__version__:
    errors.append(f"CPU torch installed — expected CUDA build: {torch.__version__}")
try:
    unsloth_ver = importlib.metadata.version("unsloth")
except importlib.metadata.PackageNotFoundError:
    errors.append("unsloth not installed"); unsloth_ver = "missing"
if errors:
    print("SETUP ERRORS:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
print(f"    torch=={torch.__version__}  vllm=={vllm_ver}  unsloth=={unsloth_ver}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"    GPU: {p.name}  VRAM: {p.total_memory // 1024**3} GB")
else:
    print("    WARN: CUDA not available to torch — verify this host has an NVIDIA GPU")
PYEOF

# ---------------------------------------------------------------------------
# 6. .env — secrets only. Seed from .env.example if missing, validate format.
# ---------------------------------------------------------------------------
env_file="${REPO_ROOT}/.env"
env_example="${REPO_ROOT}/.env.example"

if [ ! -f "$env_file" ]; then
    if [ -f "$env_example" ]; then
        cp "$env_example" "$env_file"
        echo "==> Seeded ${env_file} from .env.example — fill in real values."
    else
        echo "==> WARN: ${env_file} missing and no .env.example to copy from."
    fi
fi

if [ -f "$env_file" ]; then
    malformed=$(grep -nvE '^\s*($|#|[A-Za-z_][A-Za-z0-9_]*=)' "$env_file" || true)
    if [ -n "$malformed" ]; then
        echo "==> ERROR: ${env_file} has malformed lines:" >&2
        printf '%s\n' "$malformed" >&2
        exit 1
    fi

    # Warn (don't fail) about placeholder/short keys — pipeline stages that
    # actually need them will fail loudly with a clearer message.
    api_key=$(grep -E '^OPENAI_API_KEY=' "$env_file" | tail -1 | cut -d= -f2- || true)
    if [ -z "$api_key" ] || [ "$api_key" = "sk-..." ] || [ "${#api_key}" -lt 20 ]; then
        echo "    WARN: OPENAI_API_KEY looks unset or placeholder in ${env_file}"
    else
        echo "    OK:   OPENAI_API_KEY (${#api_key} chars)"
    fi
fi

# ---------------------------------------------------------------------------
# 7. Git identity — repo-local override at .git-identity (gitignored) so a
#    fresh pod can re-apply user.name/email without re-prompting. Falls back
#    to whatever's already in `git config --global`, prompts only if both are
#    empty and stdin is a tty.
# ---------------------------------------------------------------------------
identity_file="${REPO_ROOT}/.git-identity"
[ -f "$identity_file" ] && . "$identity_file"

GIT_USER_NAME="${GIT_USER_NAME:-$(git config --global user.name  2>/dev/null || true)}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-$(git config --global user.email 2>/dev/null || true)}"

if [ -z "$GIT_USER_NAME" ] || [ -z "$GIT_USER_EMAIL" ]; then
    if [ -t 0 ]; then
        echo "==> Git identity"
        [ -z "$GIT_USER_NAME"  ] && { printf '    Your name:  '; read -r GIT_USER_NAME;  }
        [ -z "$GIT_USER_EMAIL" ] && { printf '    Your email: '; read -r GIT_USER_EMAIL; }
    else
        echo "==> WARN: git identity not configured and stdin is not a tty — skipping prompt."
    fi
fi

if [ -n "$GIT_USER_NAME" ] && [ -n "$GIT_USER_EMAIL" ]; then
    cat > "$identity_file" <<EOF
GIT_USER_NAME="${GIT_USER_NAME}"
GIT_USER_EMAIL="${GIT_USER_EMAIL}"
EOF
    git config --global user.name  "$GIT_USER_NAME"
    git config --global user.email "$GIT_USER_EMAIL"
    echo "    git identity: ${GIT_USER_NAME} <${GIT_USER_EMAIL}>"
fi

# ---------------------------------------------------------------------------
# 8. Auto-activate the env in new interactive shells. Re-applied each run so
#    the paths track the repo's current location.
# ---------------------------------------------------------------------------
bashrc="${HOME}/.bashrc"
marker_start="# >>> baseweight-benchmark conda activation >>>"
marker_end="# <<< baseweight-benchmark conda activation <<<"

if [ -f "$bashrc" ] && grep -qF "$marker_start" "$bashrc"; then
    sed -i "\|${marker_start}|,\|${marker_end}|d" "$bashrc"
fi

cat >> "$bashrc" <<EOF

${marker_start}
export PATH="\$HOME/.local/bin:\$PATH"
if [ -r "${MINICONDA_DIR}/etc/profile.d/conda.sh" ]; then
    . "${MINICONDA_DIR}/etc/profile.d/conda.sh"
    conda activate "${ENV_PREFIX}"
fi
${marker_end}
EOF

# ---------------------------------------------------------------------------
# 9. Lock file — reproducible env snapshot. Only re-export when the env
#    actually changed; `conda env export` is several seconds even on a no-op.
# ---------------------------------------------------------------------------
if [ "$env_changed" = true ] || [ ! -f "${REPO_ROOT}/environment.lock.yml" ]; then
    conda env export --prefix "${ENV_PREFIX}" > "${REPO_ROOT}/environment.lock.yml"
fi

# ---------------------------------------------------------------------------
# 10. Fetch latest (only if remote configured; safe no-op otherwise).
# ---------------------------------------------------------------------------
if [ "$SKIP_PULL" = false ] && git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    if git -C "$REPO_ROOT" remote | grep -q .; then
        branch="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
        echo "==> Pulling latest ${branch}"
        git -C "$REPO_ROOT" pull --ff-only origin "$branch" 2>/dev/null || \
            echo "    Pull skipped (not fast-forwardable — resolve manually)."
    fi
fi

echo "==> Setup complete."
echo "    Apply to this shell:  source ~/.bashrc"
echo "    (new shells auto-activate the env)"
echo "    Pipeline:              ./run.sh"
