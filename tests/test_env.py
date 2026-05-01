"""Layer 4 — Environment sync: verify configs, imports, and environment setup."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

pytestmark = pytest.mark.integration


# ── Import smoke ───────────────────────────────────────────────────────────────

def test_utils_importable():
    import utils  # noqa: F401


def test_checkpoint_utils_importable():
    import checkpoint_utils  # noqa: F401


def test_classify_errors_importable():
    import classify_errors  # noqa: F401


def test_prepare_datasets_importable():
    import prepare_datasets  # noqa: F401


def test_generate_dashboard_data_importable():
    import generate_dashboard_data  # noqa: F401


def test_sync_artifacts_importable():
    import sync_artifacts  # noqa: F401


# ── YAML configs ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("task_id", ["banking77", "cuad", "ledgar", "fpb", "medmcqa", "mbpp"])
def test_task_config_parseable(task_id):
    path = REPO_ROOT / "configs" / "tasks" / f"{task_id}.yaml"
    assert path.exists(), f"Missing task config: {path}"
    data = yaml.safe_load(path.read_text())
    assert data["task_id"] == task_id
    assert "task_type" in data
    assert data["task_type"] in ("classification", "extraction", "code")
    assert "metric_id" in data
    assert "max_output_tokens" in data


@pytest.mark.parametrize("model_id", ["qwen3-8b", "gemma3-4b", "phi4-mini"])
def test_training_config_parseable(model_id):
    path = REPO_ROOT / "configs" / "training" / f"{model_id}.yaml"
    assert path.exists(), f"Missing training config: {path}"
    data = yaml.safe_load(path.read_text())
    assert "model_id" in data
    assert "model_short" in data
    assert "lora" in data
    assert "training" in data


def test_pricing_config_parseable():
    path = REPO_ROOT / "configs" / "pricing.yaml"
    assert path.exists()
    data = yaml.safe_load(path.read_text())
    assert "apis" in data
    assert "self_hosted" in data


def test_pricing_covers_all_api_models():
    path = REPO_ROOT / "configs" / "pricing.yaml"
    data = yaml.safe_load(path.read_text())
    api_models = list(data["apis"].keys())
    expected = ["gpt-5.4", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4.1-sft",
                "gemini-2.5-flash"]
    for model in expected:
        assert model in api_models, f"Missing pricing for {model}"


# ── Prompt files ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("task_id", ["banking77", "cuad", "ledgar", "fpb", "medmcqa", "mbpp"])
def test_prompt_file_exists_and_parseable(task_id):
    import json
    path = REPO_ROOT / "prompts" / f"{task_id}.json"
    assert path.exists(), f"Missing prompt: {path}"
    data = json.loads(path.read_text())
    assert "system" in data
    assert "user_template" in data


# ── .env.example ──────────────────────────────────────────────────────────────

def test_env_example_exists():
    env_example = REPO_ROOT / ".env.example"
    assert env_example.exists(), ".env.example not found"


def test_env_example_has_required_keys():
    env_example = REPO_ROOT / ".env.example"
    if not env_example.exists():
        pytest.skip(".env.example not found")
    content = env_example.read_text()
    for key in ["OPENAI_API_KEY", "HF_TOKEN", "NETWORK_VOLUME"]:
        assert key in content, f".env.example missing {key}"


# ── requirements.txt ──────────────────────────────────────────────────────────

def test_requirements_parseable():
    req_path = REPO_ROOT / "requirements.txt"
    assert req_path.exists()
    lines = [l.strip() for l in req_path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert len(lines) > 0
    for line in lines:
        # Each non-comment, non-empty line should have a package name
        assert line[0].isalpha() or line[0].isdigit(), f"Unexpected line: {line!r}"


