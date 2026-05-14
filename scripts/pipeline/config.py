"""Load and expose configs/pipeline.yaml. Cached after first read."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

_PIPELINE_CONFIG = Path(__file__).parent.parent.parent / "configs" / "pipeline.yaml"


@lru_cache(maxsize=1)
def _load() -> dict:
    with open(_PIPELINE_CONFIG) as f:
        return yaml.safe_load(f)


def get_tasks() -> list[str]:
    return list(_load()["tasks"])


def get_local_models() -> list[dict]:
    return list(_load()["models"]["local"])


def get_api_models() -> list[dict]:
    return list(_load()["models"]["api"])


def get_sft_models() -> list[dict]:
    return [m for m in get_api_models() if m.get("sft_base_model")]


def get_openai_models() -> dict[str, Optional[str]]:
    """Return {model_id: pinned_version} for all API models."""
    return {m["id"]: m.get("pinned_version") for m in get_api_models()}


def get_model_conditions() -> dict[str, list[str]]:
    """Return {model_id: [conditions]} for all API models."""
    return {m["id"]: m.get("conditions", []) for m in get_api_models()}


def get_reasoning_capable() -> dict[str, bool]:
    """Return {model_id: reasoning_capable} for all API models.

    True means the model accepts the `reasoning_effort` parameter. The benchmark
    sends `reasoning_effort="minimal"` for every reasoning-capable model so the
    reasoning loop is off across the board — no model gets free inference-time
    compute that another doesn't.
    """
    return {m["id"]: bool(m.get("reasoning_capable", False)) for m in get_api_models()}


def get_sft_base_models() -> dict[str, str]:
    """Return {model_id: sft_base_model} for SFT-capable API models."""
    return {m["id"]: m["sft_base_model"] for m in get_sft_models()}
