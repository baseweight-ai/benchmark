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


def get_prod_model_ids() -> set[str]:
    """Model IDs that should appear on the published (non-smoke) dashboard.

    Local: every entry in `models.local` (the smoke stand-in lives under
    `smoke_id`, not as a top-level entry).
    API: every entry NOT tagged `smoke: true`. Smoke stand-ins (gpt-5.4-nano)
    are explicitly tagged so the prod dashboard excludes them.
    """
    return (
        {m["id"] for m in get_local_models()}
        | {m["id"] for m in get_api_models() if not m.get("smoke")}
    )


def get_smoke_model_ids() -> set[str]:
    """Model IDs that should appear on the smoke dashboard.

    Local: each prod model's `smoke_id` (the small stand-in used for fast
    end-to-end checks). API: every entry tagged `smoke: true`.
    """
    return (
        {m["smoke_id"] for m in get_local_models() if m.get("smoke_id")}
        | {m["id"] for m in get_api_models() if m.get("smoke")}
    )


def get_openai_models() -> dict[str, Optional[str]]:
    """Return {model_id: pinned_version} for all API models."""
    return {m["id"]: m.get("pinned_version") for m in get_api_models()}


def get_model_conditions() -> dict[str, list[str]]:
    """Return {model_id: [conditions]} for all API models."""
    return {m["id"]: m.get("conditions", []) for m in get_api_models()}


def get_reasoning_capable() -> dict[str, bool]:
    """Return {model_id: reasoning_capable} for all API models.

    True means the model accepts the `reasoning_effort` parameter. The benchmark
    sends `reasoning_effort="none"` for every reasoning-capable model so the
    reasoning loop is off across the board — no model gets free inference-time
    compute that another doesn't.
    """
    return {m["id"]: bool(m.get("reasoning_capable", False)) for m in get_api_models()}
