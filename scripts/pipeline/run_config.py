"""Parameterized pipeline configuration — load from YAML for sweeps and ablations."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Timeouts(BaseModel):
    download_s: int = 1800
    prepare_s: int = 1800
    train_local_s: int = 43200   # 12 h
    train_api_s: int = 43200     # 12 h
    eval_local_s: int = 7200     # 2 h
    eval_api_s: int = 3600       # 1 h
    classify_s: int = 1800
    dashboard_s: int = 300


class CostCaps(BaseModel):
    train_api_usd: float = 100.0
    eval_api_usd: float = 50.0
    total_usd: float = 200.0


class RunConfig(BaseModel):
    tasks: list[str] = Field(default_factory=lambda: ["all"])
    local_models: list[str] = Field(default_factory=lambda: ["all"])
    api_models: list[str] = Field(default_factory=lambda: ["all"])
    smoke_test: bool = False
    dry_run: bool = False
    force: bool = False
    timeouts: Timeouts = Field(default_factory=Timeouts)
    cost_caps: CostCaps = Field(default_factory=CostCaps)

    @classmethod
    def from_yaml(cls, path: Path) -> "RunConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def resolved_tasks(self) -> list[str]:
        from pipeline.config import get_tasks
        if "all" in self.tasks:
            return get_tasks()
        return list(self.tasks)

    def resolved_local_models(self) -> list[str]:
        from pipeline.config import get_local_models
        if "all" in self.local_models:
            return [m["id"] for m in get_local_models()]
        return list(self.local_models)

    def resolved_api_models(self) -> list[str]:
        from pipeline.config import get_api_models
        if "all" in self.api_models:
            return [m["id"] for m in get_api_models()]
        return list(self.api_models)

    def effective_local_model(self) -> str | None:
        """Return the single local model to use, respecting smoke_test."""
        from pipeline.config import get_local_models
        models = get_local_models()
        resolved = self.resolved_local_models()
        if not resolved:
            return None
        first = resolved[0]
        if self.smoke_test:
            smoke_map = {m["id"]: m.get("smoke_id", m["id"]) for m in models}
            return smoke_map.get(first, first)
        return first

    def effective_api_model_arg(self) -> str:
        """Return 'all' or the single API model ID for CLI flags."""
        resolved = self.resolved_api_models()
        return resolved[0] if len(resolved) == 1 else "all"
