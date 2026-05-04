"""Experiment tracking — per-run manifests written to runs/."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def new_run_id() -> str:
    """Return a unique run ID: UTC timestamp + short git SHA."""
    from pipeline.versioning import git_sha
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{git_sha()}"


@dataclass
class RunManifest:
    run_id: str
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    git_sha: str = ""
    config_sha: Optional[str] = None
    stages_completed: list[str] = field(default_factory=list)
    stages_failed: list[str] = field(default_factory=list)
    notes: Optional[str] = None

    def log_stage(self, stage: str, success: bool = True) -> None:
        if success:
            self.stages_completed.append(stage)
        else:
            self.stages_failed.append(stage)


def save_manifest(manifest: RunManifest, root: Path) -> Path:
    """Write manifest to runs/{run_id}.json atomically. Returns the written path."""
    from checkpoint_utils import atomic_write_json
    from pipeline.paths import run_manifest_path
    path = run_manifest_path(root, manifest.run_id)
    atomic_write_json(asdict(manifest), path)
    return path
