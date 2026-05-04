"""CLI helper for run manifest management — called from run.sh."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import click

REPO_ROOT = Path(__file__).parent.parent


@click.group()
def cli() -> None:
    pass


@cli.command()
def init() -> None:
    """Create a new run manifest and print the run_id."""
    from pipeline.paths import run_manifest_path
    from pipeline.registry import RunManifest, save_manifest
    from pipeline.versioning import configs_sha, git_sha

    sha = git_sha()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = RunManifest(
        run_id=f"{ts}-{sha}",
        git_sha=sha,
        config_sha=configs_sha([
            REPO_ROOT / "configs" / "pipeline.yaml",
            REPO_ROOT / "environment.yml",
        ]),
    )
    save_manifest(manifest, REPO_ROOT)
    click.echo(manifest.run_id)


@cli.command()
@click.argument("run_id")
@click.argument("stage")
def log_stage(run_id: str, stage: str) -> None:
    """Append a completed stage to an existing run manifest."""
    from checkpoint_utils import atomic_write_json
    from pipeline.paths import run_manifest_path

    manifest_path = run_manifest_path(REPO_ROOT, run_id)
    if not manifest_path.exists():
        return
    data = json.loads(manifest_path.read_text())
    data.setdefault("stages_completed", []).append(stage)
    atomic_write_json(data, manifest_path)


if __name__ == "__main__":
    cli()
