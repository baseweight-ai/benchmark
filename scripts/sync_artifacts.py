"""Sync all run artifacts to HuggingFace for persistence on remote GPU instances.

Run this after any training or eval step to ensure nothing is lost if the instance
terminates. Safe to run multiple times — existing files are skipped.

Usage:
    python scripts/sync_artifacts.py                  # sync everything
    python scripts/sync_artifacts.py --what adapters  # adapters only
    python scripts/sync_artifacts.py --what data      # prepared datasets only
    python scripts/sync_artifacts.py --what all       # everything
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv

from checkpoint_utils import CHECKPOINTS_ROOT

REPO_ROOT = Path(__file__).parent.parent
load_dotenv(REPO_ROOT / ".env")

HF_ORG = "baseweight"  # update to your HF username/org
DATASET_REPO = f"{HF_ORG}/baseweight-benchmark-data"
PREDICTIONS_REPO = f"{HF_ORG}/baseweight-benchmark-predictions"


def get_api():
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:
        click.echo("ERROR: HF_TOKEN not set in .env", err=True)
        sys.exit(1)
    return HfApi(token=token)


def ensure_repo(api, repo_id: str, repo_type: str) -> None:
    from huggingface_hub import create_repo
    try:
        create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, private=False)
    except Exception as exc:
        click.echo(f"  WARNING: could not create {repo_id}: {exc}", err=True)


def upload_file_safe(api, local_path: Path, remote_path: str, repo_id: str, repo_type: str) -> bool:
    """Upload a single file. Returns True on success."""
    try:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"sync: {remote_path}",
        )
        return True
    except Exception as exc:
        click.echo(f"  ERROR uploading {remote_path}: {exc}", err=True)
        return False


def _sync_files(
    api, root: Path, files: list, remote_prefix: str,
    repo_id: str, repo_type: str, label: str, dry_run: bool,
) -> int:
    if not dry_run:
        ensure_repo(api, repo_id, repo_type)
    count = 0
    for f in files:
        remote = f"{remote_prefix}/{f.relative_to(root)}"
        if dry_run:
            click.echo(f"  [dry-run] Would upload {remote}")
            count += 1
        else:
            if upload_file_safe(api, f, remote, repo_id, repo_type):
                count += 1
    click.echo(f"  {label}: {count}/{len(files)} files synced")
    return count


def sync_prepared_data(api, dry_run: bool) -> int:
    root = REPO_ROOT / "data" / "prepared"
    if not root.exists():
        click.echo("  No prepared data found — skipping")
        return 0
    return _sync_files(api, root, list(root.rglob("*.jsonl")), "prepared", DATASET_REPO, "dataset", "Prepared data", dry_run)


def sync_adapters(api, dry_run: bool) -> int:
    """Upload LoRA adapter weights to HuggingFace model repos."""
    adapter_root = REPO_ROOT / "results" / "adapters"
    if not adapter_root.exists():
        click.echo("  No adapters found — skipping")
        return 0

    count = 0
    for model_dir in adapter_root.iterdir():
        if not model_dir.is_dir():
            continue
        for task_dir in model_dir.iterdir():
            if not task_dir.is_dir():
                continue
            for condition_dir in task_dir.iterdir():
                if not condition_dir.is_dir():
                    continue
                model_short = model_dir.name
                task_id = task_dir.name
                condition = condition_dir.name
                repo_id = f"{HF_ORG}/{model_short}-{task_id}-{condition}"

                if dry_run:
                    files = list(condition_dir.rglob("*"))
                    click.echo(f"  [dry-run] Would upload {len(files)} files to {repo_id}")
                    count += 1
                    continue

                try:
                    from huggingface_hub import create_repo
                    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
                    api.upload_folder(
                        folder_path=str(condition_dir),
                        repo_id=repo_id,
                        repo_type="model",
                        commit_message=f"sync: {model_short}/{task_id}/{condition} adapter",
                        ignore_patterns=["README.md"],
                    )
                    click.echo(f"  Synced adapter → {repo_id}")
                    count += 1
                except Exception as exc:
                    click.echo(f"  ERROR syncing {repo_id}: {exc}", err=True)

    click.echo(f"  Adapters: {count} repos synced")
    return count


def sync_predictions(api, dry_run: bool) -> int:
    root = REPO_ROOT / "results" / "predictions"
    if not root.exists():
        click.echo("  No predictions found — skipping")
        return 0
    # Include both completed (.jsonl) and in-progress (.jsonl.partial) files.
    files = sorted(list(root.rglob("*.jsonl")) + list(root.rglob("*.partial")))
    return _sync_files(api, root, files, "predictions", PREDICTIONS_REPO, "dataset", "Predictions", dry_run)


def sync_checkpoints(api, dry_run: bool) -> int:
    # Full HF Trainer checkpoints live in the repo and persist with it;
    # only train_state.json is uploaded as an offsite backup.
    root = CHECKPOINTS_ROOT
    if not root.exists():
        click.echo("  No checkpoint state found — skipping")
        return 0
    return _sync_files(api, root, list(root.rglob("train_state.json")), "checkpoints", PREDICTIONS_REPO, "dataset", "Checkpoint state", dry_run)


def sync_summaries(api, dry_run: bool) -> int:
    root = REPO_ROOT / "results" / "summaries"
    if not root.exists():
        click.echo("  No summaries found — skipping")
        return 0
    return _sync_files(api, root, list(root.rglob("*.json")), "summaries", PREDICTIONS_REPO, "dataset", "Summaries", dry_run)


def sync_training_metadata(api, dry_run: bool) -> int:
    root = REPO_ROOT / "results" / "training"
    if not root.exists():
        click.echo("  No training metadata found — skipping")
        return 0
    return _sync_files(api, root, list(root.rglob("metadata.json")), "training", PREDICTIONS_REPO, "dataset", "Training metadata", dry_run)


@click.command()
@click.option(
    "--what",
    default="all",
    type=click.Choice(["all", "data", "adapters", "predictions", "summaries", "checkpoints"]),
    help="What to sync",
)
@click.option("--dry-run", is_flag=True)
def main(what: str, dry_run: bool) -> None:
    """Sync all run artifacts to HuggingFace for remote instance persistence."""
    if dry_run:
        click.echo("  [dry-run mode — no uploads will occur]\n")
        api = None
    else:
        api = get_api()

    if what in ("all", "data"):
        sync_prepared_data(api, dry_run)

    if what in ("all", "adapters"):
        sync_adapters(api, dry_run)

    if what in ("all", "predictions"):
        sync_predictions(api, dry_run)

    if what in ("all", "summaries"):
        sync_summaries(api, dry_run)
        sync_training_metadata(api, dry_run)

    if what in ("all", "checkpoints"):
        sync_checkpoints(api, dry_run)

    click.echo("\nSync complete.")


if __name__ == "__main__":
    main()
