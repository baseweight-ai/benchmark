"""OpenAI API SFT training — idempotent, never reruns unless --force."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import click
import yaml
from dotenv import load_dotenv

from checkpoint_utils import atomic_write_json

REPO_ROOT = Path(__file__).parent.parent
load_dotenv(REPO_ROOT / ".env")

ALL_TASKS = ["banking77", "cuad", "ledgar", "fpb", "medmcqa"]

_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})

SFT_BASE_MODELS: dict[str, str] = {
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
    "gpt-5.4-mini": "gpt-5.4-mini-2026-03-17",
}

ALL_SFT_MODELS = list(SFT_BASE_MODELS.keys())

_SFT_SUFFIX_MAX_LEN = 18


def meta_path(model_id: str, task_id: str) -> Path:
    return REPO_ROOT / "results" / "training" / "api" / model_id / task_id / "api-sft" / "metadata.json"


def _find_existing_sft_job(client, model_id: str, task_id: str, smoke_test: bool):
    """Search OpenAI for a completed or in-progress fine-tuning job matching this run."""
    active = {"validating_files", "queued", "running"}
    after = None
    while True:
        page = client.fine_tuning.jobs.list(limit=100, after=after).data
        for job in page:
            if job.status not in active and job.status != "succeeded":
                continue
            meta = job.metadata or {}
            if (meta.get("model_id") == model_id
                    and meta.get("task_id") == task_id
                    and meta.get("smoke_test") == str(smoke_test)):
                return job
        if len(page) < 100:
            break
        after = page[-1].id
    return None


def _write_sft_metadata(job, sft_path: Path, model_id: str, mp: Path) -> dict:
    """Compute training cost, atomically write metadata.json, and return the metadata dict."""
    ft_model = job.fine_tuned_model
    trained_tokens = job.trained_tokens or 0
    training_time_min = round((job.finished_at - job.created_at) / 60, 1) if job.finished_at and job.created_at else None
    with open(REPO_ROOT / "configs" / "pricing.yaml") as f:
        pricing = yaml.safe_load(f)
    training_per_m = pricing.get("apis", {}).get(model_id, {}).get("training_per_m", 25.0)
    training_cost = trained_tokens * training_per_m / 1_000_000
    with open(sft_path) as f:
        n_train = sum(1 for line in f if line.strip())
    meta = {
        "ft_model_id": ft_model, "job_id": job.id,
        "trained_tokens": trained_tokens, "training_cost": training_cost,
        "training_time_min": training_time_min, "n_train": n_train,
    }
    atomic_write_json(meta, mp)
    return meta


def run_sft_train(
    model_id: str,
    task_id: str,
    dry_run: bool,
    smoke_test: bool,
    force: bool,
) -> None:
    """Upload training data and create an OpenAI fine-tuning job.

    Idempotent: skips if metadata.json already exists and --force is not set.
    Falls back to searching OpenAI for an existing job before creating a new one.
    """
    prepared = REPO_ROOT / "data" / "prepared" / task_id
    sft_path = prepared / ("smoke_train.jsonl" if smoke_test else "train.jsonl")
    base_model = SFT_BASE_MODELS[model_id]
    mp = meta_path(model_id, task_id)

    if not sft_path.exists():
        raise FileNotFoundError(f"SFT training data not found: {sft_path}")

    if dry_run:
        if mp.exists() and not force:
            click.echo(f"  [dry-run] SKIP [{model_id}/{task_id}]: metadata exists")
        else:
            click.echo(f"  [dry-run] Would upload {sft_path} and create {base_model} fine-tuning job")
        return

    try:
        cached = json.loads(mp.read_text())
        if not force:
            click.echo(f"  SKIP [{model_id}/{task_id}/api-sft]: already trained → {cached['ft_model_id']}  (use --force to retrain)")
            return
    except FileNotFoundError:
        pass

    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    job = _find_existing_sft_job(client, model_id, task_id, smoke_test)
    if job and job.status == "succeeded":
        click.echo(f"  Found completed job {job.id} on OpenAI → {job.fine_tuned_model}")
        _write_sft_metadata(job, sft_path, model_id, mp)
        click.echo(f"  Wrote metadata to {mp.relative_to(REPO_ROOT)}")
        return
    elif job:
        click.echo(f"  Found in-progress job {job.id} (status={job.status}), attaching...")
    else:
        click.echo(f"  Uploading training file for {task_id}...")
        with open(sft_path, "rb") as f:
            train_file_obj = client.files.create(file=f, purpose="fine-tune")

        n_epochs = 1 if smoke_test else 3
        batch_size = 1 if smoke_test else 8
        job = client.fine_tuning.jobs.create(
            training_file=train_file_obj.id,
            model=base_model,
            suffix=(f"bw-{task_id}-sm" if smoke_test else f"bw-{task_id}")[:_SFT_SUFFIX_MAX_LEN],
            metadata={
                "model_id": model_id,
                "task_id": task_id,
                "base_model": base_model,
                "n_epochs": str(n_epochs),
                "smoke_test": str(smoke_test),
            },
            hyperparameters={
                "n_epochs": n_epochs,
                "batch_size": batch_size,
                "learning_rate_multiplier": "auto",
            },
        )
        click.echo(f"  Fine-tuning job created: {job.id} (epochs={n_epochs}). Waiting...")

    seen_event_ids: set[str] = set()
    job_start = last_event_time = time.time()
    while job.status not in _TERMINAL_STATUSES:
        time.sleep(15)
        job = client.fine_tuning.jobs.retrieve(job.id)
        events = client.fine_tuning.jobs.list_events(job.id, limit=10)
        new_events = [e for e in reversed(events.data) if e.id not in seen_event_ids]
        for event in new_events:
            seen_event_ids.add(event.id)
            elapsed = int(time.time() - job_start)
            click.echo(f"  [{job.status}] {event.message} (+{elapsed}s)")
            last_event_time = time.time()
        if not new_events:
            since_last = int(time.time() - last_event_time)
            total = int(time.time() - job_start)
            click.echo(f"  [{job.status}] waiting... ({since_last}s since last event, {total}s total)")

    if job.status != "succeeded":
        raise RuntimeError(f"Fine-tuning job {job.id} ended with status: {job.status}")

    meta = _write_sft_metadata(job, sft_path, model_id, mp)
    click.echo(f"  Fine-tuned model: {meta['ft_model_id']}, cost: ${meta['training_cost']:.3f}")
    click.echo(f"  Wrote metadata to {mp.relative_to(REPO_ROOT)}")


@click.command()
@click.option("--model", default="all", help=f"Model ID or 'all'. Choices: {', '.join(ALL_SFT_MODELS)}")
@click.option("--task", default="all", help="Task ID or 'all'")
@click.option("--dry-run", is_flag=True)
@click.option("--smoke-test", is_flag=True)
@click.option("--force", is_flag=True, help="Retrain even if metadata.json already exists")
def main(model: str, task: str, dry_run: bool, smoke_test: bool, force: bool) -> None:
    """Run OpenAI SFT training (idempotent — skips if already trained)."""
    model_ids = ALL_SFT_MODELS if model == "all" else [model]
    task_ids = ALL_TASKS if task == "all" else [task]

    if not dry_run and not os.environ.get("OPENAI_API_KEY"):
        click.echo("  WARNING: OPENAI_API_KEY not set", err=True)

    failures = []
    for mid in model_ids:
        if mid not in SFT_BASE_MODELS:
            click.echo(f"  SKIP [{mid}]: not an SFT-capable model", err=True)
            continue
        for tid in task_ids:
            try:
                run_sft_train(mid, tid, dry_run, smoke_test, force)
            except Exception as exc:
                click.echo(f"  ERROR [{mid}/{tid}]: {exc}", err=True)
                import traceback; traceback.print_exc()
                failures.append((f"{mid}/{tid}", str(exc)))

    if failures:
        click.echo(f"\nFAILED ({len(failures)}):")
        for key, err in failures:
            click.echo(f"  {key}: {err}")
        sys.exit(1)
    click.echo("\nAll API training completed.")


if __name__ == "__main__":
    main()
