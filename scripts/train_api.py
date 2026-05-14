"""OpenAI API SFT training — idempotent, never reruns unless --force."""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import traceback as _tb

import click
import yaml
from dotenv import load_dotenv

from checkpoint_utils import atomic_write_json
from pipeline.config import get_sft_base_models, get_tasks
from pipeline.log import configure, get_logger
from pipeline.paths import training_meta_path
from pipeline.validation import require_jsonl
from pipeline.versioning import git_sha as _git_sha

_log = get_logger("train-api")

REPO_ROOT = Path(__file__).parent.parent
load_dotenv(REPO_ROOT / ".env")

ALL_TASKS: list[str] = get_tasks()

_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})

SFT_BASE_MODELS: dict[str, str] = get_sft_base_models()

ALL_SFT_MODELS = list(SFT_BASE_MODELS.keys())
SMOKE_SFT_MODELS = ["gpt-4.1-nano"]

_SFT_SUFFIX_MAX_LEN = 18


def meta_path(model_id: str, task_id: str) -> Path:
    return training_meta_path(REPO_ROOT, "api", model_id, task_id, "api-sft")


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
        "git_sha": _git_sha(),
    }
    atomic_write_json(meta, mp)
    return meta


def run_sft_train(
    model_id: str,
    task_id: str,
    dry_run: bool,
    smoke_test: bool,
    force: bool,
    submit_only: bool = False,
) -> None:
    """Upload training data and create an OpenAI fine-tuning job.

    Idempotent: skips if metadata.json already exists and --force is not set.
    Falls back to searching OpenAI for an existing job before creating a new one.
    With --submit-only, writes pending metadata and returns without polling.
    """
    prepared = REPO_ROOT / "data" / "prepared" / task_id
    sft_path = prepared / ("smoke_train.jsonl" if smoke_test else "train.jsonl")
    base_model = SFT_BASE_MODELS[model_id]
    mp = meta_path(model_id, task_id)

    # A completed run is the contract — don't re-validate sft_path, which may
    # have been mutated or removed since the upload succeeded.
    try:
        cached = json.loads(mp.read_text())
        if not force:
            if cached.get("ft_model_id"):
                click.echo(f"  SKIP [{model_id}/{task_id}/api-sft]: already trained → {cached['ft_model_id']}  (use --force to retrain)")
                _log.info("training skip", model=model_id, task=task_id, condition="api-sft",
                          event="stage_skip", reason="already trained")
            else:
                click.echo(f"  SKIP [{model_id}/{task_id}/api-sft]: pending job {cached.get('job_id')} already submitted")
                _log.info("training skip", model=model_id, task=task_id, condition="api-sft",
                          event="stage_skip", reason="pending job already submitted")
            return
    except FileNotFoundError:
        pass

    if dry_run:
        click.echo(f"  [dry-run] Would upload {sft_path} and create {base_model} fine-tuning job")
        return

    require_jsonl(sft_path, min_rows=1, check_chat_format=True)

    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    job = None if force else _find_existing_sft_job(client, model_id, task_id, smoke_test)
    if job and job.status == "succeeded":
        click.echo(f"  Found completed job {job.id} on OpenAI → {job.fine_tuned_model}")
        meta = _write_sft_metadata(job, sft_path, model_id, mp)
        click.echo(f"  Wrote metadata to {mp.relative_to(REPO_ROOT)}")
        _log.info("training complete", model=model_id, task=task_id, condition="api-sft",
                  event="stage_complete", training_cost=meta["training_cost"],
                  training_time_min=meta["training_time_min"], n_train=meta["n_train"])
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
        click.echo(f"  Fine-tuning job created: {job.id} (epochs={n_epochs}).")

    # Write pending metadata so eval_api can find the job and wait for it.
    with open(sft_path) as f:
        n_train = sum(1 for line in f if line.strip())
    mp.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json({"job_id": job.id, "status": "pending", "n_train": n_train}, mp)

    if submit_only:
        click.echo(f"  [{model_id}/{task_id}] Submitted job {job.id} — eval_api.py will wait for completion")
        return

    click.echo(f"  Waiting for job {job.id}...")
    key = f"{model_id}/{task_id}"
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
            click.echo(f"  [{key}/{job.status}] {event.message} (+{elapsed}s)")
            last_event_time = time.time()
        if not new_events:
            since_last = int(time.time() - last_event_time)
            total = int(time.time() - job_start)
            click.echo(f"  [{key}/{job.status}] waiting... ({since_last}s since last event, {total}s total)")

    if job.status != "succeeded":
        raise RuntimeError(f"Fine-tuning job {job.id} ended with status: {job.status}")

    meta = _write_sft_metadata(job, sft_path, model_id, mp)
    click.echo(f"  Fine-tuned model: {meta['ft_model_id']}, cost: ${meta['training_cost']:.3f}")
    click.echo(f"  Wrote metadata to {mp.relative_to(REPO_ROOT)}")
    _log.info("training complete", model=model_id, task=task_id, condition="api-sft",
              event="stage_complete", training_cost=meta["training_cost"],
              training_time_min=meta["training_time_min"], n_train=meta["n_train"])


@click.command()
@click.option("--model", default="all", help=f"Model ID or 'all'. Choices: {', '.join(ALL_SFT_MODELS)}")
@click.option("--task", default="all", help="Task ID or 'all'")
@click.option("--dry-run", is_flag=True)
@click.option("--smoke-test", is_flag=True)
@click.option("--force", is_flag=True, help="Retrain even if metadata.json already exists")
@click.option("--submit-only", is_flag=True, help="Submit jobs and write pending metadata, but don't poll for completion")
def main(model: str, task: str, dry_run: bool, smoke_test: bool, force: bool, submit_only: bool) -> None:
    """Run OpenAI SFT training (idempotent — skips if already trained).

    Multiple jobs are submitted and polled concurrently — wall time equals the
    slowest single job rather than the sum of all jobs.

    With --submit-only, jobs are submitted and pending metadata is written immediately,
    then the process exits. eval_api.py will wait for completion when it reaches api-sft.
    """
    configure(REPO_ROOT)
    default_sft_models = SMOKE_SFT_MODELS if smoke_test else ALL_SFT_MODELS
    model_ids = default_sft_models if model == "all" else [model]
    task_ids = ALL_TASKS if task == "all" else [task]

    if not dry_run and not os.environ.get("OPENAI_API_KEY"):
        click.echo("  WARNING: OPENAI_API_KEY not set", err=True)
        _log.warning("OPENAI_API_KEY not set")

    work = [
        (mid, tid)
        for mid in model_ids
        for tid in task_ids
        if mid in SFT_BASE_MODELS
    ]
    for mid in model_ids:
        if mid not in SFT_BASE_MODELS:
            click.echo(f"  SKIP [{mid}]: not an SFT-capable model", err=True)

    failures = []

    if dry_run or submit_only or len(work) <= 1:
        for mid, tid in work:
            try:
                run_sft_train(mid, tid, dry_run, smoke_test, force, submit_only)
            except Exception as exc:
                click.echo(f"  ERROR [{mid}/{tid}]: {exc}", err=True)
                _tb.print_exc()
                _log.error(f"training failed: {type(exc).__name__}: {exc}",
                           model=mid, task=tid, condition="api-sft",
                           exc=str(exc), traceback=_tb.format_exc())
                failures.append((f"{mid}/{tid}", str(exc)))
    else:
        click.echo(f"  Submitting {len(work)} fine-tuning job(s) concurrently...")
        with ThreadPoolExecutor(max_workers=min(len(work), 10)) as executor:
            futures = {
                executor.submit(run_sft_train, mid, tid, False, smoke_test, force, False): (mid, tid)
                for mid, tid in work
            }
            for future in as_completed(futures):
                mid, tid = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    key = f"{mid}/{tid}"
                    click.echo(f"  ERROR [{key}]: {exc}", err=True)
                    _tb.print_exc()
                    _log.error(f"training failed: {type(exc).__name__}: {exc}",
                               model=mid, task=tid, condition="api-sft",
                               exc=str(exc), traceback=_tb.format_exc())
                    failures.append((key, str(exc)))

    if failures:
        click.echo(f"\nFAILED ({len(failures)}):")
        for key, err in failures:
            click.echo(f"  {key}: {err}")
        sys.exit(1)
    if submit_only:
        click.echo("\nAll jobs submitted. Run eval_api.py to evaluate (it will wait for pending jobs).")
    else:
        click.echo("\nAll API training completed.")


if __name__ == "__main__":
    main()
