"""Artifact catalog — local registry of eval runs and results.

Usage:
    python scripts/catalog.py rebuild        # Scan summaries/ and runs/, write results/catalog.jsonl
    python scripts/catalog.py search --model gpt-5.4-mini --task fpb
    python scripts/catalog.py list-runs      # Print run manifest IDs and stage counts
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import click

from utils import load_jsonl

REPO_ROOT = Path(__file__).parent.parent
CATALOG_PATH = REPO_ROOT / "results" / "catalog.jsonl"


def _iter_summaries() -> list[dict]:
    """Yield catalog entries from every base-condition summary file."""
    summaries_root = REPO_ROOT / "results" / "summaries"
    entries = []
    if not summaries_root.exists():
        return entries
    for source_dir in sorted(summaries_root.iterdir()):
        if not source_dir.is_dir():
            continue
        for model_dir in sorted(source_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for task_dir in sorted(model_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                for f in sorted(task_dir.glob("*.json")):
                    stem = f.stem
                    if "_seed" in stem:
                        continue
                    try:
                        with open(f) as fh:
                            data = json.load(fh)
                    except Exception:
                        continue
                    entry = {
                        "type": "eval_summary",
                        "source": source_dir.name,
                        "model_id": data.get("model", model_dir.name),
                        "task_id": data.get("task_id", task_dir.name),
                        "condition": stem.replace("_agg", ""),
                        "is_aggregated": stem.endswith("_agg"),
                        "n_seeds": data.get("n_seeds"),
                        "metric_id": data.get("metric_id"),
                        "metric_value": data.get("metric_value"),
                        "metric_std": data.get("metric_std"),
                        "metric_ci_lo": data.get("metric_ci_lo"),
                        "metric_ci_hi": data.get("metric_ci_hi"),
                        "n_predictions": data.get("n_predictions"),
                        "prompt_sha": data.get("prompt_sha"),
                        "few_shot_hash": data.get("few_shot_hash"),
                        "path": str(f.relative_to(REPO_ROOT)),
                        "indexed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                    entries.append(entry)
    return entries


def _iter_runs() -> list[dict]:
    """Yield catalog entries from every run manifest in runs/."""
    runs_root = REPO_ROOT / "runs"
    entries = []
    if not runs_root.exists():
        return entries
    for f in sorted(runs_root.glob("*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
        except Exception:
            continue
        entry = {
            "type": "run_manifest",
            "run_id": data.get("run_id", f.stem),
            "started_at": data.get("started_at"),
            "git_sha": data.get("git_sha"),
            "config_sha": data.get("config_sha"),
            "stages_completed": data.get("stages_completed", []),
            "stages_failed": data.get("stages_failed", []),
            "path": str(f.relative_to(REPO_ROOT)),
            "indexed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        entries.append(entry)
    return entries


@click.group()
def cli() -> None:
    """Artifact catalog for benchmark results."""


@cli.command()
def rebuild() -> None:
    """Rebuild catalog.jsonl from all summaries and run manifests."""
    entries = _iter_summaries() + _iter_runs()
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CATALOG_PATH, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    n_evals = sum(1 for e in entries if e["type"] == "eval_summary")
    n_runs = sum(1 for e in entries if e["type"] == "run_manifest")
    click.echo(f"  Catalog rebuilt: {n_evals} eval summaries, {n_runs} run manifests → {CATALOG_PATH}")


@cli.command()
@click.option("--model", default=None, help="Filter by model_id (substring match)")
@click.option("--task", default=None, help="Filter by task_id")
@click.option("--condition", default=None, help="Filter by condition")
@click.option("--source", default=None, help="Filter by source (local|api)")
@click.option("--min-metric", default=None, type=float, help="Minimum metric_value")
def search(model: str | None, task: str | None, condition: str | None, source: str | None, min_metric: float | None) -> None:
    """Search the catalog for matching eval summaries."""
    entries = load_jsonl(CATALOG_PATH) if CATALOG_PATH.exists() else []
    if not entries:
        click.echo("Catalog is empty. Run: python scripts/catalog.py rebuild")
        sys.exit(1)

    hits = [e for e in entries if e.get("type") == "eval_summary"]
    if model:
        hits = [e for e in hits if model.lower() in e.get("model_id", "").lower()]
    if task:
        hits = [e for e in hits if e.get("task_id") == task]
    if condition:
        hits = [e for e in hits if e.get("condition") == condition]
    if source:
        hits = [e for e in hits if e.get("source") == source]
    if min_metric is not None:
        hits = [e for e in hits if (e.get("metric_value") or 0) >= min_metric]

    if not hits:
        click.echo("No matching entries.")
        return

    for e in sorted(hits, key=lambda x: (x.get("model_id", ""), x.get("task_id", ""))):
        std = f" ±{e['metric_std']:.4f}" if e.get("metric_std") is not None else ""
        seeds = f" ({e['n_seeds']} seeds)" if e.get("n_seeds") else ""
        click.echo(
            f"  {e['source']:5} {e['model_id']:20} {e['task_id']:12} {e['condition']:12} "
            f"{e.get('metric_id',''):12} "
            f"{e['metric_value']:.4f}{std}{seeds}"
            if e.get("metric_value") is not None
            else f"  {e['source']:5} {e['model_id']:20} {e['task_id']:12} {e['condition']:12}  (no data)"
        )
    click.echo(f"\n{len(hits)} result(s)")


@cli.command("list-runs")
def list_runs() -> None:
    """List run manifests with stage completion summary."""
    entries = load_jsonl(CATALOG_PATH) if CATALOG_PATH.exists() else []
    if not entries:
        click.echo("Catalog is empty. Run: python scripts/catalog.py rebuild")
        sys.exit(1)
    runs = [e for e in entries if e.get("type") == "run_manifest"]
    if not runs:
        click.echo("No run manifests in catalog.")
        return
    for r in sorted(runs, key=lambda x: x.get("started_at", "")):
        n_ok = len(r.get("stages_completed", []))
        n_fail = len(r.get("stages_failed", []))
        click.echo(f"  {r['run_id']}  started={r.get('started_at','')}  ok={n_ok}  fail={n_fail}")


if __name__ == "__main__":
    cli()
