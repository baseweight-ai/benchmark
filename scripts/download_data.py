"""Download raw datasets from HuggingFace for all benchmark tasks."""
from __future__ import annotations

import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv
from pydantic import BaseModel
import yaml

from checkpoint_utils import atomic_write_json

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# Expected minimum row counts for sanity checks (split: min_count)
EXPECTED_COUNTS: dict[str, dict[str, int]] = {
    "banking77": {"train": 8000,  "test": 2000},
    "cuad":      {"train": 10000, "test": 1000},
    "ledgar":    {"train": 50000, "test": 5000},
    "fpb":       {"train": 3000},
    "medmcqa":   {"train": 100000, "test": 4000},
}

from pipeline.config import get_tasks

REPO_ROOT = Path(__file__).parent.parent


class TaskConfig(BaseModel):
    task_id: str
    task_name: str
    dataset_path: str
    dataset_config: Optional[str] = None


def load_task_configs(task_ids: list[str]) -> list[TaskConfig]:
    configs = []
    config_dir = REPO_ROOT / "configs" / "tasks"
    for tid in task_ids:
        path = config_dir / f"{tid}.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        configs.append(TaskConfig(**{k: data[k] for k in TaskConfig.model_fields if k in data}))
    return configs


ALL_TASKS: list[str] = get_tasks()

TINY_TRAIN = 12
TINY_TEST = 5


def _hub_load(path: str, **kwargs):
    """Load from HuggingFace Hub; raise immediately if Hub is unreachable.

    The datasets library silently falls back to local cache when the Hub is
    unreachable, emitting a log message instead of raising.  We pre-check Hub
    reachability via HfApi so any failure is explicit before load_dataset runs.
    When `revision` is already in kwargs the caller already confirmed reachability
    (by fetching dataset_info to obtain the SHA), so we skip the check.
    """
    from datasets import load_dataset
    from huggingface_hub import HfApi

    if "revision" not in kwargs:
        token = kwargs.get("token")
        try:
            HfApi(token=token).dataset_info(path)
        except Exception as exc:
            raise RuntimeError(
                f"Hugging Face Hub is unreachable for '{path}'. "
                f"Cannot proceed — load_dataset would silently fall back to local cache.\n"
                f"Check network connection and HF_TOKEN.\nDetail: {exc}"
            ) from exc

    return load_dataset(path, **kwargs)


def download_task(cfg: TaskConfig, dry_run: bool, smoke_test: bool = False) -> None:
    click.echo(f"\n[{cfg.task_id}] Downloading {cfg.task_name}...")
    if dry_run:
        click.echo(f"  [dry-run] Would download {cfg.dataset_path} (config={cfg.dataset_config})")
        return

    from datasets import DatasetDict, DownloadMode  # lazy import
    from huggingface_hub import HfApi
    out_dir = REPO_ROOT / "data" / "raw" / cfg.task_id
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN") or None

    # Record the exact Hub commit so every experiment is traceable to a specific version.
    ds_info = HfApi(token=hf_token).dataset_info(cfg.dataset_path)

    sha = getattr(ds_info, "sha", None)
    load_kwargs: dict = {
        "token": hf_token,
        "download_mode": DownloadMode.FORCE_REDOWNLOAD,
    }
    if sha:
        load_kwargs["revision"] = sha
    if cfg.dataset_config:
        load_kwargs["name"] = cfg.dataset_config

    if smoke_test:
        # Probe for a test split first so we can download 2x train rows upfront for
        # tasks that have none, avoiding a redundant second download.
        test_exists = False
        for split in ("test", "validation"):
            try:
                probe = _hub_load(cfg.dataset_path, split=f"{split}[:1]", **load_kwargs)
                test_exists = True
                break
            except Exception:
                pass

        train_limit = TINY_TRAIN if test_exists else TINY_TRAIN * 2
        loaded = {}
        split_errors: list[str] = []
        for split in ("train", "test", "validation"):
            limit = train_limit if split == "train" else TINY_TEST
            try:
                ds_split = _hub_load(cfg.dataset_path, split=f"{split}[:{limit}]", **load_kwargs)
                loaded[split] = ds_split
                click.echo(f"  {split}: {len(ds_split)} rows (smoke test)")
            except Exception as e:
                split_errors.append(f"{split}: {e}")
        if not loaded:
            for err in split_errors:
                click.echo(f"  {err}", err=True)
            raise RuntimeError(
                f"Could not download any splits for {cfg.dataset_path} from the Hugging Face Hub. "
                "Check your HF_TOKEN and network connection."
            )
        if not test_exists:
            click.echo(f"  No test split — downloaded {train_limit} train rows for smoke splitting")
        ds = DatasetDict(loaded)
    else:
        ds = _hub_load(cfg.dataset_path, **load_kwargs)
        for split, dataset in ds.items():
            count = len(dataset)
            expected = EXPECTED_COUNTS.get(cfg.task_id, {}).get(split, 0)
            status = "OK" if count >= expected else f"WARNING: expected >= {expected}"
            click.echo(f"  {split}: {count:,} rows — {status}")

    ds.save_to_disk(str(out_dir))

    version_meta = {
        "dataset_id": cfg.dataset_path,
        "sha": getattr(ds_info, "sha", None),
        "last_modified": str(getattr(ds_info, "last_modified", None)),
        "downloaded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    atomic_write_json(version_meta, out_dir / "dataset_version.json")
    click.echo(f"  Saved to {out_dir} (sha={version_meta['sha']})")


@click.command()
@click.option("--task", default=None, help="Task ID to download (required; use 'all' to download every task)")
@click.option("--dry-run", is_flag=True, help="Validate config without downloading")
@click.option("--smoke-test", is_flag=True, help=f"Download only {TINY_TRAIN} train + {TINY_TEST} test rows for smoke testing")
def main(task: str, dry_run: bool, smoke_test: bool) -> None:
    """Download benchmark datasets from HuggingFace.

    You must specify --task <id> or --task all. No default — downloading all
    six datasets at once can take significant time and disk space.
    """
    if task is None:
        raise click.UsageError("--task is required. Pass a task ID or 'all' to download every task.")
    task_ids = ALL_TASKS if task == "all" else [t.strip() for t in task.split(",")]
    configs = load_task_configs(task_ids)
    failures: list[tuple[str, str]] = []

    max_workers = min(len(configs), os.cpu_count() or 4, 4)
    if len(configs) > 1:
        click.echo(f"  Downloading {len(configs)} tasks with {max_workers} workers (output may interleave)")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_to_cfg = {
            pool.submit(download_task, cfg, dry_run, smoke_test=smoke_test): cfg
            for cfg in configs
        }
        for fut in as_completed(fut_to_cfg):
            cfg = fut_to_cfg[fut]
            try:
                fut.result()
            except Exception as exc:
                click.echo(f"  ERROR [{cfg.task_id}]: {exc}", err=True)
                failures.append((cfg.task_id, str(exc)))

    if failures:
        click.echo(f"\n{'='*50}")
        click.echo(f"FAILED ({len(failures)}):")
        for tid, err in failures:
            click.echo(f"  {tid}: {err}")
        sys.exit(1)
    else:
        click.echo(f"\nAll {'validated' if dry_run else 'downloaded'} successfully.")


if __name__ == "__main__":
    main()
