"""Hyperparameter sweep runner: grid or random search over training params.

Usage:
    python scripts/run_sweep.py --config configs/sweeps/example_lr_rank.yaml
    python scripts/run_sweep.py --config configs/sweeps/example_lr_rank.yaml --dry-run
"""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Any

import click
import yaml

REPO_ROOT = Path(__file__).parent.parent

# Keys that route to model_cfg.lora rather than model_cfg.training.
_LORA_KEYS: dict[str, str] = {"lora_rank": "rank", "lora_alpha": "alpha", "lora_dropout": "dropout"}


def _make_trials(cfg: dict) -> list[dict[str, Any]]:
    """Return one param dict per trial (grid or random search)."""
    params = cfg.get("params", {})
    if not params:
        return [{}]
    keys = list(params.keys())
    combos = list(itertools.product(*[params[k] for k in keys]))
    if cfg.get("search", "grid") != "grid":
        n = cfg.get("n_trials", 10)
        rng = random.Random(cfg.get("seed", 42))
        combos = rng.sample(combos, min(n, len(combos)))
    return [dict(zip(keys, combo)) for combo in combos]


def _apply_params(model_cfg, task_cfg, params: dict[str, Any], trial_short: str):
    """Return (model_cfg, task_cfg) with sweep params layered onto task overrides.

    Sweep params go through the same `lora_overrides` / `training_overrides`
    plumbing as a task config, so the precedence is sweep > task > model. If
    we instead mutated `model_cfg.lora` directly, a task with its own
    `lora_overrides` would later clobber the sweep's value inside `train_one`.

    Only `model_short` is updated on `model_cfg`, so each trial writes to a
    unique adapter dir.
    """
    lora_overrides = dict(task_cfg.lora_overrides or {})
    training_overrides = dict(task_cfg.training_overrides or {})
    for k, v in params.items():
        if k in _LORA_KEYS:
            lora_overrides[_LORA_KEYS[k]] = v
        else:
            training_overrides[k] = v
    new_model_cfg = model_cfg.model_copy(update={"model_short": trial_short})
    new_task_cfg = task_cfg.model_copy(update={
        "lora_overrides": lora_overrides or None,
        "training_overrides": training_overrides or None,
    })
    return new_model_cfg, new_task_cfg


def _fmt_params(params: dict) -> str:
    if not params:
        return "(base)"
    parts = [f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()]
    return ", ".join(parts)


@click.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Sweep config YAML")
@click.option("--dry-run", is_flag=True, help="Print trial plan without training")
def main(config: str, dry_run: bool) -> None:
    """Run a hyperparameter sweep (grid or random search) over training params."""
    from train_local import load_model_config, load_task_config, train_one

    sweep_cfg = yaml.safe_load(Path(config).read_text())
    name = sweep_cfg.get("name", Path(config).stem)
    model_id = sweep_cfg["model"]
    task_id = sweep_cfg["task"]
    smoke_test = sweep_cfg.get("smoke_test", False)

    base_model_cfg = load_model_config(model_id)
    task_cfg = load_task_config(task_id)
    trials = _make_trials(sweep_cfg)

    click.echo(f"Sweep '{name}': {len(trials)} trial(s)  model={model_id}  task={task_id}  smoke={smoke_test}")

    if dry_run:
        for i, p in enumerate(trials):
            click.echo(f"  trial {i:02d}: {_fmt_params(p)}")
        return

    src_name = "smoke_train.jsonl" if smoke_test else "train.jsonl"
    data_path = REPO_ROOT / "data" / "prepared" / task_id / src_name
    if not data_path.exists():
        raise click.ClickException(f"Training data not found: {data_path}\nRun prepare first.")

    out_dir = REPO_ROOT / "results" / "sweeps" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_results: list[dict] = []
    for i, params in enumerate(trials):
        trial_short = f"{model_id}-sw{i:02d}"
        click.echo(f"\n── Trial {i + 1}/{len(trials)}: {_fmt_params(params)} ──")
        trial_model_cfg, trial_task_cfg = _apply_params(
            base_model_cfg, task_cfg, params, trial_short
        )
        try:
            meta = train_one(trial_model_cfg, trial_task_cfg, data_path, dry_run=False,
                             smoke_test=smoke_test, ctx=f"sw{i:02d}")
            diag = meta.get("training_diagnostics") or {}
            trial_results.append({
                "trial": i,
                "model_short": trial_short,
                "params": {k: float(v) if isinstance(v, float) else v for k, v in params.items()},
                "train_loss": meta.get("train_loss"),
                "eval_loss": meta.get("eval_loss"),
                "training_time_min": meta.get("training_time_min"),
                "converged": diag.get("converged"),
                "plateaued": diag.get("plateaued"),
                "diverged": diag.get("diverged"),
                "anomalies": diag.get("anomalies", []),
            })
        except Exception as exc:
            click.echo(f"  ERROR: {exc}", err=True)
            trial_results.append({"trial": i, "model_short": trial_short,
                                   "params": params, "error": str(exc)})

    # ── Comparison table ───────────────────────────────────────────────────
    # Sort by eval_loss (held-out val) — train_loss would pick the most-overfit
    # trial, which is the opposite of what we want from a sweep.
    # Fall back to train_loss only for trials that ran without a val split.
    click.echo(f"\n{'─' * 68}")
    click.echo(f"Sweep '{name}' — {len(trial_results)} trial(s) complete")
    click.echo(f"{'─' * 68}")
    ranked = [r for r in trial_results if r.get("eval_loss") is not None]
    no_val = [r for r in trial_results
              if r.get("eval_loss") is None and r.get("train_loss") is not None]
    ranked.sort(key=lambda r: r["eval_loss"])
    no_val.sort(key=lambda r: r["train_loss"])
    valid = ranked + no_val
    if valid:
        click.echo(
            f"{'#':>3}  {'EvalLoss':>9}  {'TrainLoss':>9}  {'Min':>6}  "
            f"{'C':>1}{'P':>1}{'D':>1}  Params"
        )
        click.echo(f"{'':>3}  {'(ranked)':>9}  {'':>9}  {'':>6}  (converged/plateaued/diverged)")
        for r in valid:
            t_min = r.get("training_time_min") or 0.0
            c = "Y" if r.get("converged") else ("?" if r.get("converged") is None else "N")
            p = "Y" if r.get("plateaued") else "."
            d = "Y" if r.get("diverged") else "."
            ev = r.get("eval_loss")
            tr = r.get("train_loss")
            ev_s = f"{ev:>9.4f}" if ev is not None else f"{'—':>9}"
            tr_s = f"{tr:>9.4f}" if tr is not None else f"{'—':>9}"
            click.echo(f"{r['trial']:>3}  {ev_s}  {tr_s}  {t_min:>6.1f}  {c}{p}{d}  {_fmt_params(r['params'])}")

    summary = {"name": name, "sweep_config": sweep_cfg, "results": trial_results}
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    click.echo(f"\nSaved: {summary_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
