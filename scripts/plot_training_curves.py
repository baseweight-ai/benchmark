#!/usr/bin/env python3
"""Plot training and validation loss curves from a training run's metadata.

Reads results/training/local/<model>/<task>/<condition>/metadata.json and writes
a PNG: training loss over steps (log y) and, when a validation split was used,
validation loss per epoch (linear y) with the best epoch and overfitting verdict
marked. Standalone analysis tool — reads pipeline outputs, writes only to plots/.
"""
from __future__ import annotations

import json
from pathlib import Path

import click

REPO_ROOT = Path(__file__).parent.parent

_VERDICT = {
    True: "overfitting detected",
    False: "no overfitting",
    None: "overfitting not assessable",
}


def plot_run(meta_path: Path, out_path: Path) -> dict:
    """Render the loss curves for one run. Returns the run's metadata dict."""
    import matplotlib
    matplotlib.use("Agg")  # headless — no display needed
    import matplotlib.pyplot as plt

    meta = json.loads(meta_path.read_text())
    train = [(d["step"], d["loss"]) for d in meta.get("loss_history", [])]
    evals = [(d["epoch"], d["step"], d["eval_loss"])
             for d in meta.get("eval_loss_history", [])]
    if not train:
        raise ValueError(f"{meta_path} has no loss_history — nothing to plot")

    diag = meta.get("training_diagnostics", {})
    run = f"{meta.get('model_id', '?')} / {meta.get('task_id', '?')} / {meta.get('condition', '?')}"
    # Prefer epochs_completed when present (early stopping shortened the run);
    # fall back to the cap for older metadata.
    ep_disp = meta.get("epochs_completed", meta.get("epochs", "?"))
    head = (f"{run}   (git {meta.get('git_sha', '?')}, "
            f"{ep_disp} epochs, n_train={meta.get('n_train', '?')})")

    if evals:
        fig, (ax_t, ax_v) = plt.subplots(1, 2, figsize=(13, 5))
    else:
        fig, ax_t = plt.subplots(figsize=(7, 5))
        ax_v = None

    # ── Panel 1: training loss over steps (log y) ──────────────────────────
    steps = [s for s, _ in train]
    losses = [l for _, l in train]
    ax_t.semilogy(steps, losses, "o-", color="#1f77b4", lw=1.6, ms=4)
    for _, st, _ in evals:
        ax_t.axvline(st, color="0.85", lw=0.8, zorder=0)  # epoch boundaries
    ax_t.set(xlabel="training step", ylabel="train loss — completion-only CE (log)",
             title="Training loss")
    ax_t.grid(alpha=0.25, which="both")

    # ── Panel 2: validation loss per epoch (linear y) ──────────────────────
    if ax_v is not None:
        eps = [e for e, _, _ in evals]
        vls = [v for _, _, v in evals]
        best = min(vls)
        best_ep = eps[vls.index(best)]
        ax_v.plot(eps, vls, "o-", color="#1f77b4", lw=2, ms=7, label="val loss")
        ax_v.scatter([best_ep], [best], marker="*", s=320, color="#2ca02c",
                     zorder=5, label=f"best — epoch {best_ep:g}")
        of = diag.get("overfitting_detected")
        title = f"Validation loss per epoch  —  {_VERDICT.get(of, of)}"
        if vls[-1] > best:
            rise = (vls[-1] / best - 1) * 100
            ax_v.axvspan(best_ep, eps[-1], color="#d62728", alpha=0.10)
            title += f"  (+{rise:.0f}% from min)"
        ax_v.set(xlabel="epoch", ylabel="validation loss (linear)", title=title)
        ax_v.legend(fontsize=8)
        ax_v.grid(alpha=0.25)

    fig.suptitle(head, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return meta


@click.command()
@click.option("--model", required=True, help="Model short id, e.g. qwen3-8b")
@click.option("--task", required=True, help="Task id, e.g. fpb")
@click.option("--condition", default="lora", show_default=True, help="Training condition")
@click.option("--out", "out", default=None, type=click.Path(),
              help="Output PNG path (default: plots/<model>__<task>__<condition>.png)")
def main(model: str, task: str, condition: str, out: str | None) -> None:
    """Plot training/validation loss curves for one training run."""
    meta_path = (REPO_ROOT / "results" / "training" / "local"
                 / model / task / condition / "metadata.json")
    if not meta_path.exists():
        raise SystemExit(f"No training metadata at {meta_path}")
    out_path = Path(out) if out else REPO_ROOT / "plots" / f"{model}__{task}__{condition}.png"
    meta = plot_run(meta_path, out_path)
    diag = meta.get("training_diagnostics", {})
    click.echo(f"Wrote {out_path}")
    click.echo(f"  converged={diag.get('converged')}  plateaued={diag.get('plateaued')}  "
               f"diverged={diag.get('diverged')}  "
               f"overfitting_detected={diag.get('overfitting_detected')}")
    click.echo(f"  train_loss={meta.get('train_loss')}  eval_loss={meta.get('eval_loss')}")


if __name__ == "__main__":
    main()
