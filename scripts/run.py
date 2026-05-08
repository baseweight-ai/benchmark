"""Pipeline DAG runner — Python entry point called by run.sh.

Can also be invoked directly:
    python scripts/run.py --stages download,prepare --task fpb --dry-run
    python scripts/run.py --config configs/my_run.yaml --smoke-test
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

REPO_ROOT = Path(__file__).parent.parent
SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS))

from pipeline.dag import DAGRunner, Stage, StageResult, StageStatus
from pipeline.log import configure, get_logger
from pipeline.run_config import RunConfig

_log = get_logger("run")

_ALL_STAGE_IDS = [
    "download", "prepare",
    "train-local", "train-api",
    "eval-local", "eval-api",
    "classify", "dashboard", "catalog",
]

_STATUS_ICON = {
    StageStatus.DONE:    "✓",
    StageStatus.FAILED:  "✗",
    StageStatus.SKIPPED: "-",
    StageStatus.RUNNING: "…",
    StageStatus.PENDING: "?",
}


def _build_stages(cfg: RunConfig, selected_ids: list[str], run_id: str | None = None) -> list[Stage]:
    """Construct Stage objects for the selected stage IDs."""
    py = sys.executable
    smoke = ["--smoke-test"] if cfg.smoke_test else []
    dry = ["--dry-run"] if cfg.dry_run else []
    force = ["--force"] if cfg.force else []
    resolved_tasks = cfg.resolved_tasks()
    if "all" in cfg.tasks:
        task_arg = "all"
    elif len(resolved_tasks) == 1:
        task_arg = resolved_tasks[0]
    else:
        task_arg = ",".join(resolved_tasks)
    task = ["--task", task_arg]

    local_model = cfg.effective_local_model() or "all"
    api_model = cfg.effective_api_model_arg()
    t = cfg.timeouts

    run_id_flag = ["--run-id", run_id] if run_id else []

    all_stages = [
        Stage(
            id="download",
            cmd=[py, str(SCRIPTS / "download_data.py"), *task, *smoke, *dry],
            depends_on=[],
            compute="cpu",
            timeout_s=t.download_s,
            description="Download raw task datasets from HuggingFace.",
        ),
        Stage(
            id="prepare",
            cmd=[py, str(SCRIPTS / "prepare_datasets.py"), *task, *smoke, *dry],
            depends_on=["download"],
            compute="cpu",
            timeout_s=t.prepare_s,
            description="Format raw data into train/test JSONL files for each task.",
        ),
        Stage(
            id="train-local",
            cmd=[py, str(SCRIPTS / "train_local.py"), *task, "--model", local_model, *smoke, *dry],
            depends_on=["prepare"],
            compute="gpu",
            timeout_s=t.train_local_s,
            description=f"Fine-tune {local_model} with QLoRA on the local GPU.",
        ),
        Stage(
            id="train-api",
            cmd=[py, str(SCRIPTS / "train_api.py"), *task, "--model", api_model, *smoke, *dry, *force],
            depends_on=["prepare"],
            compute="cloud",
            timeout_s=t.train_api_s,
            description=f"Submit and wait for OpenAI SFT fine-tuning jobs ({api_model}).",
        ),
        Stage(
            id="eval-local",
            cmd=[py, str(SCRIPTS / "eval_local.py"), *task, "--model", local_model, *smoke, *dry],
            depends_on=["train-local"],
            compute="gpu",
            timeout_s=t.eval_local_s,
            description=f"Evaluate fine-tuned {local_model} adapter via vLLM.",
        ),
        Stage(
            id="eval-api",
            cmd=[py, str(SCRIPTS / "eval_api.py"), *task, "--model", api_model, *smoke, *dry],
            depends_on=["train-api"],
            compute="cloud",
            timeout_s=t.eval_api_s,
            description=f"Evaluate fine-tuned and base API models ({api_model}).",
        ),
        Stage(
            id="classify",
            cmd=[py, str(SCRIPTS / "classify_errors.py"), *task, *dry],
            depends_on=["eval-local", "eval-api"],
            compute="cpu",
            timeout_s=t.classify_s,
            requires_all_deps=True,
            description="Classify prediction errors and generate per-task summaries.",
        ),
        Stage(
            id="dashboard",
            cmd=[py, str(SCRIPTS / "generate_dashboard_data.py"), *run_id_flag, *dry],
            depends_on=["classify"],
            compute="cpu",
            timeout_s=t.dashboard_s,
            description="Aggregate results into dashboard JSON for the web UI.",
        ),
        Stage(
            id="catalog",
            cmd=[py, str(SCRIPTS / "catalog.py"), "rebuild"],
            depends_on=["dashboard"],
            compute="cpu",
            timeout_s=t.catalog_s,
            description="Rebuild the run catalog index from all available results.",
        ),
    ]

    # Inject per-seed eval stages for seeds 1..n_eval_seeds-1
    working_ids = list(selected_ids)
    if cfg.n_eval_seeds > 1:
        classify_stage = next((s for s in all_stages if s.id == "classify"), None)
        for n in range(1, cfg.n_eval_seeds):
            if "eval-local" in working_ids:
                sid = f"eval-local-seed{n}"
                all_stages.append(Stage(
                    id=sid,
                    cmd=[py, str(SCRIPTS / "eval_local.py"), *task, "--model", local_model,
                         "--eval-seed", str(n), *smoke, *dry],
                    depends_on=["train-local"],
                    compute="gpu",
                    timeout_s=t.eval_local_s,
                ))
                working_ids.append(sid)
                if classify_stage:
                    classify_stage.depends_on.append(sid)
            if "eval-api" in working_ids:
                sid = f"eval-api-seed{n}"
                all_stages.append(Stage(
                    id=sid,
                    cmd=[py, str(SCRIPTS / "eval_api.py"), *task, "--model", api_model,
                         "--eval-seed", str(n), *smoke, *dry],
                    depends_on=["train-api"],
                    compute="cloud",
                    timeout_s=t.eval_api_s,
                ))
                working_ids.append(sid)
                if classify_stage:
                    classify_stage.depends_on.append(sid)

    selected = [s for s in all_stages if s.id in working_ids]
    selected_set = {s.id for s in selected}
    for s in selected:
        s.depends_on = [d for d in s.depends_on if d in selected_set]
    return selected


def _check_training_costs(repo_root: Path) -> float:
    total = 0.0
    for p in (repo_root / "results" / "training" / "api").glob("*/*/*/metadata.json"):
        try:
            total += json.loads(p.read_text()).get("training_cost", 0.0)
        except Exception:
            pass
    return total


def _init_run_manifest(repo_root: Path):  # -> RunManifest | None
    try:
        from pipeline.registry import RunManifest, new_run_id, save_manifest
        from pipeline.versioning import configs_sha, git_sha
        sha = git_sha()
        manifest = RunManifest(
            run_id=new_run_id(),
            git_sha=sha,
            config_sha=configs_sha([
                repo_root / "configs" / "pipeline.yaml",
                repo_root / "environment.yml",
            ]),
        )
        save_manifest(manifest, repo_root)
        return manifest
    except Exception:
        return None


def _print_results(results: dict[str, StageResult], ordered_ids: list[str]) -> None:
    click.echo("\n━━━ Pipeline Results ━━━")
    for sid in ordered_ids:
        if sid not in results:
            continue
        r = results[sid]
        icon = _STATUS_ICON[r.status]
        elapsed = f"{r.elapsed_s:7.1f}s" if r.status != StageStatus.SKIPPED else "       "
        err = f"  {r.error}" if r.error else ""
        click.echo(f"  {icon}  {sid:<14} {r.status.value:<8} {elapsed}{err}")


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True), default=None,
              help="YAML RunConfig file; CLI flags overlay its values")
@click.option("--stages", default="all",
              help="Comma-separated stage IDs or 'all': " + ",".join(_ALL_STAGE_IDS))
@click.option("--task", default=None, help="Task ID or 'all' (overrides config)")
@click.option("--local-model", "local_model", default=None,
              help="Local model ID (overrides config)")
@click.option("--api-model", "api_model", default=None,
              help="API model ID or 'all' (overrides config)")
@click.option("--smoke-test", is_flag=True)
@click.option("--dry-run", is_flag=True)
@click.option("--force", is_flag=True, help="Re-run even if outputs exist")
@click.option("--cpu", is_flag=True, help="Skip GPU stages (train-local, eval-local)")
@click.option("--test-sampling", "test_sampling", is_flag=True,
              help="Run only download + prepare with full (non-smoke) data to verify sampling. --task still applies.")
def main(
    config_path: str | None,
    stages: str,
    task: str | None,
    local_model: str | None,
    api_model: str | None,
    smoke_test: bool,
    dry_run: bool,
    force: bool,
    cpu: bool,
    test_sampling: bool,
) -> None:
    """Run pipeline stages as a dependency-aware DAG."""
    configure(REPO_ROOT)

    cfg = RunConfig.from_yaml(Path(config_path)) if config_path else RunConfig()
    for attr, val in (("smoke_test", smoke_test), ("dry_run", dry_run), ("force", force)):
        if val:
            setattr(cfg, attr, True)
    if task and task != "all":
        cfg.tasks = [t.strip() for t in task.split(",")]
    if local_model:
        cfg.local_models = [local_model]
    if api_model and api_model != "all":
        cfg.api_models = [api_model]

    # Resolve stage selection
    if test_sampling:
        selected_ids = ["download", "prepare"]
        cfg.smoke_test = False
        cfg.dry_run = False
        cfg.force = True
        click.echo("--test-sampling: running download+prepare with full production data.")
    elif stages == "all":
        selected_ids = list(_ALL_STAGE_IDS)
    else:
        selected_ids = [s.strip() for s in stages.split(",") if s.strip()]

    unknown = [s for s in selected_ids if s not in _ALL_STAGE_IDS]
    if unknown:
        raise click.UsageError(f"Unknown stage(s): {', '.join(unknown)}")

    if cpu:
        selected_ids = [s for s in selected_ids if s not in ("train-local", "eval-local")]

    manifest = _init_run_manifest(REPO_ROOT)
    stage_list = _build_stages(cfg, selected_ids, run_id=manifest.run_id if manifest else None)
    if not stage_list:
        click.echo("No stages to run.")
        return
    caps = cfg.cost_caps

    def after_stage(result: StageResult) -> bool:
        if manifest is not None:
            try:
                from pipeline.registry import save_manifest
                manifest.log_stage(result.stage_id, success=(result.status == StageStatus.DONE))
                save_manifest(manifest, REPO_ROOT)
            except Exception:
                pass
        if result.status == StageStatus.DONE and result.stage_id in ("train-api", "eval-api"):
            total = _check_training_costs(REPO_ROOT)
            if total > caps.total_usd:
                click.echo(
                    f"\n  WARNING: accumulated API cost ${total:.2f} exceeds cap ${caps.total_usd:.2f}",
                    err=True,
                )
                _log.warning("cost cap exceeded", event="cost_cap", cost_usd=round(total, 4))
        return True

    runner = DAGRunner(stage_list, REPO_ROOT, after_stage=after_stage)
    results = runner.run()

    _print_results(results, selected_ids)

    failed = [r for r in results.values() if r.status == StageStatus.FAILED]
    if failed:
        click.echo(f"\nFAILED: {', '.join(r.stage_id for r in failed)}", err=True)
        sys.exit(1)
    click.echo("\nDone.")


if __name__ == "__main__":
    main()
