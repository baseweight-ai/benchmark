"""Pipeline DAG runner — Python entry point called by run.sh.

Can also be invoked directly:
    python scripts/run.py --stages download,prepare --task fpb --dry-run
    python scripts/run.py --config configs/my_run.yaml --smoke-test
"""
from __future__ import annotations

import sys
from pathlib import Path

import click

REPO_ROOT = Path(__file__).parent.parent
SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS))

from pipeline.dag import DAGRunner, Stage, StageResult, StageStatus
from pipeline.log import configure
from pipeline.run_config import RunConfig

_ALL_STAGE_IDS = [
    "download", "prepare",
    "train-local",
    "eval-local", "eval-api",
    # "classify" is the user-facing alias; it expands to classify-local +
    # classify-api in _build_stages so each source can be scoped to its own
    # model (local_model vs api_model differ).
    "classify", "classify-local", "classify-api",
    "dashboard", "catalog",
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
            depends_on=["prepare"],
            compute="cloud",
            timeout_s=t.eval_api_s,
            description=f"Evaluate base API models zero-shot and 5-shot ({api_model}).",
        ),
        Stage(
            id="classify-local",
            # Per-source split so each classify run is scoped to its source's
            # model (local_model and api_model differ — one --model can't
            # filter both correctly).
            cmd=[py, str(SCRIPTS / "classify_errors.py"), *task,
                 "--source", "local", "--model", local_model, *smoke, *dry],
            depends_on=["eval-local"],
            compute="cpu",
            timeout_s=t.classify_s,
            description="Classify local prediction errors (scoped to the local model).",
        ),
        Stage(
            id="classify-api",
            cmd=[py, str(SCRIPTS / "classify_errors.py"), *task,
                 "--source", "api", "--model", api_model, *smoke, *dry],
            depends_on=["eval-api"],
            compute="cpu",
            timeout_s=t.classify_s,
            description="Classify API prediction errors (scoped to the API model).",
        ),
        Stage(
            id="dashboard",
            cmd=[py, str(SCRIPTS / "generate_dashboard_data.py"), *run_id_flag, *smoke, *dry],
            depends_on=["classify-local", "classify-api"],
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

    # "classify" is a user-facing alias for both per-source classify stages.
    working_ids = list(selected_ids)
    if "classify" in working_ids:
        idx = working_ids.index("classify")
        working_ids = working_ids[:idx] + ["classify-local", "classify-api"] + working_ids[idx + 1:]

    # Inject per-seed eval stages for seeds 1..n_eval_seeds-1
    if cfg.n_eval_seeds > 1:
        classify_local = next((s for s in all_stages if s.id == "classify-local"), None)
        classify_api = next((s for s in all_stages if s.id == "classify-api"), None)
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
                if classify_local:
                    classify_local.depends_on.append(sid)
            if "eval-api" in working_ids:
                sid = f"eval-api-seed{n}"
                all_stages.append(Stage(
                    id=sid,
                    cmd=[py, str(SCRIPTS / "eval_api.py"), *task, "--model", api_model,
                         "--eval-seed", str(n), *smoke, *dry],
                    depends_on=["prepare"],
                    compute="cloud",
                    timeout_s=t.eval_api_s,
                ))
                working_ids.append(sid)
                if classify_api:
                    classify_api.depends_on.append(sid)

    selected = [s for s in all_stages if s.id in working_ids]
    selected_set = {s.id for s in selected}
    for s in selected:
        s.depends_on = [d for d in s.depends_on if d in selected_set]
    return selected


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
    cpu: bool,
    test_sampling: bool,
) -> None:
    """Run pipeline stages as a dependency-aware DAG."""
    configure(REPO_ROOT)

    # `configs/run_defaults.yaml` is the source of truth for documented
    # defaults (n_eval_seeds, timeouts, etc.). When no --config is passed,
    # load it explicitly — otherwise the Python dataclass defaults silently
    # diverge from the YAML the user reads in the repo. An explicit --config
    # always wins, including when a sweep or test points at a different YAML.
    default_cfg_path = REPO_ROOT / "configs" / "run_defaults.yaml"
    if config_path:
        cfg = RunConfig.from_yaml(Path(config_path))
    elif default_cfg_path.is_file():
        cfg = RunConfig.from_yaml(default_cfg_path)
    else:
        cfg = RunConfig()
    for attr, val in (("smoke_test", smoke_test), ("dry_run", dry_run)):
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

    def after_stage(result: StageResult) -> bool:
        if manifest is not None:
            try:
                from pipeline.registry import save_manifest
                manifest.log_stage(result.stage_id, success=(result.status == StageStatus.DONE))
                save_manifest(manifest, REPO_ROOT)
            except Exception:
                pass
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
