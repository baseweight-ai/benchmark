"""Upload fine-tuned adapters and prediction logs to HuggingFace Hub."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

REPO_ROOT = Path(__file__).parent.parent
load_dotenv(REPO_ROOT / ".env")

ALL_TASKS = ["banking77", "cuad", "ledgar", "fpb", "medmcqa"]
ALL_MODELS = ["qwen3-8b"]

HF_COLLECTION_SLUG = "baseweight-ai/baseweight-benchmark-adapters"
HF_ORG = "baseweight-ai"  # 'baseweight' was taken on HF; the org handle is baseweight-ai


class ModelConfig(BaseModel):
    model_id: str
    model_short: str
    fallback_model_id: Optional[str] = None


def load_model_config(model_id: str) -> ModelConfig:
    path = REPO_ROOT / "configs" / "training" / f"{model_id}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return ModelConfig(**{k: v for k, v in data.items() if k in ModelConfig.model_fields})


def render_model_card(
    model_short: str,
    task_id: str,
    condition: str,
    base_model_id: str,
    n_train: int,
    metric_id: str,
    metric_value: Optional[float],
    training_cost: Optional[float],
    training_time_min: Optional[float],
) -> str:
    """Render model card from template."""
    template_path = REPO_ROOT / "docs" / "model_card_template.md"
    if not template_path.exists():
        return f"# {model_short}/{task_id}/{condition}\n\nBaseweight Benchmark adapter."

    with open(template_path) as f:
        template = f.read()

    metric_str = f"{metric_value:.4f}" if metric_value is not None else "N/A"
    cost_str = f"${training_cost:.3f}" if training_cost is not None else "N/A"
    time_str = f"{training_time_min:.1f} min" if training_time_min is not None else "N/A"

    return (
        template
        .replace("{{MODEL_SHORT}}", model_short)
        .replace("{{TASK_ID}}", task_id)
        .replace("{{CONDITION}}", condition)
        .replace("{{BASE_MODEL_ID}}", base_model_id)
        .replace("{{N_TRAIN}}", str(n_train))
        .replace("{{METRIC_ID}}", metric_id)
        .replace("{{METRIC_VALUE}}", metric_str)
        .replace("{{TRAINING_COST}}", cost_str)
        .replace("{{TRAINING_TIME}}", time_str)
        .replace("{{DATE}}", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    )


def upload_adapter(
    model_cfg: ModelConfig,
    task_id: str,
    condition: str,
    dry_run: bool,
    hf_org: str,
) -> Optional[str]:
    """Upload LoRA adapter to HuggingFace. Returns repo URL or None."""
    adapter_path = REPO_ROOT / "results" / "adapters" / "local" / model_cfg.model_short / task_id / condition
    meta_path = REPO_ROOT / "results" / "training" / "local" / model_cfg.model_short / task_id / condition / "metadata.json"

    if not adapter_path.exists():
        click.echo(f"  SKIP [{model_cfg.model_short}/{task_id}/{condition}]: adapter not found")
        return None

    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    repo_id = f"{hf_org}/{model_cfg.model_short}-{task_id}-{condition}"

    if dry_run:
        click.echo(f"  [dry-run] Would upload adapter to {repo_id}")
        return f"https://huggingface.co/{repo_id}"

    from huggingface_hub import HfApi, create_repo
    api = HfApi(token=os.environ.get("HF_TOKEN"))

    # Create or retrieve repo
    try:
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
    except Exception as exc:
        click.echo(f"  WARNING: could not create repo {repo_id}: {exc}", err=True)
        return None

    # Load task config for metric info
    task_cfg_path = REPO_ROOT / "configs" / "tasks" / f"{task_id}.yaml"
    with open(task_cfg_path) as f:
        task_data = yaml.safe_load(f)
    metric_id = task_data.get("metric_id", "")

    # Render and upload model card
    card = render_model_card(
        model_short=model_cfg.model_short,
        task_id=task_id,
        condition=condition,
        base_model_id=meta.get("model_used", model_cfg.model_id),
        n_train=meta.get("n_train", 0),
        metric_id=metric_id,
        metric_value=None,  # populated later from summaries
        training_cost=meta.get("training_cost"),
        training_time_min=meta.get("training_time_min"),
    )

    # Write README.md temporarily
    readme_path = adapter_path / "README.md"
    with open(readme_path, "w") as f:
        f.write(card)

    # Upload adapter folder
    try:
        api.upload_folder(
            folder_path=str(adapter_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add {model_cfg.model_short}/{task_id}/{condition} adapter",
        )
        url = f"https://huggingface.co/{repo_id}"
        click.echo(f"  Uploaded adapter to {url}")
        return url
    except Exception as exc:
        click.echo(f"  ERROR uploading {repo_id}: {exc}", err=True)
        return None
    finally:
        # Clean up README from local adapter dir
        if readme_path.exists():
            readme_path.unlink()


def upload_predictions(model_short: str, task_id: str, dry_run: bool, hf_org: str) -> Optional[str]:
    """Upload prediction log JSONL files to HuggingFace dataset repo."""
    pred_dir = REPO_ROOT / "results" / "predictions" / "local" / model_short / task_id
    if not pred_dir.exists():
        click.echo(f"  SKIP [{model_short}/{task_id}]: no predictions directory")
        return None

    files = list(pred_dir.glob("*.jsonl"))
    if not files:
        click.echo(f"  SKIP [{model_short}/{task_id}]: no prediction files")
        return None

    repo_id = f"{hf_org}/baseweight-benchmark-predictions"

    if dry_run:
        click.echo(f"  [dry-run] Would upload {len(files)} prediction files for {model_short}/{task_id} to {repo_id}")
        return f"https://huggingface.co/datasets/{repo_id}"

    from huggingface_hub import HfApi, create_repo
    api = HfApi(token=os.environ.get("HF_TOKEN"))

    try:
        create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)
    except Exception as exc:
        click.echo(f"  WARNING: {exc}", err=True)

    try:
        for f in files:
            remote_path = f"predictions/{model_short}/{task_id}/{f.name}"
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Add predictions: {model_short}/{task_id}/{f.name}",
            )
        url = f"https://huggingface.co/datasets/{repo_id}"
        click.echo(f"  Uploaded predictions for {model_short}/{task_id} to {url}")
        return url
    except Exception as exc:
        click.echo(f"  ERROR uploading predictions: {exc}", err=True)
        return None


def ensure_collection(dry_run: bool, hf_org: str) -> Optional[str]:
    """Create HF collection for all benchmark adapters if it doesn't exist."""
    if dry_run:
        click.echo(f"  [dry-run] Would ensure HF collection: {HF_COLLECTION_SLUG}")
        return f"https://huggingface.co/collections/{HF_COLLECTION_SLUG}"

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        # List existing collections
        collections = list(api.list_collections(owner=hf_org))
        slug_part = HF_COLLECTION_SLUG.split("/")[-1]
        existing = next((c for c in collections if slug_part in (c.slug or "")), None)
        if existing:
            click.echo(f"  Collection already exists: {existing.url}")
            return existing.url
        # Create collection
        collection = api.create_collection(
            title="Baseweight Benchmark Adapters",
            namespace=hf_org,
            description="QLoRA fine-tuned adapters from the Baseweight Benchmark: comparing open-source models against frontier APIs on 6 vertical SaaS tasks.",
            private=False,
        )
        click.echo(f"  Created collection: {collection.url}")
        return collection.url
    except Exception as exc:
        click.echo(f"  WARNING: could not manage collection: {exc}", err=True)
        return None


@click.command()
@click.option("--model", default="all", help="Model ID (qwen3-8b|all)")
@click.option("--task", default="all", help="Task ID or 'all'")
@click.option("--condition", default="all", help="lora|all")
@click.option("--upload-predictions", "do_predictions", is_flag=True, help="Also upload prediction logs")
@click.option("--hf-org", default=HF_ORG, help="HuggingFace org or username")
@click.option("--dry-run", is_flag=True)
def main(
    model: str,
    task: str,
    condition: str,
    do_predictions: bool,
    hf_org: str,
    dry_run: bool,
) -> None:
    """Upload fine-tuned adapters and optionally prediction logs to HuggingFace."""
    if not dry_run and not os.environ.get("HF_TOKEN"):
        click.echo("ERROR: HF_TOKEN not set", err=True)
        sys.exit(1)

    model_ids = ALL_MODELS if model == "all" else [model]
    task_ids = ALL_TASKS if task == "all" else [task]
    conditions = ["lora"] if condition == "all" else [condition]

    # Ensure collection exists
    ensure_collection(dry_run, hf_org)

    failures = []
    uploaded_repos: list[str] = []

    for mid in model_ids:
        model_cfg = load_model_config(mid)
        for tid in task_ids:
            # Upload adapters
            for cond in conditions:
                try:
                    url = upload_adapter(model_cfg, tid, cond, dry_run, hf_org)
                    if url:
                        uploaded_repos.append(url)
                except Exception as exc:
                    click.echo(f"  ERROR [{mid}/{tid}/{cond}]: {exc}", err=True)
                    failures.append((f"{mid}/{tid}/{cond}", str(exc)))

            # Upload predictions
            if do_predictions:
                try:
                    upload_predictions(model_cfg.model_short, tid, dry_run, hf_org)
                except Exception as exc:
                    click.echo(f"  ERROR [predictions/{mid}/{tid}]: {exc}", err=True)
                    failures.append((f"predictions/{mid}/{tid}", str(exc)))

    # Add uploaded repos to collection
    if uploaded_repos and not dry_run:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ.get("HF_TOKEN"))
            collections = list(api.list_collections(owner=hf_org))
            slug_part = HF_COLLECTION_SLUG.split("/")[-1]
            collection = next((c for c in collections if slug_part in (c.slug or "")), None)
            if collection:
                for repo_url in uploaded_repos:
                    repo_id = repo_url.replace("https://huggingface.co/", "")
                    try:
                        api.add_collection_item(
                            collection_slug=collection.slug,
                            item_id=repo_id,
                            item_type="model",
                            exists_ok=True,
                        )
                    except Exception:
                        pass
                click.echo(f"  Added {len(uploaded_repos)} items to collection")
        except Exception as exc:
            click.echo(f"  WARNING: could not add items to collection: {exc}", err=True)

    if failures:
        click.echo(f"\nFAILED ({len(failures)}):")
        for key, err in failures:
            click.echo(f"  {key}: {err}")
        sys.exit(1)
    click.echo(f"\nUpload complete. {len(uploaded_repos)} repos uploaded.")


if __name__ == "__main__":
    main()
