"""Evaluate frontier API models on benchmark tasks."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import traceback as _tb

import click
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from checkpoint_utils import append_jsonl, finalize_partial, load_partial_ids, partial_path
from utils import build_messages, load_jsonl, load_label_set as _load_label_set, question_id, rows_hash as _rows_hash, read_prompt_sha as _read_prompt_sha, seed_sample_questions
from pipeline.config import get_model_conditions, get_openai_models, get_reasoning_capable, get_tasks
from pipeline.log import configure, get_logger
from pipeline.cache import code_closure_hash, dict_hash, record_fingerprint, reuse_is_valid
from pipeline.paths import pred_path, prepared_path
from pipeline.providers import call_openai  # noqa: F401  # re-exported for test patching


# ── Token accounting helpers ───────────────────────────────────────────────

from functools import lru_cache


@lru_cache(maxsize=8)
def _tiktoken_encoder(model_str: str):
    """Return a tiktoken encoder for `model_str`, falling back to cl100k_base
    when the model is unknown to tiktoken (newer-than-tiktoken IDs like
    gpt-5.4-mini). None when tiktoken itself is unavailable."""
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        return tiktoken.encoding_for_model(model_str)
    except KeyError:
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            return tiktoken.get_encoding("cl100k_base")


def count_answer_tokens(answer_text: str, model_str: str) -> Optional[int]:
    """Token count of the bare answer text, with the JSON envelope stripped.

    Used to surface `envelope_overhead_tokens = output_tokens - answer_only -
    reasoning` so cost comparisons across (response_format on / response_format
    off) regimes are apples-to-apples. Returns None when tiktoken is missing
    or the answer is empty.
    """
    if not answer_text:
        return 0
    enc = _tiktoken_encoder(model_str)
    if enc is None:
        return None
    try:
        return len(enc.encode(answer_text))
    except Exception:
        return None

_log = get_logger("eval-api")

REPO_ROOT = Path(__file__).parent.parent
load_dotenv(REPO_ROOT / ".env")

ALL_TASKS: list[str] = get_tasks()
OPENAI_MODELS: dict[str, Optional[str]] = get_openai_models()
MODEL_CONDITIONS: dict[str, list[str]] = get_model_conditions()
REASONING_CAPABLE: dict[str, bool] = get_reasoning_capable()

# Reasoning fully off — the v1 benchmark holds every model to the same
# (non-reasoning) compute regime so the cost/latency comparison stays
# apples-to-apples. A later benchmark version may turn reasoning on.
REASONING_EFFORT_OFF = "none"

SMOKE_MODELS = ["gpt-5.4-nano"]
PROD_MODELS  = ["gpt-5.4-mini"]
ALL_API_MODELS = SMOKE_MODELS + PROD_MODELS

MAX_CONCURRENCY = 5


class TaskConfig(BaseModel):
    task_id: str
    max_output_tokens: int
    task_type: str
    # direct → the raw output IS the label (banking77/fpb/ledgar): the response
    #          is response_format-constrained and classify compares it directly.
    # tagged → the output is a chain-of-thought wrapped around <answer>X</answer>
    #          (medmcqa): unconstrained — a CoT cannot be pinned to a label set.
    answer_mode: str = "direct"
    skip_conditions: list[str] = []


def load_task_config(task_id: str) -> TaskConfig:
    path = REPO_ROOT / "configs" / "tasks" / f"{task_id}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return TaskConfig(**{k: v for k, v in data.items() if k in TaskConfig.model_fields})


def load_label_set(task_id: str) -> Optional[list[str]]:
    """Return the closed label set written by prepare_datasets, or None.

    Read to build the response_format schema that pins API classification
    answers to the same set eval_local constrains the local model with.
    """
    return _load_label_set(REPO_ROOT, task_id)


def build_label_response_format(labels: list[str]) -> dict:
    """OpenAI structured-output schema pinning the answer to one label.

    The API counterpart of vLLM's guided_choice: the model must return
    {"label": <one of labels>}. eval_local constrains the local model to the
    exact same set, so the closed-set comparison stays apples-to-apples.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "classification_label",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"label": {"type": "string", "enum": list(labels)}},
                "required": ["label"],
                "additionalProperties": False,
            },
        },
    }


def parse_constrained_label(text: str) -> str:
    """Unwrap the bare label from a response_format JSON object.

    response_format constrains the output to {"label": X}; storing the bare X
    keeps API predictions in the same shape as local (guided_choice) ones. On
    any parse failure the raw text is returned so the anomaly stays visible
    rather than being silently swallowed.
    """
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text
    if isinstance(obj, dict) and "label" in obj:
        return str(obj["label"])
    return text


def _eval_fingerprint(
    *,
    test_rows_hash: str,
    prompt_sha: str,
    few_shot_hash: Optional[str],
    label_set: Optional[list[str]],
    condition: str,
    eval_seed: int,
    model_str: str,
    reasoning_capable: bool,
    max_output_tokens: int,
    task_type: str,
    answer_mode: str,
) -> str:
    """Cache key for an API eval. Grouped so each section is auditable:
        code             — the eval/script closure (catches sampling-param
                           changes like temperature=0 → 0.7 in providers.py,
                           which don't appear as args here).
        data             — test rows, prompt, few-shot exemplars, label set.
        model_properties — the model and the request shape it sees.
        run              — condition + eval seed (which prompt variant / sample).
    A change in any field re-evals; nothing else does."""
    return dict_hash({
        "code": code_closure_hash(Path(__file__)),
        "data": {
            "test_rows": test_rows_hash,
            "prompt_sha": prompt_sha,
            "few_shot_hash": few_shot_hash,
            "label_set": label_set,
        },
        "model_properties": {
            "model": model_str,
            "reasoning_capable": reasoning_capable,
            "max_output_tokens": max_output_tokens,
            "task_type": task_type,
            "answer_mode": answer_mode,
        },
        "run": {
            "condition": condition,
            "eval_seed": eval_seed,
        },
    })


async def run_eval(
    model_id: str,
    task_id: str,
    condition: str,
    task_cfg: TaskConfig,
    dry_run: bool,
    smoke_test: bool = False,
    eval_seed: int = 0,
) -> None:
    """Evaluate one model/task/condition combination."""
    prepared = prepared_path(REPO_ROOT, task_id, smoke=smoke_test)
    suffix = "smoke_" if smoke_test else ""
    base_test_path = prepared / f"{suffix}test.jsonl"
    full_test_path = prepared / "test_full.jsonl"
    few_shot_path = prepared / ("smoke_train.jsonl" if smoke_test else "train.jsonl")

    if not base_test_path.exists():
        raise FileNotFoundError(f"test data not found: {base_test_path}")

    # For seed > 0, resample from the full test set when it's available.
    # Resample whole questions, not rows: chunked tasks (CUAD) have many rows per
    # question, and a question's windows must stay together. Degenerates to
    # row-level resampling for unchunked tasks (one row == one question).
    if eval_seed > 0 and full_test_path.exists() and not smoke_test:
        base_rows = load_jsonl(base_test_path)
        n_questions = len({question_id(r["id"]) for r in base_rows})
        full_rows = load_jsonl(full_test_path)
        test_rows = seed_sample_questions(full_rows, n_questions, eval_seed)
    else:
        test_rows = load_jsonl(base_test_path)

    labels_path = prepared / f"{suffix}test_labels.jsonl"
    full_labels_path = prepared / "test_full_labels.jsonl"
    if eval_seed > 0 and full_labels_path.exists() and not smoke_test:
        label_map = {r["id"]: r["label"] for r in load_jsonl(full_labels_path)}
    elif labels_path.exists():
        label_map = {r["id"]: r["label"] for r in load_jsonl(labels_path)}
    else:
        label_map = {}
    for row in test_rows:
        row["label"] = label_map.get(row["id"], "")

    # Prefer the curated few-shot pool emitted by prepare_datasets (covers
    # distinct labels) over the legacy "first 5 of train.jsonl" fallback.
    curated_few_shot = prepared / "few_shot.jsonl"
    if not smoke_test and curated_few_shot.exists():
        few_shot = load_jsonl(curated_few_shot)
    elif few_shot_path.exists():
        few_shot = load_jsonl(few_shot_path)[:5]
    else:
        few_shot = []
    prompt_sha = _read_prompt_sha(prepared)
    few_shot_hash = _rows_hash(few_shot) if few_shot else None

    # Resolve the model string. Never mutate the module-level OPENAI_MODELS dict.
    model_str = OPENAI_MODELS.get(model_id)
    if not model_str:
        raise ValueError(f"Model string not set for {model_id}")

    if dry_run:
        seed_label = f" seed={eval_seed}" if eval_seed > 0 else ""
        click.echo(f"  [dry-run] Would eval {model_id} on {task_id}/{condition}{seed_label} ({len(test_rows)} examples)")
        return

    cond_key = condition if eval_seed == 0 else f"{condition}_seed{eval_seed}"
    out_path = pred_path(REPO_ROOT, "api", model_id, task_id, cond_key, smoke=smoke_test)

    # Skip-if-unchanged: reuse the prediction file only when its fingerprint
    # still matches. The hash groups data, model_properties, run, and code —
    # see _eval_fingerprint above.
    fingerprint = _eval_fingerprint(
        test_rows_hash=_rows_hash(test_rows),
        prompt_sha=prompt_sha,
        few_shot_hash=few_shot_hash,
        label_set=load_label_set(task_id),
        condition=condition,
        eval_seed=eval_seed,
        model_str=model_str,
        reasoning_capable=REASONING_CAPABLE.get(model_id, False),
        max_output_tokens=task_cfg.max_output_tokens,
        task_type=task_cfg.task_type,
        answer_mode=task_cfg.answer_mode,
    )
    pp = partial_path(out_path)
    if reuse_is_valid(out_path, pp, fingerprint):
        click.echo(f"  SKIP [{model_id}/{task_id}/{condition}]: up-to-date")
        _log.info("eval skip", model=model_id, task=task_id, condition=condition,
                  event="stage_skip", reason="fingerprint match")
        return
    record_fingerprint(out_path, fingerprint)

    completed_ids = load_partial_ids(pp)
    pending_rows = [r for r in test_rows if r.get("id", "") not in completed_ids]

    if completed_ids:
        click.echo(f"  Resuming: {len(completed_ids)}/{len(test_rows)} rows already done")

    if not pending_rows:
        finalize_partial(pp, out_path)
        click.echo(f"  All {len(test_rows)} rows complete, finalized to {out_path.relative_to(REPO_ROOT)}")
        return
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    totals = [0, 0]  # [input_tokens, output_tokens]

    # Non-capable models reject the param entirely — only send when supported.
    reasoning_effort = REASONING_EFFORT_OFF if REASONING_CAPABLE.get(model_id, False) else None

    # Constrained decoding: classification tasks whose raw output IS the label
    # (answer_mode == "direct") get response_format pinned to labels.json — the
    # SAME closed set eval_local constrains the local model with via guided_choice.
    # Tagged/CoT tasks (medmcqa) and extraction (cuad) stay unconstrained.
    label_set = load_label_set(task_id)
    constrain = (
        task_cfg.task_type == "classification"
        and task_cfg.answer_mode == "direct"
        and label_set is not None
    )
    response_format = build_label_response_format(label_set) if constrain else None
    if constrain:
        click.echo(f"  response_format: {len(label_set)} labels for {task_id}")

    async def process_row(row: dict) -> None:
        msgs = build_messages(row, few_shot, condition)
        ground_truth = row.get("label", "")
        api_error = False
        try:
            text, in_tok, out_tok, reasoning_tok, lat, ttft, avg_logprob = await call_openai(
                client, model_str, msgs, task_cfg.max_output_tokens, semaphore,
                reasoning_effort=reasoning_effort,
                response_format=response_format,
            )
            # Unwrap {"label": X} → bare X so API predictions match the local
            # (guided_choice) output shape that classify_errors expects.
            if response_format is not None:
                text = parse_constrained_label(text)
        except Exception as exc:
            text, in_tok, out_tok, reasoning_tok, lat, ttft, avg_logprob = (
                f"ERROR: {exc}", 0, 0, 0, 0, 0.0, None
            )
            api_error = True
        totals[0] += in_tok
        totals[1] += out_tok
        # Envelope-aware accounting: count tokens of the (unwrapped) answer
        # text so the JSON envelope a response_format adds doesn't inflate the
        # apples-to-apples cost comparison vs constrained-decoding local runs.
        # For unconstrained outputs this just re-counts the response text
        # (close to the API-reported answer_tokens, modulo tokenizer drift).
        if api_error:
            answer_only = 0
        else:
            answer_only = count_answer_tokens(text, model_str)
        result = {
            "id": row.get("id", ""),
            "model": model_id,
            "condition": condition,
            "eval_seed": eval_seed,
            "prompt_sha": prompt_sha,
            "few_shot_hash": few_shot_hash,
            "input": msgs[-1]["content"] if msgs else "",
            "output": text,
            "ground_truth": ground_truth,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "reasoning_tokens": reasoning_tok,
            # Tokens of just the unwrapped answer. The complement
            # (output_tokens − reasoning_tokens − answer_only_tokens) is the
            # JSON envelope overhead and is surfaced separately downstream.
            # None when tiktoken can't tokenise (rare).
            "answer_only_tokens": answer_only,
            "latency_ms": lat,
            "ttft_ms": ttft,
            "avg_logprob": avg_logprob,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        append_jsonl(result, pp)

    # No API warmup: TLS handshake elevates TTFT on only the first row out of
    # the full eval set, which is negligible signal vs. the cost of dummy
    # billed requests. Local vLLM warmup is different — first ~20 requests
    # pay JIT + CUDA graph capture, so it's warranted there.
    seed_label = f" seed={eval_seed}" if eval_seed > 0 else ""
    _log.info("evaluating", model=model_id, task=task_id, condition=condition, eval_seed=eval_seed,
              n_rows=len(pending_rows), n_total=len(test_rows))
    click.echo(f"  Evaluating {model_id}/{task_id}/{condition}{seed_label} ({len(pending_rows)}/{len(test_rows)} rows)...")
    from tqdm.asyncio import tqdm
    await tqdm.gather(*[process_row(r) for r in pending_rows], desc=f"{model_id}/{task_id}")

    finalize_partial(pp, out_path)
    click.echo(f"  Saved {len(test_rows)} predictions to {out_path.relative_to(REPO_ROOT)}")
    _log.info("eval complete", model=model_id, task=task_id, condition=condition,
              event="stage_complete", n_rows=len(test_rows),
              total_input_tokens=totals[0], total_output_tokens=totals[1])


@click.command()
@click.option("--model", default="all", help=f"Model ID or 'all'. Choices: {', '.join(ALL_API_MODELS)}")
@click.option("--task", default="all", help="Task ID or 'all'")
@click.option("--condition", default="all", help="zero-shot|5-shot|all")
@click.option("--eval-seed", "eval_seed", default=0, type=int,
              help="Evaluation seed (0 = deterministic sample; >0 resamples from test_full.jsonl)")
@click.option("--dry-run", is_flag=True)
@click.option("--smoke-test", is_flag=True)
def main(model: str, task: str, condition: str, eval_seed: int, dry_run: bool, smoke_test: bool) -> None:
    """Evaluate frontier OpenAI models on benchmark tasks (zero-shot and 5-shot)."""
    configure(REPO_ROOT)
    default_models = SMOKE_MODELS if smoke_test else PROD_MODELS
    model_ids = default_models if model == "all" else [model]

    if not dry_run and any(m in OPENAI_MODELS for m in model_ids):
        if not os.environ.get("OPENAI_API_KEY"):
            click.echo("  WARNING: OPENAI_API_KEY not set", err=True)
            _log.warning("OPENAI_API_KEY not set")
    task_ids = ALL_TASKS if task == "all" else [task]
    failures = []

    async def run_all() -> None:
        for mid in model_ids:
            for tid in task_ids:
                try:
                    task_cfg = load_task_config(tid)
                    supported = MODEL_CONDITIONS.get(mid, [])
                    conditions_to_run = (
                        supported if condition == "all"
                        else [condition] if condition in supported
                        else []
                    )
                    conditions_to_run = [c for c in conditions_to_run if c not in task_cfg.skip_conditions]
                    if not conditions_to_run:
                        click.echo(f"  SKIP [{mid}/{tid}/{condition}]: not supported for {mid}")
                        continue
                    for cond in conditions_to_run:
                        await run_eval(mid, tid, cond, task_cfg, dry_run, smoke_test, eval_seed)
                except Exception as exc:
                    click.echo(f"  ERROR [{mid}/{tid}]: {exc}", err=True)
                    _tb.print_exc()
                    _log.error(f"eval failed: {type(exc).__name__}: {exc}",
                               model=mid, task=tid,
                               exc=str(exc), traceback=_tb.format_exc())
                    failures.append((f"{mid}/{tid}", str(exc)))

    asyncio.run(run_all())

    if failures:
        click.echo(f"\nFAILED ({len(failures)}):")
        for key, err in failures:
            click.echo(f"  {key}: {err}")
        sys.exit(1)
    click.echo("\nAll API evaluations completed.")


if __name__ == "__main__":
    main()
