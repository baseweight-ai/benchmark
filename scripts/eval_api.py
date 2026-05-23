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

from checkpoint_utils import append_jsonl, atomic_write_json, finalize_partial, load_partial_ids, partial_path
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


def _load_eval_rows(
    task_id: str, condition: str, eval_seed: int, smoke_test: bool,
) -> tuple[list[dict], list[dict], str, Optional[str]]:
    """Load the deterministic eval inputs for one (task, condition, seed).

    Shared by the streaming and batch paths so a given (task, condition, seed)
    always samples the IDENTICAL rows — the batch submit and collect phases must
    agree with each other, and batch must match what streaming would produce.
    Returns (test_rows_with_labels, few_shot, prompt_sha, few_shot_hash).
    """
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
    return test_rows, few_shot, prompt_sha, few_shot_hash


def _request_shape(
    model_id: str, task_id: str, task_cfg: TaskConfig,
) -> tuple[str, Optional[str], Optional[list[str]], Optional[dict]]:
    """Resolve the per-request shape shared by streaming and batch.

    Returns (model_str, reasoning_effort, label_set, response_format). Reasoning
    is sent as "none" on reasoning-capable models (omitted entirely otherwise so
    the API doesn't reject it); response_format pins direct classification answers
    to the closed label set, exactly as eval_local constrains the local model.
    """
    model_str = OPENAI_MODELS.get(model_id)
    if not model_str:
        raise ValueError(f"Model string not set for {model_id}")
    reasoning_effort = REASONING_EFFORT_OFF if REASONING_CAPABLE.get(model_id, False) else None
    label_set = load_label_set(task_id)
    constrain = (
        task_cfg.task_type == "classification"
        and task_cfg.answer_mode == "direct"
        and label_set is not None
    )
    response_format = build_label_response_format(label_set) if constrain else None
    return model_str, reasoning_effort, label_set, response_format


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
    test_rows, few_shot, prompt_sha, few_shot_hash = _load_eval_rows(
        task_id, condition, eval_seed, smoke_test
    )
    # Never mutates the module-level OPENAI_MODELS dict; raises if model unset.
    model_str, reasoning_effort, label_set, response_format = _request_shape(
        model_id, task_id, task_cfg
    )

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

    # reasoning_effort and response_format were resolved up-front by
    # _request_shape (classification → label-pinned schema; extraction/CoT → None).
    if response_format is not None:
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


# ── Batch API (50%-cheaper async path) ──────────────────────────────────────
# OpenAI's Batch API runs the same /v1/chat/completions requests at half price
# with a <=24h turnaround (usually 1-6h). Two phases keep it robust to the wait:
#   submit  → build + upload the request file, create the batch, record its id +
#             fingerprint in a sidecar under runs/batch/.
#   collect → poll; when complete, download the output and write predictions in
#             the EXACT shape the streaming path produces, so classify/summary
#             consume them unchanged.
# Latency/TTFT do not exist in batch and are written as the 0 sentinel (NOT
# None — classify_errors filters latency with `> 0`, which TypeErrors on None):
# the >0 filter drops them so the summary reports null latency, honestly marking
# "not measured". API cost uses token x price, not wall time, so nothing else
# downstream depends on them.

BATCH_DIR = REPO_ROOT / "runs" / "batch"


def _batch_sidecar(model_id: str, task_id: str, cond_key: str, smoke: bool) -> Path:
    prefix = "smoke__" if smoke else ""
    return BATCH_DIR / f"{prefix}{model_id}__{task_id}__{cond_key}.json"


def _build_batch_request(
    custom_id: str, msgs: list[dict], model_str: str, max_tokens: int,
    reasoning_effort: Optional[str], response_format: Optional[dict],
) -> dict:
    """One line of the batch input file: a /v1/chat/completions request mirroring
    the streaming call (temperature=0, max_completion_tokens, logprobs). Batch is
    non-streamed, so stream/stream_options are dropped; reasoning_effort is a
    plain body field here (the SDK extra_body shim isn't needed off the SDK)."""
    body: dict = {
        "model": model_str,
        "messages": msgs,
        "temperature": 0,
        "max_completion_tokens": max_tokens,
        "logprobs": True,
    }
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    if response_format is not None:
        body["response_format"] = response_format
    return {"custom_id": custom_id, "method": "POST",
            "url": "/v1/chat/completions", "body": body}


def _parse_batch_response(
    resp_line: dict,
) -> tuple[str, int, int, int, Optional[float], bool]:
    """Parse one batch output line → (text, in_tok, out_tok, reasoning_tok,
    avg_logprob, api_error), mirroring providers._openai_once's accounting.

    A row errors (api_error=True) when the request failed (line.error set or a
    non-200 status); the failure is surfaced in the text, never silently dropped.
    """
    if resp_line.get("error"):
        return f"ERROR: {resp_line['error']}", 0, 0, 0, None, True
    response = resp_line.get("response") or {}
    if response.get("status_code") != 200:
        return (f"ERROR: batch status {response.get('status_code')}: "
                f"{response.get('body')}", 0, 0, 0, None, True)
    body = response.get("body") or {}
    choices = body.get("choices") or [{}]
    choice0 = choices[0] or {}
    text = (choice0.get("message") or {}).get("content") or ""
    usage = body.get("usage") or {}
    in_tok = usage.get("prompt_tokens", 0) or 0
    out_tok = usage.get("completion_tokens", 0) or 0
    details = usage.get("completion_tokens_details") or {}
    reasoning_tok = details.get("reasoning_tokens", 0) or 0
    lps = [
        float(e["logprob"])
        for e in ((choice0.get("logprobs") or {}).get("content") or [])
        if e.get("logprob") is not None
    ]
    avg_logprob = sum(lps) / len(lps) if lps else None
    return text, in_tok, out_tok, reasoning_tok, avg_logprob, False


def _write_batch_sidecar(
    *, batch_id: str, input_file_id: Optional[str], model_id: str, task_id: str,
    condition: str, cond_key: str, eval_seed: int, smoke_test: bool,
    fingerprint: str, prompt_sha: str, few_shot_hash: Optional[str], n_pending: int,
) -> Path:
    """Persist everything collect needs to retrieve a batch later (survives a
    dropped session)."""
    sidecar = _batch_sidecar(model_id, task_id, cond_key, smoke_test)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({
        "batch_id": batch_id, "input_file_id": input_file_id,
        "model_id": model_id, "task_id": task_id, "condition": condition,
        "cond_key": cond_key, "eval_seed": eval_seed, "smoke_test": smoke_test,
        "fingerprint": fingerprint, "prompt_sha": prompt_sha,
        "few_shot_hash": few_shot_hash, "n_pending": n_pending,
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }, indent=2) + "\n")
    return sidecar


# Batch states that are dead — a new submit is allowed to replace them.
_BATCH_DEAD = {"failed", "expired", "cancelled", "cancelling"}


def _find_existing_batch(client, fingerprint: str):
    """Return a still-live/completed batch whose metadata fingerprint matches the
    one we're about to submit, else None.

    The OpenAI server — not the local sidecar — is the source of truth. This
    makes submit idempotent across a dropped session (re-running never creates a
    duplicate paid batch) AND recovers a batch whose sidecar was lost in the
    create→write window. Fails CLOSED: if the list call errors we cannot prove a
    duplicate doesn't exist, so we raise rather than risk a second paid batch."""
    try:
        for b in client.batches.list(limit=100):
            md = getattr(b, "metadata", None) or {}
            if md.get("fingerprint") == fingerprint and b.status not in _BATCH_DEAD:
                return b
    except Exception as exc:
        raise RuntimeError(
            "could not list batches to dedupe before submit (refusing to submit "
            f"and risk a duplicate paid batch): {exc}"
        ) from exc
    return None


def submit_eval_batch(
    model_id: str, task_id: str, condition: str, task_cfg: TaskConfig,
    eval_seed: int = 0, smoke_test: bool = False,
) -> Optional[str]:
    """Phase 1: build + upload the batch request file and create the batch,
    recording {batch_id, fingerprint, ...} in a sidecar for collect. Returns the
    batch id, or None when skipped (already up-to-date / nothing pending)."""
    test_rows, few_shot, prompt_sha, few_shot_hash = _load_eval_rows(
        task_id, condition, eval_seed, smoke_test
    )
    model_str, reasoning_effort, label_set, response_format = _request_shape(
        model_id, task_id, task_cfg
    )
    cond_key = condition if eval_seed == 0 else f"{condition}_seed{eval_seed}"
    out_path = pred_path(REPO_ROOT, "api", model_id, task_id, cond_key, smoke=smoke_test)
    fingerprint = _eval_fingerprint(
        test_rows_hash=_rows_hash(test_rows), prompt_sha=prompt_sha,
        few_shot_hash=few_shot_hash, label_set=label_set, condition=condition,
        eval_seed=eval_seed, model_str=model_str,
        reasoning_capable=REASONING_CAPABLE.get(model_id, False),
        max_output_tokens=task_cfg.max_output_tokens,
        task_type=task_cfg.task_type, answer_mode=task_cfg.answer_mode,
    )
    pp = partial_path(out_path)
    if reuse_is_valid(out_path, pp, fingerprint):
        click.echo(f"  SKIP [{model_id}/{task_id}/{cond_key}]: up-to-date")
        return None

    completed_ids = load_partial_ids(pp)
    pending_rows = [r for r in test_rows if r.get("id", "") not in completed_ids]
    if not pending_rows:
        finalize_partial(pp, out_path)
        click.echo(f"  All {len(test_rows)} rows already complete → finalized")
        return None

    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Crash-safety / idempotency: if a batch for these EXACT inputs is already
    # live or done server-side, adopt it instead of creating a duplicate. This
    # makes re-running submit after an interrupted session free of double-billing
    # and re-materializes a sidecar that was lost in a create→write crash window.
    existing = _find_existing_batch(client, fingerprint)
    if existing is not None:
        _write_batch_sidecar(
            batch_id=existing.id, input_file_id=getattr(existing, "input_file_id", None),
            model_id=model_id, task_id=task_id, condition=condition, cond_key=cond_key,
            eval_seed=eval_seed, smoke_test=smoke_test, fingerprint=fingerprint,
            prompt_sha=prompt_sha, few_shot_hash=few_shot_hash, n_pending=len(pending_rows),
        )
        click.echo(f"  ADOPT existing batch {existing.id} [{model_id}/{task_id}/{cond_key}]: "
                   f"status={existing.status} (not re-submitting) — collect with --collect-batch")
        return existing.id

    # BEFORE generating rows, so an interrupted run stays attributed to its inputs.
    record_fingerprint(out_path, fingerprint)

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    req_file = BATCH_DIR / f"{_batch_sidecar(model_id, task_id, cond_key, smoke_test).stem}.requests.jsonl"
    with open(req_file, "w") as f:
        for row in pending_rows:
            msgs = build_messages(row, few_shot, condition)
            f.write(json.dumps(_build_batch_request(
                row.get("id", ""), msgs, model_str, task_cfg.max_output_tokens,
                reasoning_effort, response_format,
            ), ensure_ascii=False) + "\n")

    with open(req_file, "rb") as fh:
        uploaded = client.files.create(file=fh, purpose="batch")
    # fingerprint in metadata is what _find_existing_batch matches on for dedupe.
    batch = client.batches.create(
        input_file_id=uploaded.id, endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"model": model_id, "task": task_id, "cond_key": cond_key,
                  "fingerprint": fingerprint},
    )
    sidecar = _write_batch_sidecar(
        batch_id=batch.id, input_file_id=uploaded.id, model_id=model_id,
        task_id=task_id, condition=condition, cond_key=cond_key, eval_seed=eval_seed,
        smoke_test=smoke_test, fingerprint=fingerprint, prompt_sha=prompt_sha,
        few_shot_hash=few_shot_hash, n_pending=len(pending_rows),
    )
    click.echo(f"  Submitted batch {batch.id} [{model_id}/{task_id}/{cond_key}]: "
               f"{len(pending_rows)} requests → {sidecar.relative_to(REPO_ROOT)}")
    _log.info("batch submit", model=model_id, task=task_id, condition=condition,
              eval_seed=eval_seed, event="batch_submit", batch_id=batch.id,
              n_rows=len(pending_rows))
    return batch.id


def collect_eval_batch(sidecar_path: Path) -> bool:
    """Phase 2: poll the batch; when complete, download + write predictions in
    the streaming-path schema, then finalize. Returns True when finalized,
    False when still pending / failed (caller may retry later)."""
    info = json.loads(sidecar_path.read_text())
    model_id, task_id = info["model_id"], info["task_id"]
    condition, cond_key = info["condition"], info["cond_key"]
    eval_seed, smoke_test = info["eval_seed"], info["smoke_test"]
    task_cfg = load_task_config(task_id)

    test_rows, few_shot, prompt_sha, few_shot_hash = _load_eval_rows(
        task_id, condition, eval_seed, smoke_test
    )
    model_str, reasoning_effort, label_set, response_format = _request_shape(
        model_id, task_id, task_cfg
    )
    fingerprint = _eval_fingerprint(
        test_rows_hash=_rows_hash(test_rows), prompt_sha=prompt_sha,
        few_shot_hash=few_shot_hash, label_set=label_set, condition=condition,
        eval_seed=eval_seed, model_str=model_str,
        reasoning_capable=REASONING_CAPABLE.get(model_id, False),
        max_output_tokens=task_cfg.max_output_tokens,
        task_type=task_cfg.task_type, answer_mode=task_cfg.answer_mode,
    )
    out_path = pred_path(REPO_ROOT, "api", model_id, task_id, cond_key, smoke=smoke_test)
    pp = partial_path(out_path)

    # Reproducibility guard: refuse to attribute batch outputs to data that has
    # changed since submit (e.g. a mid-flight re-prepare) — that would silently
    # pair responses with the wrong inputs.
    if fingerprint != info.get("fingerprint"):
        click.echo(f"  REFUSE [{model_id}/{task_id}/{cond_key}]: inputs changed since "
                   f"submit ({info.get('fingerprint')} → {fingerprint}); re-submit", err=True)
        return False
    if reuse_is_valid(out_path, pp, fingerprint):
        click.echo(f"  SKIP [{model_id}/{task_id}/{cond_key}]: already collected")
        return True

    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    batch = client.batches.retrieve(info["batch_id"])
    rc = batch.request_counts
    if batch.status != "completed":
        click.echo(f"  {info['batch_id']} [{model_id}/{task_id}/{cond_key}]: {batch.status} "
                   f"({getattr(rc, 'completed', '?')}/{getattr(rc, 'total', '?')} done, "
                   f"{getattr(rc, 'failed', 0)} failed)")
        if batch.status in ("failed", "expired", "cancelled"):
            click.echo("  Did not complete; sidecar kept for re-submit.", err=True)
        return False

    already = load_partial_ids(pp)  # makes re-collect idempotent
    pending = [r for r in test_rows if r.get("id", "") not in already]
    if not pending:
        if pp.exists():
            finalize_partial(pp, out_path)
            click.echo(f"  All rows already collected → {out_path.relative_to(REPO_ROOT)}")
        return True

    # The Batch API routes successes to output_file_id and per-request failures
    # to error_file_id. Read BOTH (reading only one silently drops the rest),
    # and tolerate a completed-but-all-failed batch whose output_file_id is None.
    resp_by_id: dict[str, dict] = {}
    for fid in (batch.output_file_id, batch.error_file_id):
        if not fid:
            continue
        for line in client.files.content(fid).text.splitlines():
            if line.strip():
                obj = json.loads(line)
                resp_by_id[obj.get("custom_id", "")] = obj

    # Attribute the partial to its inputs BEFORE writing rows. Covers the adopt
    # path (which never called record_fingerprint at submit) so reuse_is_valid
    # can later invalidate this output when inputs change.
    record_fingerprint(out_path, fingerprint)
    # Batch has no measurable LOCAL eval wall-time (it ran server-side); write a
    # .wall.json so classify_errors reports eval_wall_time_s=None rather than
    # deriving a meaningless value from the collect-time row timestamps — the
    # sidecar's presence suppresses the timestamp-span fallback.
    atomic_write_json({"eval_wall_time_s": None, "gpu_model": None},
                      out_path.with_suffix(".wall.json"))

    # Iterate the PENDING ROWS (not the output lines) so every request yields a
    # row — failures and no-shows are written as ERROR rows, never dropped,
    # exactly as the streaming path does.
    written = errored = missing = 0
    for row in pending:
        rid = row.get("id", "")
        resp_line = resp_by_id.get(rid)
        if resp_line is None:
            text, in_tok, out_tok, reasoning_tok, avg_logprob, api_error = (
                "ERROR: no batch response returned for this request", 0, 0, 0, None, True)
            missing += 1
        else:
            text, in_tok, out_tok, reasoning_tok, avg_logprob, api_error = _parse_batch_response(resp_line)
            if response_format is not None and not api_error:
                text = parse_constrained_label(text)
            errored += int(api_error)
        answer_only = 0 if api_error else count_answer_tokens(text, model_str)
        msgs = build_messages(row, few_shot, condition)
        append_jsonl({
            "id": rid,
            "model": model_id,
            "condition": condition,
            "eval_seed": eval_seed,
            "prompt_sha": prompt_sha,
            "few_shot_hash": few_shot_hash,
            "input": msgs[-1]["content"] if msgs else "",
            "output": text,
            "ground_truth": row.get("label", ""),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "reasoning_tokens": reasoning_tok,
            "answer_only_tokens": answer_only,
            "latency_ms": 0,      # not measured in batch (see module note)
            "ttft_ms": 0.0,       # not measured in batch
            "avg_logprob": avg_logprob,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }, pp)
        written += 1

    if pp.exists():
        finalize_partial(pp, out_path)
    click.echo(f"  Collected {written} rows ({errored} api errors) → "
               f"{out_path.relative_to(REPO_ROOT)}")
    if errored or missing:
        click.echo(f"  WARNING [{model_id}/{task_id}/{cond_key}]: {errored} api errors, "
                   f"{missing} requests had no batch response (written as ERROR rows)", err=True)
    _log.info("batch collect", model=model_id, task=task_id, condition=condition,
              eval_seed=eval_seed, event="batch_collect", batch_id=info["batch_id"],
              n_rows=written, n_errors=errored, n_missing=missing)
    return True


@click.command()
@click.option("--model", default="all", help=f"Model ID or 'all'. Choices: {', '.join(ALL_API_MODELS)}")
@click.option("--task", default="all", help="Task ID or 'all'")
@click.option("--condition", default="all", help="zero-shot|5-shot|all")
@click.option("--eval-seed", "eval_seed", default=0, type=int,
              help="Evaluation seed (0 = deterministic sample; >0 resamples from test_full.jsonl)")
@click.option("--dry-run", is_flag=True)
@click.option("--smoke-test", is_flag=True)
@click.option("--batch", is_flag=True,
              help="Submit via the 50%-cheaper Batch API (async; retrieve later with --collect-batch)")
@click.option("--collect-batch", "collect_batch", is_flag=True,
              help="Poll + write predictions for batches submitted with --batch")
def main(model: str, task: str, condition: str, eval_seed: int, dry_run: bool,
         smoke_test: bool, batch: bool, collect_batch: bool) -> None:
    """Evaluate frontier OpenAI models on benchmark tasks (zero-shot and 5-shot).

    Default streams each request. --batch submits the same requests via the
    Batch API (half price, <=24h turnaround); --collect-batch retrieves them.
    """
    configure(REPO_ROOT)
    if batch and collect_batch:
        raise click.UsageError("--batch and --collect-batch are mutually exclusive")
    default_models = SMOKE_MODELS if smoke_test else PROD_MODELS
    model_ids = default_models if model == "all" else [model]

    if not dry_run and any(m in OPENAI_MODELS for m in model_ids):
        if not os.environ.get("OPENAI_API_KEY"):
            click.echo("  WARNING: OPENAI_API_KEY not set", err=True)
            _log.warning("OPENAI_API_KEY not set")
    task_ids = ALL_TASKS if task == "all" else [task]
    failures: list[tuple[str, str]] = []

    def conditions_for(mid: str, task_cfg: TaskConfig) -> list[str]:
        supported = MODEL_CONDITIONS.get(mid, [])
        conds = (
            supported if condition == "all"
            else [condition] if condition in supported
            else []
        )
        return [c for c in conds if c not in task_cfg.skip_conditions]

    # ── Collect phase: retrieve previously-submitted batches ────────────────
    if collect_batch:
        sidecars = sorted(BATCH_DIR.glob("*.json")) if BATCH_DIR.exists() else []
        if not sidecars:
            click.echo("  No batch sidecars in runs/batch/ — nothing to collect.")
            return
        done = pending = 0
        for sidecar in sidecars:
            try:
                info = json.loads(sidecar.read_text())
            except Exception as exc:
                click.echo(f"  ERROR: unreadable batch sidecar {sidecar.name}: {exc}", err=True)
                failures.append((sidecar.name, f"unreadable sidecar: {exc}"))
                continue
            if model != "all" and info.get("model_id") != model:
                continue
            if task != "all" and info.get("task_id") != task:
                continue
            try:
                if collect_eval_batch(sidecar):
                    done += 1
                else:
                    pending += 1
            except Exception as exc:
                click.echo(f"  ERROR collecting {sidecar.name}: {exc}", err=True)
                _tb.print_exc()
                failures.append((sidecar.name, str(exc)))
        click.echo(f"\nCollect: {done} finalized, {pending} still pending/failed.")
        if failures:
            sys.exit(1)
        return

    # ── Submit phase: create batches, then exit (collect later) ─────────────
    if batch:
        submitted = 0
        for mid in model_ids:
            for tid in task_ids:
                try:
                    task_cfg = load_task_config(tid)
                    conds = conditions_for(mid, task_cfg)
                    if not conds:
                        click.echo(f"  SKIP [{mid}/{tid}/{condition}]: not supported for {mid}")
                        continue
                    for cond in conds:
                        if dry_run:
                            click.echo(f"  [dry-run] Would submit batch {mid}/{tid}/{cond}"
                                       + (f" seed={eval_seed}" if eval_seed else ""))
                            continue
                        if submit_eval_batch(mid, tid, cond, task_cfg, eval_seed, smoke_test):
                            submitted += 1
                except Exception as exc:
                    click.echo(f"  ERROR [{mid}/{tid}]: {exc}", err=True)
                    _tb.print_exc()
                    failures.append((f"{mid}/{tid}", str(exc)))
        click.echo(f"\nSubmitted {submitted} batch(es). Retrieve with: "
                   f"python scripts/eval_api.py --collect-batch"
                   + (f" --model {model}" if model != "all" else ""))
        if failures:
            for key, err in failures:
                click.echo(f"  {key}: {err}")
            sys.exit(1)
        return

    # ── Default: stream each request over the asyncio event loop ────────────
    async def run_all() -> None:
        for mid in model_ids:
            for tid in task_ids:
                try:
                    task_cfg = load_task_config(tid)
                    conditions_to_run = conditions_for(mid, task_cfg)
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
