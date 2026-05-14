"""Inference client adapters — OpenAI and vLLM.

Both call_openai and call_vllm:
  - retry with exponential backoff (logging each failed attempt)
  - are wrapped by a per-service circuit breaker that stops sending requests
    after repeated failures and re-probes after a cooldown period
  - fast-reject before the semaphore when the circuit is firmly OPEN
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import NamedTuple, Protocol, runtime_checkable

from pipeline.circuit_breaker import CircuitBreaker, CircuitOpenError  # noqa: F401
from pipeline.log import get_logger

MAX_RETRIES_OPENAI = 5
MAX_RETRIES_VLLM = 3
VLLM_HOST = "http://localhost:8000"

_log = get_logger("providers")
_openai_cb = CircuitBreaker("openai", failure_threshold=5, cooldown_s=60.0)
_vllm_cb   = CircuitBreaker("vllm",   failure_threshold=5, cooldown_s=30.0)


class InferenceResult(NamedTuple):
    """Single-request inference result.

    reasoning_tokens decomposes output_tokens (answer_tokens is derived).
    avg_logprob is None when the provider didn't return logprobs.
    """
    text: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    latency_ms: float
    ttft_ms: float
    avg_logprob: float | None


@runtime_checkable
class InferenceClient(Protocol):
    async def complete(
        self,
        messages: list[dict],
        max_tokens: int,
        semaphore: asyncio.Semaphore,
    ) -> InferenceResult:
        ...


# ── OpenAI ─────────────────────────────────────────────────────────────────

async def _openai_once(
    client, model_str: str, messages: list[dict], max_tokens: int,
    reasoning_effort: str | None = None,
) -> InferenceResult:
    t0 = time.time()
    ttft_ms = 0.0
    first_token = True
    chunks: list[str] = []
    logprobs: list[float] = []
    in_tok = out_tok = reasoning_tok = 0

    # Pass reasoning_effort via extra_body so non-reasoning models aren't sent
    # a parameter they'd reject — extra_body bypasses SDK-side validation and
    # is dropped from the request entirely when empty.
    extra: dict = {}
    if reasoning_effort is not None:
        extra["extra_body"] = {"reasoning_effort": reasoning_effort}

    stream = await client.chat.completions.create(
        model=model_str, messages=messages, temperature=0, max_tokens=max_tokens,
        stream=True, stream_options={"include_usage": True},
        logprobs=True,
        **extra,
    )
    async for chunk in stream:
        if chunk.usage:
            in_tok = chunk.usage.prompt_tokens
            out_tok = chunk.usage.completion_tokens
            # completion_tokens_details only set on reasoning models.
            details = getattr(chunk.usage, "completion_tokens_details", None)
            if details is not None:
                reasoning_tok = getattr(details, "reasoning_tokens", 0) or 0
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        content = choice.delta.content or ""
        if content:
            if first_token:
                ttft_ms = (time.time() - t0) * 1000
                first_token = False
            chunks.append(content)
        lp_obj = getattr(choice, "logprobs", None)
        if lp_obj is not None:
            content_lps = getattr(lp_obj, "content", None) or []
            for entry in content_lps:
                lp = getattr(entry, "logprob", None)
                if lp is not None:
                    logprobs.append(float(lp))

    avg_logprob = sum(logprobs) / len(logprobs) if logprobs else None

    return InferenceResult(
        text="".join(chunks),
        input_tokens=in_tok,
        output_tokens=out_tok,
        reasoning_tokens=reasoning_tok,
        latency_ms=(time.time() - t0) * 1000,
        ttft_ms=ttft_ms,
        avg_logprob=avg_logprob,
    )


async def _openai_with_retries(
    client, model_str: str, messages: list[dict], max_tokens: int,
    reasoning_effort: str | None = None,
) -> InferenceResult:
    for attempt in range(MAX_RETRIES_OPENAI):
        try:
            return await _openai_once(client, model_str, messages, max_tokens, reasoning_effort)
        except Exception as exc:
            if attempt == MAX_RETRIES_OPENAI - 1:
                raise
            delay = 2 ** attempt
            _log.warning(
                "OpenAI request failed, retrying",
                model=model_str,
                attempt=attempt + 1,
                max_attempts=MAX_RETRIES_OPENAI,
                delay_s=float(delay),
                exc=str(exc),
            )
            await asyncio.sleep(delay)
    return InferenceResult("", 0, 0, 0, 0.0, 0.0, None)  # unreachable


async def call_openai(
    client,
    model_str: str,
    messages: list[dict],
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    reasoning_effort: str | None = None,
) -> InferenceResult:
    """Stream one OpenAI request with retries and circuit-breaker protection.

    reasoning_effort: pass "minimal" (or "low"/"medium"/"high") on a
    reasoning-capable model. Pass None on a non-reasoning model — the parameter
    is omitted entirely so the API doesn't reject the request.
    """
    _openai_cb.raise_if_open()
    async with semaphore:
        return await _openai_cb.call(
            _openai_with_retries(client, model_str, messages, max_tokens, reasoning_effort)
        )


# ── vLLM ───────────────────────────────────────────────────────────────────

_THINK_END = "</think>"


def _estimate_reasoning_tokens(text: str, total_output_tokens: int) -> int:
    """Approximate token split by character ratio when text contains <think>...</think>.

    vLLM reports only total completion_tokens, not a phase breakdown. Returns 0
    when no </think> marker is present (expected with enable_thinking=False).
    """
    if _THINK_END not in text or not text:
        return 0
    boundary = text.index(_THINK_END) + len(_THINK_END)
    return round(total_output_tokens * boundary / len(text))


async def _vllm_once(
    session, model_name: str, messages: list[dict], max_tokens: int, host: str,
    chat_template_kwargs: dict | None = None,
    guided_choice: list[str] | None = None,
) -> InferenceResult:
    import aiohttp

    payload = {
        "model": model_name, "messages": messages, "temperature": 0,
        "max_tokens": max_tokens, "stream": True,
        "stream_options": {"include_usage": True},
        "logprobs": True,
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    if guided_choice:
        # vLLM-specific extension: constrains decoding to one of these strings.
        # Pinned at the OpenAI-compatible endpoint as a top-level field.
        payload["guided_choice"] = guided_choice

    t0 = time.time()
    ttft_ms = 0.0
    chunks: list[str] = []
    logprobs: list[float] = []
    first_token = True
    in_tok = out_tok = 0

    async with session.post(
        f"{host}/v1/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=300),
    ) as resp:
        resp.raise_for_status()
        async for raw_line in resp.content:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            usage = data.get("usage") or {}
            if usage.get("prompt_tokens"):
                in_tok = usage["prompt_tokens"]
            if usage.get("completion_tokens"):
                out_tok = usage["completion_tokens"]
            choice = (data.get("choices") or [{}])[0]
            content = choice.get("delta", {}).get("content", "")
            if content:
                if first_token:
                    ttft_ms = (time.time() - t0) * 1000
                    first_token = False
                chunks.append(content)
            lp_obj = choice.get("logprobs")
            if lp_obj:
                content_lps = lp_obj.get("content") or []
                for entry in content_lps:
                    lp = entry.get("logprob")
                    if lp is not None:
                        logprobs.append(float(lp))

    text = "".join(chunks)
    avg_logprob = sum(logprobs) / len(logprobs) if logprobs else None
    return InferenceResult(
        text=text,
        input_tokens=in_tok,
        output_tokens=out_tok,
        reasoning_tokens=_estimate_reasoning_tokens(text, out_tok),
        latency_ms=(time.time() - t0) * 1000,
        ttft_ms=ttft_ms,
        avg_logprob=avg_logprob,
    )


async def _vllm_with_retries(
    session, model_name: str, messages: list[dict], max_tokens: int, host: str,
    chat_template_kwargs: dict | None = None,
    guided_choice: list[str] | None = None,
) -> InferenceResult:
    for attempt in range(MAX_RETRIES_VLLM):
        try:
            return await _vllm_once(
                session, model_name, messages, max_tokens, host,
                chat_template_kwargs, guided_choice,
            )
        except Exception as exc:
            if attempt == MAX_RETRIES_VLLM - 1:
                raise
            delay = 2 ** attempt
            _log.warning(
                "vLLM request failed, retrying",
                model=model_name,
                attempt=attempt + 1,
                max_attempts=MAX_RETRIES_VLLM,
                delay_s=float(delay),
                exc=str(exc),
            )
            await asyncio.sleep(delay)
    return InferenceResult("", 0, 0, 0, 0.0, 0.0, None)  # unreachable


async def call_vllm(
    session,
    model_name: str,
    messages: list[dict],
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    host: str = VLLM_HOST,
    chat_template_kwargs: dict | None = None,
    guided_choice: list[str] | None = None,
) -> InferenceResult:
    """Stream one vLLM request with retries and circuit-breaker protection."""
    _vllm_cb.raise_if_open()
    async with semaphore:
        return await _vllm_cb.call(
            _vllm_with_retries(
                session, model_name, messages, max_tokens, host,
                chat_template_kwargs, guided_choice,
            )
        )


# ── Adapter wrappers ───────────────────────────────────────────────────────

class OpenAIAdapter:
    """InferenceClient that wraps call_openai."""

    def __init__(self, client, model_str: str,
                 reasoning_effort: str | None = None) -> None:
        self._client = client
        self._model_str = model_str
        self._reasoning_effort = reasoning_effort

    async def complete(
        self, messages: list[dict], max_tokens: int, semaphore: asyncio.Semaphore
    ) -> InferenceResult:
        return await call_openai(
            self._client, self._model_str, messages, max_tokens, semaphore,
            self._reasoning_effort,
        )


class VLLMAdapter:
    """InferenceClient that wraps call_vllm."""

    def __init__(self, session, model_name: str, host: str = VLLM_HOST,
                 chat_template_kwargs: dict | None = None,
                 guided_choice: list[str] | None = None) -> None:
        self._session = session
        self._model_name = model_name
        self._host = host
        self._chat_template_kwargs = chat_template_kwargs
        self._guided_choice = guided_choice

    async def complete(
        self, messages: list[dict], max_tokens: int, semaphore: asyncio.Semaphore
    ) -> InferenceResult:
        return await call_vllm(
            self._session, self._model_name, messages, max_tokens, semaphore, self._host,
            self._chat_template_kwargs, self._guided_choice,
        )
