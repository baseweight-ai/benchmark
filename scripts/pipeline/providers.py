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
from typing import Protocol, runtime_checkable

from pipeline.circuit_breaker import CircuitBreaker, CircuitOpenError  # noqa: F401
from pipeline.log import get_logger

MAX_RETRIES_OPENAI = 5
MAX_RETRIES_VLLM = 3
VLLM_HOST = "http://localhost:8000"

_log = get_logger("providers")
_openai_cb = CircuitBreaker("openai", failure_threshold=5, cooldown_s=60.0)
_vllm_cb   = CircuitBreaker("vllm",   failure_threshold=5, cooldown_s=30.0)


@runtime_checkable
class InferenceClient(Protocol):
    async def complete(
        self,
        messages: list[dict],
        max_tokens: int,
        semaphore: asyncio.Semaphore,
    ) -> tuple[str, int, int, float, float]:
        """Return (text, in_tokens, out_tokens, latency_ms, ttft_ms)."""
        ...


# ── OpenAI ─────────────────────────────────────────────────────────────────

async def _openai_once(
    client, model_str: str, messages: list[dict], max_tokens: int
) -> tuple[str, int, int, float, float]:
    t0 = time.time()
    ttft_ms = 0.0
    first_token = True
    chunks: list[str] = []
    in_tok = out_tok = 0

    stream = await client.chat.completions.create(
        model=model_str, messages=messages, temperature=0, max_tokens=max_tokens,
        stream=True, stream_options={"include_usage": True},
    )
    async for chunk in stream:
        if chunk.usage:
            in_tok = chunk.usage.prompt_tokens
            out_tok = chunk.usage.completion_tokens
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content or ""
        if content:
            if first_token:
                ttft_ms = (time.time() - t0) * 1000
                first_token = False
            chunks.append(content)

    return "".join(chunks), in_tok, out_tok, (time.time() - t0) * 1000, ttft_ms


async def _openai_with_retries(
    client, model_str: str, messages: list[dict], max_tokens: int
) -> tuple[str, int, int, float, float]:
    for attempt in range(MAX_RETRIES_OPENAI):
        try:
            return await _openai_once(client, model_str, messages, max_tokens)
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
    return "", 0, 0, 0.0, 0.0  # unreachable


async def call_openai(
    client,
    model_str: str,
    messages: list[dict],
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> tuple[str, int, int, float, float]:
    """Stream one OpenAI request with retries and circuit-breaker protection."""
    _openai_cb.raise_if_open()
    async with semaphore:
        return await _openai_cb.call(
            _openai_with_retries(client, model_str, messages, max_tokens)
        )


# ── vLLM ───────────────────────────────────────────────────────────────────

async def _vllm_once(
    session, model_name: str, messages: list[dict], max_tokens: int, host: str,
    chat_template_kwargs: dict | None = None,
) -> tuple[str, int, int, float, float]:
    import aiohttp

    payload = {
        "model": model_name, "messages": messages, "temperature": 0,
        "max_tokens": max_tokens, "stream": True,
        "stream_options": {"include_usage": True},
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs

    t0 = time.time()
    ttft_ms = 0.0
    chunks: list[str] = []
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
            content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if content:
                if first_token:
                    ttft_ms = (time.time() - t0) * 1000
                    first_token = False
                chunks.append(content)

    return "".join(chunks), in_tok, out_tok, (time.time() - t0) * 1000, ttft_ms


async def _vllm_with_retries(
    session, model_name: str, messages: list[dict], max_tokens: int, host: str,
    chat_template_kwargs: dict | None = None,
) -> tuple[str, int, int, float, float]:
    for attempt in range(MAX_RETRIES_VLLM):
        try:
            return await _vllm_once(session, model_name, messages, max_tokens, host, chat_template_kwargs)
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
    return "", 0, 0, 0.0, 0.0  # unreachable


async def call_vllm(
    session,
    model_name: str,
    messages: list[dict],
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    host: str = VLLM_HOST,
    chat_template_kwargs: dict | None = None,
) -> tuple[str, int, int, float, float]:
    """Stream one vLLM request with retries and circuit-breaker protection."""
    _vllm_cb.raise_if_open()
    async with semaphore:
        return await _vllm_cb.call(
            _vllm_with_retries(session, model_name, messages, max_tokens, host, chat_template_kwargs)
        )


# ── Adapter wrappers ───────────────────────────────────────────────────────

class OpenAIAdapter:
    """InferenceClient that wraps call_openai."""

    def __init__(self, client, model_str: str) -> None:
        self._client = client
        self._model_str = model_str

    async def complete(
        self, messages: list[dict], max_tokens: int, semaphore: asyncio.Semaphore
    ) -> tuple[str, int, int, float, float]:
        return await call_openai(self._client, self._model_str, messages, max_tokens, semaphore)


class VLLMAdapter:
    """InferenceClient that wraps call_vllm."""

    def __init__(self, session, model_name: str, host: str = VLLM_HOST,
                 chat_template_kwargs: dict | None = None) -> None:
        self._session = session
        self._model_name = model_name
        self._host = host
        self._chat_template_kwargs = chat_template_kwargs

    async def complete(
        self, messages: list[dict], max_tokens: int, semaphore: asyncio.Semaphore
    ) -> tuple[str, int, int, float, float]:
        return await call_vllm(
            self._session, self._model_name, messages, max_tokens, semaphore, self._host,
            self._chat_template_kwargs,
        )
