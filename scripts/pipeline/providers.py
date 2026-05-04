"""Inference client adapters — OpenAI and vLLM."""
from __future__ import annotations
import asyncio
import time
from typing import Protocol, runtime_checkable

MAX_RETRIES_OPENAI = 5
MAX_RETRIES_VLLM = 3
VLLM_HOST = "http://localhost:8000"


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


async def call_openai(
    client,
    model_str: str,
    messages: list[dict],
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> tuple[str, int, int, float, float]:
    """Stream one OpenAI request. Returns (text, in_tokens, out_tokens, latency_ms, ttft_ms)."""
    async with semaphore:
        for attempt in range(MAX_RETRIES_OPENAI):
            try:
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

            except Exception:
                if attempt == MAX_RETRIES_OPENAI - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
    return "", 0, 0, 0, 0.0


async def call_vllm(
    session,
    model_name: str,
    messages: list[dict],
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    host: str = VLLM_HOST,
) -> tuple[str, int, int, float, float]:
    """Stream one vLLM request. Returns (text, in_tokens, out_tokens, latency_ms, ttft_ms)."""
    import json as _json
    import aiohttp

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    async with semaphore:
        for attempt in range(MAX_RETRIES_VLLM):
            try:
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
                            data = _json.loads(data_str)
                        except _json.JSONDecodeError:
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

            except Exception:
                if attempt == MAX_RETRIES_VLLM - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
    return "", 0, 0, 0.0, 0.0


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

    def __init__(self, session, model_name: str, host: str = VLLM_HOST) -> None:
        self._session = session
        self._model_name = model_name
        self._host = host

    async def complete(
        self, messages: list[dict], max_tokens: int, semaphore: asyncio.Semaphore
    ) -> tuple[str, int, int, float, float]:
        return await call_vllm(self._session, self._model_name, messages, max_tokens, semaphore, self._host)
