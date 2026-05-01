"""Inject stub modules for openai/aiohttp/tqdm so API tests run without those packages."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock


async def _tqdm_gather(*coros, desc=None, **kwargs):
    return await asyncio.gather(*coros, **kwargs)


for _name in ("openai", "aiohttp"):
    if _name not in sys.modules:
        sys.modules[_name] = MagicMock()

if "tqdm" not in sys.modules:
    _tqdm_stub = MagicMock()
    _tqdm_stub.asyncio.tqdm.gather = _tqdm_gather
    sys.modules["tqdm"] = _tqdm_stub
    sys.modules["tqdm.asyncio"] = _tqdm_stub.asyncio
else:
    import tqdm.asyncio as _tqdm_async
    _tqdm_async.tqdm.gather = _tqdm_gather  # type: ignore[attr-defined]
