"""Inject stub modules for openai/aiohttp/tqdm so API tests run without those packages.

If a package is already installed, the real module is used and these stubs are no-ops.
Tests that need to prevent real API calls use explicit patch() context managers.
"""
from __future__ import annotations

import asyncio
import importlib
import sys
from unittest.mock import MagicMock


async def _tqdm_gather(*coros, desc=None, **kwargs):
    return await asyncio.gather(*coros, **kwargs)


for _name in ("openai", "aiohttp"):
    if _name not in sys.modules:
        try:
            importlib.import_module(_name)
        except ImportError:
            sys.modules[_name] = MagicMock()

try:
    import tqdm.asyncio as _tqdm_async
    _tqdm_async.tqdm.gather = _tqdm_gather  # type: ignore[attr-defined]
except ImportError:
    _tqdm_stub = MagicMock()
    _tqdm_stub.asyncio.tqdm.gather = _tqdm_gather
    sys.modules["tqdm"] = _tqdm_stub
    sys.modules["tqdm.asyncio"] = _tqdm_stub.asyncio
