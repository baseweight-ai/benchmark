"""Artifact versioning helpers — git SHA and file content hashing."""
from __future__ import annotations
import hashlib
import subprocess
from pathlib import Path


def git_sha() -> str:
    """Return the short git SHA of HEAD, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def file_sha256(path: Path) -> str:
    """Return the first 16 hex chars of the SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def configs_sha(paths: list[Path]) -> str:
    """Return a combined SHA of multiple config files (sorted by path for stability)."""
    h = hashlib.sha256()
    for p in sorted(paths, key=str):
        h.update(str(p).encode())
        h.update(b"\n")
        if p.exists():
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
    return h.hexdigest()[:16]
