"""Content-hash utilities for skip-if-unchanged logic."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Optional

from pipeline.versioning import file_sha256 as file_content_hash  # re-exported, no duplicate


def rows_sha(rows: list[dict]) -> str:
    """SHA-256 (first 16 hex) of rows serialized as JSONL (mirrors write_jsonl byte output)."""
    h = hashlib.sha256()
    for row in rows:
        h.update((json.dumps(row, ensure_ascii=False) + "\n").encode())
    return h.hexdigest()[:16]


def dict_hash(d: dict) -> str:
    """SHA-256 (first 16 hex chars) of a stable JSON serialization (keys sorted)."""
    serialized = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def training_inputs_hash(data_path: Path, hyperparams: dict) -> str:
    """Combined hash of training data content and hyperparameters."""
    parts = [file_content_hash(data_path), dict_hash(hyperparams)]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def read_stored_hash(meta_path: Path) -> Optional[str]:
    """Read input_hash from metadata.json. Returns None if file absent or key missing."""
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            return json.load(f).get("input_hash")
    except Exception:
        return None


def inputs_changed(current_hash: str, meta_path: Path) -> bool:
    """Return True only when a stored hash exists AND differs from current_hash.

    No stored hash → False, preserving backward-compatible skip for runs that
    pre-date content hashing.
    """
    stored = read_stored_hash(meta_path)
    if stored is None:
        return False
    return stored != current_hash


# ── Code & directory fingerprinting ────────────────────────────────────────

def _local_module_file(modname: str, scripts_root: Path) -> Optional[Path]:
    """Resolve a dotted module name to a source file under scripts_root.

    Returns None for stdlib / third-party modules — their versions are pinned
    separately (conda/pip), so they are intentionally outside the closure.
    """
    rel = modname.replace(".", "/")
    for cand in (scripts_root / f"{rel}.py", scripts_root / rel / "__init__.py"):
        if cand.is_file():
            return cand.resolve()
    return None


def _imported_modules(tree: ast.AST) -> set[str]:
    """Every dotted module name referenced by import statements in `tree`."""
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
            # `from pkg import sub` where sub is itself a submodule.
            for alias in node.names:
                mods.add(f"{node.module}.{alias.name}")
    return mods


def code_closure_hash(entry: Path) -> str:
    """Content hash of a script plus every pipeline module it transitively imports.

    `entry` is a top-level pipeline script (scripts/<name>.py); its directory is
    the scripts root. Walks import statements from `entry`, following only
    modules that resolve to a file under that root (stdlib / third-party imports
    are ignored — their versions are pinned separately). The digest changes iff
    any source file in that closure changes, so a stage keyed on it re-runs
    exactly when its own code — or shared code it depends on — was edited, and
    not when an unrelated script changes.
    """
    entry = entry.resolve()
    scripts_root = entry.parent
    seen: set[Path] = set()
    queue: list[Path] = [entry]
    while queue:
        path = queue.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for modname in _imported_modules(tree):
            resolved = _local_module_file(modname, scripts_root)
            if resolved and resolved not in seen:
                queue.append(resolved)
    h = hashlib.sha256()
    for path in sorted(seen):
        h.update(path.relative_to(scripts_root).as_posix().encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


def tree_hash(root: Path) -> str:
    """Content hash of every file under `root`, keyed by relative path.

    Fingerprints a directory-shaped input (e.g. a downloaded raw dataset) so a
    downstream stage can detect when that input changed. Empty / missing dir
    hashes to a stable constant.
    """
    h = hashlib.sha256()
    for path in sorted((p for p in root.rglob("*") if p.is_file()), key=str):
        h.update(path.relative_to(root).as_posix().encode())
        h.update(b"\0")
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()[:16]


def _meta_sidecar(out_path: Path) -> Path:
    """Path of the fingerprint sidecar for a produced artifact file."""
    return out_path.with_suffix(".meta.json")


def reuse_is_valid(out_path: Path, partial_path: Path, fingerprint: str) -> bool:
    """Whether a completed stage output can be reused instead of regenerated.

    True iff `out_path` exists and its recorded fingerprint still matches — or
    predates fingerprinting (an output with no sidecar is grandfathered in,
    preserving the old skip-if-exists behaviour). On a fingerprint MISMATCH the
    stale output, its partial file and its sidecar are deleted so the caller
    regenerates cleanly rather than resuming stale rows.
    """
    meta = _meta_sidecar(out_path)
    if meta.exists() and inputs_changed(fingerprint, meta):
        for stale in (out_path, partial_path, meta):
            stale.unlink(missing_ok=True)
        return False
    return out_path.exists()


def record_fingerprint(out_path: Path, fingerprint: str) -> None:
    """Write the fingerprint sidecar for an artifact.

    Call this BEFORE generating rows so an interrupted run's partial file stays
    correctly attributed to the inputs it was produced from.
    """
    meta = _meta_sidecar(out_path)
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({"input_hash": fingerprint}) + "\n")
