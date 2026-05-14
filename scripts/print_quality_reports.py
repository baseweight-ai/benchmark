"""Pretty-print every data/prepared/*/quality_report.json."""
from __future__ import annotations

import json
import signal
from pathlib import Path

# Behave like a normal shell tool: exit silently when our pipe consumer closes
# (e.g. piping into `less` or `head`).
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

REPO_ROOT = Path(__file__).parent.parent


def fmt(n):
    return f"{n:,}" if isinstance(n, int) else str(n)


def print_labels(label_dist, indent="    "):
    items = sorted(
        ((k, v["count"], v["pct"]) for k, v in label_dist.items()),
        key=lambda x: -x[1],
    )
    n = len(items)
    max_len = max((len(str(l)) for l, _, _ in items), default=0)
    if max_len > 80 or n > 200:
        print(f"{indent}Labels: {n:,} unique  (top 5 by count, label preview truncated)")
        for i, (l, c, p) in enumerate(items[:5], 1):
            preview = str(l).replace("\n", " ")[:80]
            print(f"{indent}  {i:>3}. {preview}…  count={c}  pct={p}%")
        return
    pad = min(max_len + 2, 50)
    print(f"{indent}Labels ({n} classes, ordered by count desc):")
    for i, (l, c, p) in enumerate(items, 1):
        print(f"{indent}  {i:>3}. {str(l)[:48].ljust(pad)} count={fmt(c):>7}  pct={p:>5}%")


def main():
    reports = sorted((REPO_ROOT / "data" / "prepared").glob("*/quality_report.json"))
    if not reports:
        print(f"No quality reports found under {REPO_ROOT / 'data' / 'prepared'}")
        return
    for fp in reports:
        r = json.loads(fp.read_text())
        task = r.get("task_id", fp.parent.name)
        raw = r.get("raw", {})
        flt = r.get("filtering", {})
        prep = r.get("prepared", {})
        print()
        print("=" * 84)
        print(f"  TASK: {task}")
        print("=" * 84)
        print(f"  Raw         train={fmt(raw.get('train_n', 0)):<10}  test={fmt(raw.get('test_n', 0))}")
        if flt:
            print("  Filtering   " + "  ".join(f"{k}={fmt(v)}" for k, v in flt.items()))
        for split in ("train", "test"):
            if split not in prep:
                continue
            s = prep[split]
            print()
            print(f"  Prepared {split}  n={fmt(s.get('n', 0))}  n_classes={s.get('n_classes', '-')}")
            for stat in ("char_length", "word_length"):
                d = s.get(stat)
                if not d:
                    continue
                print(
                    f"    {stat:11}  mean={d['mean']:<7.1f}std={d['std']:<6.1f}"
                    f"p5={d['p5']:<6}p50={d['p50']:<6}p95={d['p95']:<6}max={d['max']}"
                )
            nd = s.get("within_near_dupes", {})
            if nd:
                print(
                    f"    near_dupes   {nd.get('n_near_dup_pairs', 0)} pairs at threshold {nd.get('threshold')}"
                )
            if "label_distribution" in s:
                print_labels(s["label_distribution"])
        cs = prep.get("cross_split")
        if cs:
            print(f"\n  Cross-split: {cs}")
    print()


if __name__ == "__main__":
    main()
