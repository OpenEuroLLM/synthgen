"""JSONL I/O helpers shared by every pipeline stage."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict], *, append: bool = False) -> int:
    mode = "a" if append else "w"
    n = 0
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_done_ids(path: str | Path, key: str = "id") -> set[str]:
    """IDs already present in a JSONL output file (resume support)."""
    p = Path(path)
    if not p.exists():
        return set()
    out: set[str] = set()
    for line in p.open("r", encoding="utf-8"):
        try:
            out.add(json.loads(line)[key])
        except Exception:
            continue
    return out


def load_done_ids_ok(path: str | Path, key: str = "id") -> set[str]:
    """IDs with at least one *successful* (non-error) record.

    Unlike `load_done_ids`, rows whose only record carries an ``error`` are
    *not* counted as done, so they remain eligible for retry.
    """
    p = Path(path)
    if not p.exists():
        return set()
    out: set[str] = set()
    for line in p.open("r", encoding="utf-8"):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if "error" in rec:
            continue
        if key in rec:
            out.add(rec[key])
    return out


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")
