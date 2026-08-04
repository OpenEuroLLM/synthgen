"""JSONL I/O helpers shared by every pipeline stage."""
from __future__ import annotations

import json
import re
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


def extract_json(text: str | None) -> dict | None:
    """Parse the LAST balanced ``{...}`` object in `text`.

    Drops any ``<think>`` block first (reasoning models emit reasoning before
    the answer) and code fences, so this works directly on raw model output.
    """
    if not text:
        return None
    t = text.strip()
    if "</think>" in t:
        t = t.split("</think>")[-1]
    t = re.sub(r"```(?:json)?", "", t).strip()
    end = t.rfind("}")
    while end != -1:
        depth = 0
        for i in range(end, -1, -1):
            if t[i] == "}":
                depth += 1
            elif t[i] == "{":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[i:end + 1])
                    except json.JSONDecodeError:
                        break
        end = t.rfind("}", 0, end)
    return None


_ID_RE = re.compile(r"^(.*)-(\d+)$")


def next_id_index(path: str | Path, lang_code: str) -> int:
    """Next free numeric suffix for ``{lang_code}-{index}`` ids in a JSONL file.

    Scans existing ids for this language and returns ``max(index) + 1`` (0 if
    the file doesn't exist or has none) — lets a top-up round append new rows
    without ever colliding with ids already generated/judged/kept.
    """
    p = Path(path)
    if not p.exists():
        return 0
    best = -1
    prefix = f"{lang_code}-"
    for line in p.open("r", encoding="utf-8"):
        try:
            rid = json.loads(line).get("id", "")
        except Exception:
            continue
        if not isinstance(rid, str) or not rid.startswith(prefix):
            continue
        m = _ID_RE.match(rid)
        if m and m.group(1) == lang_code:
            best = max(best, int(m.group(2)))
    return best + 1
