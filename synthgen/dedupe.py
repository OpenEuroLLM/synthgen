"""Dedupe an SFT JSONL by (lang, normalized prompt); drop empty/degenerate responses."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from synthgen.log import get_logger

log = get_logger("synthgen.dedupe")

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


def _key(lang: str, prompt: str) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(lang.encode("utf-8"))
    h.update(b"\x1f")
    h.update(_norm(prompt).encode("utf-8"))
    return h.digest()


def run(*, input: Path, output: Path,
        min_response_chars: int = 8, report: Path | None = None) -> dict:
    seen: set[bytes] = set()
    kept = dup = empty = short = 0
    by_lang_kept: Counter[str] = Counter()
    by_lang_drop: Counter[str] = Counter()

    with open(input, "r", encoding="utf-8") as fin, \
         open(output, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                empty += 1
                continue

            lang = r.get("lang", "")
            prompt = r.get("prompt", "") or ""
            resp = (r.get("response", "") or "").strip()
            if not resp:
                empty += 1
                by_lang_drop[lang] += 1
                continue
            if len(resp) < min_response_chars:
                short += 1
                by_lang_drop[lang] += 1
                continue
            k = _key(lang, prompt)
            if k in seen:
                dup += 1
                by_lang_drop[lang] += 1
                continue
            seen.add(k)

            fout.write(line if line.endswith("\n") else line + "\n")
            kept += 1
            by_lang_kept[lang] += 1

            if (i + 1) % 200_000 == 0:
                log.info("processed %d | kept %d | dup %d", i + 1, kept, dup)

    summary = {
        "input": str(input), "output": str(output),
        "total_read": kept + dup + empty + short,
        "kept": kept,
        "dropped_duplicate": dup,
        "dropped_empty": empty,
        "dropped_short": short,
        "min_response_chars": min_response_chars,
        "kept_by_lang": dict(by_lang_kept),
        "dropped_by_lang": dict(by_lang_drop),
    }
    if report:
        report.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("kept=%d dup=%d empty=%d short=%d", kept, dup, empty, short)
    return summary
