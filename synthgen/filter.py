"""Filter generations where the model produced an answer alongside the prompt.

Observed: Gemma sometimes emits the user prompt plus an example assistant
response. Catch with constraint-aware signals (length, leaked phrases, JSON
objects, list items, all-caps prompts).

Inputs:  outputs/gen_*.jsonl
Outputs: outputs/gen_*.filtered.jsonl   (kept rows)
         outputs/gen_*.leaked.jsonl     (rejected rows + reason)
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from synthgen.config import SynthConfig
from synthgen.log import get_logger

log = get_logger("synthgen.filter")

MAX_WORDS = 300


def detect_leakage(row: dict) -> tuple[bool, str | None]:
    text = (row.get("generated_prompt") or "").strip()
    if not text:
        return True, "empty"
    if "error" in row:
        return True, "error"

    cname = row.get("constraint_name") or ""
    ctext = row.get("constraint_text") or ""
    words = text.split()
    if len(words) > MAX_WORDS:
        return True, f"too_long({len(words)})"

    if cname in ("start_with", "end_with"):
        for p in re.findall(r'"([^"]+)"', ctext):
            if text.count(p) >= 2:
                return True, f"answer_includes_{cname}_phrase"

    if cname == "format_json":
        if re.search(r'\{\s*"\w+"\s*:\s*("[^"]*"|\[|\{|\d)', text):
            return True, "contains_json_answer"

    if cname == "format_bullets":
        bullets = sum(1 for ln in text.splitlines() if re.match(r"^\s*[-*•]\s+\S", ln))
        if bullets >= 3:
            return True, f"prompt_contains_{bullets}_bullets"
    if cname == "format_numbered":
        numd = sum(1 for ln in text.splitlines() if re.match(r"^\s*\d+[.)]\s+\S", ln))
        if numd >= 3:
            return True, f"prompt_contains_{numd}_numbered_items"

    if cname == "casing_all_caps":
        letters = [c for c in text if c.isalpha()]
        if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.9:
            return True, "prompt_is_all_caps"

    return False, None


def filter_file(path: Path) -> tuple[int, int, Counter]:
    keep_path = path.with_suffix(".filtered.jsonl")
    drop_path = path.with_suffix(".leaked.jsonl")
    reasons: Counter[str] = Counter()
    kept = dropped = 0
    with path.open("r", encoding="utf-8") as fin, \
         keep_path.open("w", encoding="utf-8") as fk, \
         drop_path.open("w", encoding="utf-8") as fd:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            leaked, reason = detect_leakage(row)
            if leaked:
                row["leak_reason"] = reason
                fd.write(json.dumps(row, ensure_ascii=False) + "\n")
                reasons[reason or "unknown"] += 1
                dropped += 1
            else:
                fk.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
    total = kept + dropped
    pct = 100 * dropped / total if total else 0.0
    log.info("%s: kept=%d dropped=%d (%.1f%%)", path.name, kept, dropped, pct)
    for reason, n in reasons.most_common():
        log.info("  %s: %d", reason, n)
    return kept, dropped, reasons


def run(cfg: SynthConfig, *, gen: list[Path] | None = None) -> None:
    files = list(gen) if gen else sorted(cfg.paths.outputs.glob("gen_*.jsonl"))
    files = [f for f in files if ".filtered" not in f.name and ".leaked" not in f.name]
    for f in files:
        filter_file(f)
