"""Judge-score quality filter for localized one-shot generations.

Joins a `generate.py` mode="localized" output (outputs/loc_<model>.jsonl) with
its mode="judge" scores (outputs/judged_<model>.jsonl) by id, and keeps rows
scoring at or above the per-language threshold.

Output rows are shaped `{id, lang, prompt, response, ...}` — `instruction` is
aliased to `prompt` so the result drops straight into the existing
`dedupe.py` / `qc.py` / `to_open_instruct.py` stages unchanged.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from synthgen.io import iter_jsonl
from synthgen.log import get_logger

log = get_logger("synthgen.quality_filter")


def threshold_for(thresholds: dict, lang: str) -> float:
    """`thresholds` is either {"global": x} or {lang: x, ...} (optionally with
    a "global" fallback for languages not listed)."""
    if lang in thresholds:
        return thresholds[lang]
    if "global" in thresholds:
        return thresholds["global"]
    raise KeyError(f"no threshold for lang={lang!r} and no 'global' fallback "
                   f"in {thresholds!r}")


def run(*, gen: Path, judged: Path, thresholds: dict, output: Path,
        report: Path | None = None) -> dict:
    scores: dict[str, float] = {}
    for r in iter_jsonl(judged):
        if r.get("error") or not isinstance(r.get("score"), (int, float)):
            continue
        scores[r["id"]] = r["score"]

    kept = dropped_low = dropped_unscored = 0
    kept_by_lang: Counter[str] = Counter()
    dropped_by_lang: Counter[str] = Counter()

    with open(output, "w", encoding="utf-8") as fout:
        for r in iter_jsonl(gen):
            if r.get("error") or not r.get("instruction") or not r.get("response"):
                continue
            lang = r["lang"]
            score = scores.get(r["id"])
            if score is None:
                dropped_unscored += 1
                dropped_by_lang[lang] += 1
                continue
            if score < threshold_for(thresholds, lang):
                dropped_low += 1
                dropped_by_lang[lang] += 1
                continue
            row = {
                "id": r["id"], "lang": lang,
                "prompt": r["instruction"], "response": r["response"],
                "domain": r.get("domain"), "is_local": r.get("is_local"),
                "intent": r.get("intent"), "role": r.get("role"),
                "quality_score": score, "model": r.get("model"),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            kept_by_lang[lang] += 1

    summary = {
        "gen": str(gen), "judged": str(judged), "output": str(output),
        "thresholds": thresholds,
        "kept": kept, "dropped_low_score": dropped_low,
        "dropped_unscored": dropped_unscored,
        "kept_by_lang": dict(kept_by_lang),
        "dropped_by_lang": dict(dropped_by_lang),
    }
    if report:
        report.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("kept=%d dropped_low=%d dropped_unscored=%d",
             kept, dropped_low, dropped_unscored)
    return summary
