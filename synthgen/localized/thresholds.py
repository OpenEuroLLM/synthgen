"""Decide a judge quality-filter threshold from pilot score distributions.

Policy: if per-language cut points are close together, use one global
threshold; if the judge scores meaningfully differently across languages, use
a per-language threshold instead. See studies/localized_bootstrap for the
pilot that produces the input `records.jsonl` (rows with `lang`, `label`,
`score`).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from synthgen.io import iter_jsonl
from synthgen.log import get_logger

log = get_logger("synthgen.thresholds")


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        raise ValueError("no values")
    i = int(q * (len(sorted_vals) - 1))
    return sorted_vals[i]


def per_language_cut(records: list[dict], *, label: str = "good",
                     percentile: float = 0.10) -> dict[str, float]:
    """Per-language cut point: the `percentile` quantile of judge scores among
    records with the given label (default: the lower tail of "good" scores —
    a conservative "a legitimately good example rarely scores below this")."""
    by_lang: dict[str, list[float]] = {}
    for r in records:
        if r.get("label") != label or not isinstance(r.get("score"), (int, float)):
            continue
        by_lang.setdefault(r["lang"], []).append(r["score"])
    if not by_lang:
        raise ValueError(f"no scored label={label!r} records found")
    return {lang: _quantile(sorted(xs), percentile) for lang, xs in by_lang.items()}


def decide_thresholds(records: list[dict], *, label: str = "good",
                      percentile: float = 0.10, spread_tolerance: float = 1.5) -> dict:
    """Returns {"mode", "thresholds", "per_language_cut", "spread"}.

    `thresholds` is directly consumable by quality_filter.run(): either
    {"global": x} (spread <= spread_tolerance) or {lang: x, ...} otherwise.
    """
    cuts = per_language_cut(records, label=label, percentile=percentile)
    spread = max(cuts.values()) - min(cuts.values())
    if spread <= spread_tolerance:
        thresholds = {"global": round(statistics.mean(cuts.values()), 1)}
        mode = "global"
    else:
        thresholds = {lang: round(v, 1) for lang, v in cuts.items()}
        mode = "per_language"
    return {"mode": mode, "thresholds": thresholds,
            "per_language_cut": {k: round(v, 2) for k, v in cuts.items()},
            "spread": round(spread, 2)}


def run(*, records: Path, output: Path, label: str = "good",
       percentile: float = 0.10, spread_tolerance: float = 1.5,
       report: Path | None = None) -> dict:
    rows = list(iter_jsonl(records))
    decision = decide_thresholds(rows, label=label, percentile=percentile,
                                 spread_tolerance=spread_tolerance)
    output.write_text(json.dumps(decision["thresholds"], indent=2, ensure_ascii=False))
    if report:
        report.write_text(json.dumps(decision, indent=2, ensure_ascii=False))
    log.info("mode=%s spread=%.2f thresholds=%s",
             decision["mode"], decision["spread"], decision["thresholds"])
    return decision
