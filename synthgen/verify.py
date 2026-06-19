"""Lightweight constraint + language-ID smoke checks. Emits per-model yield report.

Inputs:  outputs/gen_*.jsonl (prefers .filtered.jsonl when present)
Outputs: review/yield_<model>.json, review/yield_summary.json
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from synthgen.config import LANGUAGES_PHASE3, SynthConfig
from synthgen.io import iter_jsonl
from synthgen.log import get_logger

log = get_logger("synthgen.verify")

try:
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0
    _HAVE_LD = True
except Exception:
    _HAVE_LD = False


def _lang_ok(text: str, target: str) -> bool | None:
    if not _HAVE_LD or not text:
        return None
    try:
        return detect(text) == target
    except Exception:
        return False


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _constraint_ok(row: dict) -> bool | None:
    text = row.get("generated_prompt") or ""
    cname = row.get("constraint_name") or ""
    ct = row.get("constraint_text") or ""
    if not text or not cname or not ct:
        return None
    nums = re.findall(r"\d+", ct)
    quoted = re.findall(r'"([^"]+)"', ct)
    score = 0
    if len(text) > 40:
        score += 1
    if nums and any(n in text for n in nums):
        score += 1
    if quoted and any(q in text for q in quoted):
        score += 1
    if not nums and not quoted:
        score += 1
    return score >= 2


def yield_report(gen_path: Path) -> dict:
    by_lang: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "errors": 0, "lang_ok": 0, "lang_unknown": 0,
                 "constraint_ok": 0, "constraint_unknown": 0, "avg_words": 0.0})
    for r in iter_jsonl(gen_path):
        lang = r.get("lang", "")
        b = by_lang[lang]
        b["n"] += 1
        if "error" in r:
            b["errors"] += 1
            continue
        l_ok = _lang_ok(r.get("generated_prompt", ""), lang)
        if l_ok is None:
            b["lang_unknown"] += 1
        elif l_ok:
            b["lang_ok"] += 1
        c_ok = _constraint_ok(r)
        if c_ok is None:
            b["constraint_unknown"] += 1
        elif c_ok:
            b["constraint_ok"] += 1
        b["avg_words"] += _word_count(r.get("generated_prompt", ""))
    for lang, b in by_lang.items():
        n = max(1, b["n"] - b["errors"])
        n_constr = max(1, n - b["constraint_unknown"])
        b["avg_words"] = round(b["avg_words"] / n, 1)
        b["lang_ok_pct"] = round(100 * b["lang_ok"] / n, 1)
        b["constraint_ok_pct"] = round(100 * b["constraint_ok"] / n_constr, 1)
        b["n_constrained"] = n_constr
        b["language"] = LANGUAGES_PHASE3.get(lang, lang)
    return dict(by_lang)


def _pick_files(cfg: SynthConfig, gen: list[Path] | None) -> list[Path]:
    if gen:
        return list(gen)
    all_files = sorted(cfg.paths.outputs.glob("gen_*.jsonl"))
    filtered = [f for f in all_files if f.name.endswith(".filtered.jsonl")]
    if not filtered:
        return [f for f in all_files
                if not f.name.endswith((".filtered.jsonl", ".leaked.jsonl"))]
    stems = {f.name.replace(".filtered.jsonl", "") for f in filtered}
    rest = [f for f in all_files
            if not f.name.endswith((".filtered.jsonl", ".leaked.jsonl"))
            and f.name.replace(".jsonl", "") not in stems]
    return filtered + rest


def run(cfg: SynthConfig, *, gen: list[Path] | None = None) -> None:
    cfg.paths.ensure()
    summary: dict[str, dict] = {}
    for gen_path in _pick_files(cfg, gen):
        report = yield_report(gen_path)
        out = cfg.paths.review / (gen_path.stem.replace("gen_", "yield_") + ".json")
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        summary[gen_path.name] = report
        log.info("== %s ==", gen_path.name)
        for lang, b in report.items():
            log.info("  %-10s n=%-3d lang_ok=%s%%  constr_ok=%s%% (of %d)  "
                     "avg_words=%s  err=%d",
                     b["language"], b["n"], b["lang_ok_pct"], b["constraint_ok_pct"],
                     b["n_constrained"], b["avg_words"], b["errors"])
    (cfg.paths.review / "yield_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
