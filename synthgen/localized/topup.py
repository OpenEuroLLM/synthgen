"""Resumable top-up loop: generate -> judge -> quality-filter -> dedupe,
repeated until every language has `target_per_lang` survivors (or `max_rounds`
is hit).

All state lives on disk in the accumulated JSONL files:
    prompts/localized_prompts.jsonl        (grows every round)
    outputs/loc_<gen_model_slug>.jsonl      (generate.py mode="localized")
    outputs/judged_<judge_model_slug>.jsonl (generate.py mode="judge")
    outputs/loc_full.kept.jsonl             (quality-filter survivors, all gens)
    outputs/loc_full.dedup.jsonl            (final deduped survivors)

If the whole loop is killed mid-round, calling `run()` again just picks back
up: unfinished generate/judge rows resume via generate.py's own id-checkpoint
(unaffected rows are skipped, not re-run), and quality-filter/dedupe are cheap
enough (pure CPU, no LLM calls, exact-hash dedup) to always recompute fresh
over everything accumulated so far rather than needing their own checkpoint.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from synthgen.pipeline import dedupe, generate
from synthgen.backends import Backend
from synthgen.config import LANGUAGES_PHASE3, SynthConfig
from synthgen.io import iter_jsonl, load_done_ids_ok, next_id_index, write_jsonl
from synthgen.localized import prompts as prompts_localized
from synthgen.localized import quality_filter
from synthgen.log import get_logger

log = get_logger("synthgen.topup")


def _lang_counts(path: Path) -> Counter:
    c: Counter = Counter()
    if not path.exists():
        return c
    for r in iter_jsonl(path):
        c[r.get("lang", "")] += 1
    return c


def _survival_rate(gen_out_paths: list[Path], final_path: Path) -> float | None:
    """Observed (deduped survivors) / (raw successful generations) so far,
    across every round and every generator model — used to calibrate how many
    extra raw rows the next batch needs."""
    raw_ok = sum(len(load_done_ids_ok(p)) for p in gen_out_paths)
    if raw_ok == 0 or not final_path.exists():
        return None
    survivors = sum(1 for _ in iter_jsonl(final_path))
    return survivors / raw_ok if survivors else None


def run(
    cfg: SynthConfig,
    *,
    gen_backends: list[Backend],
    judge_backend: Backend,
    target_per_lang: int,
    thresholds: dict,
    lang_codes: list[str] | None = None,
    overgen_factor_default: float = 1.6,
    max_rounds: int = 20,
    seed: int = 0,
    max_attempts: int = 3,
    retry_delay: float = 15.0,
    split: bool | None = None,
) -> dict:
    """Top up every language in `lang_codes` (default: all LANGUAGES_PHASE3) to
    `target_per_lang` deduped, quality-filtered survivors.

    `thresholds` is consumed by quality_filter.threshold_for — either
    {"global": x} or {lang: x, ...} (see synthgen.thresholds).
    """
    langs = lang_codes or list(LANGUAGES_PHASE3)
    cfg.paths.ensure()

    prompts_path = cfg.paths.prompts / "localized_prompts.jsonl"
    kept_all_path = cfg.paths.outputs / "loc_full.kept.jsonl"
    final_path = cfg.paths.outputs / "loc_full.dedup.jsonl"
    judge_out = generate.output_path_for("judge", cfg, judge_backend.model)

    rnd = 0
    for rnd in range(1, max_rounds + 1):
        survivors = _lang_counts(final_path)
        deficits = {l: target_per_lang - survivors.get(l, 0) for l in langs
                    if survivors.get(l, 0) < target_per_lang}
        if not deficits:
            log.info("[topup] target reached for all %d languages after %d round(s)",
                     len(langs), rnd - 1)
            break

        gen_out_paths = [generate.output_path_for("localized", cfg, be.model)
                         for be in gen_backends]
        generated_ok = set()
        for p in gen_out_paths:
            generated_ok |= load_done_ids_ok(p)
        prompts_have = _lang_counts(prompts_path)
        rate = _survival_rate(gen_out_paths, final_path)
        factor = min(5.0, max(1.1, 1.0 / rate)) if rate else overgen_factor_default

        appended = 0
        for lang, deficit in deficits.items():
            backlog = prompts_have.get(lang, 0) - sum(
                1 for i in generated_ok if i.startswith(f"{lang}-"))
            if backlog > 0:
                log.info("[topup] round %d: %s has %d rows still in flight, "
                        "not adding more yet", rnd, lang, backlog)
                continue
            n_new = math.ceil(deficit * factor)
            start = next_id_index(prompts_path, lang)
            rows = prompts_localized.build_rows(n_new, lang, start_index=start, seed=seed)
            write_jsonl(prompts_path, rows, append=True)
            appended += n_new
            log.info("[topup] round %d: %s deficit=%d factor=%.2f -> +%d rows "
                     "(ids %d..%d)", rnd, lang, deficit, factor, n_new,
                     start, start + n_new - 1)

        # generate + judge: resume-safe, only new/errored ids get (re)run
        loc_outputs = generate.run(cfg, backends=gen_backends, mode="localized",
                                   split=split, max_attempts=max_attempts,
                                   retry_delay=retry_delay)
        for p in loc_outputs:
            generate.run(cfg, backends=[judge_backend], mode="judge", in_path=p,
                        max_attempts=max_attempts, retry_delay=retry_delay)

        # quality filter + dedupe: cheap, always recomputed fresh over everything
        kept_paths = []
        for p in loc_outputs:
            kp = cfg.paths.outputs / f"kept_{p.stem.replace('loc_', '', 1)}.jsonl"
            quality_filter.run(gen=p, judged=judge_out, thresholds=thresholds, output=kp)
            kept_paths.append(kp)
        with open(kept_all_path, "w", encoding="utf-8") as fout:
            for kp in kept_paths:
                for line in kp.open("r", encoding="utf-8"):
                    fout.write(line)
        dedupe.run(input=kept_all_path, output=final_path)

        new_survivors = _lang_counts(final_path)
        log.info("[topup] round %d done: survivors=%s", rnd,
                 {l: new_survivors.get(l, 0) for l in langs})
        if appended == 0 and new_survivors == survivors:
            log.warning("[topup] round %d made no progress (nothing appended, no "
                       "new survivors) — check for a stuck backlog", rnd)
    else:
        log.error("[topup] max_rounds=%d reached before every language hit target",
                  max_rounds)

    final_survivors = _lang_counts(final_path)
    return {
        "rounds": rnd,
        "target_per_lang": target_per_lang,
        "survivors_by_lang": dict(final_survivors),
        "met_target": all(final_survivors.get(l, 0) >= target_per_lang for l in langs),
        "final_output": str(final_path),
    }
