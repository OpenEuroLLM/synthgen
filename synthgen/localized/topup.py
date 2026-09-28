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
enough (pure CPU, no LLM calls -- exact-hash by default, optionally also
embedding-based near-dup via `embedding_dedupe=True`, see `near_dup.py`) to
always recompute fresh over everything accumulated so far rather than
needing their own checkpoint.
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
    embedding_dedupe: bool = False,
    embedding_threshold: float | None = None,
    embedding_model: str | None = None,
    persona_p: float | None = None,
    gen_concurrency: int | None = None,
    judge_concurrency: int | None = None,
) -> dict:
    """Top up every language in `lang_codes` (default: all LANGUAGES_PHASE3) to
    `target_per_lang` deduped, quality-filtered survivors.

    `thresholds` is consumed by quality_filter.threshold_for — either
    {"global": x} or {lang: x, ...} (see synthgen.thresholds).

    `persona_p`, if given, overrides `build_rows()`'s default for every row
    appended by this call. Left unset (None), rows use build_rows()'s own
    module-default PERSONA_P.

    No `general_p` override anymore: domain/locality is no longer a
    controllable split (see taxonomy.py's docstring) -- `build_rows()` draws
    domain from one real-data-weighted taxonomy, and whether a given example
    gets a local/cultural angle is an emergent, per-example, model-judged
    decision, not a knob this loop can force. There is no way to run a
    "dedicated local-content top-up" anymore; if you need to guarantee a
    minimum amount of local-flavored content, that has to happen by
    inspecting generated rows post-hoc (e.g. re-classifying/tagging), not by
    biasing the sampler up front.

    `embedding_dedupe=True` makes this loop's diversity guarantee real rather
    than aspirational: a paraphrased near-duplicate dropped by
    `near_dup.py`'s embedding check shrinks that round's survivor count the
    same as any other dropped row, which the deficit calculation above picks
    up automatically — so the loop keeps generating until `target_per_lang`
    truly-distinct rows exist, not just `target_per_lang` exact-string-unique
    ones. Needs `sentence-transformers` installed wherever this runs (not
    just the login node); if it isn't, each round logs a warning and falls
    back to exact-match-only, same as `dedupe.run()` does standalone.

    `gen_concurrency`/`judge_concurrency`, if given, override `cfg.concurrency`
    for just the gen-phase or judge-phase `generate.run()` call respectively.
    Matters because `synthgen/backends/vllm.py`'s `EndpointPool` is a plain
    round-robin over however many `*.endpoint` files are registered — a fixed
    concurrency budget dilutes across every replica in the pool, so a gen
    pool with many more replicas than the judge pool (the realistic shape for
    a production run, since gen needs far more raw compute per example) needs
    a proportionally higher concurrency to actually saturate its replicas,
    while the SAME value applied to the judge phase would wildly oversaturate
    its much smaller pool. Left unset, both phases fall back to
    `cfg.concurrency` unchanged (today's behavior).
    """
    langs = lang_codes or list(LANGUAGES_PHASE3)
    cfg.paths.ensure()

    prompts_path = cfg.paths.prompts / "localized_prompts.jsonl"
    kept_all_path = cfg.paths.outputs / "loc_full.kept.jsonl"
    final_path = cfg.paths.outputs / "loc_full.dedup.jsonl"
    judge_out = generate.output_path_for("judge", cfg, judge_backend.model)

    embedding_near_dup_dropped_total = 0
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
            build_kwargs = {}
            if persona_p is not None:
                build_kwargs["persona_p"] = persona_p
            rows = prompts_localized.build_rows(n_new, lang, start_index=start, seed=seed,
                                                **build_kwargs)
            write_jsonl(prompts_path, rows, append=True)
            appended += n_new
            log.info("[topup] round %d: %s deficit=%d factor=%.2f -> +%d rows "
                     "(ids %d..%d)", rnd, lang, deficit, factor, n_new,
                     start, start + n_new - 1)

        # generate + judge: resume-safe, only new/errored ids get (re)run.
        # cfg.concurrency is temporarily overridden per phase (see
        # gen_concurrency/judge_concurrency's docstring above) and restored
        # after -- safe because these two calls are sequential, never
        # concurrent, within a single-threaded asyncio process. try/finally
        # guarantees the restore even if a call raises (VLLMBackend.chat()
        # itself never does -- it exhausts its own retries and returns an
        # error dict -- but a different backend or an unrelated failure
        # shouldn't be able to leave cfg.concurrency polluted for whatever
        # reuses this same cfg object next).
        original_concurrency = cfg.concurrency
        try:
            cfg.concurrency = gen_concurrency if gen_concurrency is not None else original_concurrency
            loc_outputs = generate.run(cfg, backends=gen_backends, mode="localized",
                                       split=split, max_attempts=max_attempts,
                                       retry_delay=retry_delay)
            cfg.concurrency = judge_concurrency if judge_concurrency is not None else original_concurrency
            for p in loc_outputs:
                generate.run(cfg, backends=[judge_backend], mode="judge", in_path=p,
                            max_attempts=max_attempts, retry_delay=retry_delay)
        finally:
            cfg.concurrency = original_concurrency

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
        dedupe_summary = dedupe.run(input=kept_all_path, output=final_path,
                                    embedding_dedupe=embedding_dedupe,
                                    embedding_threshold=embedding_threshold,
                                    embedding_model=embedding_model)
        embedding_near_dup_dropped_total += dedupe_summary.get("dropped_near_dup_embedding", 0)

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
    gen_out_paths = [generate.output_path_for("localized", cfg, be.model)
                     for be in gen_backends]
    total_raw_generated = sum(len(load_done_ids_ok(p)) for p in gen_out_paths)
    return {
        "rounds": rnd,
        "target_per_lang": target_per_lang,
        "survivors_by_lang": dict(final_survivors),
        "met_target": all(final_survivors.get(l, 0) >= target_per_lang for l in langs),
        "final_output": str(final_path),
        "total_raw_generated": total_raw_generated,
        # Cumulative rows dropped by the embedding near-dup pass across every
        # round (0 if embedding_dedupe=False) -- each one already triggered a
        # replacement generation via the deficit loop above, so this is
        # exactly the extra generate/judge-call cost that setting bought you.
        "embedding_near_dup_dropped_total": embedding_near_dup_dropped_total,
    }
