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


def _raw_ok_by_lang(gen_out_paths: list[Path]) -> Counter:
    """Per-language count of successful (non-error) raw generations, summed
    across every gen_out_path -- handles split=True sharding across multiple
    generator backends/models the same way the official end-of-round pass
    already does (concatenate-then-count, not per-path)."""
    c: Counter = Counter()
    for p in gen_out_paths:
        if not p.exists():
            continue
        for r in iter_jsonl(p):
            if "error" not in r:
                c[r.get("lang", "")] += 1
    return c


def _interim_survivors_and_rates(
    *, gen_out_paths: list[Path], judge_out: Path, thresholds: dict,
    interim_kept_path: Path, interim_final_path: Path,
    embedding_dedupe: bool, embedding_threshold: float | None,
    embedding_model: str | None, embedding_workers: int | None,
) -> tuple[Counter, dict[str, float]]:
    """Lazy, cheap (CPU-only, no LLM calls) re-run of the SAME quality_filter
    + dedupe pipeline the official end-of-round pass uses, against whatever
    gen+judge data already exists on disk RIGHT NOW -- not just the last
    *completed* round's `final_path` (which may not exist at all if no round
    has ever reached dedupe, as happened for weeks across this run's first
    three job submissions). Written to throwaway interim_* paths so the
    official kept_all_path/final_path are never touched.

    Must run the real dedupe pipeline (not a quality-filter-only shortcut):
    a paraphrased near-duplicate that only embedding_dedupe catches would
    otherwise look like a "survivor" here and wrongly suppress a replacement
    generation that's actually still needed -- the same class of blindness
    this function exists to fix, just relocated.

    Returns (interim_survivors_by_lang, interim_rate_by_lang). Guards the one
    case quality_filter.run()/dedupe.run() do NOT guard themselves (both
    raise FileNotFoundError on a missing input path): if no gen_out_paths
    exist yet, or judge_out doesn't exist, returns ({}, {}) without touching
    either -- exactly preserves old behavior (overgen_factor_default
    fallback, deficit computed from scratch) for a genuinely fresh run.
    """
    existing_gen_paths = [p for p in gen_out_paths if p.exists()]
    if not existing_gen_paths or not judge_out.exists():
        return Counter(), {}

    kept_paths = []
    for p in existing_gen_paths:
        kp = interim_kept_path.with_name(
            f"{interim_kept_path.stem}_{p.stem}{interim_kept_path.suffix}")
        quality_filter.run(gen=p, judged=judge_out, thresholds=thresholds, output=kp)
        kept_paths.append(kp)
    with open(interim_kept_path, "w", encoding="utf-8") as fout:
        for kp in kept_paths:
            for line in kp.open("r", encoding="utf-8"):
                fout.write(line)

    dedupe.run(input=interim_kept_path, output=interim_final_path,
               embedding_dedupe=embedding_dedupe,
               embedding_threshold=embedding_threshold,
               embedding_model=embedding_model,
               embedding_workers=embedding_workers)

    interim_survivors = _lang_counts(interim_final_path)
    raw_ok = _raw_ok_by_lang(existing_gen_paths)
    interim_rate = {l: interim_survivors.get(l, 0) / n
                    for l, n in raw_ok.items() if n > 0 and interim_survivors.get(l, 0) > 0}
    return interim_survivors, interim_rate


def run(
    cfg: SynthConfig,
    *,
    gen_backends: list[Backend],
    judge_backend: Backend,
    target_per_lang: int,
    thresholds: dict,
    lang_codes: list[str] | None = None,
    overgen_factor_default: float = 1.6,
    max_overgen_factor: float = 5.0,
    max_rounds: int = 20,
    seed: int = 0,
    max_attempts: int = 3,
    retry_delay: float = 15.0,
    split: bool | None = None,
    embedding_dedupe: bool = False,
    embedding_threshold: float | None = None,
    embedding_model: str | None = None,
    embedding_workers: int | None = None,
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

    Per-language overgen factor: each round, before deciding whether a
    backlog-cleared language needs new prompts, a lazy "interim" re-run of
    quality_filter+dedupe against whatever's on disk right now (see
    `_interim_survivors_and_rates`) checks (a) whether that language already
    has enough judge-passing, deduped rows to meet `target_per_lang` --
    skipping it entirely if so, even though the old backlog-only check would
    have blindly requested more -- and (b) that language's OWN observed
    survival rate, clamped to `[1.1, max_overgen_factor]`, instead of one
    rate pooled across every language. A language with a much lower pass
    rate than the rest (e.g. an 8% outlier when most languages pass 90%+)
    needs a correspondingly larger factor to close its gap in one round;
    raise `max_overgen_factor` above its 5.0 default if you have a known
    severe outlier and want it caught up faster rather than over several
    rounds.

    Each round also drains any already-generated-but-unjudged backlog via an
    extra judge pass at the TOP of the round (before the above decision),
    so judge's dedicated endpoints aren't idle through an entire round's gen
    phase when there's already real judge work waiting from a prior
    interrupted run -- cheap/no-op via generate.run()'s own id-checkpointing
    whenever there's nothing new to judge.

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
    interim_kept_path = cfg.paths.outputs / "loc_full.interim_kept.jsonl"
    interim_final_path = cfg.paths.outputs / "loc_full.interim_dedup.jsonl"
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

        # Drain any already-generated-but-unjudged backlog FIRST, before
        # deciding what (if anything) new to generate this round -- so
        # judge's dedicated endpoints aren't idle through this round's gen
        # phase when there's already real judge work waiting (e.g. from a
        # prior interrupted run). Cheap/no-op via generate.run()'s own
        # id-checkpointing whenever there's nothing new to judge.
        gen_out_paths_so_far = [p for p in (
            generate.output_path_for("localized", cfg, be.model) for be in gen_backends)
            if p.exists()]
        if gen_out_paths_so_far:
            original_concurrency = cfg.concurrency
            try:
                cfg.concurrency = judge_concurrency if judge_concurrency is not None else original_concurrency
                for p in gen_out_paths_so_far:
                    generate.run(cfg, backends=[judge_backend], mode="judge", in_path=p,
                                max_attempts=max_attempts, retry_delay=retry_delay)
            finally:
                cfg.concurrency = original_concurrency

        gen_out_paths = [generate.output_path_for("localized", cfg, be.model)
                         for be in gen_backends]
        generated_ok = set()
        for p in gen_out_paths:
            generated_ok |= load_done_ids_ok(p)
        prompts_have = _lang_counts(prompts_path)
        backlogs = {lang: prompts_have.get(lang, 0) - sum(
            1 for i in generated_ok if i.startswith(f"{lang}-")) for lang in deficits}

        # Lazy: only pay for the interim quality_filter+dedupe pass if at
        # least one language is actually a candidate for new prompts this
        # round (backlog cleared) -- the helper's own existence-guard makes
        # it a no-op anyway on a data-free first round, but this skips even
        # the attempt when every language still has backlog outstanding.
        interim_survivors, interim_rate = (
            _interim_survivors_and_rates(
                gen_out_paths=gen_out_paths, judge_out=judge_out, thresholds=thresholds,
                interim_kept_path=interim_kept_path, interim_final_path=interim_final_path,
                embedding_dedupe=embedding_dedupe, embedding_threshold=embedding_threshold,
                embedding_model=embedding_model, embedding_workers=embedding_workers,
            ) if any(b <= 0 for b in backlogs.values()) else (Counter(), {})
        )

        appended = 0
        for lang, deficit in deficits.items():
            backlog = backlogs[lang]
            if backlog > 0:
                log.info("[topup] round %d: %s has %d rows still in flight, "
                        "not adding more yet", rnd, lang, backlog)
                continue
            interim_n = interim_survivors.get(lang, 0)
            if interim_n >= target_per_lang:
                log.info("[topup] round %d: %s backlog clear but interim judge-"
                         "pass+dedup estimate already has %d >= target %d -- "
                         "skipping, no new prompts", rnd, lang, interim_n, target_per_lang)
                continue
            rate = interim_rate.get(lang)
            factor = (min(max_overgen_factor, max(1.1, 1.0 / rate)) if rate
                      else overgen_factor_default)
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
                                    embedding_model=embedding_model,
                                    embedding_workers=embedding_workers)
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
