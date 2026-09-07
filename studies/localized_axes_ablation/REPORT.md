# localized_axes_ablation — results

**Status: all 10 ablations complete at full scale (n≈30-40/lang × 5 languages,
`es fr de pl uk`)**, run on LUMI `standard-g`, gen=`google/gemma-4-31b-it`,
judge=`Qwen/Qwen3.6-27B`. Two real bugs were found and fixed in `01`'s script
before this run — see `PLAN.md`/`README.md` for detail — so `01`'s numbers
below are post-fix; the pre-fix smoke-scale read for `01` should be
discarded, it measured a different (broken) comparison.

This is the deliverable the study exists to produce: concrete numbers to
hand back for a decision on `synthgen.localized` defaults. It does not
itself change any defaults in `synthgen/localized/` — that's a follow-up
step now that these numbers exist.

**Metrics fix, applied after the full-scale run (no regeneration needed):**
every diversity number below originally shipped as lexical-only — the
embedding-based near-dup metric never ran during the batch itself
(`sentence-transformers` wasn't installed in the vLLM container; every
`summary.json` recorded `embedding_unavailable` and silently fell back). This
was closed with a login-node venv + cached model, re-scoring every existing
`records.jsonl` with no GPU/regeneration required. First pass surfaced a real
instrument bug: the default embedder (`all-MiniLM-L6-v2`) is English-only and
flagged a spurious **33% near-dup rate on Ukrainian text** that vanished to
**0%** once switched to a proper multilingual model
(`paraphrase-multilingual-MiniLM-L12-v2`, now `_common/diversity_metrics.py`'s
default). With the corrected model, **every ablation's embedding near-dup
rate is ≤0.1%** — the lexical "zero collapse" reading now has independent,
paraphrase-sensitive confirmation, not just an untested assumption. The
per-domain sample-size caveat (need ~400-750 rows/lang to fully trust a
per-domain rate) still applies — this fix closes the "wrong instrument" gap,
not the "too few rows" one. See `_common/rescore_embeddings.py` and
`_common/embedding_rescore_results.json`.

## 04 — diversity baseline

- Judge score: n=148, mean=9.7 (123/148 scored 10, 18 scored 9, only 4 at 5)
- Near-duplicate rate: **zero** lexical near-duplicates detected in any of
  the 5 languages at n≈30/lang. No mode collapse visible at this scale — but
  per this study's own sizing note, a trustworthy *per-domain* near-dup rate
  needs ~400-750 rows/lang, well beyond this pilot. Treat "zero collapse" as
  encouraging, not conclusive.

## 01 — domain grounding (broadened taxonomy + grounding mode)

- Score by arm (n≈146-149/arm): `none`=9.78, `light`=9.76, `deep`=9.64.
  **Essentially flat, and if anything reversed** from what heavier grounding
  would predict — no evidence that deeper grounding improves judge-assessed
  quality at this scale, once the RNG-seed and unweighted-sampling bugs were
  fixed.
- Diversity delta vs. baseline: none detected (0.000 lexical near-dup in all
  three arms, all 5 languages) — same caveat on per-domain trust as `04`.
- Validation loop: not yet formally re-run (comparing generated-instruction
  classification back against the WildChat/LMArena target distribution) —
  `domain_distribution/` was produced and is now actually consumed by the
  weighted sampler (confirmed via unit test), but the closed-loop check
  itself wasn't executed this round.
- Rows clamped from deep → light (domain ceiling): recorded per-arm in
  `outputs/mode_deep/summary.json`'s `clamped_to_light` field — not
  hand-tallied here; check that file directly for the exact count.

## 02 — persona rate

- Score by arm (n≈146-149/arm): 0.5=9.6, 0.15=9.56, 0.10=9.61, 0.0=9.66.
  **All four arms land within 0.1 of each other, and dropping persona
  entirely (0.0) scores marginally the highest of all four.**
- Diversity delta: none detected (0.000 lexical near-dup, all arms/languages).
- **Recommendation: drop to a much lower persona rate, or 0.0.** At this
  scale, the current 50% suggestion rate shows no measurable quality
  benefit over 0% — directly supporting the supervisor's original
  suspicion. Note the known caveat (documented in `02`'s README): arms
  aren't bit-for-bit row-aligned past the first persona-triggering row, so
  this is a solid aggregate/distributional read, not a strict paired
  comparison — consistent with treating this as directional-but-strong
  evidence rather than a definitive proof.

## 03 — intent necessity

- Score, intent-injected vs. intent-stripped (n≈147-149 pairs): 9.69 vs. 9.59
  — **within noise of each other.** (The n=10 smoke read had shown stripped
  scoring higher; that direction did not hold at full scale — a useful
  demonstration of exactly the kind of small-n trap this study exists to
  catch.)
- Self-reported intent echo rate (with intent given): **32%** — the model's
  self-reported `intent_used` matched the sampled intent only about a third
  of the time, consistent with the smoke-scale reading (30%).
- **Recommendation: simplify or drop the intent axis.** It neither helps nor
  hurts judge-assessed quality, and the model mostly does not follow the
  suggested label anyway (it appears to pick its own intent from the
  domain/instruction context instead). Keeping it is not harmful, but it is
  not doing the conditioning work its presence implies.

## 05 — country grounding

- **Country-specificity rate: with real exemplar (n=9) = 78%; without real
  exemplar (n=139) = 89%.** The opposite of what real-data grounding was
  supposed to achieve — exemplar-grounded rows scored *lower* on
  country-specificity, not higher. Caveat: only 9 of 148 `deep`-arm rows had
  a real exemplar available at all (the classifier only found exemplars for
  a subset of domains per language), so this is a small, noisy sample — but
  it gives no support for the exemplar mechanism as currently implemented.
- Diversity delta, single- vs. multi-country sampling: single-country
  baseline (`04`, always the `LANG_COUNTRY` default) vs. diversified
  sampling (n=117, mean=9.54) — comparable score, and diversified sampling
  produced real spread across every listed alternate country in every
  language (e.g. fr: Canada 10, Belgium 9, Switzerland 8, France 3; es:
  Colombia 11, Argentina 8, Mexico 6, Spain 5) with no score penalty.
- **Recommendation: add multi-country sampling for fr/de/es/pt/nl** (the
  languages with known alternates) — no measured downside, and it directly
  fixes a real mode-collapse source. **Do not expect the current exemplar
  mechanism in `01` to improve country-specificity** without further work —
  the small sample here at least doesn't support it.

## 06 — domain split ratio

- Raw vs. post-filter ratio, per arm (n=200/arm): 0.2→17%/17%,
  0.4→35%/37%, 0.6→55%/55%, 0.8→79%/79%. Raw and post-filter track within 2
  points of every arm — Pool B does not survive the judge threshold at a
  meaningfully different rate than Pool A here (194-198/200 kept in every
  arm, no systematic gap).
- **Recommendation: to hit a real 40% general / 60% local final
  composition, set `GENERAL_P` to roughly 0.42-0.45** (interpolating between
  the 0.4→37% and 0.6→55% arms) — not 0.6 as originally proposed (that
  overshoots to ~55%), and a large change from the current 0.2 (→17%).

## 07 — judge-model sensitivity

- Mean |delta| between judges (Qwen3.6-27B vs. gemma-4-31b, n=148): 0.27
- Agreement within 1 point: **96.6%**
- **Recommendation: trust the score deltas seen elsewhere in this study.**
  Two judges from different model families read the same 148 examples
  almost identically — the findings above are unlikely to be an artifact of
  Qwen3.6-27B specifically.

## 08 — script effects

- Score delta, Latin (n=119) vs. non-Latin (n=29, `uk` only): 9.69 vs. 9.76
  — no meaningful gap.
- Diversity delta: identical (0.000 lexical near-dup, both groups).
- **Recommendation: no script-specific treatment appears necessary** for
  any axis in this study, though this is a 2-way comparison (`uk` is
  currently the only non-Latin-script language in the study's 5-language
  set) — worth re-checking if Greek (`el`) is added later.

## 09 — salt necessity

- Score, salt-present vs. salt-removed (n≈147-149/arm): 9.73 vs. 9.76 — no
  penalty from removing salt.
- Diversity delta (the primary outcome here): **zero lexical near-dup in
  both arms**, all 5 languages, at n≈30/lang.
- **Recommendation: inconclusive at this scale, lean toward "not proven
  useful."** Salt shows no detectable diversity benefit here, but this
  ablation's own near-dup rate needs the same ~400-750 rows/lang the study's
  sizing note calls for before a real per-domain collapse would even be
  visible — the honest reading is "no evidence salt helps, not yet strong
  evidence it doesn't," and a temperature-only comparison (never built this
  round) would be needed to fully settle it.

## 10 — persona specificity

- Score, generic vs. concrete persona (n≈146-147/arm, same rate): 9.67 vs.
  9.71 — within 0.04, no meaningful difference.
- Diversity delta: not separately reported (score was the primary outcome
  for this ablation).
- **Recommendation: persona specificity does not appear to matter at this
  scale, in either direction.** The n=10 smoke read (concrete scoring
  dramatically lower, 8.5 vs. 9.7) was noise — full scale reversed it. If
  `02`'s recommendation to drop/lower the persona rate is adopted, this
  ablation's finding is moot for the remaining low-rate personas; if
  personas are kept, generic labels appear to work as well as concrete ones.

## Overall recommendation

Taken together, the axes tested split into three groups:

1. **Not earning their keep** — `02` (persona rate) and `03` (intent) show
   no measurable quality benefit over a much simpler default (persona rate
   near 0, or dropping intent). `10` shows persona *specificity* doesn't
   matter either, so if personas are kept, generic labels are fine.
2. **Needs a different default, not removal** — `06` (pool split): the
   proposed `GENERAL_P=0.6` overshoots the stated 40% general target; use
   ~0.42-0.45 instead. `05` (country): add multi-country sampling for
   fr/de/es/pt/nl — clear win, no downside measured.
3. **Not delivering on its premise as currently built** — `01` (domain
   grounding) shows no score benefit from deeper grounding once the sampling
   bugs are fixed, and `05`'s accuracy check suggests the real-exemplar
   mechanism isn't improving country-specificity either (small sample,
   worth a larger follow-up before concluding this definitively). `09`
   (salt) shows no measured diversity benefit, but at a scale too small to
   fully trust that reading.

`07` (judge-model sensitivity) and `08` (script effects) are both
confidence checks rather than defaults to change: they support trusting the
other nine findings as real rather than judge-model or script artifacts.

**Suggested next steps, not decided here:** (a) a larger `01`/`05`
follow-up specifically on the real-exemplar mechanism, since the current
finding rests on only 9 exemplar-grounded rows; (b) the deferred
generator-model-sensitivity and interaction-effects ablations, now that
each axis's solo effect is known; (c) an actual decision on `PERSONA_P`,
intent, `GENERAL_P`, and multi-country sampling defaults in
`synthgen/localized/`, informed by but not automatically dictated by this
pilot-scale study.
