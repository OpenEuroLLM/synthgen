# localized_axes_ablation

Eleven pilot-scale ablations of `synthgen.localized`'s conditioning axes
(domain, grounding depth, persona rate, intent, country, pool split, salt,
persona specificity, judge model, script, dedupe cost), run independently
and compared against a shared diversity/quality baseline. Full design
rationale in [`PLAN.md`](PLAN.md) — read that first.

**Status: ablations 01-10 complete at full scale; `11_topup_dedupe_cost` is
built but not yet run** (needs a fresh SLURM submission — see its README).
Real findings, numbers, and recommendations for 01-10 are in
[`REPORT.md`](REPORT.md). Headline: `02`
(persona rate) and `03` (intent) show no measurable quality benefit over
simpler defaults; `01` (domain grounding) shows no score benefit from deeper
grounding once its sampling bugs were fixed, and `05`'s accuracy check
suggests the real-exemplar mechanism isn't improving country-specificity
either (small sample); `06` needs `GENERAL_P≈0.42-0.45`, not the proposed
0.6, to hit a real 40% general composition; `05`'s multi-country sampling is
a clear win with no measured downside; `07`/`08` (judge-model and script
sensitivity) support trusting the other findings.

**Two bugs were found and fixed in `01_domain_grounding/prompts_grounded.py`
during a correctness pass before the full-scale run** (see `PLAN.md` for
detail): the per-arm RNG seed included `grounding_mode`, so `none`/`light`/
`deep` were silently comparing different sampled rows instead of the same
rows under different treatment; and Pool A domain sampling never actually
used `ground_domains.py`'s real-data weights (plain uniform `rng.choice`).
Both fixed and unit-verified before the full-scale job launched — the
`01`'s smoke-test data collected before this fix is stale and was not reused.

**A third, cross-cutting metrics bug was found and fixed after the full-scale
run**: every ablation's diversity report had silently fallen back to
lexical-only near-dup detection (`sentence-transformers` wasn't in the vLLM
container), and the embedder module's default model turned out to be
English-only — it flagged a spurious 33% near-dup rate on Ukrainian text that
vanished to 0% under a proper multilingual model. Fixed in
`_common/diversity_metrics.py` (default model swapped to
`paraphrase-multilingual-MiniLM-L12-v2`) and re-scored against every existing
`records.jsonl`, no regeneration needed — see `REPORT.md`'s "Metrics fix"
note and `_common/rescore_embeddings.py`.

**Acted on since: `09`'s finding was carried into production code.** The
`salt` text nudge showed no measurable diversity benefit (before or after the
metrics fix above — a fresh salt-vs-no-salt re-score with the corrected
multilingual model still landed both arms at the same ~0.02% pooled
embedding near-dup rate) and had no programmatic backstop anyway — it just
asked the model to be non-obvious, with nothing checking compliance. It's
been removed from `GENERATION_PROMPT` (`synthgen/localized/prompts.py`; the
`salt` RNG draw is kept for stream stability and row provenance, just no
longer surfaced in the prompt text). The real backstop is new:
`synthgen/pipeline/near_dup.py`, an embedding-based near-dup filter wired
into `dedupe.py` and, structurally, into `topup.run()`'s
`embedding_dedupe=True` — a dropped near-duplicate there shrinks that
round's survivor count, which the topup loop's own deficit calculation turns
into a real replacement generation, not just a flagged row. `11` (below)
measures what enabling that actually costs before it's turned on for a full
production run.

## Subfolders

| # | What | Depends on |
|---|---|---|
| [`04_diversity_baseline`](04_diversity_baseline/) | Reference judge-score + near-dup numbers for the current, unmodified pipeline | — (run first) |
| [`01_domain_grounding`](01_domain_grounding/) | Broadened taxonomy + orthogonal none/light/deep grounding mode + real-data anchoring (WildChat/LMArena), Pool A weighted by real-data frequency | — |
| [`02_persona_rate`](02_persona_rate/) | `PERSONA_P` sweep: 0.5 (current) / 0.15 / 0.10 / 0.0 | `04` (for comparison) |
| [`03_intent_necessity`](03_intent_necessity/) | Intent-injected vs. intent-stripped, paired | — |
| [`05_country_grounding`](05_country_grounding/) | Country-accuracy check + single- vs. multi-country sampling | `01` (accuracy check only) |
| [`06_domain_split_ratio`](06_domain_split_ratio/) | Pool split (`GENERAL_P`) sweep: 0.2 / 0.4 / 0.6 / 0.8, raw vs. post-filter ratio | `04` (for comparison) |
| [`07_judge_model_sensitivity`](07_judge_model_sensitivity/) | Re-judges existing generations with a second judge model (`google/gemma-4-31b-it`) — no new generation | `04` (or any completed ablation's `records.jsonl`) |
| [`08_script_effects`](08_script_effects/) | Regroups existing generations by script (Latin vs. non-Latin) instead of language — no new generation, no GPU needed | `04` (or any completed ablation's `records.jsonl`) |
| [`09_salt_necessity`](09_salt_necessity/) | Salt-present vs. salt-removed, paired — diversity is the primary outcome | — |
| [`10_persona_specificity`](10_persona_specificity/) | Generic persona label vs. concrete situated persona, same suggestion rate | — |
| [`11_topup_dedupe_cost`](11_topup_dedupe_cost/) | Cost of enabling `topup.run(embedding_dedupe=True)`: extra generate/judge calls vs. `embedding_dedupe=False`, same target | `09` (motivates it), needs `sentence-transformers` in the vLLM container |

`_common/` (`diversity_metrics.py`, `report.py`) is shared infrastructure, not
an ablation — every subfolder imports it rather than reimplementing
near-duplicate detection or histogram reporting.

## Deliberately deferred (not built this round)

**Generator-model sensitivity** — needs a second generator model chosen and
staged first (a real decision, not defaulted). **Interaction effects**
(factorial design across axes) — designed to run only after each axis's
individual effect is known; the cheapest useful design is still ~8 cells ×
150 rows, the single largest line item in the study's compute budget.

## Running

Each subfolder is self-contained: its own `README.md`, its own script(s), its
own `run.slurm` (same serve-vLLM-then-run pattern as
`studies/localized_bootstrap/smoke_pipeline.slurm`), its own `outputs/`.
Suggested order: `04` first, then `01` (needed by `05`'s accuracy check),
then the rest in any order. `01_domain_grounding/ground_domains.py` is the
one step that needs internet — run it on a login node before any SLURM job.

## Branch

This whole directory lives on a dedicated `studies` branch, never on `main`
— see the repo's `CLAUDE.md` for why (`.claude/`-adjacent convention: study
code stays out of the package history). Confirm `git branch`/`git status`
before committing.

