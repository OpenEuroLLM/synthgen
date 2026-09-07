# Study: ablating the localized-generation grounding axes

**Status: design only — nothing in this study has been built yet.** This
file is the working design doc; update it as each subfolder gets built
rather than letting implementation drift away from what's written here.

## Context

`synthgen.localized` conditions generation on four axes — `domain`, `role`
(persona), `intent`, and a `salt` "diversity seed" — sampled uniformly by
`synthgen/localized/prompts.py:build_rows()`. Working through each axis
surfaced concrete, testable concerns:

1. **Domain** (`synthgen/localized/taxonomy.py`) is a hand-authored, 30-label
   taxonomy with no tie to real data and no cultural grounding beyond a
   `{country}` string — the LLM invents its own "specific" facts. It's also
   missing the topic categories real IF datasets are full of (coding, math,
   summarization, translation, data_analysis, generic qa, creative_writing,
   roleplay, brainstorming — literally `synthgen/pipeline/topics.py`'s
   `TOPIC_NAMES`, which exists for the *other* pipeline but was never ported
   to the localized one).
2. **Cultural grounding is currently welded to domain choice** (`is_local`
   gates `_SCOPE_LOCAL` vs `_SCOPE_GENERAL` in `prompts.py`) rather than
   being an orthogonal axis — so the 10 `GENERAL_DOMAINS` get zero cultural
   treatment, and there's no way to ask "is grounding worth anything on a
   coding/math prompt" separately from "is grounding worth anything at all."
3. **`salt`** (the diversity seed) is a random integer text-nudge only — no
   programmatic dedup beyond exact-string match
   (`synthgen/pipeline/dedupe.py:17-26`). Real risk of semantic mode collapse
   (paraphrased near-duplicates) the pipeline currently cannot detect.
4. **`role`/persona** is suggested 50% of the time (`PERSONA_P = 0.5`,
   `prompts.py:22`); the user's supervisor suspects persona conditioning
   doesn't matter much and a 10–15% rate might do as well.
5. **`intent`** is already fully automatic and always injected — the open
   question is whether it's *necessary*, answered by an ablation.
6. **The general/local domain split** (`GENERAL_P = 0.2` today) is an
   unvalidated fixed constant. Target composition under discussion is 60%
   general / 40% local — a large swing from the current 80% local default,
   worth measuring rather than assuming.
7. **`LANG_COUNTRY`** (`synthgen/config.py:42-46`) maps each language to one
   representative country (`"pt": "Portugal"`, dropping Brazil; `"fr":
   "France"`, dropping Belgium/Canada/Switzerland) — a second, independent
   mode-collapse source on top of `salt`.

**Design decision this study is built around — two pools, not one axis:**
Matching a real dataset's topic distribution is the right goal for general
capability mix (how much coding vs. math vs. creative writing), and the
*wrong* goal for how much culturally-specific content to generate — real
chat logs barely contain "public holidays specific to my country" as a
request shape, not because it's low-value, but because `synthgen.localized`
exists specifically to manufacture coverage that's naturally scarce in
scraped data. So:

- **Pool A — general/dataset-common domains** (broadened list, includes
  `topics.py`'s `TOPIC_NAMES`): sampled with weights from a real dataset's
  empirical distribution.
- **Pool B — local-culture domains** (existing 20): stays on a protected,
  dataset-independent allocation — real data informs *content grounding*
  here (real dish names, real institution names), never *frequency*.
- **The split between A and B** (`GENERAL_P`, renamed conceptually to "pool
  split") is an explicit, tested dial — not derived from the real dataset.

**Grounding depth has a ceiling set by topic type, not a uniform coin-flip:**
topics with a real-world referent to source from (`qa`, `creative_writing`,
`roleplay`, `brainstorming`, all of Pool B) can take any of three tiers;
abstract/self-contained topics (`coding`, `math`, `data_analysis`) can only
sensibly take the lighter two:

| Tier | What it means | Applies to |
|---|---|---|
| `none` | Ungrounded, plain general-capability prompt | any domain |
| `light` | Surface texture only: currency, units, dates, names, formality register | any domain |
| `deep` | Country-specific fact/institution/custom as the substance of the task | domains with a real cultural referent (Pool B; `qa`, `creative_writing`, `roleplay`, `brainstorming` in Pool A) |

This is a `studies/`-only investigation (not `main`, per repo convention) —
its job is to produce evidence (judge-score distributions + diversity
metrics) that informs whether/how to change `synthgen.localized`'s defaults,
not to change them directly.

## Where this lives

New top-level directory `studies/localized_axes_ablation/` (this directory),
alongside the existing `studies/localized_bootstrap/` (whose
`run_judge_validation.py` / `taxonomy.py` / `prompts.py` / judge-prompt
machinery this reuses via a shared helper module — see `_common/` — not
re-derived; do not duplicate the good/medium/bad harness). Every ablation
gets its **own self-contained subfolder** — separate script(s), separate
outputs, separate mini-README — so any one can be run, read, or handed off
independently of the rest:

```
studies/localized_axes_ablation/
├── PLAN.md                    # this file
├── README.md                  # overview of all ablations, links to each
├── _common/                   # shared helpers, imported by every study
│   ├── diversity_metrics.py   # lexical (n-gram Jaccard/self-BLEU) +
│   │                          # embedding (sentence-transformers cosine
│   │                          # similarity) near-duplicate detection
│   └── report.py              # histogram/report helpers — thin wrapper
│                              # around localized_bootstrap's existing
│                              # _hist/print_report, not a rewrite
├── 01_domain_grounding/
│   ├── README.md
│   ├── ground_domains.py      # offline, login-node, needs internet:
│   │                          # broadened-domain classifier over
│   │                          # WildChat-1M/LMArena-100k -> per-domain
│   │                          # exemplars AND per-domain frequency
│   │                          # (Pool A only) -> domain_exemplars/,
│   │                          # domain_distribution/
│   ├── taxonomy_broadened.py  # Pool A (topics.py's TOPIC_NAMES) + Pool B
│   │                          # (existing LOCAL_DOMAINS), tier-ceiling
│   │                          # table above
│   ├── prompts_grounded.py    # grounding-mode axis (none/light/deep),
│   │                          # feature-flagged so each mode can run
│   │                          # side-by-side on identical
│   │                          # (lang, domain, role, intent, salt) draws
│   ├── domain_exemplars/      # ground_domains.py output (static artifact)
│   ├── domain_distribution/   # ground_domains.py output (Pool A weights,
│   │                          # per-language + aggregate fallback)
│   └── run.slurm
├── 02_persona_rate/
│   ├── README.md
│   ├── persona_sweep.py       # PERSONA_P in {0.5 (baseline), 0.15, 0.10, 0.0}
│   └── run.slurm
├── 03_intent_necessity/
│   ├── README.md
│   ├── intent_ablation.py     # intent-injected vs. intent-stripped, paired
│   └── run.slurm
├── 04_diversity_baseline/
│   ├── README.md
│   ├── measure_baseline.py    # unmodified pipeline + _common/diversity_metrics
│   │                          # -> the reference every other ablation diffs against
│   └── run.slurm
├── 05_country_grounding/
│   ├── README.md
│   ├── check_country_accuracy.py  # does grounded content land on the
│   │                               # intended country, or genericize/drift?
│   ├── country_diversify.py       # single-country vs. multi-country-
│   │                               # per-language sampling
│   └── run.slurm
├── 06_domain_split_ratio/
│   ├── README.md
│   ├── split_sweep.py         # GENERAL_P (pool split) in
│   │                          # {0.2 (current), 0.4, 0.6 (target), 0.8},
│   │                          # reports RAW sampled ratio AND POST-FILTER
│   │                          # survivor ratio per arm (these can diverge —
│   │                          # see below)
│   └── run.slurm
└── REPORT.md                  # final cross-study summary (§ Verification)
```

`04_diversity_baseline/` stays independent (not folded into `01`) because
`01`, `02`, and `06` all need the same "how bad is mode collapse today"
reference number to diff against — it must exist on its own rather than
being computed once inside another subfolder's script.

**Branch**: all of `studies/localized_axes_ablation/` is developed on its
own branch off `main` (never on `main` itself, per the existing `studies/`
convention in `CLAUDE.md`) — same branch `studies/localized_bootstrap/`
lives on if that already exists, otherwise a new `studies` branch. Nothing
here touches `synthgen/` package code.

## 1. `01_domain_grounding/` — broadened taxonomy + orthogonal grounding mode + real-data anchoring

- `taxonomy_broadened.py`: defines Pool A (existing `GENERAL_DOMAINS` +
  `topics.py`'s `TOPIC_NAMES`, deduped) and Pool B (existing
  `LOCAL_DOMAINS`, unchanged), plus the tier-ceiling table from Context
  (which domains may take `deep` grounding, which are capped at `light`).
- `ground_domains.py`: streams `allenai/WildChat-1M` (fallback
  `lmsys/lmarena-human-preference-100k` if per-language coverage for
  `LANGUAGES_PHASE3` is too thin), classifies sampled real user turns into
  Pool A domains (same shape as `topics.py::CLASSIFY_PROMPT`, adapted label
  set). Two outputs:
  - `domain_exemplars/<lang>.jsonl` — real prompts bucketed by domain, used
    as `light`/`deep` grounding content (both pools).
  - `domain_distribution/<lang>.json` (+ an aggregate-across-languages
    fallback for thin-coverage languages) — **Pool A only** empirical
    weights, smoothed with a floor so it doesn't just reproduce WildChat's
    own skew; **never applied to Pool B**, which stays on its own protected
    allocation (§06). Both are static artifacts, produced once from a
    login node with internet, consumed offline afterward.
- `prompts_grounded.py`: extends
  `synthgen.localized.prompts.generation_prompt()` with the grounding-mode
  axis (`none`/`light`/`deep`, capped per the tier table) and an optional
  "Grounding example (real; for topical/stylistic anchoring only — do not
  copy verbatim)" block sourced from `domain_exemplars`. Feature-flagged so
  every combination of grounding mode can run side-by-side on identical
  `(lang, domain, role, intent, salt)` draws.
- Validation loop: re-classify *generated* instructions with the same
  classifier and compare the resulting distribution back against the
  WildChat/LMArena target (simple per-domain proportion diff) — confirms
  weighted sampling actually reproduces the target shape, not just in theory.
- Comparison: judge-score distributions and `_common/diversity_metrics.py`
  output across grounding modes, per language and per Pool A/B, against the
  `04_diversity_baseline/` reference.

## 2. `_common/diversity_metrics.py` — diversity / mode-collapse measurement

Shared module, not a standalone ablation — imported by `01`, `02`, `03`,
`04`, `05`, `06`:

- **Lexical**: pairwise n-gram Jaccard / self-BLEU over instructions,
  grouped by `(lang, domain)` — cheap, no model download, catches
  near-identical phrasing.
- **Embedding**: cosine similarity via a small local sentence-embedding
  model (`sentence-transformers/all-MiniLM-L6-v2`, downloaded once and
  cached, then loaded fully offline on LUMI) — catches paraphrased/semantic
  collapse lexical checks miss.
- Output: near-duplicate rate at a couple of similarity thresholds, plus a
  pairwise-similarity histogram per `(lang, domain)` bucket, via
  `_common/report.py`'s wrapper around `run_judge_validation.py`'s existing
  `_hist`/`print_report` helpers.

## 3. `02_persona_rate/` — persona rate sweep

`persona_sweep.py`: arms at `PERSONA_P` in `{0.5 (baseline), 0.15, 0.10,
0.0}`, domain/intent/salt sampling identical across arms (same seed
stream), built on `studies/localized_bootstrap/`'s existing generation +
judge machinery. Per arm: judge-score distribution (does dropping persona
hurt quality?) and `_common/diversity_metrics.py` output (does persona
conditioning meaningfully diversify surface form, or is it cosmetic?),
against the `04` reference. `run.slurm` runs one arm at a time, results kept
separate per arm.

## 4. `03_intent_necessity/` — intent necessity ablation

`intent_ablation.py`: paired rows with identical `(lang, domain, role,
salt)` — intent injected (current) vs. intent clause stripped entirely —
compare judge scores + diversity metrics. Also records whether the model's
self-reported `intent_used` (when intent *is* given) ever diverges from the
sampled intent, as a cheap signal on whether the axis does real conditioning
work or is just echoed back. `run.slurm` runs both arms side by side.

## `04_diversity_baseline/` — reference numbers

`measure_baseline.py`: runs the current, unmodified `synthgen.localized`
pipeline (default `PERSONA_P`, current domain taxonomy, intent always-on,
`GENERAL_P = 0.2`) at pilot scale, applies `_common/diversity_metrics.py`.
The number `01`, `02`, and `06` each diff against — generated first, lives
independently.

## 5. `05_country_grounding/` — does grounding land on the right country?

- `check_country_accuracy.py`: takes `01`'s grounded-vs-ungrounded output
  and checks whether content is actually specific to `LANG_COUNTRY[lang]`,
  or generic/wrong-country — a second judge call ("is this response
  specific to {country}, or could it describe any country?"), reusing the
  judge rather than inventing a new labeling scheme. Measures whether `01`'s
  real-data grounding improves *country*-specificity, not just topical
  relevance.
- `country_diversify.py`: for languages spoken across multiple countries,
  samples from a short list of plausible countries per language instead of
  the single `LANG_COUNTRY` default, compares judge scores + diversity
  metrics against the single-country baseline — tests whether the
  single-country simplification is itself a mode-collapse source.
- `run.slurm`: `check_country_accuracy.py` depends on `01`'s output;
  `country_diversify.py` is independent and can run standalone.

## 6. `06_domain_split_ratio/` — Pool A / Pool B split ratio

`split_sweep.py`: arms at `GENERAL_P` (pool split) in `{0.2 (current
default), 0.4, 0.6 (proposed target), 0.8}`. Per arm, report **both**:
- the raw sampled A/B ratio (what `GENERAL_P` directly controls), and
- the **post-quality-filter survivor ratio** — Pool B rows plausibly survive
  the judge threshold at a different rate than Pool A rows (harder to get
  local facts right than a generic coding/math prompt), so a 60/40 *raw*
  split could land far from 60/40 *after* filtering. This is the actual
  number that determines final dataset composition, so it's reported
  explicitly rather than assumed to match the sampling-time knob.

Per arm also report judge-score distribution and `_common/diversity_metrics`
against the `04` reference. `run.slurm` runs one arm at a time.

## Other angles worth ablating (not built this round — logged for later)

Documented in the top-level `README.md` as candidate follow-ups, so they
aren't lost, without expanding this round's scope:

- **Salt necessity itself**: does the `salt` text nudge do anything
  measurable, or would raising generation `temperature` alone achieve the
  same diversity? Nobody has ablated salt with salt removed entirely.
- **Script effects**: do these axes (grounding, persona rate, intent, pool
  split) behave differently for non-Latin-script languages (`el`, `uk` per
  `LANG_SCRIPT`) vs. Latin-script ones — the judge's leniency policy already
  treats them differently, so the axes might interact with script too.
- **Persona specificity**: current personas are generic labels (`"parent"`,
  `"retiree"`) — does a concrete, situated persona ("a 34-year-old teacher
  in Kraków") outperform the generic label at the same suggestion rate, or
  is specificity orthogonal to rate?
- **Generator-model sensitivity**: do findings hold across different
  generator models, or are they an artifact of one model's idiosyncrasies?
- **Judge-model sensitivity**: does a second judge model agree on the score
  deltas, or is a finding a property of `Qwen/Qwen3.6-27B` specifically?
  (`localized_bootstrap`'s `outputs_v5_judge2/` already started on this —
  reuse rather than restart.)
- **Interaction effects**: are gains from grounding + lower persona rate +
  intent stripping + pool-split change additive, or do they interact (e.g.
  does grounding matter less once persona is dropped)? Needs a small
  factorial design rather than one-axis-at-a-time sweeps — worth doing only
  after each axis's individual effect is known.

## Running it

Each subfolder's `run.slurm` follows the existing
`studies/localized_bootstrap/smoke_pipeline.slurm` pattern: serve gen +
judge vLLM servers on `dev-g`, poll `/v1/models`, write endpoint files, then
run that subfolder's script(s) — independently of the other subfolders.
`01_domain_grounding/ground_domains.py` (the only step needing internet)
runs separately beforehand, on a login node, producing the static
`domain_exemplars/`/`domain_distribution/` artifacts ahead of any SLURM job.

Suggested order: `04_diversity_baseline` first (reference numbers), then
`01_domain_grounding` (needed by `05_country_grounding`'s
`check_country_accuracy.py`), then `02`, `03`, `05`, `06` in any order.

Scale: pilot-sized (tens of rows per condition per language, matching
`localized_bootstrap`'s existing scale), not a production run — this is
about getting a directional read on all six ablations before touching
`synthgen.localized` defaults.

## Verification

- `01_domain_grounding/ground_domains.py` producing non-degenerate
  per-domain exemplar counts and a sane (non-collapsed) frequency table per
  language (spot-check a couple of `domain_exemplars/<lang>.jsonl` and
  `domain_distribution/<lang>.json` files).
- `01`'s validation loop (re-classify generated output, diff against
  WildChat/LMArena target distribution) shows the weighted sampler actually
  approximates the target shape, not just in the sampling code.
- Each subfolder's `run.slurm` runs end-to-end against the smoke-test vLLM
  servers (gen + judge) for at least one language before the full sweep,
  and can be run in isolation (deleting the other subfolders should not
  break it, except `05`'s dependency on `01`'s output, which is documented).
- `06_domain_split_ratio` explicitly reports both raw and post-filter
  ratios per arm — confirm they're both present in `split_sweep.py`'s
  output before treating any arm's "60/40" label as accurate.
- Final deliverable: a short top-level `REPORT.md` in
  `studies/localized_axes_ablation/` summarizing all six ablations with
  concrete numbers (score deltas, near-duplicate rates, raw-vs-post-filter
  split ratios), to hand back for a decision on `synthgen.localized`
  defaults — this study does not itself change any defaults in
  `synthgen/localized/`.
- All of this work happens on a dedicated `studies` branch, never on
  `main` — confirm `git status`/`git branch` before any commit.
