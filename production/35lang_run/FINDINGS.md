# Findings: the 35-language production run's provisioning bugs

Captured 2026-10-19, after ~3 weeks / 3 SLURM resubmissions on LUMI, before
applying the fix designed in response to these findings (see git history
around this commit for the actual code change). A data backup from this
point is at `/scratch/project_465002530/abhasjha/
oellm_production_35lang_backup_20261019_1446/`.

## Run history

- **Job 22406130** (first submission, 16 nodes, 12 gen + 4 judge): hit the
  48h `standard-g` time limit on 2026-10-02, having finished the entire gen
  phase (5.6M raw rows requested) but only ~30 minutes / 36,000 rows into
  judge.
- **Job 22736864** (resubmit, resumed from checkpoint): hit the 48h limit
  again on 2026-10-16, this time 71.3% through judge (3,964,000 /
  5,559,456).
- **Job 22895003** (resubmit, `--embedding-dedupe` removed to isolate the
  judge bottleneck): as of this writing, still running — but discovered
  along the way that `topup`'s round loop had never reached its
  quality-filter + dedupe step even once across all three jobs, because
  every single submission died mid-gen or mid-judge first.

## Judge score distribution (as of ~4.0M rows judged, partial — 8 of 35
languages not yet reached by judge at all)

- Overall average score: **8.07** / 10. Pass rate (score ≥ 7, the
  `thresholds.json` cutoff): **78.1%**.
- Distribution is bimodal, not smooth: 60.9% score exactly 10, but there's
  also a distinct cluster at score 2 (10.6%) — a real quality cliff, not
  noise in the tail.

### Per-language pass rate and % of the 100k/language target already met
(pre-dedup estimate: `judge-passing-count / 100,000`)

| Lang | Pass rate | % of target | | Lang | Pass rate | % of target |
|---|---|---|---|---|---|---|
| sv | 99.3% | 149.6% | | sk | 78.5% | 118.2% |
| de | 99.0% | 149.9% | | cs | 79.6% | 118.0% |
| pt | 98.9% | 150.1% | | gl | 85.1% | 106.2% |
| uk | 98.8% | 150.8% | | **sl** | 61.3% | **92.4%** |
| da | 98.8% | 149.8% | | **hu** | 47.7% | **72.3%** |
| fr | 98.6% | 151.1% | | **lt** | 51.6% | **76.9%** |
| pl | 97.7% | 149.8% | | **et** | 35.8% | **52.6%** |
| hr | 96.4% | 149.2% | | **lv** | 49.9% | **56.0%** |
| it | 96.2% | 147.7% | | **mt** | 34.0% | **50.8%** |
| bg | 95.7% | 148.5% | | **eu** | 33.9% | **37.5%** |
| nl | 95.2% | 147.7% | | **ga (Irish)** | **8.3%** | **12.5%** |
| es | 93.5% | 144.8% | | | | |
| ro | 92.7% | 141.6% | | | | |
| ca | 82.4% | 127.5% | | | | |
| el | 77.9% | 119.0% | | | | |
| fi | 97.2% | 147.3% | | | | |

Not yet judged at all: `bs, ka, mk, sq, sr, tr, is, no`.

**`ga` (Irish) is a severe outlier** — roughly 1 in 12 rows pass versus
~19 in 20 for most Western European languages. Worth a dedicated
investigation (is the judge unreliable in Irish, or is the generator
producing genuinely poor Irish text, or is there a parsing/prompt issue
specific to this language?) rather than just throwing more raw volume at
it.

## Raw generation volume (before any judge filtering) — never the bottleneck

Every one of the 35 languages already has 165-254% of the 100k target in
raw generations, confirmed directly from `loc_gen.jsonl`'s per-language row
counts (e.g. `lv`: 253,761; `eu`: 237,910; down to `sr`: 164,552 as the
smallest). **Raw volume was never short anywhere, including for `ga`**
(168,956 raw rows = 169% of target already). The shortfall is entirely a
quality/pass-rate problem, not a quantity problem.

## The two provisioning bugs found

1. **`topup`'s round-start decision to write new generation prompts for a
   language is blind to judge data.** It only checks `loc_full.dedup.jsonl`
   (which has never existed — no round has ever reached dedupe) and a
   "does this language have unprocessed prompts queued" backlog check. It
   has no way to see the judge-score data above, so it can't tell an
   already-satisfied language (`de`, `bg`, `da` at ~150%) from a desperately
   short one (`ga` at 12.5%) — both just show "survivors=0" since the file
   doesn't exist.
2. **The overgen factor is pooled across all 35 languages, not computed
   per-language.** A single rate (driven mostly by the 90%+-pass-rate
   majority) gets applied uniformly, so `ga`'s real need (~12× overgen to
   close its gap) gets the same ~1.6× as everyone else — wildly
   insufficient.

**Concrete observed consequence:** in job 22895003's first real
round-start decision, 9 languages happened to have a fully-cleared gen
backlog (`de, bg, da, ga, hr, ca, gl, bs, tr`) and got a blind
`+160,000`-row batch each — 6 of those 9 were already at 106-150% of
target (pure waste), while 7 genuinely-short languages (`eu, mt, et, lv,
lt, hu, sl`) got nothing because they had a small nonzero backlog (1-3,495
leftover rows, unrelated to actual need) at that exact moment.

## Dedup cost sizing (separate finding, already acted on this session)

`embedding_dedupe_mask`'s O(n²) near-dup scan, benchmarked for real on a
LUMI debug node (`production/35lang_run/_dedup_sizing_check.py`,
`_dedup_sizing_check2.py`): `t ≈ 2.68e-07 · n²` seconds per language
bucket. At production scale (~140-250k kept rows/language), that's
~15-54 hours **total** if run serially across all 35 languages — bigger
than the judge phase itself. Fixed by parallelizing across the
(independent) language buckets via a process pool
(`synthgen/pipeline/near_dup.py`'s `max_workers` parameter, `--embedding-
workers` CLI flag) — wall-clock drops from "sum of all buckets" to
roughly "the single slowest bucket" (~195 min worst case). Verified
correct (parallel and serial produce identical kept-row sets) on a real
debug-node run.
