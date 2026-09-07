# 04 — diversity baseline

Reference numbers every other ablation in this study diffs against: judge
score distribution + near-duplicate rate (lexical and embedding) of the
**current, unmodified** `synthgen.localized` pipeline at pilot scale.

Run this first — `01`, `02`, and `06` all compare their own arms back to
this baseline.

## Run

```
python measure_baseline.py \
    --gen-endpoints  $EP/gen  --judge-endpoints  $EP/judge \
    --n-per-lang 30 --langs es fr de pl uk
```

See `run.slurm` for the full serve-vLLM-then-run pattern (same shape as
`studies/localized_bootstrap/smoke_pipeline.slurm`).

## Output

`outputs/records.jsonl` — one row per generated example, with judge score.
`outputs/summary.json` — score stats + `_common/diversity_metrics.py`'s
per-`(lang, domain)` near-duplicate rates.

## What "done" looks like

A `summary.json` with a non-degenerate score distribution (not everything
piled at 0 or 10) and near-duplicate rates that are neither implausibly 0%
(metric probably broken) nor implausibly high (real collapse, or threshold
too loose — sanity-check a few flagged pairs by hand before trusting the
number).
