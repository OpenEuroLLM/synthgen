# 06 — domain split ratio (Pool A / Pool B)

Tests the pool-split composition (`GENERAL_P` today) at
`{0.2 (current default), 0.4, 0.6 (proposed target), 0.8}`, using
`synthgen.localized.prompts.build_rows(general_p=...)` directly — production
code, not a fork.

## Why this reports two numbers, not one

Local-culture (Pool B) rows plausibly survive the judge threshold at a
different rate than general (Pool A) rows — it's harder to get a specific
local fact right than to write a competent generic coding/math response. So
a 60/40 **raw** split (what `general_p` directly controls) is not the same
as the 60/40 **post-filter** split you'd actually end up training on.
`split_sweep.py` runs the real production
`synthgen.localized.quality_filter.run()` on each arm and reports both
ratios explicitly — never trust a "target split" number that only reflects
the sampling-time knob.

## Run

```
python split_sweep.py \
    --gen-endpoints $EP/gen --judge-endpoints $EP/judge \
    --n-per-lang 40 --langs es fr de --threshold 6.0
```

`--threshold` is a single human-reviewed global cutoff by default, not
`decide-thresholds` output — per `CLAUDE.md`'s documented caveat, percentile
thresholds degenerate at pilot scale (~10 labeled rows/language).

## Output

`outputs/general_p_<value>/{gen,judged,filtered}.jsonl` + `summary.json` per
arm (raw ratio, post-filter ratio, filter_summary from the real
`quality_filter.run()`, score stats, diversity). `outputs/arm_comparison.json`
for the raw-vs-post-filter table across all four arms.

## What "done" looks like

A table showing, for each raw split, what it actually becomes after
filtering — enough to pick a `general_p` value that *targets* the desired
final composition (e.g. if Pool B survives filtering at 70% the rate of
Pool A, hitting a true 60/40 final split needs a different raw split than
0.6).
