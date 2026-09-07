# 02 — persona rate sweep

Tests the supervisor's hypothesis: does `PERSONA_P = 0.5` (current default —
persona suggested on half of all rows) actually earn its keep over a much
lower rate (10–15%), or is persona conditioning mostly cosmetic?

Arms: `PERSONA_P in {0.5 (baseline), 0.15, 0.10, 0.0}`, using
`synthgen.localized.prompts.build_rows(persona_p=...)` directly — production
code, not a fork — so nothing about domain/intent/salt sampling changes
except the persona rate.

## Known limitation — read before trusting a small score delta

`build_rows()`'s per-row RNG only consumes an *extra* `rng.choice(ROLES)` call
when the persona coin-flip actually lands true. Two arms with different
`persona_p` will therefore land on that coin-flip differently row-by-row,
which shifts how many random calls have been consumed by the time later rows
in the same batch draw their domain/intent/salt — so **domain/intent/salt
across arms are correlated but not bit-for-bit identical past the first
divergent row**, contrary to the "same seed stream" framing above. This is
a real but second-order effect (it reshuffles *which* domain a given row
index gets, not the aggregate domain distribution) — good enough for a
directional read at pilot scale, but don't over-interpret a small
(`< ~0.3` mean-score) delta between arms as persona-attributable without
checking the diversity metrics agree.

## Run

```
python persona_sweep.py \
    --gen-endpoints $EP/gen --judge-endpoints $EP/judge \
    --n-per-lang 30 --langs es fr de
```

## Output

`outputs/persona_p_<value>/records.jsonl` + `summary.json` per arm,
`outputs/arm_comparison.json` for the headline mean-score table across arms.

## What "done" looks like

A judge-score delta between `persona_p=0.5` and `persona_p=0.10/0.15` small
enough to justify dropping the suggestion rate (supports the supervisor's
hypothesis), or large enough to keep it — either way, a number instead of a
guess. Also check whether the *diversity* metrics move with persona rate —
if personas are the main thing preventing surface-form collapse, dropping
the rate should show up as a higher near-duplicate rate even if judge scores
don't move.
