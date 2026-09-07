# 03 — intent necessity ablation

`intent` is already fully automatic in production (always sampled, always
injected — there's no toggle to build). The open question is whether it's
*necessary*: does telling the model "suggested intent: ask for a
recommendation" change anything, or would the model land on a sensible
intent from domain + role alone?

Paired design: each sampled row (`lang, domain, role, salt`) is generated
**twice** — once with the current template (intent injected), once with
`no_intent_prompt()` (this script's only forked template — everything else
is imported straight from `synthgen.localized`) — so the comparison is
within-row, not aggregate-only.

Also recorded: whether the model's self-reported `intent_used` (when intent
*is* given) matches the intent it was told to use. A high mismatch rate
would suggest the model treats intent as a loose suggestion rather than a
real constraint — worth knowing regardless of the main ablation result.

## Run

```
python intent_ablation.py \
    --gen-endpoints $EP/gen --judge-endpoints $EP/judge \
    --n-per-lang 30 --langs es fr de
```

## Output

`outputs/records.jsonl` — one row per sampled `(lang, domain, role, salt)`,
containing both the `with_intent` and `without_intent` generation+judge
results side by side. `outputs/summary.json` — score stats for each arm,
the self-reported-intent echo rate, and diversity metrics for each arm.

## What "done" looks like

If `intent_stripped` scores and diversity are statistically indistinguishable
from `intent_injected`, that's a real signal the axis isn't pulling its
weight and could be simplified or dropped. If scores drop meaningfully
without it, intent is doing real conditioning work and should stay.
