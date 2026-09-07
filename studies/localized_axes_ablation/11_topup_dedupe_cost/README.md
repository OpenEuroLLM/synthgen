# 11 — topup embedding-dedupe cost

Does turning on `synthgen.localized.topup.run(embedding_dedupe=True)` for
real cost anything in practice, or is it free at the collapse rates this
study has actually measured?

## Why this exists

`09_salt_necessity` found no measurable diversity benefit from the `salt`
prompt nudge, which was then removed from `GENERATION_PROMPT` (see
`synthgen/localized/prompts.py`) — it was never a real check, just a request
the model could ignore silently. The actual backstop now lives in
`synthgen/pipeline/near_dup.py`: an embedding-based near-dup filter wired
into `topup.run()` as `embedding_dedupe=True`. Because `topup`'s loop
recomputes each language's deficit from the *deduped* survivor count every
round, a row the embedding filter drops isn't just flagged — it triggers a
real replacement generation next round. That makes the diversity guarantee
structural instead of a text hint, but every drop costs one extra
generate+judge call pair. This ablation puts a number on that cost before
anyone enables it for a full production run.

## Design

Run `topup.run()` to the **same `--target`** twice, same seed, same gen/judge
backends, into two isolated `SynthConfig` roots (`outputs/run_off/`,
`outputs/run_on/`) so neither arm's on-disk state touches the other:

- **off** — `embedding_dedupe=False`, i.e. current production default
  (exact-match dedup only).
- **on** — `embedding_dedupe=True`.

Reports, per arm: `total_raw_generated` (every successful `generate()` call,
summed across rounds) and `rounds` to hit target. For **on** only:
`embedding_near_dup_dropped_total` — the actual number of rows the embedding
filter caught across the whole run. The headline number is the delta:

```
cost_delta_pct = (on.total_raw_generated - off.total_raw_generated)
                 / off.total_raw_generated * 100
```

This is the real answer to "does this cost anything" — not a guess from the
pilot-scale near-dup *rate* (which only measures collapse in what was
already generated, not what an active filter-and-regenerate loop would
actually spend catching it).

```
python measure_dedupe_cost.py --gen-endpoints $EP/gen --judge-endpoints $EP/judge \
    --target 30 --langs es fr de pl uk
```

## Caveat

Same sample-size ceiling as the rest of this study: `--target 30` per
language is pilot scale, well under the ~400-750 rows/lang this study's own
sizing note says is needed to trust a per-domain collapse rate. A near-zero
cost delta here is consistent with (not proof of) a near-zero cost at
production scale — `04`/`09`'s embedding near-dup rates were already ≤0.1%
at this same scale, so a large delta would be a surprise worth digging into,
not dismissing.

## Output

`outputs/run_off/`, `outputs/run_on/` (full topup working directories, each
arm's own `prompts/`, `outputs/loc_full.dedup.jsonl`, etc.) and
`outputs/summary.json` (the comparison: both arms' `topup.run()` return
dicts, `extra_raw_generations`, `cost_delta_pct`).
