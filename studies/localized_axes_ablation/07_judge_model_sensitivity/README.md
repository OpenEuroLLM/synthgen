# 07 — judge-model sensitivity

Does a second judge model agree with `Qwen/Qwen3.6-27B` (the study's primary
judge) on the same instruction/response pairs, or is a score delta seen
elsewhere in this study a property of one judge model rather than the axis
being tested?

No new generation: `rejudge.py` reads an already-generated `records.jsonl`
(default: `04_diversity_baseline`'s full-scale output) and re-judges each row
with a second judge server, using the same production `judge_quality_prompt`
so only the model differs. Judge-only calls, no gen server needed — this
subfolder's `run.slurm` uses 2 GPUs, half of every other subfolder's.

The second judge served is `google/gemma-4-31b-it` (the study's *generator*
model) — a different architecture/family from Qwen3.6-27B, already cached
locally from being served as "gen" elsewhere, so no extra download.

**Depends on**: `04_diversity_baseline`'s full-scale `outputs/records.jsonl`.

```
python rejudge.py --judge-endpoints $EP/judge2 --judge-model judge2 \
    --records ../04_diversity_baseline/outputs/records.jsonl
```

## Output

`outputs/rejudge.jsonl` (per-row original vs. second-judge score) +
`outputs/summary.json` (mean |delta|, % agreement within 1 point).
