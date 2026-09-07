# 08 — script effects

Do these axes behave differently for non-Latin-script languages (`el`, `uk`
per `synthgen.config.LANG_SCRIPT`) than Latin-script ones — the production
judge already applies a different leniency policy by language, so the axes
this study tests might interact with script too.

**No new generation.** This study's five languages (`es fr de pl uk`) already
span both script groups (`uk` = non_latin; the rest = latin), so
`04_diversity_baseline`'s full-scale run already contains both — this script
just regroups an existing `records.jsonl` by script instead of by language.
Runs anywhere with the repo importable (no GPU/vLLM needed).

```
python analyze_by_script.py --records ../04_diversity_baseline/outputs/records.jsonl
```

**Depends on**: any completed ablation's `records.jsonl` (default: `04`'s).

## Output

`outputs/summary.json` — score stats + diversity, grouped by script instead
of language.
