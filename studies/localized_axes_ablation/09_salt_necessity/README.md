# 09 — salt necessity

Does the `salt` "diversity seed" text nudge do anything measurable, or is it
cosmetic? Item 1 in the top-level README's "logged for later" list, now built.

Paired design (same shape as `03_intent_necessity`): each row generated twice
with identical `(lang, domain, role, intent)` — once with the production
`Diversity seed: {salt}` clause, once with a stripped fork that removes it
entirely. Both judge score AND `_common/diversity_metrics` are compared —
diversity is the *primary* outcome here, not a secondary check, since
fighting near-duplication is salt's whole reason for existing.

```
python salt_ablation.py --gen-endpoints $EP/gen --judge-endpoints $EP/judge \
    --n-per-lang 30 --langs es fr de pl uk
```

## Output

`outputs/records.jsonl` (paired) + `outputs/summary.json` (score stats +
diversity for both arms).
