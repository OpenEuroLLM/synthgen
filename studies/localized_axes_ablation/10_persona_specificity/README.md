# 10 — persona specificity

Does a concrete, situated persona ("a 34-year-old teacher in Kraków...")
outperform the current generic label ("parent") at the SAME suggestion rate
(`PERSONA_P=0.5`, unchanged), or is specificity orthogonal to rate (per `02`'s
finding on rate itself)?

`CONCRETE_ROLES` in `specificity_sweep.py` pairs one concrete description
per production `taxonomy.ROLES` entry. Both arms draw the SAME generic role
via the same RNG call (`rng.choice(T.ROLES)`) — the `concrete` arm just maps
it to the longer description afterward — so domain/intent/salt/persona-coin
draws stay identical across arms at every row index (verified: see the
per-row comparison in this file's git history / commit message).

```
python specificity_sweep.py --gen-endpoints $EP/gen --judge-endpoints $EP/judge \
    --n-per-lang 30 --langs es fr de pl uk
```

## Output

`outputs/{generic,concrete}/records.jsonl` + `summary.json` per arm,
`outputs/arm_comparison.json` for the headline table.
