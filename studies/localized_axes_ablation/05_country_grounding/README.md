# 05 — country grounding

Two related but distinct questions about `LANG_COUNTRY`
(`synthgen/config.py:42-46`) mapping each language to one representative
country (`"pt": "Portugal"`, dropping Brazil; `"fr": "France"`, dropping
Belgium/Canada/Switzerland):

## `check_country_accuracy.py` — does grounding land on the right country?

**Depends on `01_domain_grounding`'s output — run that first.** Takes the
`deep`-mode records and asks the judge a second, different question: is this
response actually specific to `{country}`, or generic/could-be-anywhere?
Compares the country-specific rate between rows that got a real grounding
exemplar vs. rows that didn't (deep mode with no exemplar available for that
domain/language). If real-data grounding isn't lifting the country-specific
rate, it's improving topical relevance without improving what it was meant
to improve.

```
python check_country_accuracy.py \
    --judge-endpoints $EP/judge \
    --records ../01_domain_grounding/outputs/mode_deep/records.jsonl
```

## `country_diversify.py` — is the single-country default itself a mode-collapse source?

**Standalone, no dependency on `01`.** Samples from a short list of plausible
countries per language (`COUNTRY_ALTERNATES` in the script — e.g. Portugal
*and* Brazil for `pt`) instead of always `LANG_COUNTRY`'s single default, via
`synthgen.localized.prompts.generation_prompt(country=...)` called directly
(production code, just with a different argument — no fork). Compare its
`summary.json` against `04_diversity_baseline`'s (same `n_per_lang`/`seed`)
to see whether diversifying country moves the diversity metrics.

```
python country_diversify.py \
    --gen-endpoints $EP/gen --judge-endpoints $EP/judge \
    --n-per-lang 30 --langs fr pt de es
```

Languages without a listed alternate (most of `LANGUAGES_PHASE3` — el, uk,
ro, cs, pl, it) are skipped with a warning; there's nothing to diversify for
them under this taxonomy.

## What "done" looks like

For `check_country_accuracy.py`: a clear answer on whether real-data
grounding actually improves country-specificity (not just topical
relevance). For `country_diversify.py`: a diversity-metric delta against the
single-country baseline large enough to justify adding multi-country
sampling to production, or small enough to conclude the single-country
simplification isn't costing much.
