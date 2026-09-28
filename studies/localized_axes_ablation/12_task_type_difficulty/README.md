# 12 — task-type & difficulty layers

Do the two newest optional prompt layers — task type (edit/rewrite,
extraction/classification vs. open generation) and difficulty (prefer a
harder, more expert-level version when one fits) — actually do anything, or
are they ignored the way the old suggested-`intent` was (`03_intent_necessity`:
followed only ~32% of the time)?

## Why this exists

Both layers were added this session on top of the already-validated
general/local angle pair (see `synthgen/localized/prompts.py`'s docstring),
motivated by inference, not measurement: task type fills a real taxonomy gap
(no existing axis distinguishes generation from edit/rewrite/extraction);
difficulty was motivated by checking the actual eval harness directly (Arena-
Hard's published selection criteria are about prompt difficulty/specificity,
not topic). Neither claim was ever tested. This ablation tests them.

**Prerequisite, already applied**: the judge prompt's "not a genuine
request: a textbook/quiz item" disqualifier could plausibly fire on
legitimate edit/extraction requests or genuinely hard tasks, which would
confound this ablation's score comparison (can't tell "the layer doesn't
work" from "the judge wrongly zeroed the output"). A clarifying note was
added to `JUDGE_QUALITY_PROMPT` first — see `synthgen/localized/prompts.py`.

## Design

Paired, same shape as `03`/`09`: each row generated twice with identical
`(lang, domain, role, salt)` — once with the full production prompt (all
four optional layers), once with a stripped fork that removes ONLY the
task-type and difficulty clauses (general/local angle, persona, everything
else byte-identical). Three comparisons:

1. **Judge score**, paired — does removing the two clauses cost anything?
2. **Task-type distribution** — a follow-up classifier call (same judge
   model, cheap one-word-answer call) tags each generated instruction as
   `generation` / `edit_rewrite` / `extraction_classification`. If the
   "full" arm's distribution doesn't skew away from pure `generation`
   relative to "stripped", the clause isn't doing real work.
3. **Difficulty distribution** — a follow-up classifier call rates each
   instruction 1-5 for genuine expertise/complexity required. If "full"'s
   mean doesn't exceed "stripped"'s, the clause isn't doing real work either.

```
python task_type_difficulty_ablation.py \
    --gen-endpoints $EP/gen --judge-endpoints $EP/judge \
    --n-per-lang 30 --langs es fr de pl uk
```

## Caveat

Same sample-size ceiling as the rest of this study: n≈30/lang is a
directional pilot read, not a final answer. The follow-up classifier calls
are themselves LLM judgments (not ground truth) — a systematic bias in how
the judge model tags task type or rates difficulty would show up identically
in both arms, so the *comparison* between arms is more trustworthy than
either arm's absolute numbers.

## Output

`outputs/records.jsonl` (paired, both arms plus their follow-up
classifications) + `outputs/summary.json` (score stats, task-type
distribution, mean difficulty — both arms).
