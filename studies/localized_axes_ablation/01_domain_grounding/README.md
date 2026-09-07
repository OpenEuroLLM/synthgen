# 01 — domain grounding (broadened taxonomy + orthogonal grounding mode)

Three things bundled together because they're motivated by the same finding
(see `../PLAN.md` Context items 1–2):

1. **Broadened taxonomy** (`taxonomy_broadened.py`): Pool A (general/
   dataset-common — includes `synthgen/pipeline/topics.py`'s `TOPIC_NAMES`,
   ported here for the first time) + Pool B (existing local-culture domains,
   unchanged).
2. **Grounding mode as its own axis** (`prompts_grounded.py`), decoupled from
   which domain was sampled: `none` / `light` (surface texture — currency,
   units, dates, names, register) / `deep` (country-specific fact as the
   task's substance, using a real grounding example when available). Capped
   per domain by `taxonomy_broadened.max_grounding_mode()`.
3. **Real-data anchoring** (`ground_domains.py`, run separately beforehand):
   classifies real WildChat-1M/LMArena-100k prompts into Pool A domains,
   producing (a) real exemplars for the `deep` grounding block and (b) a
   Pool-A-only empirical frequency table — **never** applied to Pool B (see
   `../PLAN.md`'s "two pools, not one axis" design decision).

## Step 0 — warm_texts_cache.py (login node, needs internet, run once)

SLURM compute nodes on LUMI have no internet access, so raw text sampling
from WildChat-1M/LMArena-100k must happen on the login node first:

```
singularity exec --bind /scratch/project_462001516:/scratch/project_462001516 \
  --env HF_HOME=$HF_HOME --env PYTHONPATH=$(pwd) \
  /scratch/project_465002530/containers/vllm-openai-rocm.sif \
  python3 warm_texts_cache.py --n-per-lang 400 --langs es fr de pl uk \
  --cache-dir texts_cache
```

Writes `texts_cache/<lang>.json` — raw sampled texts only, no classification,
no API key needed. `ground_domains.py` (below) reuses this cache instead of
re-streaming, so it can run classification-only from inside a SLURM job.

## Step 1 — ground_domains.py (classification; now runs inside run.slurm)

Also runnable standalone from the login node with `OPENROUTER_API_KEY` set:

```
python ground_domains.py --n-per-lang 400 --out . --texts-cache texts_cache
```

Or, with no API key, against a running vLLM server (this is what `run.slurm`
now does automatically as its first step, reading from `texts_cache/`):

```
python ground_domains.py --classifier-backend vllm --vllm-endpoints $EP/gen \
  --texts-cache texts_cache --n-per-lang 400 --out .
```

Writes `domain_exemplars/<lang>.jsonl` and `domain_distribution/<lang>.json`
(+ `_aggregate.json` fallback for thin-coverage languages). These are static
artifacts.

## Step 2 — prompts_grounded.py (SLURM, vLLM servers)

```
python prompts_grounded.py \
    --gen-endpoints $EP/gen --judge-endpoints $EP/judge \
    --n-per-lang 30 --langs es fr de --exemplar-dir domain_exemplars
```

Sweeps `{none, light, deep}`, judges each arm, reports score + diversity per
arm, and flags how many `deep` requests got clamped to `light` because the
sampled domain's ceiling didn't allow `deep`
(`taxonomy_broadened.POOL_A_DEEP_ELIGIBLE`).

## Validation loop (do this, don't skip it)

Re-classify the *generated* instructions from the `deep`-with-empirical-
weights condition using the same classifier prompt `ground_domains.py` used,
and diff the resulting per-domain distribution against the WildChat/LMArena
target. If they don't match, the weighted sampler isn't doing what it claims
— check `taxonomy_broadened.POOL_A` labels are consistent between
`ground_domains.py`'s classifier and `build_grounded_rows()`'s sampler before
trusting any downstream comparison.

## What this does NOT test

Whether grounded content actually lands on the right *country* — that's
`05_country_grounding/check_country_accuracy.py`, which consumes this
subfolder's `deep`-arm output. Run `01` before `05`.

## Output

`outputs/mode_{none,light,deep}/records.jsonl` + `summary.json` per arm
(including `pool_a_rows`/`pool_b_rows`/`clamped_to_light` counts).
`outputs/arm_comparison.json` for the headline table.
