# synthgen

Multilingual synthetic instruction-following (IF) data generation, with pluggable
generation backends.

`synthgen` has two generation routes that share backends, the `generate`
runner and the dedupe stage:

- **Topic route** (`synthgen.pipeline`): builds prompts across
  **language × topic × persona × constraint** axes, generates them with one or
  more chat-completions models, filters leakage, verifies lang-id and
  constraint compliance, then generates responses.
- **Localized route** (`synthgen.localized`): one call writes an instruction
  *and* its response natively in the target language, conditioned on
  **domain × role × intent × language**; an LLM judge scores every row,
  low scores are filtered out, and `topup` loops until each language has
  enough survivors. This is the route behind the released dataset
  [`openeurollm/EuroMultiSynth-Instruct`](https://huggingface.co/datasets/openeurollm/EuroMultiSynth-Instruct)
  (4.16M rows, 35 languages) — see
  [`production/35lang_run/RUN.md`](production/35lang_run/RUN.md) for exactly
  how that run was done.

Two generation routes are supported, selected per run with `--backend`:

| Backend       | Where it calls                                | Auth                    |
|---------------|-----------------------------------------------|-------------------------|
| `openrouter`  | `https://openrouter.ai/api/v1/chat/completions` | `OPENROUTER_API_KEY`  |
| `vllm`        | Local vLLM servers (OpenAI wire format)       | None — directory of `*.endpoint` files |

Both backends implement the same `Backend` protocol, so the rest of the
pipeline (filter / verify / back-translate / dedupe / qc) is route-agnostic.

## Install

```bash
pip install -e .                   # core
pip install -e ".[topics]"         # + empirical-topic estimator (needs `datasets`)
pip install -e ".[dev]"            # + ruff, pytest
```

Requires Python ≥ 3.10.

## Pipeline (topic route)

```
build-prompts → generate (--mode prompt) → filter → verify
              → generate (--mode response)
              → dedupe → qc → export-openinstruct
              (back-translate at any point for human review)
```

Outputs land under `$SYNTHGEN_ROOT` (defaults to cwd):

```
$SYNTHGEN_ROOT/
├── prompts/prompts.jsonl
├── outputs/
│   ├── gen_<model>.jsonl              # mode=prompt
│   ├── gen_<model>.filtered.jsonl
│   ├── gen_<model>.leaked.jsonl
│   ├── sft_<model>.jsonl              # mode=response
│   ├── bt_<model>.jsonl
│   └── sft_full.dedup.jsonl
├── review/
│   ├── review_<model>.tsv
│   ├── yield_<model>.json
│   └── yield_summary.json
└── logs/
```

## Quick start

### Route A — OpenRouter (hosted)

```bash
export OPENROUTER_API_KEY=sk-...
export SYNTHGEN_ROOT=/path/to/run

synthgen build-prompts --phase 3
synthgen generate --backend openrouter --mode prompt --phase 3
synthgen filter
synthgen verify
synthgen generate --backend openrouter --mode response --phase 3
```

### Route B — local vLLM

Start one or more vLLM servers and have each write its `host:port` to a file
in `endpoints/`. Example for one server:

```bash
echo "node03:8000" > endpoints/node03_8000.endpoint
```

Then point the client at the directory:

```bash
synthgen generate \
    --backend vllm \
    --mode prompt \
    --models google/gemma-4-26b-a4b-it \
    --endpoints-dir endpoints \
    --min-endpoints 1

# second pass — generated_prompt → response
synthgen generate \
    --backend vllm \
    --mode response \
    --models google/gemma-4-26b-a4b-it \
    --endpoints-dir endpoints
```

The pool re-reads `endpoints/*.endpoint` every 30s, so servers that come up
later are picked up automatically, and removing a file evicts an endpoint.

### Mixing modes

`--mode prompt` (default) reads `prompts/prompts.jsonl` and writes
`gen_<model>.jsonl`. `--mode response` reads `gen_<model>.filtered.jsonl`
(or `gen_<model>.jsonl` if no filter step has run) and writes
`sft_<model>.jsonl` — one SFT pair per row.

### Resume & retry

Every `generate` run is checkpointed: it appends one record per row and, on
restart, **skips rows that already have a successful (non-error) record** — so
re-running the same command resumes where it left off and re-attempts any rows
that errored.

To make a run keep itself whole automatically, pass `--max-attempts`: after a
pass, if the output is short of its target (the input row count), it re-runs
only the missing rows, up to N attempts, pausing `--retry-delay` seconds
between them.

```bash
synthgen generate --backend vllm --mode response \
    --models google/gemma-4-26b-a4b-it \
    --endpoints-dir endpoints \
    --max-attempts 5 --retry-delay 30
```

If it's still short after the last attempt, it logs an `INCOMPLETE` error with
the missing count (a later re-run will pick those up).

### Common follow-up

```bash
synthgen back-translate --sample 1000          # eyeball review TSV (OpenRouter)
synthgen dedupe --input outputs/sft_full.jsonl \
                --output outputs/sft_full.dedup.jsonl \
                --report outputs/dedupe_report.json
synthgen qc     --input outputs/sft_full.dedup.jsonl \
                --output outputs/qc_stats.json
```

### Export for open-instruct

Turn a finished SFT JSONL into per-language parquet under a named source dir,
ready to drop into an open-instruct training mix:

```bash
synthgen export-openinstruct \
    --input outputs/sft_full.dedup.jsonl \
    --out-dir by_language \
    --source synthgen-if \
    --report outputs/export_report.json
# writes by_language/synthgen-if/<lang>.parquet
```

### Near-duplicate removal (embeddings)

`dedupe` always drops exact duplicates of `(lang, normalized prompt)`.
`--embedding-dedupe` adds a second pass that also drops paraphrases: prompts
are embedded and a prompt is dropped if its cosine similarity to an earlier
*kept* prompt **of the same language** is at or above the threshold
(languages are never compared with each other).

```bash
synthgen dedupe --input outputs/loc_full.kept.jsonl \
                --output outputs/loc_full.dedup.jsonl \
                --report outputs/dedupe_report.json \
                --embedding-dedupe --embedding-threshold 0.97 \
                --embedding-device cuda --embedding-workers 4 \
                --lang-overrides embed_dedupe_lang_overrides.json
```

- `--embedding-model` — default `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  (multilingual on purpose; English-only embedders inflate similarity on
  other languages). Default threshold 0.85 is aggressive; the 35-language
  release used 0.97 after inspecting example pairs per similarity band.
- `--embedding-device cuda` runs it on GPUs (one worker per GPU, blocked
  matmuls; same result as the CPU path). `--embedding-workers` is then the
  number of GPUs; on CPU it is the number of languages scanned in parallel.
  At production scale use the GPU — CPU encoding is very slow.
- `--lang-overrides` (GPU only) gives languages their own model/threshold,
  e.g. LaBSE for languages MiniLM does not cover:
  `{"mt": {"model": "/path/to/labse", "threshold": 0.96}}`.
- Needs `sentence-transformers`. If it can't load, the pass is **skipped with
  a warning** and only exact dedupe is applied — check
  `embedding_dedupe_skipped` / `dropped_near_dup_embedding` in the summary.

### Empirical topic distribution

```bash
synthgen topics --dataset allenai/WildChat-1M --n 2000   # needs `.[topics]` + OPENROUTER_API_KEY
```

## Localized route (generate + judge + filter)

Used for the 35-language production run. The vLLM servers' served names
(`--served-model-name gen` / `judge`) become the file names below, and every
stage resumes by row id, so keep them stable across resubmits.

```
build-localized-prompts → generate --mode localized → generate --mode judge
                        → quality-filter → dedupe (exact [+ embedding])
```

```bash
synthgen build-localized-prompts --n 1000 --langs es fr     # prompts/localized_prompts.jsonl
synthgen generate --backend vllm --mode localized --models gen \
    --endpoints-dir endpoints/gen                          # outputs/loc_gen.jsonl
synthgen generate --backend vllm --mode judge --models judge \
    --in outputs/loc_gen.jsonl --endpoints-dir endpoints/judge   # outputs/judged_judge.jsonl
synthgen quality-filter --gen outputs/loc_gen.jsonl --judged outputs/judged_judge.jsonl \
    --thresholds thresholds.json --output outputs/loc_full.kept.jsonl   # {"global": 7}
synthgen dedupe --input outputs/loc_full.kept.jsonl --output outputs/loc_full.dedup.jsonl
```

`synthgen topup` runs that whole chain in rounds until every language has
`--target` survivors: each round first judges any unjudged backlog, then
estimates each language's survival rate and generates only what that
language still needs (`--overgen-factor`, capped by `--max-overgen-factor`).

```bash
synthgen topup --target 100000 --thresholds thresholds.json \
    --gen-backend vllm --gen-models gen --gen-endpoints-dir endpoints/gen \
    --judge-backend vllm --judge-model judge --judge-endpoints-dir endpoints/judge \
    --gen-concurrency 512 --judge-concurrency 256 --max-rounds 10
```

Notes:

- **Judge sampling.** Reasoning judges (e.g. `Qwen/Qwen3.6-27B`) are called
  with `temperature=0` and thinking disabled; otherwise they spend
  `max_tokens` thinking and the score never appears. This is set centrally
  in `generate.py` for `--mode judge`.
- **Thresholds.** `decide-thresholds` derives a per-language or global cut
  from a labelled pilot; with few labelled rows per language it is noisy, so
  check its output by hand. The release used `{"global": 7}`.
- **Concurrency.** Tune `--concurrency` per replica on your hardware; on
  H100s with vLLM, ~32 in-flight requests per judge replica was the stable
  point (256 caused connection-failure collapse).
- Per-language parquet for open-instruct: `export-openinstruct` (above); the
  Hugging Face release format is built by
  `production/35lang_run/build_hf_release.py`.

## Configuration

`synthgen.config.SynthConfig` is a dataclass holding all runtime knobs (phase,
generators, sampling, concurrency, persona/unconstrained mix). Constants
(`LANGUAGES_PHASE3`, `PERSONAS`, `CONSTRAINTS`, …) are module-level.

Defaults (`--phase 3`):
- **persona_p = 0.2** — topic > persona; persona kept for stylistic variety.
- **unconstrained_p = 0.3** — some prompts are hard-to-check by design.
- **Two generators** (`google/gemma-4-26b-a4b-it`, `openai/gpt-oss-120b`)
  split round-robin (`--split`, on by default with >1 model).

## Languages

- Phase 1 (pilot): 5 languages — `el cs pl ro uk`.
- Phase 3 (`LANGUAGES_PHASE3`, default for `--langs`): OpenEuroLLM's 35
  target languages (its priority list minus English) — `es fr de it pt pl nl
  cs ro el uk bg da et fi ga hr hu lv lt mt sk sl sv ca eu gl bs ka mk sq sr
  tr is no`. The original 11 (`es fr de it pt pl nl cs ro el uk`) were the
  first production set: 8 Dolci-trained + 3 held-out (`ro el uk`).

## Library usage

```python
from synthgen import SynthConfig, Paths
from synthgen.backends import OpenRouterBackend, VLLMBackend
from synthgen.pipeline import prompts, generate, filter as flt, verify

cfg = SynthConfig(phase=3, paths=Paths.from_root("/path/to/run"))
prompts.build(cfg)

# OpenRouter
generate.run(cfg, backends=[
    OpenRouterBackend(model="google/gemma-4-26b-a4b-it", cfg=cfg),
    OpenRouterBackend(model="openai/gpt-oss-120b", cfg=cfg),
], mode="prompt")

# Or local vLLM
generate.run(cfg, backends=[
    VLLMBackend(model="google/gemma-4-26b-a4b-it",
                endpoints_dir="endpoints", cfg=cfg),
], mode="prompt")

flt.run(cfg)
verify.run(cfg)
```

### Adding a new backend

Implement the `Backend` protocol in `synthgen/backends/base.py`:

```python
class Backend(Protocol):
    name: str
    model: str
    async def chat(self, client, *, messages, sampling) -> GenResult: ...
```

Return `{"content": ..., "usage": ..., "error": ...}` — always return, never
raise. The runner records the error alongside successes in the output JSONL.

## Layout

```
synthgen/
├── cli.py                   # `synthgen` entrypoint (both routes)
├── config.py                # SynthConfig, Paths, languages, constants
├── log.py, io.py
├── backends/
│   ├── base.py              # Backend protocol
│   ├── openrouter.py
│   └── vllm.py              # EndpointPool + VLLMBackend
├── pipeline/                # topic route + shared stages
│   ├── prompts.py
│   ├── generate.py          # backend-agnostic runner: prompt / response / localized / judge
│   ├── filter.py, verify.py, back_translate.py, qc.py
│   ├── dedupe.py            # exact dedupe (+ optional embedding pass)
│   ├── near_dup.py          # embedding near-dup removal, CPU and GPU
│   ├── to_open_instruct.py  # export SFT JSONL → per-language parquet source
│   └── topics.py            # empirical topic distribution
└── localized/               # localized route
    ├── taxonomy.py          # domain / role / intent / language axes
    ├── prompts.py           # generation + judge prompt builders
    ├── quality_filter.py    # keep rows scoring >= threshold
    ├── thresholds.py        # decide-thresholds
    └── topup.py             # resumable generate → judge → filter → dedupe loop
production/35lang_run/       # SLURM scripts + RUN.md for the 35-language release
tests/
```

## Environment

- `OPENROUTER_API_KEY` — required for `openrouter` backend, `back-translate`, `topics`.
- `SYNTHGEN_ROOT` — project root (default: cwd).
- `SYNTHGEN_LOG` — log level (default: `INFO`).
