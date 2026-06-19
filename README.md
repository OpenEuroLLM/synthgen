# synthgen

Multilingual synthetic instruction-following (IF) data generation, with pluggable
generation backends.

`synthgen` builds prompts across **language × topic × persona × constraint**
axes, generates them with one or more chat-completions models for diversity,
filters leakage, verifies lang-id and constraint compliance, and produces a
clean SFT JSONL suitable for fine-tuning.

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

## Pipeline

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

### Empirical topic distribution

```bash
synthgen topics --dataset allenai/WildChat-1M --n 2000   # needs `.[topics]` + OPENROUTER_API_KEY
```

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
- Phase 3 (production): 11 EU languages — 8 Dolci-trained
  (`es fr de it pt pl nl cs`) + 3 held-out (`ro el uk`).

## Library usage

```python
from synthgen import SynthConfig, Paths
from synthgen.backends import OpenRouterBackend, VLLMBackend
from synthgen import prompts, generate, filter as flt, verify

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
├── cli.py               # `synthgen` entrypoint
├── config.py            # SynthConfig, Paths, constants
├── log.py, io.py
├── prompts.py
├── backends/
│   ├── base.py          # Backend protocol
│   ├── openrouter.py
│   └── vllm.py          # EndpointPool + VLLMBackend
├── generate.py          # backend-agnostic runner, prompt + response modes
├── filter.py
├── verify.py
├── back_translate.py
├── dedupe.py
├── qc.py
├── to_open_instruct.py  # export SFT JSONL → per-language parquet source
└── topics.py            # empirical topic distribution
```

## Environment

- `OPENROUTER_API_KEY` — required for `openrouter` backend, `back-translate`, `topics`.
- `SYNTHGEN_ROOT` — project root (default: cwd).
- `SYNTHGEN_LOG` — log level (default: `INFO`).
