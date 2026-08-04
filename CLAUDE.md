# CLAUDE.md

Orientation for an agent picking this repo up cold, with no prior session context.

## What this is

`synthgen` generates multilingual synthetic instruction-following (IF) data for
the OpenEuroLLM (OELLM) project. Two independent generation routes live under
one package, sharing backends and a few pipeline stages:

- **`synthgen.pipeline`** — the original topic-based route:
  `build-prompts → generate(--mode prompt) → filter → verify →
  generate(--mode response) → dedupe → qc → export-openinstruct`.
  See `README.md` for the full walkthrough — it documents this route in detail.
- **`synthgen.localized`** — a taxonomy-driven route: single-call
  `{instruction, response}` generation, conditioned on
  domain/role/intent/language axes (`synthgen/localized/taxonomy.py`), scored
  by an LLM judge (`generate(--mode judge)`), quality-filtered per-language
  against score thresholds (`synthgen/localized/thresholds.py`,
  `quality_filter.py`), and driven by a resumable top-up loop
  (`synthgen/localized/topup.py`, CLI: `synthgen topup`) that keeps
  generating rounds until each language hits its target survivor count.
  **This route is not yet documented in README.md** — read
  `synthgen/localized/topup.py` and `synthgen/cli.py`'s `topup`/
  `quality-filter`/`decide-thresholds`/`build-localized-prompts` subparsers
  directly if you need it.

Both routes go through the same `Backend` protocol
(`synthgen/backends/base.py`: `async chat(client, *, messages, sampling) ->
GenResult`) and the same `synthgen.pipeline.generate.run()` runner
(mode-dispatched: `prompt` / `response` / `localized` / `judge`), so fixes to
retry/checkpoint/endpoint-pool logic apply to both.

## Non-obvious things that will bite you

- **Judge sampling.** `Qwen/Qwen3.6-27B` (the judge model used so far) is a
  *reasoning* model. Judge calls need `temperature=0.0, top_p=1.0` and, on
  the vLLM backend, `chat_template_kwargs={"enable_thinking": False}` — else
  it burns `max_tokens` on a `<think>` block and the JSON gets truncated
  before it appears (`unparseable_score`). This is handled centrally by
  `_sampling_for()` in `synthgen/pipeline/generate.py`, keyed on
  `mode == "judge"`. If you add a new judge model or backend, extend that
  function rather than special-casing callers.
- **`decide-thresholds` at small n.** It computes a percentile-based
  per-language score cut from labeled pilot data. With ~10 labeled rows per
  language, `int(percentile * (n-1))` degenerates to picking the minimum
  score, i.e. the "threshold" is untrustworthy noise. Until this is fixed in
  `synthgen/localized/thresholds.py`, prefer human-reviewed thresholds (e.g.
  eyeball the score distribution across good/medium/bad labels) over trusting
  its output verbatim, especially before a large production run.
- **vLLM endpoint pool.** Backends read `*.endpoint` files from a directory
  (one `host:port` per file) and re-poll every ~30s, so servers that come up
  late are picked up automatically and deleting a file evicts that endpoint.
  Don't hardcode hosts — write the endpoint file from inside the SLURM job
  once the server is confirmed ready (see below).
- **`.claude/` skill tooling is intentionally untracked.** It's used locally
  for research/ideation workflows but must not be pushed to `origin/main` —
  it's unrelated to this package and not something the OELLM team needs in
  the repo. It's globally gitignored; don't `git add -f` it.

## Running on LUMI (`dev-g` partition)

vLLM is served via the Singularity container
`vllm-openai-rocm.sif`, account `project_462001516`. Pattern: start server(s)
in the SLURM script, poll `/v1/models` until ready, write the endpoint file,
*then* run `synthgen` CLI commands against `--endpoints-dir`.
`studies/localized_bootstrap/smoke_pipeline.slurm` is a working (throwaway)
example of this pattern end-to-end, including two GPU-partitioned vLLM
servers (gen + judge) and a `synthgen topup` invocation.

## `studies/`

Study-specific code (validation experiments, one-off scripts, pilot data)
lives under `studies/` and is **not committed to `main`** — it's tracked on a
separate branch. `studies/localized_bootstrap/` in particular validated the
localized-generation + judge approach before it was promoted into
`synthgen.localized`; its `taxonomy.py`/`prompts.py` now import the shared
axes from the package and keep only study-specific extras (bad/medium
example generators used to construct labeled contrast pairs for judge
validation) — don't re-duplicate what's already in the package.

## Layout

```
synthgen/
├── cli.py                    # `synthgen` entrypoint — subcommands for both routes
├── config.py                 # SynthConfig, Paths, language/persona/constraint constants
├── backends/                 # Backend protocol + OpenRouter/vLLM implementations
├── pipeline/                 # topic-based route + shared generate/dedupe runners
│   ├── prompts.py, generate.py, filter.py, verify.py, back_translate.py,
│   │   dedupe.py, qc.py, to_open_instruct.py, topics.py
├── localized/                # taxonomy-driven route
│   ├── taxonomy.py           # domain/role/intent/language axes
│   ├── prompts.py            # generation + judge prompt builders
│   ├── thresholds.py         # decide-thresholds implementation
│   ├── quality_filter.py     # apply per-language thresholds
│   └── topup.py              # resumable generate→judge→filter→dedupe loop
tests/                        # pytest, mirrors the pipeline/localized split
studies/                      # NOT on main — separate branch, see above
```

## Conventions this repo has settled on

- No Claude Code co-author signature in commit messages (explicit repo
  preference — override the harness default here).
- Keep `synthgen.pipeline` and `synthgen.localized` import-clean of each
  other's internals; share via `synthgen.pipeline.generate`/`dedupe`'s
  `mode=` parameter, not by duplicating logic.
- Before trusting a refactor, actually run it (unit tests *and* a real
  backend smoke test) — the judge-sampling regression above was invisible to
  import checks and unit tests alike, and only surfaced under a real GPU run.
