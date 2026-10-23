# How the 35-language run was actually produced

End-to-end record of how the data released as
[`openeurollm/EuroMultiSynth-Instruct`](https://huggingface.co/datasets/openeurollm/EuroMultiSynth-Instruct)
was made: which scripts ran, where, with what settings, and what each stage
produced. Written 2026-10-23, after the release. For the earlier problems
(topup provisioning bugs, judge pass rates) see `FINDINGS.md`;
`../HANDOFF_MN5.md` is the LUMI → MN5 handoff snapshot from 2026-10-19 and is
superseded by this file.

The run spanned two clusters: **generation and the first ~4M judgments on LUMI**
(AMD MI250X), **the rest of judging, filtering, de-duplication and the release
on MareNostrum 5** (BSC, NVIDIA H100, partition `acc`, account `ehpc532`).

## Pipeline at a glance

| # | stage | where | script | job(s) | output | rows |
|---|---|---|---|---|---|---|
| 1 | generate (Gemma 4 31B-it) + judge (Qwen3.6-27B) via `synthgen topup` | LUMI, 16 nodes | `run_production.slurm` | 22406130, 22736864, 22895003 | `loc_gen.jsonl`, `judged_judge.jsonl` | 7,040,000 unique generated; ~4.0M judged |
| 2 | port smoke test | MN5, 1 node `acc_debug` | `smoke_mn5.slurm` | 47548821 | throwaway `_smoke_data/` | — |
| 3 | judge the remaining backlog | MN5, 8 nodes | `run_judge_pass_mn5_sup.slurm` | 47635229 | `judged_judge.jsonl` | 7,037,796 scored rows |
| 4 | quality filter (score ≥ 7) | MN5 (same job) | ↑ step 2/3 | 47635229 | `loc_full.kept.jsonl` | 5,688,783 |
| 5 | exact de-dup | MN5 (same job / repeated in 6) | ↑ step 3/3 | 47635229 | `loc_full.dedup.jsonl` | 5,595,396 |
| 6 | exact + embedding near-dup, within language | MN5, 1 node, 4 GPUs | `run_embed_dedupe_mn5.slurm` | 47805911 | `loc_full.dedup_embed.jsonl` | **4,160,598** |
| 7 | Hugging Face release build | MN5, 1 node `acc_debug` | `build_hf_release.slurm` | 47816337 | `hf_release/synthgen-35lang-if/` | 4,160,598 |
| 8 | upload | MN5 login node `alogin1` | `hf upload-large-folder` (manual) | — | HF repo (private) | — |

All MN5 data lives under `/gpfs/scratch/ehpc532/eit234648/`. The final outputs
are in `oellm_production_35lang_sup/outputs/` (the supervised run's isolated
root, see stage 3); `oellm_production_35lang/` holds the earlier, partially
judged copy transferred from LUMI.

## Models

| role | model | serving |
|---|---|---|
| generator | `google/gemma-4-31B-it` | vLLM, served as `gen`, TP=2, `--max-model-len 16384`, `--gpu-memory-utilization 0.90` |
| judge | `Qwen/Qwen3.6-27B` (reasoning model; judged with `temperature=0`, thinking disabled — see `synthgen/CLAUDE.md`) | vLLM, served as `judge`; TP=2 on LUMI, TP=4 on MN5 |
| near-dup embeddings (31 langs) | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | in-process, GPU |
| near-dup embeddings (mt, ga, eu, is) | `sentence-transformers/LaBSE` | in-process, GPU |

The served names `gen` / `judge` matter: `synthgen` names its output files
after them (`loc_gen.jsonl`, `judged_judge.jsonl`) and resumes by row id from
those files, so keep them identical when resubmitting or moving clusters.

## Stage 1 — generation + judging on LUMI

`run_production.slurm` (LUMI `standard-g`, 16 nodes × 8 GCDs): 12 nodes run
4 gen replicas each (48 total), 4 nodes run 4 judge replicas each (16 total);
each node writes `host:port` endpoint files that `synthgen`'s pool re-polls.
It then runs

```bash
python3 -m synthgen.cli topup --root $DATA_ROOT --target 100000 \
  --thresholds production/35lang_run/thresholds.json --langs <35 langs> \
  --gen-backend vllm --gen-models gen   --gen-endpoints-dir $EP/gen \
  --judge-backend vllm --judge-model judge --judge-endpoints-dir $EP/judge \
  --max-rounds 30 ...
```

Generation is `synthgen.localized`: one call writes an instruction *and* its
response directly in the target language, conditioned on a task domain
(18 domains), optionally a user persona (8 personas, ~13% of rows) and an
intent. Languages: `es fr de it pt pl nl cs ro el uk bg da et fi ga hr hu lv
lt mt sk sl sv ca eu gl bs ka mk sq sr tr is no`.

What happened: three 48 h submissions. The first finished generation but timed
out at the start of judging; the second reached ~71% of judging; during the
third, the topup provisioning bugs in `FINDINGS.md` were found and fixed
(`bb879f8`), and the job was cancelled to continue on MN5. `topup`'s
filter/dedupe step never ran on LUMI. Hand-over state (rsynced to MN5):
`loc_gen.jsonl` 7,575,024 lines = **7,040,000 unique ids** (160k or 320k per
language; the rest are repeated ids from resumes) and ~4.0M judgments.

## Stage 2 — porting to MN5 (smoke test)

`smoke_mn5.slurm`: one `acc_debug` node, 1 gen replica (GPUs 0,1) + 1 judge
replica (GPUs 2,3), tiny `topup` (`es fr`, target 50, 2 rounds) into a
throwaway root. Passed end to end (gen → judge → filter → dedupe).

MN5-specific settings, all baked into the `*_mn5.slurm` scripts:

- `singularity exec --nv` with `vllm-v0.22.1.sif`, `CUDA_VISIBLE_DEVICES`
  (not `ROCR_…`), `--bind /gpfs/scratch/ehpc532`.
- `SINGULARITYENV_CC=gcc`, `VLLM_DISABLE_COMPILE_CACHE=1` (the host's Intel
  compiler env otherwise leaks into torch.compile).
- `unset HTTP_PROXY HTTPS_PROXY …` at the top of every job — the login
  proxy otherwise leaks in and hijacks the localhost endpoint traffic.
- No `--mem` / `--mem-per-cpu` (rejected on MN5; memory scales with
  `--cpus-per-task`). The `ehpc532` account is ACC-only (`acc_debug`,
  `acc_ehpc`); there is no `gp` access.
- Compute nodes have no internet: models and Python extras are fetched on the
  login node beforehand.

## Stage 3 — judging the backlog on MN5 (supervised)

Only judging was needed (generation volume was already enough; the low-yield
languages are a judge-pass-rate problem, see `FINDINGS.md`), so instead of
`topup` this uses a judge-only chain. The script that finished is
**`run_judge_pass_mn5_sup.slurm`**, `sbatch -N 8`, 13 h 57 min:

1. `synthgen.cli generate --mode judge --in loc_gen.jsonl` — resumable; skips
   ids already in `judged_judge.jsonl`.
2. `synthgen.cli quality-filter --thresholds thresholds.json` (`{"global": 7}`)
   → `loc_full.kept.jsonl`.
3. `synthgen.cli dedupe` (exact only) → `loc_full.dedup.jsonl`.

Settings that made it work, learned the hard way:

- **1 judge replica per node at TP=4** (`--disable-custom-all-reduce`),
  **32 concurrent requests per replica**. 256/replica (the LUMI value)
  collapsed with ~89% connection errors (`run_judge_pass_mn5.slurm`, job
  47549930); the `debug_conntest_mn5.slurm` / `debug_c32_mn5.slurm` sweeps
  found the knee between 32 and 64.
- **A per-node supervisor around vLLM.** Under sustained load the judge
  engines die with an NCCL collective timeout
  (`RPC call to sample_tokens timed out`), roughly one every 6 min at 2×TP=2
  per node and every ~21 min at 1×TP=4; vLLM cannot recover from that by
  itself. The supervisor in this script watches for process exit, the death
  markers in the log, or 3 failed health checks; it then removes the node's
  endpoint file, kills everything on the GPUs, relaunches vLLM on the same
  port and rewrites the endpoint file, and the pool picks the replica up
  again within ~30 s. Job 47635229 logged 89 revives (≈45 engine deaths)
  and still averaged ~70 req/s.
- **Isolated data root.** The supervised job ran alongside an earlier,
  degrading unsupervised job, so it uses its own `endpoints_sup/` and
  `DATA_ROOT=oellm_production_35lang_sup`, symlinks the shared
  `loc_gen.jsonl`, and seeds `judged_judge.jsonl` from the main root only if
  absent. Error rows from failed runs were stream-filtered out of
  `judged_judge.jsonl` before reuse.

Result (`logs/quality_filter_47635229.json`, `logs/dedupe_47635229.json`):
7,037,796 rows accounted for → **5,688,783 kept (score ≥ 7)**,
1,347,250 dropped as low-score, 1,763 with no valid score after retries
(2,204 of the 7.04M ids never got a usable judgment). Exact dedupe:
−93,387 → 5,595,396. Weakest languages after filtering: ga 26k, eu 51k,
mt 54k, et 58k, lv 74k, hu 77k.

## Stage 4 — embedding near-duplicate removal (within language)

`run_embed_dedupe_mn5.slurm` → `run_embed_dedupe.py` → `synthgen.pipeline.dedupe.run(embedding_dedupe=True, ...)`,
1 node with 4 GPUs (`acc_ehpc`), 22 min:

- Input `loc_full.kept.jsonl` (5,688,783); exact pass again (→ 5,595,396),
  then the near-dup pass → **`loc_full.dedup_embed.jsonl`, 4,160,598 rows**
  (−1,434,798). Report: `loc_full.dedup_embed.report.json`.
- Strictly **within each language** — never across languages. Prompts are
  embedded; a prompt is dropped if its cosine similarity to an earlier *kept*
  prompt of the same language is ≥ threshold (greedy, first occurrence wins).
  `near_dup.embedding_dedupe_gpu` does this with blocked matmuls, one worker
  process per GPU, largest language buckets first; it is tested equal to the
  CPU path (`_gpu_dedupe_equiv_test.*`).
- **MiniLM at 0.97** for 31 languages; **LaBSE at 0.96** for `mt ga eu is`
  (`embed_dedupe_lang_overrides.json`), because MiniLM does not cover them and
  flagged ~89% of their rows as duplicates. LaBSE 0.96 was calibrated on de/hr
  to remove the same share as MiniLM 0.97. Thresholds come from
  `_dedupe_threshold_sweep.*` / `_dedupe_labse_sweep.slurm`, which print
  example pairs per similarity band: at these values removed pairs are
  rewordings of the same request.
- Python extras (sentence-transformers, scipy, scikit-learn, …) are not in
  the vLLM image; they were `pip install --target <overlay> --no-deps` on the
  login node, inside the SIF, and put on `PYTHONPATH`
  (`/gpfs/scratch/ehpc532/eit234648/st_overlay`). The CPU path
  (`--embedding-workers`) works but was far too slow at this scale (MiniLM on
  CPU ~18 texts/s/worker).

## Stage 5 — Hugging Face release

`build_hf_release.slurm` → `build_hf_release.py` (CPU, `acc_debug`, 4 min;
needs pyarrow from `/gpfs/scratch/ehpc532/eit234648/hf_overlay`):

- Streams `loc_full.dedup_embed.jsonl` into
  `data/<lang>/train-00000-of-00001.parquet` (zstd), one config per language
  plus an `all` default config.
- Columns: `id`, `language`, `messages` (`[user, assistant]`), `domain`,
  `persona` (null for ~87%), `judge_score` (int8, 7–10). Drops `model`,
  `intent`, `is_local`.
- Verifies row counts against the dedupe report, writes `stats.json` and the
  dataset card (`README.md`). Language codes in the card YAML are quoted —
  unquoted `no` (Norwegian) parses as `false` and breaks the dataset viewer.
- License **Apache-2.0**, because every model involved is Apache-2.0
  (Gemma 4 31B-it, Qwen3.6-27B, MiniLM, LaBSE); the card credits Gemma 4 as the
  author of all text.

Upload: MN5 has no direct internet (`glogin1`, `alogin1`, `transfer1` all
blocked). It went from `alogin1` through an SSH reverse tunnel to a laptop
proxy (`RemoteForward 18888 localhost:8888`):

```bash
export HTTPS_PROXY=http://localhost:18888 HF_HUB_DISABLE_XET=1
hf upload-large-folder openeurollm/EuroMultiSynth-Instruct \
  /gpfs/scratch/ehpc532/eit234648/hf_release/synthgen-35lang-if \
  --repo-type dataset --num-workers 4
```

Plain `hf upload` (Xet, 64 MB chunks) kept timing out through the tunnel;
`upload-large-folder` without Xet took ~3 min for 4.85 GB and is resumable.
All 35 Parquet files were checked against the local copies by sha256 after
upload.

## Final numbers

| step | rows |
|---|---|
| generated (unique ids) | 7,040,000 |
| with a usable judge score | 7,037,796 |
| score ≥ 7 | 5,688,783 |
| after exact dedup | 5,595,396 |
| after within-language near-dup | **4,160,598** |

Per-language counts, domain and score distributions are in the dataset card
and `stats.json`. Judge scores of released rows: 10 → 78%, 9 → 11%,
8 → 4%, 7 → 7%.

## Scripts in this folder

| file | used for |
|---|---|
| `run_production.slurm` | stage 1 (LUMI) |
| `run_production_mn5.slurm` | MN5 port of the `topup` job; prepared, not used for this release |
| `smoke_mn5.slurm` | stage 2 |
| `run_judge_pass_mn5.slurm` | first MN5 judge pass, no supervisor (failed / degraded; kept for reference) |
| `run_judge_pass_mn5_sup.slurm` | stage 3 — the run that finished |
| `debug_conntest_mn5.slurm`, `debug_c32_mn5.slurm` | concurrency sweeps behind the 32/replica setting |
| `run_embed_dedupe_mn5.slurm`, `run_embed_dedupe.py`, `embed_dedupe_lang_overrides.json` | stage 4 |
| `_dedupe_threshold_sweep.*`, `_dedupe_labse_sweep.slurm`, `_dedupe_dryrun.*`, `_gpu_dedupe_equiv_test.*`, `_override_check.py` | threshold choice and correctness checks for stage 4 |
| `build_hf_release.py`, `build_hf_release.slurm` | stage 5 |
| `thresholds.json` | judge cut-off (`{"global": 7}`) |

The multi-node scripts generate their `_per_node_*.sh` / `_in_container_*.sh`
helpers at submit time; those generated files are not tracked.
